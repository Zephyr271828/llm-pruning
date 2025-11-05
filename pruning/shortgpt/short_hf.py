import os
import json
import torch
import argparse
import numpy as np


from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)

from tqdm import tqdm
from typing import List, Optional
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import default_data_collator, Trainer, TrainingArguments

from metrics import *


class ShortHFModel():

    def __init__(self, model_name: str, layers_path: str, n_prune_layers: Optional[int] = None):
        """
        HuggingFace Model Wrapper

        Args:
            model_name (str): HuggingFace model name
            layers_path (str): String in dot notation demonstrating how to access layers of the model. Ex: "model.layers"
            (Optional) n_prune_layers (int): Number of layers to prune. Defaults to None.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        # self.model.params = self.model.to_fp16(self.model.params)
        self.model.to("cuda")

        modules = layers_path.split(".")
        mod = self.model
        for m in modules:
            mod = getattr(mod, m)
        self.layers = mod

        self.n_prune_layers = n_prune_layers
        self.importances = [0 for _ in self.layers]  # layer-wise importance scores

    def remove_layers(
        self,
        layers_to_remove: Optional[List[int]] = [],
        angular: Optional[bool] = False
    ):
        if angular:
            assert self.importances, "Need to compute importances with eval_importance()"
            assert self.n_prune_layers, "Need number of layers to prune, set `n_prune_layers`"
            start_layer = np.argsort(np.array(self.importances[:-self.n_prune_layers+1]))[0]
            layers_to_remove = list(range(start_layer, start_layer + self.n_prune_layers))
        elif not layers_to_remove and self.n_prune_layers:
            assert self.importances, "Need to compute importances with eval_importance()"
            layers_to_remove = np.argsort(np.array(self.importances))[:self.n_prune_layers].tolist()

        # remove layers in reverse to avoid indexing errors
        for layer_idx in sorted(layers_to_remove, reverse=True):
            try:
                del self.layers[layer_idx]
            except IndexError:
                print(f"layer {layer_idx} does not exist, function may have already been called")
                return []
        
        return layers_to_remove
    
    def compute_bi(self, hiddens: List[torch.Tensor], angular: bool):
        n = 1
        if angular:
            assert self.n_prune_layers is not None, "Set number of layers to prune to use angular importance"
            n = self.n_prune_layers

        for i in range(len(hiddens) - n):
            in_hidden = hiddens[i]
            out_hidden = hiddens[i+n]
            if angular:
                # use only last token for angular distance as described in section 3.2
                # https://arxiv.org/pdf/2403.17887.pdf
                in_hidden = in_hidden[:,-1:]
                out_hidden = out_hidden[:,-1:]
            
            self.importances[i] += block_influence(
                in_hidden,
                out_hidden,
                angular=angular
            ).sum().cpu().item()

    @torch.inference_mode()
    def eval_importance(
        self,
        prompts: List[str],
        max_seq_len: int,
        stride: int = 256,
        max_gen_len: int = 0,
        temperature: float = 0.6,
        top_p: float = 0.9,
        angular: Optional[bool] = False
    ):
        """
        Computes layer-wise importances over input texts.

        NOTE: ShortGPT paper performs no generation during importance computation, which suggests a `max_gen_len`= 0.

        Args:
            prompts (List[str]): List of prompts.
            max_seq_len (int): Maximum sequence length for model input, the sliding window size.
            (Optional) stride (int): Number of tokens to skip/shift between each window inference.
            (Optional) max_gen_len (int): Maximum length of the generated text sequence.
            (Optional) temperature (float): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            (Optional) top_p (float): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            (Optional) angular (bool): Whether to ues angular distance. Defaults to False.

        Returns:
            None
        """
        prompt_tokens = self.tokenizer(
            prompts,
            padding=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = prompt_tokens.input_ids
        attn_mask = prompt_tokens.attention_mask

        max_prompt_len = max(len(t) for t in input_ids)

        # authors use a sliding window of size 1024 with a shift of 256
        for start in range(0, max_prompt_len, stride):
            seq_ids = (attn_mask.sum(dim=-1) > start).nonzero().squeeze()
            seq_ids = seq_ids.unsqueeze(0) if seq_ids.dim() == 0 else seq_ids  # ensure 2d
            inputs = input_ids[seq_ids, start:start+max_seq_len]
            attn = attn_mask[seq_ids, start:start+max_seq_len]

            if max_gen_len == 0:
                outputs = self.model(
                    input_ids=inputs.to("cuda"),
                    attention_mask=attn.to("cuda"),
                    output_hidden_states=True,
                )
            else:
                outputs = self.model.generate(
                    input_ids=inputs.to("cuda"),
                    attention_mask=attn.to("cuda"),
                    max_new_tokens=max_gen_len, 
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
            
            self.compute_bi(outputs.hidden_states, angular=angular)

        return

def main(args):
    
    data = load_dataset("emozilla/pg19", split="validation")  # authors sample 10,000 texts to compute block influences
    dataloader = DataLoader(
        data,
        batch_size=1,
        shuffle=True,
    )
    
    MAX_SEQ_LEN = 1024
    short_model = ShortHFModel(
        model_name=args.model_path,
        layers_path="model.layers",
        n_prune_layers=args.n_prune_layers
    )
    
    for i, batch in enumerate(tqdm(dataloader)):
        prompts = batch['text']

        short_model.eval_importance(
            prompts=prompts,
            max_seq_len=MAX_SEQ_LEN,
            stride=256,
            max_gen_len=0
        )
        if args.debug:
            break
        
    short_model.remove_layers()
    
    for layer_idx, module in enumerate(short_model.layers):
        module.self_attn.layer_idx = layer_idx
        
    pruned_dir = os.path.join(args.output_dir, 'shortgpt', 'llama2_4.5b')
    # pruned_dir = "/n/fs/vision-mix/yx1168/pruning/ckpts/shortgpt/llama2_7b_pruned"
    
    print(f"Saving pruned model to {pruned_dir}...")
    short_model.model.save_pretrained(pruned_dir)
    short_model.tokenizer.save_pretrained(pruned_dir)
    print(f"Pruned model & tokenizer saved to {pruned_dir}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--output_dir")
    parser.add_argument("--n_prune_layers", type=int, default=9)
    parser.add_argument("--debug", action='store_true')
   
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=2))
    main(args)
        