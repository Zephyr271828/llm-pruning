import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_dir, '..', 'lm-evaluation-harness'))
sys.path.append(os.path.join(cur_dir, '..', 'transformers', 'src'))

import json
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse
import numpy as np

from tqdm import tqdm
from functools import partial
from datasets import load_dataset, load_from_disk
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval import evaluator, tasks, models

from fms.models.llama import LLaMA
from fms.models.hf import to_hf_api
from fms_fsdp.utils.config_utils import get_model_config

PPL_TASKS = [
    "c4",
    "wikitext",
    "wikitext2",
    "cnn_dailymail",
    "dclm"
]

ACC_TASKS = [
    {
        "name": "winogrande",
        "num_fewshot": 0,
        "acc_key": "acc,none",
    },
    {
        "name": "winogrande",
        "num_fewshot": 5,
        "acc_key": "acc,none",
    },
    {
        "name": "arc_easy",
        "num_fewshot": 0,
        "acc_key": "acc_norm,none",
    },
    {
        "name": "arc_challenge",
        "num_fewshot": 0,
        "acc_key": "acc_norm,none",
    },
    {
        "name": "arc_challenge",
        "num_fewshot": 25,
        "acc_key": "acc_norm,none",
    },
    {
        "name": "hellaswag",
        "num_fewshot": 0,
        "acc_key": "acc_norm,none",
    },
    {
        "name": "hellaswag",        
        "num_fewshot": 10,
        "acc_key": "acc_norm,none",
    },
    {
        "name": "truthfulqa_mc1",
        "num_fewshot": 0,
        "acc_key": "acc,none",
    },
    {
        "name": "truthfulqa_mc2",
        "num_fewshot": 0,
        "acc_key": "acc,none",
    },
    {
        "name": "piqa",
        "num_fewshot": 0,
        "acc_key": "acc_norm,none",
    },
    {
        "name": "sciq",
        "num_fewshot": 0,
        "acc_key": "acc,none",
    },
    {
        "name": "boolq",
        "num_fewshot": 0,
        "acc_key": "acc,none",
    },
    {
        "name": "anli_r1",
        "num_fewshot": 0,
        "acc_key": None,
    },
    {
        "name": "anli_r2",
        "num_fewshot": 0,
        "acc_key": None,
    },
    {
        "name": "anli_r3",
        "num_fewshot": 0,
        "acc_key": None,
    },
    {
        "name": "openbookqa",
        "num_fewshot": 0,
        "acc_key": None,
    },
    {
        "name": "rte",
        "num_fewshot": 0,
        "acc_key": None,
    },
    {
        "name": "mmlu",
        "num_fewshot": 0,
        "acc_key": None,
    },
    {
        "name": "mmlu",
        "num_fewshot": 5,
        "acc_key": None,
    },
    {
        "name": "record",
        "num_fewshot": 0,
        "acc_key": None,
    },
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def smart_load_dataset(
    dataset_name: str,
    config: str = None,
    data_files: dict = None,
    split: str = None,
    cache_dir: str = None,
    **kwargs
):
    dataset_fs_name = dataset_name.replace("/", "___")
    dataset_base_dir = os.path.join(cache_dir, dataset_fs_name)
    if config is not None:
        dataset_base_dir = os.path.join(dataset_base_dir, config)
    # if data_files is not None:
    #     pass
    # else:
    return load_from_disk(os.path.join(dataset_base_dir, split))

def get_ppl_enc(task, tokenizer):
    if task == 'wikitext':
        dataset = load_dataset(
            "wikitext", 
            "wikitext-103-v1", 
            split="train", 
        )
        # dataset = load_from_disk("/scratch/gpfs/yx1168/datasets")
        text_column = "text"
        testenc = tokenizer.encode("\n\n".join(dataset[:32768][text_column]), return_tensors='pt')
    elif task == 'wikitext2':
        dataset = load_dataset(
            "wikitext", 
            "wikitext-2-raw-v1", 
            split="train", 
        )
        # dataset = load_from_disk("/scratch/gpfs/yx1168/datasets")
        text_column = "text"
        testenc = tokenizer.encode("\n\n".join(dataset[:32768][text_column]), return_tensors='pt')
    elif task == 'cnn_dailymail':
        dataset = load_dataset(
            "cnn_dailymail", 
            "3.0.0", 
            split="train", 
        )
        text_column = "article"
        testenc = tokenizer.encode(" ".join(dataset[:8000][text_column]), return_tensors='pt')
    elif task == 'c4':
        dataset = load_dataset(
            "allenai/c4", 
            data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, 
            split="train", 
            verification_mode="no_checks", 
        )
        text_column = "text"
        testenc = tokenizer.encode(" ".join(dataset[:8000][text_column]), return_tensors='pt')
    elif task == 'dclm':
        dataset = load_dataset(
            'json', 
            # data_files="/vast/yx3038/datasets/dclm/dclm_baseline_1.0_shuffled/dclm_baseline_1.0.val.jsonl",
            # data_files="/vast/yx3038/datasets/dclm/dclm_baseline_1.0_shuffled/dclm_baseline_1.0.chunk.00.jsonl",
            data_files="/n/fs/vision-mix/yx1168/datasets/dclm/dclm_baseline_1.0.val.jsonl",
            split="train",
            verification_mode="no_checks",
        )
        text_column = "text"
        testenc = tokenizer.encode(" ".join(dataset[:1400][text_column]), return_tensors='pt')
    else:
        raise NotImplementedError(f"Unsupported task: {task}")
    return testenc

def get_ppl(
    fms_model, 
    tokenizer, 
    tasks,
    batch_size: int = 1,
    calib_size: int = 256,
    max_length: int = 8192
):
    try:
        model = to_hf_api(fms_model)
    except:
        model = fms_model
    ppl_res = {}
    for task in tasks:
        testenc = get_ppl_enc(task, tokenizer)
        model.eval()
        tot_loss = 0
        tot_tokens = 0
        bs = batch_size
        seq_len = max_length
        nsamples = min(testenc.numel() // seq_len, calib_size)
        device = model.device
        with torch.no_grad():
            for i in tqdm(range(0, nsamples, bs), desc=f"Evaluating PPL for {task}"):
                j = min(i + bs, nsamples)
                inputs = testenc[:,(i * seq_len):(j * seq_len)].to(device)
                inputs = inputs.reshape(j - i, seq_len)
                
                outputs = model(inputs)
                if hasattr(outputs, "logits"):
                    lm_logits = outputs.logits
                else:
                    lm_logits = outputs
                
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = inputs[:, 1:]
                
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
                
                tot_loss += loss.item() * seq_len * (j - i)
                tot_tokens += seq_len * (j - i)
                
            ppl_res[task] = torch.exp(torch.tensor(tot_loss / tot_tokens)).item()
            print(f"{task} ppl: {ppl_res[task]}")
                
    return ppl_res

def get_acc(fms_model, tokenizer, tasks, task_range=[]):
    try:
        model = to_hf_api(fms_model)
    except:
        model = fms_model
    lm_eval_model = models.huggingface.HFLM(
        pretrained=model, 
        tokenizer=tokenizer,
        generation_kwargs={
            "do_sample": True,
            "temperature": 0.2,
            "top_p": 0.95,
        }
    )
    
    if task_range:
        tasks = (cfg for cfg in tasks if cfg["name"] in task_range)
    print("tasks to evaluate:")
    print(json.dumps(tasks, indent=2))
    
    acc_res = {}
    for cfg in tasks:
        task = cfg["name"]
        print("evaluating with config:")
        print(json.dumps(cfg, indent=2))
        res = evaluator.simple_evaluate(
            model=lm_eval_model,
            tasks=[task],
            num_fewshot=cfg["num_fewshot"],
            max_batch_size=64,
            log_samples=True,
            confirm_run_unsafe_code=True,
            # limit=1,
        )
        
        print(res['results'][task])
        acc_key = cfg["acc_key"]
        if acc_key is not None:
            acc_res[task] = res['results'][task][acc_key]

    return acc_res

def main(args):
    hf_path = args.hf_path
    fms_path = args.fms_path
    tokenizer_path = args.tokenizer_path
    model_variant = args.model_variant
    
    assert (hf_path is None) ^ (fms_path is None)
    
    # if fms_path is not None:
    assert model_variant is not None
    llama_config = get_model_config(model_variant)
    model = LLaMA(llama_config)
    state_dict = torch.load(fms_path, weights_only=False)
    model_state_dict = {}
    for k, v in state_dict['model_state'].items():
        if k.startswith("_orig_mod."):
            newk = k[len("_orig_mod."):]
        else:
            newk = k
        model_state_dict[newk] = v
    
    model.load_state_dict(model_state_dict)
    print(f"loaded fms model from {fms_path}")
    # else:
    #     model = AutoModelForCausalLM.from_pretrained(hf_path)
    #     print(f"loaded hf model from {hf_path}")
    
    model = model.to(torch.bfloat16).to('cuda')
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
    )

    ppl_res = get_ppl(model, tokenizer, tasks=PPL_TASKS)
    # ppl_res = get_ppl(model, tokenizer, tasks=['wikitext'])
    print(ppl_res)

    acc_res = get_acc(model, tokenizer, tasks=ACC_TASKS)
    # acc_res = get_acc(model, tokenizer, tasks=['winogrande'])
    print(acc_res)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_path", type=str, default=None, help="Path to HF checkpoint (.pth)")
    parser.add_argument("--fms_path", type=str, default=None, help="Path to FMS checkpoint (.pth)")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to Hugging Face config/tokenizer")
    parser.add_argument("--model_variant", type=str, default=None, help="FMS model variant name (e.g., llama3_4b_depth)")

    args = parser.parse_args()

    set_seed(42)
    main(args)