import os
import sys

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

def str2bool(v):
    if isinstance(v, bool):
        return v
    val = v.lower()
    if val in ("yes", "true", "t", "y", "1"):
        return True
    elif val in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (yes/no/true/false)")

PPL_TASKS = [
    "c4",
    "wikitext",
    "wikitext2",
    "cnn_dailymail",
    # "dclm"
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

def get_ppl_enc(task, tokenizer):
    if task == 'wikitext':
        dataset = load_dataset(
            "wikitext", 
            "wikitext-103-v1", 
            split="train", 
        )
        text_column = "text"
        testenc = tokenizer.encode("\n\n".join(dataset[:32768][text_column]), return_tensors='pt', add_special_tokens=True)
    elif task == 'wikitext2':
        dataset = load_dataset(
            "wikitext", 
            "wikitext-2-raw-v1", 
            split="train", 
        )
        text_column = "text"
        testenc = tokenizer.encode("\n\n".join(dataset[:32768][text_column]), return_tensors='pt', add_special_tokens=True)
    elif task == 'cnn_dailymail':
        dataset = load_dataset(
            "cnn_dailymail", 
            "3.0.0", 
            split="train", 
        )
        text_column = "article"
        testenc = tokenizer.encode(" ".join(dataset[:8000][text_column]), return_tensors='pt', add_special_tokens=True)
    elif task == 'c4':
        dataset = load_dataset(
            "allenai/c4", 
            data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, 
            split="train", 
            verification_mode="no_checks", 
        )
        text_column = "text"
        testenc = tokenizer.encode(" ".join(dataset[:8000][text_column]), return_tensors='pt', add_special_tokens=True)
    # elif task == 'dclm':
    #     dataset = load_dataset(
    #         'json', 
    #         # data_files="/vast/yx3038/datasets/dclm/dclm_baseline_1.0_shuffled/dclm_baseline_1.0.val.jsonl",
    #         # data_files="/vast/yx3038/datasets/dclm/dclm_baseline_1.0_shuffled/dclm_baseline_1.0.chunk.00.jsonl",
    #         data_files="/n/fs/vision-mix/yx1168/datasets/dclm/dclm_baseline_1.0.val.jsonl",
    #         split="train",
    #         verification_mode="no_checks",
    #     )
    #     text_column = "text"
    #     testenc = tokenizer.encode(" ".join(dataset[:1400][text_column]), return_tensors='pt', add_special_tokens=True)
    else:
        raise NotImplementedError(f"Unsupported task: {task}")
    return testenc

def get_ppl(
    model, 
    tokenizer, 
    tasks,
    batch_size: int = 1,
    calib_size: int = 256,
    max_length: int = 8192,
):
    ppl_res = {}
    for task in tasks:
        testenc = get_ppl_enc(task, tokenizer)
        model.eval()
        tot_loss = 0
        tot_tokens = 0
        bs = batch_size
        seq_len = min(max_length, model.config.max_position_embeddings)
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

def get_acc(
        model, 
        tokenizer, 
        tasks, 
        limit=100000
    ):
    
    print("tasks to evaluate:")
    print(json.dumps(tasks, indent=2))
    
    lm_eval_model = models.huggingface.HFLM(
        pretrained=model, 
        tokenizer=tokenizer,
        generation_kwargs={
            "do_sample": True,
            "temperature": 0.2,
            "top_p": 0.95,
        }
    )
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
            limit=limit,
        )
        
        print(res['results'])
        acc_key = cfg["acc_key"]
        if acc_key is not None:
            acc_res[task] = res['results'][task][acc_key]

    return acc_res

def main(args):
    hf_path = args.hf_path  
    tokenizer_path = args.tokenizer_path
    
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    print(f"loaded hf model from {hf_path}")
    
    model = model.to(torch.bfloat16).to('cuda')
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
    )

    ppl_tasks = [t for t in PPL_TASKS if t in args.tasks]
    ppl_res = get_ppl(model, tokenizer, calib_size=256, tasks=ppl_tasks)

    acc_tasks = [t for t in ACC_TASKS if t['name'] in args.tasks]
    acc_res = get_acc(model, tokenizer, limit=args.limit, tasks=acc_tasks)
    
    if args.eval_noise:
        for noise_scale in [0.001, 0.01, 0.1]:
            print(f"running eval with noise scale {noise_scale}")
            get_ppl(model, tokenizer, tasks=PPL_TASKS, noise_scale=noise_scale)
            get_acc(
                model, 
                tokenizer, 
                tasks={k:v for k, v in TASK_CONFIG.items() if k in ['winogrande', 'arc_challenge', 'arc_easy', 'hellaswag']},
                noise_scale=noise_scale
            )
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_path", type=str, default=None, help="Path to HF checkpoint (.pth)")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to Hugging Face config/tokenizer")
    
    parser.add_argument("--tasks", type=lambda x: [] if not x else x.split(","), default=[])
    parser.add_argument("--limit", type=int, default=100000, help="Limit the number of samples per task")

    args = parser.parse_args()
    
    print(json.dumps(vars(args), indent=2))

    set_seed(42)
    main(args)