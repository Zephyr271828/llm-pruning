import os
import pdb
import sys
import math
import json
import torch
import argparse
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from utils import get_ppl, get_acc, set_seed
from prune_depth import depth_prune_BI, depth_prune_score
from prune_width import width_prune

def main(args):
    
    # NOTE print config
    print(json.dumps(vars(args), indent=2))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16, 
        device_map='auto', 
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(f'loaded model and tokenizer from {args.model_path}')
    
    model_name = os.path.basename(args.model_path).replace('_', '-')
    if args.mode in ['depth', 'bi']:
        exp_name = f"{model_name}_{args.mode}_task_{args.prune_task}_nlayers_{args.num_layers}"
    else:
        exp_name = f"{model_name}_{args.mode}_task_{args.prune_task}_hidden_size_{args.hidden_size}_ffn_hidden_size_{args.ffn_hidden_size}"

    if args.prune_task in ['c4', 'wikitext', 'wikitext2', 'cnn_dailymail', 'pg19']:
        exp_name = f"{exp_name}_calib_size_{args.calib_size}_seqlen_{args.seq_len}"
        prune_scorer = partial(
            get_ppl, 
            task=args.prune_task, 
            batch_size=1,
            calib_size=args.calib_size, 
            max_length=args.seq_len
        )
    elif args.prune_task in ['winogrande']:
        exp_name = f"{exp_name}_calib_size_{args.calib_size}_fewshot_{args.num_fewshot}"
        prune_scorer = partial(
            get_acc, 
            task=args.prune_task, 
            num_fewshot=args.num_fewshot,
            acc_key='acc,none',
            limit=args.calib_size,
        )
    else:
        raise NotImplementedError(f"Unsupported prune task: {args.prune_task}")
    
    base_line = prune_scorer(model, tokenizer)
    print("Before pruning:")
    print(f"task: {args.prune_task}, score: {base_line}")
    
    data = {
        "model_name": model_name,
        "prune_task": args.prune_task,
        "base_line": base_line,
    }
    
    if args.mode.lower() == 'bi':
        assert args.num_layers is not None, "num_layers must be specified for depth pruning"
        all_scores, layer_idx_to_drop = depth_prune_BI(model, tokenizer, prune_scorer, args)
        
        data.update({
            # "total_layers": model.config.num_hidden_layers,
            "num_layers": args.num_layers,
            "bi_scores": all_scores,
            "layer_idx_to_drop": layer_idx_to_drop,
        })
        
    elif args.mode.lower() == 'depth':
        assert args.num_layers is not None, "num_layers must be specified for depth pruning"
        all_scores, layer_idx_to_drop = depth_prune_score(model, tokenizer, prune_scorer, base_line, args)
        
        data.update({
            # "total_layers": model.config.num_hidden_layers,
            "num_layers": args.num_layers,
            "base_line": base_line,
            "scores": all_scores,
        })
            
    elif args.mode.lower() == 'width':
        assert args.hidden_size is not None or args.ffn_hidden_size is not None, "hidden_size or ffn_hidden_size must be specified for width pruning"
        width_prune(model, tokenizer, prune_scorer, args)
    
    score = prune_scorer(model, tokenizer)
    print("After pruning:")
    print(f"task: {args.prune_task}, score: {score}")
    
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_path = os.path.join(args.log_dir, f"{exp_name}.json")
        data.update({
            "final_score": score,
        })
    
    with open(log_path, 'w') as f:
                json.dump(data, f, indent=2)
    
    if args.save_dir is not None:
        save_path = os.path.join(args.save_dir, exp_name)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", type=str, choices=['depth', 'width', 'bi']
    )
    
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--prune_task", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    
    # pruning config
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--ffn_hidden_size", type=int, default=None)
    
    # calib data config
    parser.add_argument("--calib_size", type=int, default=1024)
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--num_fewshot", type=int, default=0)
    
    args = parser.parse_args()
    
    set_seed(42)
    main(args)