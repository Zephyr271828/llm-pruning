import os
import pdb
import sys
import fire
import math
import json
import torch
import argparse
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from fms.models import get_model
from fms.models.llama import LLaMA
from fms.models.hf import to_hf_api
from fms_fsdp.utils.config_utils import get_model_config

from eval import set_seed, get_ppl, get_acc, PPL_TASKS, TASK_CONFIG
from prune_depth import drop_layers, depth_prune_BI, depth_prune_score
from prune_width import width_prune


def print_config(args):
    print("=" * 50)
    print("🚀 Run Configuration")
    print("=" * 50)
    for key, value in vars(args).items():
        print(f"{key:30}: {value}")
    print("=" * 50)
    
def run_eval(model, tokenizer, args):
    ppl_res = get_ppl(
        model, 
        tokenizer, 
        # calib_size=args.calib_size, 
        calib_size=256,
        max_length=8192,
        tasks=[task for task in args.eval_tasks if task in PPL_TASKS]
    )
    print(ppl_res)
    acc_res = get_acc(model, tokenizer, tasks=[task for task in args.eval_tasks if task in TASK_CONFIG.keys()])
    print(acc_res)
    ppl_res.update(acc_res)
    return ppl_res

def save_model(model, args):
    model_type = 'llama'
    
    if args.drop_layers is not None:
        prune_method = 'drop'
        fms_save_dir = os.path.join(args.save_dir, f'{model_type}_drop_{args.drop_layers[0]}-{args.drop_layers[-1]}_fms/')
    else:
        prune_method = args.mode.lower()
        prune_task = args.prune_task
        fms_save_dir = os.path.join(args.save_dir, f'{model_type}_{prune_method}_{prune_task}_open_source')
        
    fms_save_path = os.path.join(fms_save_dir, 'ckpt.pth')
    os.makedirs(fms_save_dir, exist_ok=True)
    save_dict = {
        "model_state": model.state_dict(),  
        "step": 0,                      
    }
    torch.save(save_dict, fms_save_path)
    print(f"fms model saved to {fms_save_path}")

def main(args):
    
    # NOTE print config
    print_config(args)
    
    assert (args.fms_path is None) ^ (args.hf_path is None)
    
    if args.fms_path is not None:
        # NOTE load fms model
        llama_config = get_model_config(args.model_variant)
        model = LLaMA(llama_config)
        state_dict = torch.load(args.fms_path, weights_only=True, map_location="cpu")
        model_state_dict = {}
        for k, v in state_dict['model_state'].items():
            if k.startswith("_orig_mod."):
                newk = k[len("_orig_mod."):]
            else:
                newk = k
            model_state_dict[newk] = v
        model.load_state_dict(model_state_dict)
    else:
        model = get_model("hf_pretrained", args.hf_path)
            
    model = model.to(torch.bfloat16).to('cuda')
    print(f'loaded model from {args.fms_path}')
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(f'loaded tokenizer from {args.tokenizer_path}')
    
    print("Before pruning:")
    eval_res = run_eval(model, tokenizer, args)
    base_line = eval_res[args.prune_task]
    
    if args.prune_task in PPL_TASKS:
        prune_scorer = partial(get_ppl, tasks=[args.prune_task], calib_size=args.calib_size, max_length=512)
    elif args.prune_task in TASK_CONFIG.keys():
        prune_scorer = partial(get_acc, tasks=[args.prune_task], )
    
    if args.drop_layers is not None:
        drop_layers(model, args)
    elif args.mode.lower() == 'bi':
        depth_prune_BI(model, tokenizer, prune_scorer, args)
    elif args.mode.lower() == 'depth':
        # if args.prune_task == 'wikitext':
        #     args.drop_layers = [str(i) for i in range(13, 29)]
        #     drop_layers(model, args)
        # elif args.prune_task == 'winogrande':
        #     args.drop_layers = [str(i) for i in range(15, 31)]
        #     drop_layers(model, args)
        all_scores = depth_prune_score(model, tokenizer, prune_scorer, args)
        if args.prune_task in PPL_TASKS:
            base_line = math.log(base_line)
            all_scores = [math.log(score) for score in all_scores]
        
        print(all_scores)
        if args.log_path and os.path.exists(args.log_path):
            data = {
                "prune_task": args.prune_task,
                "num_layers": args.num_layers,
                "base_line": base_line,
                "scores": all_scores
            }
            with open(args.log_path, 'a') as f:
                f.write(json.dumps(data) + '\n')
                
    elif args.mode.lower() == 'width':
        width_prune(model, tokenizer, prune_scorer, args)
    else:
        raise NotImplementedError(f'unimplemented prune mode: {args.prune.mode}')
    
    print("After pruning:")
    run_eval(model, tokenizer, args)
    
    if args.save_dir is not None:
        save_model(model, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, choices=['depth', 'width', 'bi'], required=False
    )
    parser.add_argument("--fms_path", type=str, default=None)
    parser.add_argument("--hf_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--model_variant", type=str, required=True)
    parser.add_argument("--prune_task", type=str, required=False)
    parser.add_argument("--eval_tasks", type=lambda s: s.split(","), required=True)
    parser.add_argument("--drop_layers", type=lambda s: s.split(","), default=None)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--log_path", type=str, default=None)
    
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--ffn_hidden_size", type=int, default=None)
    parser.add_argument("--calib_size", type=int, default=1024)
    
    args = parser.parse_args()
    
    set_seed(42)
    main(args)