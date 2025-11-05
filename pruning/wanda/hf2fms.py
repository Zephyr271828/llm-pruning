import os
import pdb
import torch
import torch.nn as nn
import argparse
from fms.models import get_model
from fms.models.llama import LLaMA, SparseLLaMA
from fms.models.hf import to_hf_api
from fms_fsdp.utils.config_utils import get_model_config

def main(args):
    
    fms_save_dir = args.hf_path.replace('hf', 'fms')
    os.makedirs(fms_save_dir, exist_ok=True)

    variant = '2-7b' if '7b' in args.hf_path.lower() else '3-8b'
    model = get_model(
        architecture='sparse_llama',
        variant=variant,
        model_path=args.hf_path,
        source='hf'
    )
    print(f"loaded fms model from {args.hf_path}")

    save_dict = {
        "model_state": model.state_dict(),  
        "step": 0,                      
    }
    torch.save(save_dict, os.path.join(fms_save_dir, 'ckpt.pth'))
    print(f"fms model saved to {fms_save_dir}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_path', type=str, required=True)
    args = parser.parse_args()
    
    main(args)