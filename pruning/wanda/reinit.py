import torch
import torch.nn as nn
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer


def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        # print(f"layer {i} sparsity {float(sub_count)/sub_params:.4f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def reinit_nonzero_weights(model, method="xavier_uniform", seed=None):
    """
    Reinitialize only non-zero weights of all Linear layers in a model.

    Args:
        model (nn.Module): Loaded model whose linear weights to modify.
        method (str): One of {"normal", "uniform", "xavier_uniform", "xavier_normal"}.
        seed (int, optional): Random seed for reproducibility.
    """
    if seed is not None:
        torch.manual_seed(seed)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            W = module.weight.data
            mask = (W != 0)

            if mask.sum() == 0:
                continue  # all zeros → skip

            # Initialize a fresh tensor of the same shape
            if method == "normal":
                new_vals = torch.randn_like(W)
            elif method == "uniform":
                new_vals = torch.rand_like(W).mul_(2).sub_(1)  # [-1, 1]
            elif method == "xavier_uniform":
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(W)
                bound = (6.0 / (fan_in + fan_out)) ** 0.5
                new_vals = torch.empty_like(W).uniform_(-bound, bound)
            elif method == "xavier_normal":
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(W)
                std = (2.0 / (fan_in + fan_out)) ** 0.5
                new_vals = torch.empty_like(W).normal_(0, std)
            elif method == "trunc_norm":
                new_vals = torch.empty_like(W)
                nn.init.trunc_normal_(new_vals, mean=0.0, std=0.02)
            else:
                raise ValueError(f"Unknown init method: {method}")

            # Replace only non-zero positions
            W[mask] = new_vals[mask]

            print(f"[Reinit] {name}.weight — {mask.sum().item()} / {mask.numel()} elements reinitialized")

    print("✅ Reinitialization complete.")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_path", type=str, required=True)
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.hf_model_path, torch_dtype=torch.float16, device_map="auto")
    sparsity_before = check_sparsity(model)
    print("Sparsity before reinit:", sparsity_before)
    mask_before = model.model.layers[0].mlp.gate_proj.weight.data == 0
    
    model = reinit_nonzero_weights(model, method="trunc_norm", seed=42)
    
    sparsity_after = check_sparsity(model)
    print("Sparsity after reinit:", sparsity_after)
    mask_after = model.model.layers[0].mlp.gate_proj.weight.data == 0
    print("Mask difference:", (mask_before != mask_after).sum().item())  # should be 0
    
    for name, module in model.named_modules():
        if hasattr(module, "sparse_mask"):
            delattr(module, "sparse_mask")
            
    print(model)
    print("num params:", sum(p.numel() for p in model.parameters()))
    
    save_path = args.hf_model_path + "_reinit"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Model saved to", save_path)
    