import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from eval import PPL_TASKS, TASK_CONFIG, ACC_TASKS

def drop_layers(model, args):
    layer_idx_to_drop = [int(each) for each in args.drop_layers]
    assert all([idx in range(model.config.nlayers) for idx in layer_idx_to_drop]), 'invalid layer idx to drop!'
    print('Layers to drop(0-indexed):', layer_idx_to_drop)
    
    layers_to_drop = [
        layer for i, layer in enumerate(model.layers) if i in layer_idx_to_drop
    ]
    model.layers = nn.ModuleList(
        [layer for i, layer in enumerate(model.layers) if i not in layer_idx_to_drop]
    )
    model.config.nlayers = len(model.layers)
    torch.cuda.empty_cache()
    
def depth_prune_BI(model, tokenizer, scorer, args):
    BI_scores = [0 for _ in model.layers]
    hooks = []
    
    def get_BI_hook(layer_idx):
        def calculate_BI_hook(module, inputs, outputs):
            hidden_states = inputs[0]
            output = outputs[0]
            with torch.no_grad():
                BI = 1 - F.cosine_similarity(hidden_states, output, dim=2).mean()
                BI_scores[layer_idx] += BI
        return calculate_BI_hook
    
    for i, layer in enumerate(model.layers):
        hooks.append(
            layer.register_forward_hook(get_BI_hook(i))
        )
        
    _ = scorer(model, tokenizer)
            
    for hook in hooks:
        hook.remove()

    sorted_idx = sorted(range(len(BI_scores)), key=lambda i: BI_scores[i], reverse=True)
    
    layer_idx_to_keep = sorted(sorted_idx[:args.num_layers])
    layer_idx_to_drop = sorted(sorted_idx[args.num_layers:])
    
    # NOTE test
    # layer_idx_to_drop = list(range(15, 31))
    # layer_idx_to_keep = [i for i in range(32) if i not in layer_idx_to_drop]
    
    layers_to_drop = [
        layer for i, layer in enumerate(model.layers) if i in layer_idx_to_drop
    ]
    
    print('Layers to drop(0-indexed):', layer_idx_to_drop)
    
    model.layers = nn.ModuleList(
        [layer for i, layer in enumerate(model.layers) if i in layer_idx_to_keep]
    )
    model.config.nlayers = args.num_layers
    
    for layer in layers_to_drop:
        del layer
        
def depth_prune_score(model, tokenizer, scorer, args):
    
    num_layers_to_prune = model.config.nlayers - args.num_layers
    all_layers = model.layers  # Direct access in FMS

    best_i = -1
    if scorer.keywords["tasks"][0] in PPL_TASKS:
        best_score = float('inf')
    elif scorer.keywords["tasks"][0] in TASK_CONFIG.keys():
        best_score = float('-inf')
    all_scores = []

    for i in range(model.config.nlayers):
        if i > args.num_layers:
            break
        model.layers = all_layers[:i] + all_layers[i + num_layers_to_prune:]
        score = scorer(model, tokenizer)[scorer.keywords['tasks'][0]]
        print(f"i(0-indexed): {i} score: {score}")
        if scorer.keywords["tasks"][0] in PPL_TASKS and score < best_score:
            best_i, best_score = i, score
        elif scorer.keywords["tasks"][0] in ACC_TASKS and score > best_score:
            best_i, best_score = i, score
        all_scores.append(score)

    print('best i(0-indexed):', best_i)
    print(f'best score:', best_score)
    print('layers_to_drop(0-indexed)', list(range(best_i, best_i + num_layers_to_prune)))

    model.layers = all_layers[:best_i] + all_layers[best_i + num_layers_to_prune:]
    model.config.nlayers = args.num_layers  # still keep for consistency

    return all_scores