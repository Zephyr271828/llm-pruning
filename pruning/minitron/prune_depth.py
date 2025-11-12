import torch
import torch.nn as nn
import torch.nn.functional as F

def depth_prune_BI(model, tokenizer, scorer, args):
    BI_scores = [[] for _ in model.model.layers]
    hooks = []
    
    def get_BI_hook(layer_idx):
        def calculate_BI_hook(module, inputs, outputs):
            hidden_states = inputs[0]
            output = outputs[0]
            with torch.no_grad():
                BI = 1 - F.cosine_similarity(hidden_states, output, dim=2).mean()
                BI_scores[layer_idx].append(BI.item())
        return calculate_BI_hook
    
    for i, layer in enumerate(model.model.layers):
        hooks.append(
            layer.register_forward_hook(get_BI_hook(i))
        )
        
    _ = scorer(model, tokenizer)
            
    for hook in hooks:
        hook.remove()
        
    BI_scores = [sum(s)/len(s) if len(s) > 0 else 0 for s in BI_scores]

    sorted_idx = sorted(range(len(BI_scores)), key=lambda i: BI_scores[i], reverse=True)
    
    layer_idx_to_keep = sorted(sorted_idx[:args.num_layers])
    layer_idx_to_drop = sorted(sorted_idx[args.num_layers:])
    
    layers_to_drop = [
        layer for i, layer in enumerate(model.model.layers) if i in layer_idx_to_drop
    ]
    
    print('Layers to drop(0-indexed):', layer_idx_to_drop)
    
    model.model.layers = nn.ModuleList(
        [layer for i, layer in enumerate(model.model.layers) if i in layer_idx_to_keep]
    )
    model.config.num_hidden_layers = args.num_layers
    
    for layer in layers_to_drop:
        del layer
        
    return BI_scores, layer_idx_to_drop
        
def depth_prune_score(model, tokenizer, scorer, base_line, args):
    num_layers_to_prune = model.config.num_hidden_layers - args.num_layers
    all_layers = model.model.layers  # Direct access in FMS

    best_i = -1
    best_score = None
    min_abs_diff = float('inf')
    all_scores = []

    for i in range(model.config.num_hidden_layers):
        if i > args.num_layers:
            break
        model.model.layers = all_layers[:i] + all_layers[i + num_layers_to_prune:]
        score = scorer(model, tokenizer)
        print(f"i(0-indexed): {i} score: {score}")
        abs_diff = abs(score - base_line)
        if abs_diff < min_abs_diff:
            best_i = i
            best_score = score
            min_abs_diff = abs_diff
        all_scores.append(score)

    layer_idx_to_drop = list(range(best_i, best_i + num_layers_to_prune))
    print('best i(0-indexed):', best_i)
    print(f'best score:', best_score)
    print('layers_to_drop(0-indexed)', layer_idx_to_drop)

    model.model.layers = all_layers[:best_i] + all_layers[best_i + num_layers_to_prune:]
    model.config.num_hidden_layers = args.num_layers  # still keep for consistency

    return all_scores, layer_idx_to_drop