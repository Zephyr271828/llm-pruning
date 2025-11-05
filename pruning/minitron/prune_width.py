import pdb
import torch
import torch.nn.functional as F

def get_idx(importance, size, layer_idx=0):
    _, idx = torch.sort(importance[layer_idx, :], descending=True)
    return idx[:size]
    
def prune_linear_module(module, idx, size, dim='in'):
    if dim == 'in':
        module.weight.data = module.weight.data[:, idx]
        module.in_features = size
    elif dim == 'out':
        module.weight.data = module.weight.data[idx, :]
        module.out_features = size
        if module.bias is not None:
            module.bias.data = module.bias.data[idx]

def width_prune(model, tokenizer, scorer, args):
    
    device = next(model.parameters()).device
    intermediate_size = int(model.config.hidden_grow_factor * model.config.emb_dim)
    hidden_size_importance = torch.zeros(1, model.config.emb_dim, device=device)
    ffn_importance = torch.zeros(model.config.nlayers, intermediate_size, device=device)
    attn_importance = torch.zeros(model.config.nlayers, model.config.emb_dim, device=device)
    
    def get_ffn_hook(layer_idx):
        def ffn_hook(module, inputs, outputs):
            ffn_input = inputs[0]
            activations = ffn_input.abs().mean(dim=0) 
            activations = activations.pow(2).sum(dim=0) # ffn_hidden_size
            ffn_importance[layer_idx, :] += activations
        return ffn_hook
    
    def LN_hook(module, inputs, outputs):
        activations = outputs.abs().mean(dim=0)
        activations = activations.pow(2).sum(dim=0) # hidden_size
        hidden_size_importance[0, :] += activations
    
    hooks = []
    for i, layer in enumerate(model.layers):
        hooks.append(layer.ln.register_forward_hook(LN_hook))
        hooks.append(layer.ff_ln.register_forward_hook(LN_hook))
        hooks.append(layer.ff_sub_layer.w2.register_forward_hook(get_ffn_hook(i)))
    
    _ = scorer(model, tokenizer)
    
    # print(hidden_size_importance)
    # print(ffn_importance)
    # pdb.set_trace()
        
    hidden_idx = get_idx(importance=hidden_size_importance, size=args.hidden_size)
    for i, layer in enumerate(model.layers):
        # NOTE sort by importance
        ffn_idx = get_idx(importance=ffn_importance, size=args.ffn_hidden_size, layer_idx=i).view(-1)
        
        # NOTE ATTN pruning
        layer.ln.weight.data = layer.ln.weight.data[hidden_idx]
        prune_linear_module(module=layer.attn.in_proj.qkv_fused, idx=hidden_idx, size=args.hidden_size, dim='in')
        prune_linear_module(module=layer.attn.dense, idx=hidden_idx, size=args.hidden_size, dim='out')
        
        # NOTE MLP pruning
        layer.ff_ln.weight.data = layer.ff_ln.weight.data[hidden_idx]
        prune_linear_module(module=layer.ff_sub_layer.wg1_fused, idx=hidden_idx, size=args.hidden_size, dim='in')
        combined_idx = torch.cat([ffn_idx, ffn_idx + intermediate_size])
        prune_linear_module(module=layer.ff_sub_layer.wg1_fused, idx=combined_idx, size=2 * args.ffn_hidden_size, dim='out')
        layer.ff_sub_layer.hidden_dim = args.ffn_hidden_size
        
        prune_linear_module(module=layer.ff_sub_layer.w2, idx=ffn_idx, size=args.ffn_hidden_size, dim='in')
        prune_linear_module(module=layer.ff_sub_layer.w2, idx=hidden_idx, size=args.hidden_size, dim='out')
    
    # NOTE prune embedding   
    model.shared.emb.weight.data = model.shared.emb.weight.data[:, hidden_idx]
    model.shared.emb.embedding_dim = args.hidden_size
    # NOTE prune model norm and lm_head
    model.dec_norm.weight.data = model.dec_norm.weight.data[hidden_idx]
    prune_linear_module(module=model.shared.head, idx=hidden_idx, size=args.hidden_size, dim='in')
        
    for hook in hooks:
        hook.remove()
        
    model.config.emb_dim = args.hidden_size
    model.config.hidden_grow_factor = args.ffn_hidden_size / args.hidden_size