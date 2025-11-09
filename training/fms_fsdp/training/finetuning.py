import math
import os
import glob
import random
import numpy as np

import fire
import torch
import torch.optim as optim
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import LambdaLR

from fms_fsdp import config
from fms_fsdp.utils.checkpointing_utils import Checkpointer
from fms.models.llama import LLaMA, LLaMABlock, LLaMAConfig, param_init_function
from fms_fsdp.utils.config_utils import get_model_config, update_config
from fms_fsdp.utils.dataset_utils import DistributedDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from fms_fsdp.utils.train_utils import (
    get_policies,
    get_profiler,
    setup,
    setup_environ_flags,
    train,
)
from fms.models import get_model

# from transformers import AutoTokenizer, AutoModelForCausalLM
# from fms_to_hf_llama import convert_to_hf 
from functools import partial

def main(**kwargs):
    # get configs
    cfg = config.finetune_config()
    update_config(cfg, **kwargs)

    # ensure reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"--> running with these configs {cfg}")

    # some setups
    setup()
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    setup_environ_flags()

    # get policy
    block = LLaMABlock
    (
        mixed_precision_policy,
        wrapping_policy,
        sharding_strategy_policy,
        apply_selective_ac,
        _
    ) = get_policies(cfg, rank, block)

    # get fms model
    llama_config = get_model_config(cfg.model_variant)

    if rank == 0:
        print(f"--> llama config: {llama_config}")

    # NOTE load fms model from hf ckpt
    # model = get_model(
    #     architecture="hf_pretrained", 
    #     model_path=cfg.ckpt_load_path,
    #     device_type=torch.cuda.current_device(),
    #     data_type=torch.bfloat16,
    #     distributed_strategy='fsdp',
    # )
    
    # NOTE load fms model from fms ckpt
    with torch.device("meta"):
        model = LLaMA(llama_config)
    
    # NOTE if no previous finetuning ckpts
    if not glob.glob(os.path.join(cfg.ckpt_save_path, 'checkpoints', 'step_*_ckp')):
        print()
        model.to_empty(device="cpu")
        state_dict = torch.load(cfg.ckpt_load_path, weights_only=False, map_location="cpu")
        model_state_dict = {}
        for k, v in state_dict['model_state'].items():
            if k.startswith("_orig_mod."):
                newk = k[len("_orig_mod."):]
            else:
                newk = k
            model_state_dict[newk] = v
        model.load_state_dict(model_state_dict)
        
    # model = model.to(torch.bfloat16)
   
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> model has {total_params / 1e6} Million params\n")

    # get data loader
    if rank == 0:
        print("Constructing datasets...")
    
    # handle batch size ramp up
    
    # handle data loader reverse
    if cfg.training_stage == "initial":
        reverse = True
    elif cfg.training_stage == "finetuning":
        reverse = False
    else:
        raise ValueError(f"Invalid training stage \"{cfg.training_stage}\"")    
    
    dataset = DistributedDataset(cfg.data_path, rank, world_size, cfg.batch_size, cfg.seq_length, cfg.bos_token, cfg.eos_token, reverse)
    train_loader = StatefulDataLoader(dataset, batch_size=None, num_workers=0)
    if rank == 0:
        print("Datasets constructed!")

    if rank == 0:
        print(model)
        
    # eval
    # from minitron.eval import get_ppl
    # from transformers import AutoTokenizer
    # tasks = ['c4', 'wikitext', 'cnn_dailymail', 'dclm']
    # tokenizer = AutoTokenizer.from_pretrained('/scratch/yx3038/model_ckpt/Llama-3.1-8B')
    # ppl_res = get_ppl(model.to('cuda'), tokenizer, tasks)
    # print(ppl_res)

    model = FSDP(
        model,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=sharding_strategy_policy,
        use_orig_params=cfg.use_torch_compile,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        # param_init_fn=param_init_fn,
    )
    # we need this post-fsdp call to avoid graph break with torch.compile, until we figure out a better solution.
    model.rot_emb.compute_freqs_cis(
        torch.device("cuda", torch.cuda.current_device()),
        model.config.max_expected_seq_len,
    )

    # fsdp activation checkpointing
    if cfg.fsdp_activation_checkpointing:
        if rank == 0:
            print(f"--> applying FSDP activation checkpointing...")
        apply_selective_ac(model, p=cfg.selective_checkpointing)

    # torch compile
    if cfg.use_torch_compile:
        if rank == 0:
            print(f"--> enabling torch compile...")
        # the default accumulated_cache_size_limit=64 is not enough for 70b model, so we make it 128 here
        torch._dynamo.config.accumulated_cache_size_limit = 128
        model = torch.compile(model)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95), weight_decay=0.1
    )
    
    # optionally load from checkpoint (when continue pretraining)
    # NOTE if there exists previous finetuning ckpts
    # if glob.glob(os.path.join(cfg.ckpt_save_path, 'checkpoints', 'step_*_ckp')):
    cfg.ckpt_load_path = cfg.ckpt_save_path
    checkpointer = Checkpointer(
        cfg.ckpt_save_path, 0, cfg.sharding_strategy, rank, local_rank
    )
    _, optimizer, train_loader, start_step, tokens_seen, is_resuming = checkpointer.load(
        model=model,
        optimizer=optimizer,
        dataloader=train_loader,
        path=os.path.join(cfg.ckpt_load_path, "checkpoints/") 
        if not os.path.isfile(cfg.ckpt_load_path) 
        else cfg.ckpt_load_path, 
        strict=False,
    )
    # else:
    #     tokens_seen = 0
    #     is_resuming = False
    if not is_resuming:
        start_step = 0
        # Override loaded optim hyperparams with the current values
        for g in optimizer.param_groups:
            g["initial_lr"] = cfg.learning_rate

    # LR schedule
    if cfg.training_stage == "annealing":
        schedule = lambda x: 1 - x / cfg.num_steps
    else:
        warmup_interval = min(2000, int(cfg.num_steps * cfg.warmup_ratio))
        if warmup_interval > 0:
            schedule = lambda x: min(
                1 - (1 - min(x, warmup_interval) / warmup_interval) ** 2,
                cfg.min_learning_rate_ratio
                + 0.5
                * (1 - cfg.min_learning_rate_ratio)
                * (1 + math.cos(min(x, cfg.num_steps) / cfg.num_steps * math.pi)),
            )
        else:
            schedule = lambda x: cfg.min_learning_rate_ratio \
                + 0.5 * (1 - cfg.min_learning_rate_ratio) \
                * (1 + math.cos(min(x, cfg.num_steps) / cfg.num_steps * math.pi))
    scheduler = LambdaLR(optimizer, lambda x: schedule(x + start_step * cfg.grad_accum_steps))

    # profiler
    profiler = get_profiler(cfg, rank)

    # Train
    if rank == 0:
        print(f"Training for {cfg.num_steps} steps")
    train(
        cfg,
        model,
        local_rank,
        rank,
        train_loader,
        optimizer,
        scheduler,
        profiler,
        checkpointer,
        start_step,
        tokens_seen,
    )

    checkpointer.save_single_file(cfg.num_steps, model)

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    fire.Fire(main)
