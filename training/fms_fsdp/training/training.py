import math
import os

import fire
import torch
import torch.optim as optim
from fms.models.llama import LLaMA, LLaMABlock
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import LambdaLR

from fms_fsdp import config
from fms_fsdp.utils.checkpointing_utils import Checkpointer
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


def main(**kwargs):
    # get configs
    cfg = config.train_config()
    update_config(cfg, **kwargs)

    # ensure reproducibility
    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # dist.init_process_group(backend="nccl", init_method="env://",
    #                             rank=rank, world_size=world_size)
    
    print(f"Hello from rank {rank} / local rank {local_rank}!")

    if rank == 0:
        print(f"--> running with these configs {cfg}")

    # some setups
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    setup()
    setup_environ_flags()

    # get policy
    block = LLaMABlock
    (
        mixed_precision_policy,
        wrapping_policy,
        sharding_strategy_policy,
        apply_selective_ac,
        param_init_fn,
    ) = get_policies(cfg, rank, block)

    # get fms model
    llama_config = get_model_config(cfg.model_variant)
    if cfg.low_cpu_fsdp:
        with torch.device("meta"):
            model = LLaMA(llama_config)
    else:
        model = LLaMA(llama_config)
        model.reset_parameters()
        
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> model has {total_params / 1e6} Million params\n")

    # get data loader
    if rank == 0:
        print("Constructing datasets...")
    dataset = DistributedDataset(cfg.data_path, rank, world_size, cfg.batch_size, cfg.seq_length, cfg.bos_token, cfg.eos_token, reverse=False)
    train_loader = StatefulDataLoader(dataset, batch_size=None, num_workers=0)
    if rank == 0:
        print("Datasets constructed!")

    # FSDP
    model = FSDP(
        model,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=sharding_strategy_policy,
        use_orig_params=cfg.use_torch_compile,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        param_init_fn=param_init_fn,
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
    checkpointer = Checkpointer(
        cfg.ckpt_save_path, 0, cfg.sharding_strategy, rank, local_rank
    )
    model, optimizer, train_loader, start_step, tokens_seen, is_resuming = checkpointer.load(
        model,
        optimizer,
        train_loader,
        path=os.path.join(cfg.ckpt_load_path, "checkpoints/")
        if not os.path.isfile(cfg.ckpt_load_path)
        else cfg.ckpt_load_path,
        strict=False,
    )
    print("Resumed from checkpoint:", is_resuming, "at step:", start_step, "on rank:", rank)

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
    scheduler = LambdaLR(optimizer, lambda x: schedule(x + start_step))

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
