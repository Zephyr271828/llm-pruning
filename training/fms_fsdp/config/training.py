from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class train_config:
    # model
    model_variant: str = "7b"
    ckpt_load_path: str = "/fsx/output/ckpt"
    ckpt_save_path: str = "/fsx/output/ckpt"

    # dataset and dataloader
    data_path: str = "/fsx/data"
    seq_length: int = 4096
    bos_token: int = 128000
    eos_token: int = 128001

    # fsdp policies
    sharding_strategy: str = "hsdp"
    fsdp_activation_checkpointing: bool = False
    selective_checkpointing: Union[float, str] = 1  # percentage of blocks to apply ac
    mixed_precision: bool = True
    low_cpu_fsdp: bool = False

    # training spec
    batch_size: int = 2
    grad_accum_steps: Optional[int] = 1
    num_steps: int = 1000000
    training_stage: str = "initial"
    learning_rate: float = 3e-4
    min_learning_rate_ratio: float = 0.1
    warmup_ratio: float = 0.05
    grad_clip_thresh: float = 1.0
    seed: int = 2023

    # profiling
    use_profiler: bool = False
    profiler_rank0_only: bool = True

    # logging
    report_interval: int = 10
    checkpoint_interval: int = 10000
    tracker: Optional[str] = None  # None, "wandb", "aim"
    tracker_dir: str = "/fsx/aim_logs/llama"
    tracker_project_name: str = "llama"  # project name for a group of runs
    tracker_run_name: Optional[str] = None
    debug: bool = False
    debug_output_path: str = "./all_logs.jsonl"

    # compile
    use_torch_compile: bool = True

    