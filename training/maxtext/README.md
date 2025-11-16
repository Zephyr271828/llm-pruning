# MaxText Modified

This is a modified version of MaxText for LLM Pruning Training.
For the native README of MaxText see [README_ORIGINAL.md](README_ORIGINAL.md).
NOTE: you should run all the following commands on a TPU VM.

## Installation
```
bash scripts/install.sh
```

## Data and Model Preparation
Since TPU VM involves bucket storage, the provided scripts are not ready to run unless you have configured the buckect layout and directories in the scripts. Please kindly configure the following scripts before running them:
- [ ] [scripts/get_tpu_info.sh](scripts/get_tpu_info.sh)
- [ ] [scripts/training.sh](scripts/training.sh)
- [ ] [scripts/convert.sh](scripts/convert.sh)
- [ ] [scripts/convert.sh](scripts/convert.sh)

For example, the layout of my bucket is mounted at `/home/zephyr/gcs-bucket`, and the layout looks like the following:
```bash
.
├── datasets
│   └── dclm
├── model_ckpts
│   ├── direct
│   ├── hf
│   └──maxtext
└── wandb_run_ids
```

For a more detailed introduction to buckets, please refer to the [TPU manual](https://github.com/TaiMingLu/TPU-Manual?tab=readme-ov-file#storage) developed by Taiming Lu.

## Data Preparation
You may direct download training data for llama3 models from [here](https://huggingface.co/datasets/Zephyr271828/dclm-260b/tree/main), which contains 260B tokenized tokens from [DCLM-Baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) in array_record format.

You may also want to refer to the [data preparation doc](getting_started/Data_Input_Pipeline.md) by MaxText to create your own customized dataset.

## Pretraining
```bash
bash scripts/training.sh \
    --model=llama3.1-8b \
    --num_steps=50000 \
    --lr=3e-4 \
    --global_batch_size=512 \
    --data_files=/home/zephyr/gcs-bucket/datasets/dclm/llama3_64_array_record/*.array_record 
```

## Fine-tuning
```bash
bash scripts/finetuning.sh \
    --model=llama3.1-8b \
    --num_steps=50000 \
    --lr=3e-4 \
    --global_batch_size=512 \
    --data_files=/home/zephyr/gcs-bucket/datasets/dclm/llama3_64_array_record/*.array_record \
    --load_parameters_path=gs://${BUCKET_NAME}/model_ckpts/maxtext/llama3.1_8b
```
More examples can be found in [scripts/examples](scripts/examples).


## Model Conversion

MaxText involves 3 types of model checkpoints: 
- huggingface checkpoints: compatible with HF libraries and torch-xla training frameworks on TPUs.
- orbax checkpoints: a JAX-based model format. Used by MaxText for training.
- param-only (direct checkpoints): a JAX-based model format that only contains the parameters. Used for evaluation and inference.

The following commands provide an easy-to-use interface for converting between these formats, as well as evaluation and testing that the outputs(logits) of different formats are consistent.  
Before using this script, please kindly check [scripts/convert.sh](scripts/convert.sh) for the configuration of the root dirs of the checkpoints.
```bash
MODEL= # the model architecture
ORBAX_CKPT_NAME= # the basename of the orbax checkpoint
HF_MODEL_NAME= # the basename of the huggingface checkpoint
DIRECT_RUN_NAME= # the basename of the param-only checkpoint
STEP= # the step to choose in the **orbax checkpoint**
TASKS= # optionally to only eval the model on a subset of tasks (comma-separated list of task names)
scripts/convert.sh hf_to_orbax     --model=$MODEL --orbax_ckpt_name=$ORBAX_CKPT_NAME --hf_model_name=$HF_MODEL_NAME
scripts/convert.sh gen_param_ckpt  --model=$MODEL --orbax_ckpt_name=$ORBAX_CKPT_NAME --step=$STEP                    --direct_run_name=$DIRECT_RUN_NAME
scripts/convert.sh orbax_to_hf     --model=$MODEL --orbax_ckpt_name=$ORBAX_CKPT_NAME --step=$STEP                    --hf_model_name=$HF_MODEL_NAME
scripts/convert.sh logits_test     --model=$MODEL --direct_run_name=$DIRECT_RUN_NAME --hf_model_name=$HF_MODEL_NAME
scripts/convert.sh eval            --model=$MODEL --direct_run_name=$DIRECT_RUN_NAME --hf_model_name=$HF_MODEL_NAME [--tasks=$TASKS]
```



