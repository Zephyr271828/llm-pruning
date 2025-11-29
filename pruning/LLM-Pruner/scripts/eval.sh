#!/bin/bash

#SBATCH --job-name=eval_%j
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=neu306

#SBATCH --mail-type=all
#SBATCH --mail-user=yx1168@princeton.edu

source $(conda info --base)/etc/profile.d/conda.sh
conda activate llm-pruner

PROJ_DIR=$(pwd)

export PYTHONPATH=$PROJ_DIR/lm-evaluation-harness:$PYTHONPATH
ckpt_dir=$PROJ_DIR/../../checkpoints/llm-pruner
log_dir=$PROJ_DIR/outputs

mapfile -t bin_dirs < <(find "$ckpt_dir" -name "*.bin" -type f | xargs -I {} dirname {} | sort -u)

for bin_dir in "${bin_dirs[@]}"; do
    model_name=$(basename $bin_dir)
    echo "[INFO] evaluating ${model_name}"

    if [[ -f ${log_dir}/${model_name}.json ]]; then
        echo "[INFO] ${model_name} already evaluated, skipping..."
        continue
    fi

    python eval_ppl.py \
        --model_path="$bin_dir/pytorch_model.bin" 

    if [[ $model_name == *llama-7b* ]]; then
        base_model=/n/fs/vision-mix/yx1168/model_ckpts/llama-7b
    elif [[ $model_name == *Llama-2-7b-hf* ]]; then
        base_model=meta-llama/Llama-2-7b-hf
    elif [[ $model_name == *Llama-3.1-8B* ]]; then
        base_model=meta-llama/Llama-3.1-8B
    fi

    python lm-evaluation-harness/main.py \
        --model hf-causal-experimental \
        --model_args checkpoint=$bin_dir/pytorch_model.bin,config_pretrained=$base_model \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
        --device cuda:0 \
        --output_path ${log_dir}/${model_name}.json \
        --no_cache

done