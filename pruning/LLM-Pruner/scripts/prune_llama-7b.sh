#!/bin/bash


#SBATCH --job-name=prune_llama-7b_%j
#SBATCH --output=logs/prune_llama-7b_%j.out
#SBATCH --error=logs/prune_llama-7b_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

#SBATCH --mail-type=all
#SBATCH --mail-user=yx1168@princeton.edu

set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate llm-pruner

PROJ_DIR=$(pwd)
base_model=/n/fs/vision-mix/yx1168/model_ckpts/llama-7b

export PYTHONPATH=$(pwd):${PYTHONPATH:-''}
run_exp() {
    dim=${1:-block_wise}
    type=${2:-taylor}
    taylor=${3:-param_first}
    ratio=${4:-0.25}
    exp_name=
    CUDA_VISIBLE_DEVICES=0,1 python hf_prune.py \
        --base_model ${base_model} \
        --pruning_ratio ${ratio} \
        --device cuda  \
        --eval_device cuda \
        --${dim} \
        --block_mlp_layer_start 4 \
        --block_mlp_layer_end 30 \
        --block_attention_layer_start 4 \
        --block_attention_layer_end 30 \
        --log_dir ${PROJ_DIR}/../../checkpoints \
        --save_ckpt_log_name "${exp_name}" \
        --pruner_type ${type} \
        --taylor ${taylor} \
        --save_model \
        --test_before_train \
        --test_after_train 
}

run_exp "block_wise" "taylor" "param_first" 0.25
run_exp "channel_wise" "taylor" "param_first" 0.25
run_exp "block_wise" "l2" "param_first" 0.25
run_exp "block_wise" "random" "param_first" 0.25
run_exp "channel_wise" "l2" "param_first" 0.25
run_exp "channel_wise" "random" "param_first" 0.25
run_exp "block_wise" "taylor" "param_second" 0.25
run_exp "block_wise" "taylor" "vectorize" 0.25