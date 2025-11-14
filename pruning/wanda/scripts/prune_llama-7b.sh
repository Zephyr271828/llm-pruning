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
conda activate wanda

methods=(
    wanda
    sparsegpt
    magnitude
)

sparsities=(
    unstructured
    2:4
    4:8
)

ratio=0.5

PROJ_DIR=$(pwd)
export PYTHONPATH=${PROJ_DIR}/src/lib:${PYTHONPATH:-''}
export PYTHONPATH=${PROJ_DIR}/lm-evaluation-harness:${PYTHONPATH}
# model_path=huggyllama/llama-7b
model_path=/n/fs/vision-mix/yx1168/model_ckpts/llama-7b
model_name=$(basename ${model_path})
save_dir=${PROJ_DIR}/../../checkpoints
log_dir=${PROJ_DIR}/outputs

cd $PROJ_DIR/src
for method in "${methods[@]}"; do
    for sparsity in "${sparsities[@]}"; do
        echo "[INFO] Pruning with method: $method and sparsity: $sparsity and ratio: $ratio"
        python main.py \
            --model ${model_path} \
            --prune_method ${method} \
            --sparsity_ratio ${ratio} \
            --sparsity_type ${sparsity} \
            --save ${log_dir}/${method}/${sparsity}/ \
            --save_model ${save_dir}/${method}/${model_name}_${method}_${sparsity}_${ratio} \
            --eval_zero_shot
    done
done