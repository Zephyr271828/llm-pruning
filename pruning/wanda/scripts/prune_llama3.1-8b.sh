#!/bin/bash

set -euo pipefail

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
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

PROJ_DIR=$(pwd)
# model_path=meta-llama/Llama-3.1-8B
model_path=/n/fs/vision-mix/yx1168/model_ckpts/Llama-3.1-8B
save_dir=${PROJ_DIR}/../../checkpoints
log_dir=${PROJ_DIR}/outputs

cd $PROJ_DIR/src
for method in "${methods[@]}"; do
    hf_dir=${save_dir}/${method}
    mkdir -p ${hf_dir}
    for sparsity in "${sparsities[@]}"; do
        echo "Pruning with method: $method and sparsity: $sparsity"
        python main.py \
            --model ${model_path} \
            --prune_method ${method} \
            --sparsity_ratio 0.5 \
            --sparsity_type ${sparsity} \
            --save outputs/${method}/${sparsity} \
            --save_model ${hf_dir} \
            --eval_zero_shot
    done
done