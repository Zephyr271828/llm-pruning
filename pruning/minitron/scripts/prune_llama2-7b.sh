#!/bin/bash

#SBATCH --job-name=prune_llama2-7b_%j
#SBATCH --output=logs/prune_llama2-7b_%j.out
#SBATCH --error=logs/prune_llama2-7b_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

#SBATCH --mail-type=all
#SBATCH --mail-user=yx1168@princeton.edu

#SBATCH --chdir=.

set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate minitron

PROJ_DIR=$(pwd)

model_path=meta-llama/Llama-2-7b-hf
save_dir=${PROJ_DIR}/../../checkpoints
log_dir=${PROJ_DIR}/outputs
calib_size=1024

python $PROJ_DIR/main.py bi \
    --model_path ${model_path} \
    --save_dir ${save_dir}/shortgpt \
    --log_dir ${log_dir} \
    --prune_task pg19 \
    --num_layers 23 \
    --calib_size ${calib_size} \
    --seq_len 4096

python $PROJ_DIR/main.py depth \
    --model_path ${model_path} \
    --save_dir ${save_dir}/minitron \
    --log_dir ${log_dir} \
    --prune_task pg19 \
    --num_layers 23 \
    --calib_size ${calib_size} \
    --seq_len 4096

python $PROJ_DIR/main.py depth \
    --model_path ${model_path} \
    --save_dir ${save_dir}/minitron \
    --log_dir ${log_dir} \
    --prune_task pg19 \
    --num_layers 31 \
    --calib_size ${calib_size} \
    --seq_len 4096