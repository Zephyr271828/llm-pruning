#!/bin/bash
#SBATCH --job-name=test_eq_%j
#SBATCH --output=logs/test_eq_%j.out
#SBATCH --error=logs/test_eq_%j.err
#SBATCH --tasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

#SBATCH --account=bdhh-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --mem=128GB
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=all
#SBATCH --mail-user=yx3038@nyu.edu
#SBATCH --no-requeue

source ../configs/setup.sh

# export HF_MODEL_NAME="${MODEL_DIR}/Llama-2-7b-hf"
# export OUTPUT_PATH="${PROJ_DIR}/ckpts/Llama-2-7b-composer/state_dict.pt"
# export OUTPUT_PATH="${PROJ_DIR}"/outputs/llama2_7b_pruning_scaling_doremi_to2.7b_sl2048/
export OUTPUT_PATH="${PROJ_DIR}/ckpts/llama2_7b_pruning_scaling_doremi_to2.7b_sl4096/pruned-latest-rank0.pt"
export HF_MODEL_NAME2="${PROJ_DIR}/ckpts/Llama-2-1.3b-hf"
export HF_MODEL_NAME="${PROJ_DIR}/ckpts/Sheared-LLaMA-2.7B-Pruned"
export MODEL_SIZE=2.7B

python3 -m llmshearing.utils.test_composer_hf_eq $HF_MODEL_NAME $OUTPUT_PATH $MODEL_SIZE

# python3 -m llmshearing.utils.test_hf2_eq $HF_MODEL_NAME $HF_MODEL_NAME2