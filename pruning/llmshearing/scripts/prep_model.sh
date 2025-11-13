#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate llmshearing

PROJ_DIR=$(pwd)
# HF_MODEL_NAME=meta-llama/Llama-2-7b-hf
HF_MODEL_NAME=/n/fs/vision-mix/yx1168/model_ckpts/Llama-2-7b-hf
OUTPUT_PATH=${PROJ_DIR}/../../checkpoints/llmshearing/Llama-2-7b-composer/state_dict.pt

# Create the necessary directory if it doesn't exist
mkdir -p $(dirname $OUTPUT_PATH)

# Convert the Hugging Face model to Composer key format
python3 -m llmshearing.utils.composer_to_hf save_hf_to_composer $HF_MODEL_NAME $OUTPUT_PATH