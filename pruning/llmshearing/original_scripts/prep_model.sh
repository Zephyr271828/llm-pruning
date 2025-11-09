#!/bin/bash

export PROJ_DIR='/scratch/yx3038/Research/pruning/LLM-Shearing'

# Define the Hugging Face model name and the output path
export HF_MODEL_NAME='/scratch/yx3038/model_ckpt/Llama-2-7b-hf'
export OUTPUT_PATH="${PROJ_DIR}/ckpts/Llama-2-7b-composer/state_dict.pt"

# Create the necessary directory if it doesn't exist
mkdir -p $(dirname $OUTPUT_PATH)

# Convert the Hugging Face model to Composer key format
python3 -m llmshearing.utils.composer_to_hf save_hf_to_composer $HF_MODEL_NAME $OUTPUT_PATH