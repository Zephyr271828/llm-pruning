#!/bin/bash

PROJ_DIR=$(pwd)
hf_model_path=${PROJ_DIR}/../../checkpoints/wanda/Llama-3.1-8B_wanda_unstructured_0.5

python reinit.py \
    --hf_model_path $hf_model_path