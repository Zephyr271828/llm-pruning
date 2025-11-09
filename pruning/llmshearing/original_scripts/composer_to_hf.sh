#!/bin/bash

source ../configs/setup.sh
check_sbash composer_to_hf_%j 16 128 4 1 "tandon_h100_1,tandon_a100_1,tandon_a100_2"

MODEL_SIZE=2.7
MODEL_PATH="${PROJ_DIR}/ckpts/llama2_7b_pruning_scaling_doremi_to${MODEL_SIZE}b_sl4096/latest-rank0.pt"
PRUNED_MODEL_PATH="${PROJ_DIR}/ckpts/llama2_7b_pruning_scaling_doremi_to${MODEL_SIZE}b_sl4096/pruned-latest-rank0.pt"
OUTPUT_PATH="${PROJ_DIR}/ckpts/Llama-2-${MODEL_SIZE}b-hf"
MODEL_CLASS=LlamaForCausalLM
if [[ $MODEL_SIZE == 1.3 ]]; then
    HIDDEN_SIZE=2048; NUM_ATTENTION_HEADS=16; NUM_HIDDEN_LAYERS=24; INTERMEDIATE_SIZE=5504
elif [[ $MODEL_SIZE == 2.7 ]]; then
    HIDDEN_SIZE=2560; NUM_ATTENTION_HEADS=20; NUM_HIDDEN_LAYERS=32; INTERMEDIATE_SIZE=6912
fi
MODEL_NAME=Sheared-Llama-${MODEL_SIZE}B


# # MODEL_PATH=$MODEL_DIR/latest-rank0.pt
python3 -m llmshearing.utils.post_pruning_processing prune_and_save_model $MODEL_PATH

python3 -m llmshearing.utils.composer_to_hf save_composer_to_hf $PRUNED_MODEL_PATH $OUTPUT_PATH \
    model_class=${MODEL_CLASS} \
    hidden_size=${HIDDEN_SIZE} \
    num_attention_heads=${NUM_ATTENTION_HEADS} \
    num_hidden_layers=${NUM_HIDDEN_LAYERS} \
    intermediate_size=${INTERMEDIATE_SIZE} \
    num_key_value_heads=${NUM_ATTENTION_HEADS} \
    _name_or_path=${MODEL_NAME}

