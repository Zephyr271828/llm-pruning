#!/bin/bash

hf_model_path=meta-llama/Llama-3.1-8B
hf_model_path=/n/fs/vision-mix/yx1168/model_ckpts/Llama-3.1-8B
tasks=(
    c4
    wikitext
    wikitext2
    cnn_dailymail
    winogrande
    arc_easy
    arc_challenge
    hellaswag
    truthfulqa_mc1
    truthfulqa_mc2
    piqa
    sciq
    boolq
    anli_r1
    anli_r2
    anli_r3
    openbookqa
    rte
    mmlu
    record
)

PROJ_DIR=$(pwd)
export PYTHONPATH=$PROJ_DIR/lm-evaluation-harness:$PYTHONPATH

tasks_str=$(IFS=, ; echo "${tasks[*]}")
# echo "Evaluating tasks: $tasks_str"

python eval.py \
    --hf_path $hf_model_path \
    --tokenizer_path $hf_model_path \
    --tasks $tasks_str
