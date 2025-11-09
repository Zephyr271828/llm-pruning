#!/bin/bash

#SBATCH --job-name=pruning_%j
#SBATCH --output=pruning_%j.out
#SBATCH --error=pruning_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate llmshearing

# pruning llama2 7b -> 2.7b or 1.3b

# pruning llama2 7b -> 2.7b or 1.3b
PROJ_DIR=/n/fs/vision-mix/yx1168/pruning/fms-llmshearing/llmshearing
LAUNCH_SCRIPT=${PROJ_DIR}/llmshearing/scripts/launch.sh
DATA_DIR=${PROJ_DIR}/llmshearing/data/orig_data/for_prune
OUTPUT_DIR=${PROJ_DIR}/ckpts/
TRAIN_SCRIPT=${PROJ_DIR}/llmshearing/train.py
# MODEL_PATH=${PROJ_DIR}/ckpts/Llama-2-7b-composer

# Specify $PROJ_DIR in scripts/launch.sh and scripts/srun_launch.sh if using slurm

test=False

from_model=7b # source model size
to_model=2.7b # target model size
config_file=${PROJ_DIR}/llmshearing/configs/llama2/${from_model}.yaml
path=${PROJ_DIR}/ckpts/Llama-2-7b-composer/state_dict.pt

# data setup
data_local=${DATA_DIR}

# basic setup
num_tokens=$((2 * 100 * 2**20))         
max_seq_len=4096
device_train_microbatch_size=4
global_train_batch_size=32
device_eval_batch_size=8
precision=amp_bf16

num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
tokens_per_batch=$(( num_gpus * device_train_microbatch_size * max_seq_len ))
steps=$(( (num_tokens + tokens_per_batch - 1) / tokens_per_batch ))
warmup_steps=$(( steps / 10 ))
lag_warmup_steps=$(( steps / 5 ))

# learning setup
lr=1e-4 # learning rate for the main parameters
max_duration=${steps}ba # 0.42B tokens
save_interval=${steps}ba # save in the end
t_warmup=${warmup_steps}ba # 10% learning rate warmup 

# dynamic loading setup
dynamic=True
set_names=[cc,github,book,stackexchange,wiki,arxiv,c4-rp] # domain names
proportion=[0.67,0.045,0.045,0.02,0.045,0.025,0.15] # initial proportion of RP, make sure that the sum(proportion) = 1
# doremi: update weights with exponential descent
# constant: keep the weights constant
update_type=doremi 
if [[ $to_model == 1.3b ]]; then
    target_loss=[1.9643,0.7459,2.1393,1.6117,1.7590,1.4449,2.1251] # 1.3b predicted loss from scaling law
elif [[ $to_model == 2.7b ]]; then
    target_loss=[1.8712,0.6883,2.0325,1.5353,1.6297,1.3560,2.0328] # 2.7b predicted loss from scaling law
elif [[ $to_model == 370m ]]; then
    target_loss=[2.1401,0.8694,2.3625,1.7791,2.047,1.6637,2.3139] # 410m predicted loss from scaling law
fi
eval_split_name=eval_merge # eval on all domains
eval_target_model=false # evaluate on the current model, not the target model, otherwise the loss will be inaccurate
eval_interval=50ba # eval every 50 batches and update the loading proportion


# pruning setup
lag_lr=1.0 # learning rate or l0_module
lagr_warmup=${lag_warmup_steps}ba # 20% sparsity warmup
if [[ $to_model == 1.3b ]]; then
    target_d_model=2048; target_n_heads=16; target_n_layers=24; target_intermediate_size=5504
elif [[ $to_model == 2.7b ]]; then
    target_d_model=2560; target_n_heads=20; target_n_layers=32; target_intermediate_size=6912
elif [[ $to_model == 370m ]]; then
    target_d_model=1024; target_n_heads=8; target_n_layers=24; target_intermediate_size=2816
fi
attn_impl=flash

# save directroy
run_name=llama2_${from_model}_pruning_scaling_${update_type}_to${to_model}_sl${max_seq_len}
save_dir=${OUTPUT_DIR}/${run_name}
wandb_dir=${save_dir} # save locally

if [[ $test == True ]]; then t=00-01:00:00; else t=00-20:00:00; fi

# Run in bash, it will automatically use resources available in the current environment
# composer $TRAIN_SCRIPT \

# Run with slurm    
# bash $LAUNCH_SCRIPT \
composer $PROJ_DIR/llmshearing/train.py \
    $config_file \
    run_name=${run_name} \
    data_local=${data_local} \
    eval_loader.dataset.split=${eval_split_name} \
    global_train_batch_size=${global_train_batch_size} \
    device_train_microbatch_size=${device_train_microbatch_size} \
    device_eval_batch_size=${device_eval_batch_size} \
    precision=${precision} \
    max_seq_len=${max_seq_len} \
    max_duration=${max_duration} \
    eval_first=false \
    scheduler.t_warmup=${t_warmup} \
    save_folder=${save_dir} \
    loggers.wandb.init_kwargs.dir=${wandb_dir} \
    eval_interval=${eval_interval} \
    save_interval=${save_interval} \
    optimizer.lr=${lr} \
    optimizer.lag_lr=${lag_lr} \
    model.path=${path} \
    model.l0_module.lagrangian_warmup_steps=${lagr_warmup} \
    model.l0_module.pruning_modules='[head,intermediate,layer,hidden]' \
    model.l0_module.eval_target_model=${eval_target_model} \
    model.l0_module.target_model.d_model=${target_d_model} \
    model.l0_module.target_model.n_heads=${target_n_heads} \
    model.l0_module.target_model.n_layers=${target_n_layers} \
    model.l0_module.target_model.intermediate_size=${target_intermediate_size} \
    model.attn_impl=${attn_impl} \
    callbacks.data_loading.dynamic=${dynamic} \
    callbacks.data_loading.set_names=${set_names} \
    callbacks.data_loading.proportion=${proportion} \
    callbacks.data_loading.update_type=${update_type} \
    callbacks.data_loading.target_loss=${target_loss} \
    train_loader.num_workers=0 \
    train_loader.prefetch_factor=null \
    train_loader.persistent_workers=false \
    autoresume=false