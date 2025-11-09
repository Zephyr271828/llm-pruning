#!/bin/bash

#SBATCH --job-name=llama3.1_4b_%j
#SBATCH --output=logs/llama3.1_4b_%j.out
#SBATCH --error=logs/llama3.1_4b_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --gres=gpu:4
#SBATCH --time=4:00:00

#SBATCH --mail-type=all
#SBATCH --mail-user=yx1168@princeton.edu

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate fms

SHUFFLE=False

export OMP_NUM_THREADS=10

export CKPT_DIR=../checkpoints

NUM_STEPS=12500
LEARNING_RATE=0.0003
MIN_LEARNING_RATE_RATIO=0.1
BATCH_SIZE=2
GLOBAL_BATCH_SIZE=64
REPORT_INTERVAL=1
CHECKPOINT_INTERVAL=1
WARMUP_RATIO=0.05

MODEL_VARIANT='llama3_4b_width'

if [[ "$MODEL_VARIANT" == *"llama3"* ]]; then
    SEQ_LENGTH=8192
    BOS_TOKEN=128000
    EOS_TOKEN=128001
    DATA_DIR=<llama3_data_dir>
elif [[ "$MODEL_VARIANT" == *"llama2"* ]]; then
    SEQ_LENGTH=4096
    BOS_TOKEN=1
    EOS_TOKEN=2
    DATA_DIR=<llama2_data_dir>
fi

get_free_port() {
    python3 -c "import socket; s = socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
}

num_nodes=${SLURM_JOB_NUM_NODES}
num_gpus=$(nvidia-smi -L | wc -l)

if [[ $num_nodes -gt 1 ]]; then
    echo "Running on multiple nodes is not supported yet."
    node_rank=${SLURM_NODEID}    
    master_addr=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
    head_node_ip=( $(srun --nodes=1 --ntasks=1 -w "$master_addr" hostname --ip-address) )

    echo "SLURM_JOB_NUM_NODES: $num_nodes"
    echo "SLURM_NODEID (node rank): $node_rank"
    echo "GPUs on this node: $num_gpus"
    echo "Master address: $master_addr"
    echo "Head node ip: $head_node_ip" 
fi

GRAD_ACCUM_STEPS=$(( GLOBAL_BATCH_SIZE / (BATCH_SIZE * num_nodes * num_gpus) ))

run_exp() {

    EXPERIMENT_NAME=${MODEL_VARIANT}_seq_${SEQ_LENGTH}_bs_${BATCH_SIZE}_global_bs_${GLOBAL_BATCH_SIZE}_steps_${NUM_STEPS}_lr_${LEARNING_RATE}_minlr_ratio_${MIN_LEARNING_RATE_RATIO}_warmup_${WARMUP_RATIO}
    if [[ "$SHUFFLE" == "True" ]]; then
        EXPERIMENT_NAME=${EXPERIMENT_NAME}_shuffle
    fi

    SAVE_DIR="${CKPT_DIR}/${EXPERIMENT_NAME}"
    mkdir -p ${SAVE_DIR}


    export WANDB_API_KEY='7d11bbca76b3081b6bd1efbbcf1572aab26c5d56'

    SCRIPT_ARGS="\
    training/finetuning.py \
    --model_variant=${MODEL_VARIANT} \
    --ckpt_load_path=${SAVE_DIR} \
    --ckpt_save_path=${SAVE_DIR} \
    --data_path=${DATA_DIR} \
    --bos_token=${BOS_TOKEN} \
    --eos_token=${EOS_TOKEN} \
    --sharding_strategy=hsdp \
    --fsdp_activation_checkpointing=True \
    --selective_checkpointing=1 \
    --mixed_precision=True \
    --low_cpu_fsdp=True \
    --use_torch_compile=True \
    --seq_length=${SEQ_LENGTH} \
    --batch_size=${BATCH_SIZE} \
    --grad_accum_steps=${GRAD_ACCUM_STEPS} \
    --num_steps=${NUM_STEPS} \
    --learning_rate=${LEARNING_RATE} \
    --min_learning_rate_ratio=${MIN_LEARNING_RATE_RATIO} \
    --warmup_ratio=${WARMUP_RATIO} \
    --report_interval=${REPORT_INTERVAL} \
    --checkpoint_interval=${CHECKPOINT_INTERVAL} \
    --tracker=wandb \
    --tracker_project_name=llm_pruning \
    --tracker_run_name=${EXPERIMENT_NAME}
    "

    port=$(get_free_port)
    num_gpus=$(nvidia-smi -L | wc -l)

    export PROJ_DIR=$(pwd)
    export PYTHONPATH=$(pwd):${PYTHONPATH}

    if [[ $num_nodes -gt 1 ]]; then
        srun torchrun \
            --nnodes=${num_nodes} \
            --nproc_per_node=${num_gpus} \
            --node_rank=${node_rank} \
            --rdzv_id=${SLURM_JOB_ID} \
            --rdzv_backend=c10d \
            --rdzv_endpoint=${head_node_ip}:54224 \
            ${SCRIPT_ARGS}
    else
        echo "PORT: $port"
        MASTER_PORT=${port} \
            torchrun \
            --master_port ${port} \
            --nproc_per_node ${num_gpus} \
            ${SCRIPT_ARGS}
    fi

}

run_exp

