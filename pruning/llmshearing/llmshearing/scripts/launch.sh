#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=gputest
#SBATCH --time=1:00:00

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=512gb
#SBATCH --constraint gpu80
#SBATCH --output=/scratch/gpfs/mengzhou/space2/out/logs/%x-%j.out

PROJ_DIR='/scratch/yx3038/Research/pruning/LLM-Shearing'
LOG_DIR="${PROJ_DIR}/logs"

# num_nodes=$(scontrol show job $SLURM_JOB_ID | grep NodeList=della | wc -l)
num_nodes=$(scontrol show hostnames $(hostname) | wc -l)
master_addr=$(scontrol show hostnames $(hostname) | head -n 1)
SLURM_GPUS_PER_NODE=$(nvidia-smi -L | wc -l)

export MASTER_ADDR=$master_addr
echo $SLURM_GPUS_PER_NODE

export WORLD_SIZE=$(( $num_nodes * $SLURM_GPUS_PER_NODE ))
export MASTER_PORT=$(( 10000 + RANDOM % 10000 ))

echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE
echo "num_nodes="$num_nodes

# composer $PROJ_DIR/llmshearing/train.py "$@" 

torchrun \
  --nproc_per_node=$SLURM_GPUS_PER_NODE \
  --nnodes=$num_nodes \
  --node_rank=${SLURM_NODEID:-0} \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  $PROJ_DIR/llmshearing/train.py "$@" 
 
