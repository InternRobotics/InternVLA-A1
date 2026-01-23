#!/usr/bin/env bash

###############################################################################
################################# ENV config ##################################

export HF_HOME=${HF_HOME}

WANDB_TOKEN=${WANDB_TOKEN}
CONDA_ROOT=${_CONDA_ROOT}
CONDA_ENV=internvla_a1
#CONDA_ENV=lerobot_lab
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

###############################################################################

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-6379}
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

PROC_PER_NODE="${PROC_PER_NODE:-2}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_HOME="/usr/local/cuda-12.8"
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

###############################################################################
############################## TRAINING config ################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"

cd ${PROJ_ROOT}

PRETRAINED_CKPT=InternRobotics/InternVLA-A1-3B-RoboTwin
BASE_OUTPUT_PATH=${PROJ_ROOT}/evaluation/RoboTwin/output
TASK_CONFIG=demo_clean
TASK_IDX=0 # adjust_bottle

OUTPUT_PATH=${BASE_OUTPUT_PATH}/${TASK_CONFIG}/${TASK_IDX}

cd ${PROJ_ROOT}/third_party/RoboTwin
python ../../evaluation/RoboTwin/inference.py \
    --args.ckpt_path $PRETRAINED_CKPT \
    --args.video_dir $OUTPUT_PATH \
    --args.task_config $TASK_CONFIG \
    --args.task_idx $TASK_IDX