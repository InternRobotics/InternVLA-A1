#!/usr/bin/env bash
set -euo pipefail

###############################################################################
################################# ENV config ##################################

export HF_HOME=${HF_HOME}

WANDB_TOKEN=${WANDB_TOKEN}
CONDA_ROOT=${_CONDA_ROOT}
CONDA_ENV=internvla_a1

source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

wandb login ${WANDB_TOKEN}

###############################################################################

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-6379}
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

PROC_PER_NODE="${PROC_PER_NODE:-2}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

export CUDA_HOME="/usr/local/cuda-12.8"
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

###############################################################################
############################## TRAINING config ################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"

cd ${PROJ_ROOT}

# 1. policy config
POLICY="qwena1"

# 2. dataset config
DATASET_REPO_ID="$1"
ACTION_TYPE=delta  # abs | delta
USE_EXTERNAL_STATS=true

# 3. output config
BASE_OUTPUT_DIR="outputs/${POLICY}"
JOB_NAME="$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-${DATASET_REPO_ID//[\/ ]/_}-${ACTION_TYPE}-scratch"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"

ARGS=(
    --multi_gpu
    --num_processes="${NUM_PROCESSES}"
    --num_machines="${NODE_COUNT}"
    --machine_rank="${NODE_RANK}"
    --main_process_ip="${MASTER_ADDR}"
    --main_process_port="${MASTER_PORT}" 
    src/lerobot/scripts/lerobot_train.py

    --output_dir="${OUTPUT_DIR}"
    --num_workers=12
    --job_name="${JOB_NAME}"

    # ---- Policy ----
    --policy.type=${POLICY}
    --policy.repo_id=lerobot_lab/${POLICY}
    --policy.push_to_hub=false
    --policy.gradient_checkpointing=false
    --policy.dtype=bfloat16
    --policy.optimizer_lr=1e-4
    --policy.scheduler_warmup_steps=2000
    --policy.scheduler_decay_steps=60000
    --policy.scheduler_decay_lr=2.5e-6
    --policy.freeze_vision_encoder=false
    --policy.train_expert_only=false
    --policy.train_vlm_only=false
    --policy.qwen3_vl_variant=qwen3_vl_28l
    --policy.action_expert_variant=qwen3_28l

    # ---- Dataset ----
    --dataset.type=${POLICY}
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.action_mode="${ACTION_TYPE}"
    --dataset.use_external_stats=true
    --dataset.external_stats_path=${HF_HOME}/lerobot/stats/${ACTION_TYPE}/${DATASET_REPO_ID}/stats.json

    # ---- Training ----
    --seed=42
    --batch_size=8
    --steps=60000
    # --eval_freq=60000
    --save_freq=20000
    --log_freq=200

    # ---- Logging ----
    --wandb.enable=true
    --wandb.project=lerobot_lab_${POLICY}
    --wandb.mode=offline
)

accelerate launch "${ARGS[@]}"