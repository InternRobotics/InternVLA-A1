#!/usr/bin/env bash
set -euo pipefail

source /mnt/shared-storage-user/caijunhao/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_qwen3

cd /mnt/shared-storage-user/caijunhao/lerobot

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/mnt/shared-storage-user/internvla/Users/caijunhao/cache/huggingface

# DATASET_REPO_ID="$(
#     find data/lift2_sim_* data/piper_sim_* data/a2d_sim_* \
#         -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
#         | sed 's#^data/##' \
#         | sort
# )"

DATASET_REPO_ID="$(
    find data/piper_sim_other* \
        -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
        | sed 's#^data/##' \
        | sort
)"

echo "RAW MULTILINE:"
echo "$DATASET_REPO_ID"

# convert to single line
DATASET_REPO_ID=$(echo "$DATASET_REPO_ID" | tr '\n' ' ' | xargs)

echo "FINAL ONELINE:"
echo "$DATASET_REPO_ID"

python src/lerobot/scripts/lerobot_aggregate.py --repo-ids ${DATASET_REPO_ID} --aggr-repo-id piper_sim/other_part2