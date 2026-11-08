#!/usr/bin/env bash
# Stage 3: generate predictions on the test split.
# Usage: bash scripts/run_inference.sh [dataset] [rewards] [model]   (same arguments as train_grpo.sh)
set -euo pipefail

DATASET=${1:-slake-vqa}
REWARDS=${2:-"accuracy format dtw_embedding"}
MODEL=${3:-Qwen/Qwen2.5-VL-3B-Instruct}
TAG=${REWARDS// /_}
SFT_CHECKPOINT=${SFT_CHECKPOINT:-checkpoints/sft/${DATASET}/checkpoint-best}
GRPO_CHECKPOINT=${GRPO_CHECKPOINT:-checkpoints/grpo/${DATASET}/${TAG}/checkpoint-best}

python scripts/inference.py \
    --base_model_name "${MODEL}" \
    --sft_checkpoint_path "${SFT_CHECKPOINT}" \
    --checkpoint_path "${GRPO_CHECKPOINT}" \
    --data_path "data/${DATASET}_all.json" \
    --output_path "results/${DATASET}/${TAG}.json" \
    --max_new_tokens 512 \
    --seed 42 \
    "${@:4}"
