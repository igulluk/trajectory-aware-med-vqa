#!/usr/bin/env bash
# Stage 2: GRPO with trajectory-aware rewards on top of the SFT model.
# Usage: bash scripts/train_grpo.sh [dataset] [rewards] [model]
#   dataset: vqa-rad | slake-vqa | path-vqa | pmc-vqa | omnimed-vqa | vqa-med       (default: slake-vqa)
#   rewards: space-separated subset of "accuracy format dtw_embedding nw_alignment"  (default: "accuracy format dtw_embedding")
#   model:   Hugging Face model id                                                  (default: Qwen/Qwen2.5-VL-3B-Instruct)
# The SFT adapter is read from $SFT_CHECKPOINT (default: checkpoints/sft/<dataset>/checkpoint-best).
set -euo pipefail

DATASET=${1:-slake-vqa}
REWARDS=${2:-"accuracy format dtw_embedding"}
MODEL=${3:-Qwen/Qwen2.5-VL-3B-Instruct}
NUM_GPUS=${NUM_GPUS:-2}
SFT_CHECKPOINT=${SFT_CHECKPOINT:-checkpoints/sft/${DATASET}/checkpoint-best}
TAG=${REWARDS// /_}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/grpo/${DATASET}/${TAG}}

# shellcheck disable=SC2086  # REWARDS is intentionally word-split into several arguments
torchrun --nproc_per_node="${NUM_GPUS}" scripts/grpo.py \
    --model_name_or_path "${MODEL}" \
    --sft_checkpoint_path "${SFT_CHECKPOINT}" \
    --data_json_path "data/${DATASET}_all.json" \
    --dataset_name MedVQA \
    --reward_funcs ${REWARDS} \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "$(basename "${MODEL}")-${DATASET}-GRPO-${TAG}" \
    --deepspeed configs/zero2.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant \
    --num_train_epochs 10 \
    --bf16 \
    --attn_implementation sdpa \
    --gradient_checkpointing false \
    --eval_strategy steps \
    --eval_steps 10 \
    --save_steps 10 \
    --save_total_limit 10 \
    --save_only_model false \
    --load_best_model_at_end true \
    --metric_for_best_model eval_reward \
    --greater_is_better true \
    --early_stopping_patience 25 \
    --logging_steps 1 \
    --dataloader_num_workers 8 \
    --report_to wandb \
    --seed 42 \
    "${@:4}"
