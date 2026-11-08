#!/usr/bin/env bash
# Stage 1: supervised fine-tuning on reasoning-augmented data.
# Usage: bash scripts/train_sft.sh [dataset] [model]
#   dataset: vqa-rad | slake-vqa | path-vqa | pmc-vqa | omnimed-vqa | vqa-med  (default: slake-vqa)
#   model:   Hugging Face model id                                         (default: Qwen/Qwen2.5-VL-3B-Instruct)
set -euo pipefail

DATASET=${1:-slake-vqa}
MODEL=${2:-Qwen/Qwen2.5-VL-3B-Instruct}
NUM_GPUS=${NUM_GPUS:-1}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/sft/${DATASET}}

torchrun --nproc_per_node="${NUM_GPUS}" scripts/sft.py \
    --model_name_or_path "${MODEL}" \
    --data_json_path "data/${DATASET}_all.json" \
    --dataset_name MedVQA \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "$(basename "${MODEL}")-${DATASET}-SFT" \
    --deepspeed configs/zero3.json \
    --max_seq_length 512 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 6 \
    --learning_rate 1e-4 \
    --num_train_epochs 200 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --gradient_checkpointing false \
    --eval_strategy steps \
    --eval_steps 10 \
    --save_steps 10 \
    --save_total_limit 10 \
    --save_only_model false \
    --load_best_model_at_end true \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --early_stopping_patience 20 \
    --logging_steps 1 \
    --dataloader_num_workers 8 \
    --report_to wandb \
    --seed 42 \
    "${@:3}"
