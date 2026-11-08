# Adapted from R1-V (https://github.com/StarsfieldAI/R1-V) and open-r1,
# Copyright 2025 The HuggingFace Team, licensed under the Apache License 2.0.
"""GRPO fine-tuning of a Qwen-VL model with trajectory-aware process rewards.

The policy is initialised from the base model with the SFT LoRA adapter merged
in, and a fresh LoRA adapter is trained on top with GRPO.

Example:
    torchrun --nproc_per_node=2 scripts/grpo.py \\
        --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \\
        --sft_checkpoint_path checkpoints/sft/slake-vqa/checkpoint-best \\
        --data_json_path data/slake-vqa_all.json \\
        --reward_funcs accuracy format dtw_embedding \\
        --output_dir checkpoints/grpo/slake-vqa/accuracy_format_dtw \\
        --deepspeed configs/zero2.json ...

See ``scripts/train_grpo.sh`` for the full configuration used in the paper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified
from transformers import EarlyStoppingCallback, set_seed
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from medvqa.checkpointing import load_model_weights, plan_resume
from medvqa.data import load_splits
from medvqa.modeling import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    LORA_TARGET_MODULES,
    get_model_class,
    merge_adapter,
)
from medvqa.prompts import build_messages
from medvqa.rewards import AVAILABLE_REWARDS, build_reward_funcs

logger = logging.getLogger(__name__)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "dtw_embedding"],
        metadata={"help": f"Rewards to combine (summed). Choices: {', '.join(AVAILABLE_REWARDS)}."},
    )
    data_json_path: Optional[str] = field(default=None, metadata={"help": "Path to a data/*_all.json file."})
    image_root: Optional[str] = field(
        default=None, metadata={"help": "Directory the relative image paths are resolved against (default: cwd)."}
    )
    sft_checkpoint_path: Optional[str] = field(
        default=None, metadata={"help": "SFT LoRA checkpoint merged into the base model before GRPO."}
    )
    max_pixels: Optional[int] = field(default=DEFAULT_MAX_PIXELS, metadata={"help": "Max image pixels."})
    min_pixels: Optional[int] = field(default=DEFAULT_MIN_PIXELS, metadata={"help": "Min image pixels."})
    early_stopping_patience: Optional[int] = field(
        default=None, metadata={"help": "Stop after this many evaluations without improvement."}
    )
    sentence_transformer_model: str = field(
        default="all-mpnet-base-v2", metadata={"help": "Sentence encoder for the trajectory rewards."}
    )
    nw_gap_open: float = field(default=-0.5, metadata={"help": "NW gap-opening penalty (negative)."})
    nw_gap_extend: float = field(default=-0.1, metadata={"help": "NW gap-extension penalty (negative)."})
    nw_mismatch_penalty: float = field(
        default=-0.3, metadata={"help": "NW penalty added to pairs with similarity below the threshold."}
    )
    nw_similarity_threshold: float = field(default=0.3, metadata={"help": "NW mismatch similarity threshold."})


def to_grpo_prompt(example):
    # The trainer reads the image from the `image` column, so only a placeholder is needed here.
    return {"prompt": build_messages(example, include_reasoning=True, image_placeholder=True)}


def build_policy(script_args, model_args):
    """Return the base model id, or the base model with the SFT adapter merged in."""
    if script_args.sft_checkpoint_path is None:
        return model_args.model_name_or_path
    # Load on CPU; DeepSpeed moves the weights to the GPUs.
    base_model = get_model_class(model_args.model_name_or_path).from_pretrained(
        model_args.model_name_or_path,
        torch_dtype="auto",
        attn_implementation=model_args.attn_implementation,
        device_map="cpu",
    )
    return merge_adapter(base_model, script_args.sft_checkpoint_path, torch_device="cpu")


def build_callbacks(script_args, training_args):
    if script_args.early_stopping_patience is None:
        return None
    if not training_args.load_best_model_at_end:
        logger.warning("Early stopping requires load_best_model_at_end; enabling it.")
        training_args.load_best_model_at_end = True
    if training_args.metric_for_best_model is None:
        training_args.metric_for_best_model = "eval_reward"
        training_args.greater_is_better = True
    return [EarlyStoppingCallback(early_stopping_patience=script_args.early_stopping_patience)]


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    resume = plan_resume(training_args.output_dir, training_args.resume_from_checkpoint)
    if resume.weights_only_checkpoint is not None:
        training_args.ignore_data_skip = True

    reward_funcs = build_reward_funcs(
        script_args.reward_funcs,
        sentence_encoder=script_args.sentence_transformer_model,
        nw_gap_open=script_args.nw_gap_open,
        nw_gap_extend=script_args.nw_gap_extend,
        nw_mismatch_penalty=script_args.nw_mismatch_penalty,
        nw_similarity_threshold=script_args.nw_similarity_threshold,
    )

    dataset = load_splits(script_args.data_json_path, image_root=script_args.image_root).map(to_grpo_prompt)
    logger.info("Dataset: %s", dataset)

    trainer_cls = Qwen2VLGRPOVLLMTrainerModified if training_args.use_vllm else Qwen2VLGRPOTrainer
    trainer = trainer_cls(
        model=build_policy(script_args, model_args),
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split],
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        callbacks=build_callbacks(script_args, training_args),
    )

    if resume.weights_only_checkpoint is not None:
        try:
            step = load_model_weights(resume.weights_only_checkpoint, trainer.model)
            logger.info("Loaded model weights from step %s; optimizer restarts.", step)
        except Exception as exc:
            logger.error("Could not load weights from %s (%s); training from scratch.",
                         resume.weights_only_checkpoint, exc)

    trainer.train(resume_from_checkpoint=resume.checkpoint)
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    model_args.use_peft = True
    model_args.lora_target_modules = LORA_TARGET_MODULES
    main(script_args, training_args, model_args)
