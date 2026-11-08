# Adapted from R1-V (https://github.com/StarsfieldAI/R1-V) and open-r1,
# Copyright 2025 The HuggingFace Team, licensed under the Apache License 2.0.
"""Supervised fine-tuning (LoRA) of a Qwen-VL model on reasoning-augmented medical VQA.

Targets have the form ``<think> reasoning </think><answer> answer </answer>``,
or just the answer with ``--include_reasoning false``.

Example:
    torchrun --nproc_per_node=1 scripts/sft.py \\
        --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \\
        --data_json_path data/slake-vqa_all.json \\
        --output_dir checkpoints/sft/slake-vqa \\
        --deepspeed configs/zero3.json ...

See ``scripts/train_sft.sh`` for the full configuration used in the paper.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
import transformers
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, EarlyStoppingCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from medvqa.data import load_splits
from medvqa.modeling import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    LORA_TARGET_MODULES,
    get_model_class,
    is_qwen3_vl,
)
from medvqa.prompts import FINAL_ANSWER_MARKER, build_messages, format_target

logger = logging.getLogger(__name__)


@dataclass
class SFTScriptArguments(ScriptArguments):
    data_json_path: Optional[str] = field(default=None, metadata={"help": "Path to a data/*_all.json file."})
    image_root: Optional[str] = field(
        default=None, metadata={"help": "Directory the relative image paths are resolved against (default: cwd)."}
    )
    include_reasoning: bool = field(
        default=True, metadata={"help": "Train on <think>/<answer> targets (True) or on the bare answer (False)."}
    )
    min_pixels: int = field(default=DEFAULT_MIN_PIXELS, metadata={"help": "Min image pixels."})
    max_pixels: int = field(default=DEFAULT_MAX_PIXELS, metadata={"help": "Max image pixels."})
    early_stopping_patience: Optional[int] = field(
        default=None, metadata={"help": "Stop after this many evaluations without eval_loss improvement."}
    )


def build_sft_conversation(example, include_reasoning: bool = True):
    """System + user turns followed by the assistant target."""
    if include_reasoning:
        reasoning = example["reasoning_answer"].split(FINAL_ANSWER_MARKER)[0]
        target = format_target(reasoning, example["gt_answer"])
    else:
        target = f"{example['gt_answer']}"
    messages = build_messages(example, include_reasoning=include_reasoning)
    messages.append({"role": "assistant", "content": f"\n\n{target}"})
    return messages


class VQACollator:
    """Tokenise conversations and images; padding and image tokens are masked from the loss."""

    def __init__(self, processor, max_seq_length: int, include_reasoning: bool = True):
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.include_reasoning = include_reasoning
        self.image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)

    def __call__(self, examples):
        conversations = [build_sft_conversation(ex, self.include_reasoning) for ex in examples]
        # NOTE: add_generation_prompt=True matches the configuration the paper's checkpoints were trained with.
        texts = [
            self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in conversations
        ]
        images = [process_vision_info(conv)[0] for conv in conversations]
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
        )
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.image_token_id] = -100
        batch["labels"] = labels
        return batch


def setup_logging(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def load_processor(model_args, script_args):
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        min_pixels=script_args.min_pixels,
        max_pixels=script_args.max_pixels,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def load_model(model_args, training_args):
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ("auto", None) else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    if not is_qwen3_vl(model_args.model_name_or_path):  # Qwen3-VL does not accept use_cache at init
        model_kwargs["use_cache"] = not training_args.gradient_checkpointing
    return get_model_class(model_args.model_name_or_path).from_pretrained(model_args.model_name_or_path, **model_kwargs)


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)
    setup_logging(training_args)
    logger.info("Model arguments: %s", model_args)
    logger.info("Script arguments: %s", script_args)

    dataset = load_splits(script_args.data_json_path, image_root=script_args.image_root)
    processor = load_processor(model_args, script_args)
    model = load_model(model_args, training_args)

    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.remove_unused_columns = False

    callbacks = None
    if script_args.early_stopping_patience is not None:
        if not training_args.load_best_model_at_end:
            logger.warning("Early stopping requires load_best_model_at_end; enabling it.")
            training_args.load_best_model_at_end = True
        callbacks = [EarlyStoppingCallback(early_stopping_patience=script_args.early_stopping_patience)]

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split],
        processing_class=processor.tokenizer,
        data_collator=VQACollator(processor, training_args.max_seq_length, script_args.include_reasoning),
        peft_config=get_peft_config(model_args),
        callbacks=callbacks,
    )

    checkpoint = training_args.resume_from_checkpoint
    if checkpoint is None and os.path.isdir(training_args.output_dir):
        checkpoint = get_last_checkpoint(training_args.output_dir)
        if checkpoint is not None:
            logger.info("Resuming training from %s", checkpoint)

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info("Model saved to %s", training_args.output_dir)

    card_kwargs = {"dataset_name": script_args.dataset_name, "tags": ["R1-V", "medical-vqa"]}
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**card_kwargs)
        trainer.model.config.use_cache = True  # restore the KV cache for inference
        trainer.model.config.save_pretrained(training_args.output_dir)

    if training_args.push_to_hub:
        trainer.push_to_hub(**card_kwargs)
        processor.push_to_hub(training_args.hub_model_id)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    model_args.use_peft = True
    model_args.lora_target_modules = LORA_TARGET_MODULES
    main(script_args, training_args, model_args)
