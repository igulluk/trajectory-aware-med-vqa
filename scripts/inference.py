"""Generate answers on the test split with a fine-tuned (SFT or GRPO) Qwen-VL model.

For a GRPO checkpoint, pass the SFT adapter too: GRPO was trained on top of
``base + SFT`` merged weights, so both adapters are merged in that order.

Example:
    python scripts/inference.py \\
        --base_model_name Qwen/Qwen2.5-VL-3B-Instruct \\
        --sft_checkpoint_path checkpoints/sft/slake-vqa/checkpoint-best \\
        --checkpoint_path checkpoints/grpo/slake-vqa/accuracy_format_dtw/checkpoint-best \\
        --data_path data/slake-vqa_all.json \\
        --output_path results/slake-vqa_inference.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, set_seed

from medvqa.data import load_splits
from medvqa.modeling import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS, get_model_class, merge_adapter
from medvqa.prompts import build_messages, format_target

logger = logging.getLogger(__name__)

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def str2bool(value: str) -> bool:
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint_path", required=True,
                   help="Fine-tuned LoRA adapter, or a full model if --is_peft false.")
    p.add_argument("--base_model_name", default="Qwen/Qwen2.5-VL-3B-Instruct",
                   help="Base model the adapters were trained on.")
    p.add_argument("--sft_checkpoint_path", default=None,
                   help="SFT adapter to merge first (required for GRPO checkpoints).")
    p.add_argument("--data_path", required=True, help="Path to a data/*_all.json file.")
    p.add_argument("--image_root", default=None, help="Directory the relative image paths are resolved against.")
    p.add_argument("--output_path", default="results/inference_results.json")
    p.add_argument("--num_samples", type=int, default=None, help="Only run on the first N test samples.")
    p.add_argument("--include_reasoning", type=str2bool, default=True, help="Use the <think>/<answer> system prompt.")
    p.add_argument("--is_peft", type=str2bool, default=True, help="Whether --checkpoint_path is a LoRA adapter.")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--do_sample", type=str2bool, default=True)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_pixels", type=int, default=DEFAULT_MIN_PIXELS)
    p.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS)
    p.add_argument("--torch_dtype", default="bfloat16", choices=list(DTYPES))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--trust_remote_code", type=str2bool, default=True)
    return p.parse_args()


def load_model_and_processor(args):
    processor = AutoProcessor.from_pretrained(
        args.base_model_name,
        trust_remote_code=args.trust_remote_code,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model_class = get_model_class(args.base_model_name)
    load_kwargs = dict(
        torch_dtype=DTYPES[args.torch_dtype],
        trust_remote_code=args.trust_remote_code,
        device_map="auto" if args.device == "cuda" else None,
    )
    if args.is_peft:
        model = model_class.from_pretrained(args.base_model_name, **load_kwargs)
        if args.sft_checkpoint_path is not None:
            model = merge_adapter(model, args.sft_checkpoint_path)
        model = merge_adapter(model, args.checkpoint_path)
    else:
        model = model_class.from_pretrained(args.checkpoint_path, **load_kwargs)

    model.eval()
    if args.device != "cuda" or not hasattr(model, "hf_device_map"):
        model = model.to(args.device)
    return model, processor


@torch.no_grad()
def generate(model, processor, example, args) -> str:
    messages = build_messages(example, include_reasoning=args.include_reasoning)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs or None,
        videos=video_inputs or None,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature if args.do_sample else None,
        num_return_sequences=1,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]


def reference_text(example, include_reasoning: bool) -> str:
    if include_reasoning:
        return format_target(example.get("reasoning_answer"), example.get("gt_answer"))
    return f"{example.get('gt_answer')}"


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    args = parse_args()
    set_seed(args.seed)
    logger.info("Arguments: %s", vars(args))

    test_set = load_splits(args.data_path, image_root=args.image_root)["test"]
    if args.num_samples is not None:
        test_set = test_set.select(range(min(args.num_samples, len(test_set))))
    logger.info("Test samples: %d", len(test_set))

    model, processor = load_model_and_processor(args)

    passthrough_skip = {"reasoning_answer", "gt_answer", "image", "split", "system"}
    results = []
    for idx, example in enumerate(tqdm(test_set, desc="Generating")):
        record = {
            "idx": idx,
            "image": example.get("image", ""),
            "ground_truth": reference_text(example, args.include_reasoning),
            "include_reasoning": args.include_reasoning,
        }
        try:
            record["generated"] = generate(model, processor, example, args)
            record.update({k: v for k, v in example.items() if k not in passthrough_skip})
        except Exception as exc:
            logger.error("Sample %d failed: %s", idx, exc)
            record.update({
                "reas_ans": example.get("reasoning_answer"),
                "final_ans": example.get("gt_answer"),
                "generated": f"ERROR: {exc}",
                "error": True,
            })
        results.append(record)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n_errors = sum(r.get("error", False) for r in results)
    logger.info("Saved %d predictions to %s (%d errors)", len(results), args.output_path, n_errors)


if __name__ == "__main__":
    main()
