"""Model-class resolution and adapter loading for the Qwen-VL family."""

from __future__ import annotations

import logging

from transformers import Qwen2VLForConditionalGeneration

try:  # transformers >= 4.49
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:  # pragma: no cover - depends on installed version
    Qwen2_5_VLForConditionalGeneration = None

try:  # transformers >= 4.57
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:  # pragma: no cover - depends on installed version
    Qwen3VLForConditionalGeneration = None

logger = logging.getLogger(__name__)

# Pixel budget used throughout the paper (~32 to ~128 visual tokens per image).
DEFAULT_MIN_PIXELS = 32 * 28 * 28
DEFAULT_MAX_PIXELS = 128 * 28 * 28

# LoRA is applied to the attention projections of the language model.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def is_qwen3_vl(model_name_or_path: str) -> bool:
    name = model_name_or_path.lower()
    return "qwen3-vl" in name or "qwen3_vl" in name


def is_qwen2_5_vl(model_name_or_path: str) -> bool:
    name = model_name_or_path.lower()
    return any(tag in name for tag in ("qwen2.5-vl", "qwen2_5-vl", "qwen2.5_vl"))


def get_model_class(model_name_or_path: str):
    """Return the ``*ForConditionalGeneration`` class matching a Qwen-VL checkpoint name."""
    if is_qwen3_vl(model_name_or_path):
        if Qwen3VLForConditionalGeneration is None:
            raise ImportError("Qwen3-VL requires a newer `transformers` release (>= 4.57).")
        cls = Qwen3VLForConditionalGeneration
    elif is_qwen2_5_vl(model_name_or_path):
        if Qwen2_5_VLForConditionalGeneration is None:
            raise ImportError("Qwen2.5-VL requires `transformers` >= 4.49.")
        cls = Qwen2_5_VLForConditionalGeneration
    else:
        cls = Qwen2VLForConditionalGeneration
    logger.info("Using %s for %s", cls.__name__, model_name_or_path)
    return cls


def merge_adapter(base_model, adapter_path: str, **peft_kwargs):
    """Load a LoRA adapter onto ``base_model`` and merge it into the weights."""
    from peft import PeftModel

    logger.info("Merging LoRA adapter from %s", adapter_path)
    return PeftModel.from_pretrained(base_model, adapter_path, **peft_kwargs).merge_and_unload()
