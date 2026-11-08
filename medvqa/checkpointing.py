"""Checkpoint resumption, including resuming a DeepSpeed run on a different number of GPUs.

DeepSpeed ZeRO optimizer states are sharded per rank and cannot be loaded with
a different world size. In that case we restore only the model weights and
let the optimizer start fresh.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

import torch
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class ResumePlan:
    """How to resume: through the Trainer (``checkpoint``) or by loading weights only."""

    checkpoint: Optional[str] = None
    weights_only_checkpoint: Optional[str] = None


def checkpoint_world_size(checkpoint_path: str) -> Optional[int]:
    """Number of ranks a DeepSpeed checkpoint was saved with, or ``None`` if unknown."""
    for pattern in ("*_optim_states.pt", "bf16_zero_pp_rank_*"):
        shards = glob.glob(os.path.join(checkpoint_path, "global_step*", pattern))
        if shards:
            return len(shards)
    if os.path.exists(os.path.join(checkpoint_path, "zero_to_fp32.py")):
        shards = glob.glob(os.path.join(checkpoint_path, "global_step*", "mp_rank_*"))
        if shards:
            return len(shards)
    return None


def checkpoint_step(checkpoint_path: str) -> Optional[int]:
    match = re.search(r"checkpoint-(\d+)", checkpoint_path)
    return int(match.group(1)) if match else None


def plan_resume(output_dir: str, resume_from_checkpoint: Optional[str] = None) -> ResumePlan:
    """Choose how to resume from the latest checkpoint in ``output_dir``."""
    if resume_from_checkpoint is not None:
        return ResumePlan(checkpoint=resume_from_checkpoint)
    if not os.path.isdir(output_dir):
        return ResumePlan()
    last = get_last_checkpoint(output_dir)
    if last is None:
        return ResumePlan()

    saved_world_size = checkpoint_world_size(last)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if saved_world_size is not None and saved_world_size != world_size:
        logger.warning(
            "Checkpoint %s was saved with world_size=%d but the current world_size is %d. "
            "Loading model weights only; optimizer states will be reinitialised.",
            last, saved_world_size, world_size,
        )
        return ResumePlan(weights_only_checkpoint=last)

    logger.info("Resuming training from %s", last)
    return ResumePlan(checkpoint=last)


def load_model_weights(checkpoint_path: str, model) -> Optional[int]:
    """Load model weights (no optimizer state) from a HF, PEFT or DeepSpeed ZeRO checkpoint.

    Returns the global step of the checkpoint.
    """
    step = checkpoint_step(checkpoint_path)

    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file

        model.load_state_dict(load_file(safetensors_path, device="cpu"), strict=False)
        return step

    bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.exists(bin_path):
        model.load_state_dict(torch.load(bin_path, map_location="cpu"), strict=False)
        return step

    adapter_files = ("adapter_model.safetensors", "adapter_model.bin")
    if any(os.path.exists(os.path.join(checkpoint_path, f)) for f in adapter_files):
        if hasattr(model, "load_adapter"):
            model.load_adapter(checkpoint_path, adapter_name="default")
            return step
        raise RuntimeError(f"Model cannot load the PEFT adapter in {checkpoint_path}")

    zero_to_fp32 = os.path.join(checkpoint_path, "zero_to_fp32.py")
    if os.path.exists(zero_to_fp32) or glob.glob(os.path.join(checkpoint_path, "global_step*")):
        try:
            from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

            model.load_state_dict(get_fp32_state_dict_from_zero_checkpoint(checkpoint_path), strict=False)
            return step
        except Exception as exc:
            logger.warning("Direct ZeRO load failed (%s); converting with zero_to_fp32.py", exc)
            with tempfile.TemporaryDirectory() as tmpdir:
                output_file = os.path.join(tmpdir, "pytorch_model.bin")
                subprocess.run([sys.executable, zero_to_fp32, checkpoint_path, output_file],
                               check=True, capture_output=True)
                model.load_state_dict(torch.load(output_file, map_location="cpu"), strict=False)
                return step

    raise RuntimeError(f"Could not find model weights in checkpoint: {checkpoint_path}")
