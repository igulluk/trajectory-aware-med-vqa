"""Reward functions for GRPO.

=================  ==========================================================
``format``         R_form: output follows ``<think>...</think><answer>...</answer>``
``accuracy``       R_exact: case-insensitive exact match of the final answer
``dtw_embedding``  R_DTW: DTW alignment of sentence-embedding trajectories
``nw_alignment``   R_NW: Needleman-Wunsch alignment with affine gaps
=================  ==========================================================
"""

from __future__ import annotations

from typing import Callable, Sequence

from .outcome import accuracy_reward, format_reward
from .trajectory import load_sentence_encoder, make_dtw_reward, make_nw_reward

AVAILABLE_REWARDS = ("accuracy", "format", "dtw_embedding", "nw_alignment")
TRAJECTORY_REWARDS = ("dtw_embedding", "nw_alignment")


def build_reward_funcs(
    names: Sequence[str],
    sentence_encoder: str = "all-mpnet-base-v2",
    nw_gap_open: float = -0.5,
    nw_gap_extend: float = -0.1,
    nw_mismatch_penalty: float = -0.3,
    nw_similarity_threshold: float = 0.3,
    encoder=None,
) -> list[Callable]:
    """Instantiate reward functions by name, in the given order.

    The sentence encoder is loaded once, and only if a trajectory reward is requested.
    """
    unknown = [n for n in names if n not in AVAILABLE_REWARDS]
    if unknown:
        raise ValueError(f"Unknown reward(s) {unknown}; choose from {list(AVAILABLE_REWARDS)}")

    if encoder is None and any(n in TRAJECTORY_REWARDS for n in names):
        encoder = load_sentence_encoder(sentence_encoder)

    factories = {
        "accuracy": lambda: accuracy_reward,
        "format": lambda: format_reward,
        "dtw_embedding": lambda: make_dtw_reward(encoder),
        "nw_alignment": lambda: make_nw_reward(
            encoder, nw_gap_open, nw_gap_extend, nw_mismatch_penalty, nw_similarity_threshold
        ),
    }
    return [factories[name]() for name in names]


__all__ = [
    "AVAILABLE_REWARDS",
    "accuracy_reward",
    "build_reward_funcs",
    "format_reward",
    "load_sentence_encoder",
    "make_dtw_reward",
    "make_nw_reward",
]
