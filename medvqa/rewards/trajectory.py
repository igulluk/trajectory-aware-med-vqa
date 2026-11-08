"""Trajectory-aware process rewards (R_DTW and R_NW).

A completion and its reference are each turned into a sequence of steps
(reasoning sentences followed by the final answer), embedded with a
Sentence-Transformer, and compared with a sequence-alignment score. A
completion that does not follow the ``<think>/<answer>`` format gets 0.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from .alignment import dtw_reward, nw_reward
from .logging_utils import debug_enabled, debug_log
from .parsing import completion_texts, extract_answer, extract_think, is_well_formatted, trajectory_steps

logger = logging.getLogger(__name__)

TrajectoryScore = Callable[[np.ndarray, np.ndarray], float]


def load_sentence_encoder(model_name: str = "all-mpnet-base-v2", device: str = "cpu"):
    """Load the Sentence-Transformer used to embed reasoning steps.

    It runs on CPU by default to leave GPU memory to the policy model.
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence encoder %s on %s", model_name, device)
    return SentenceTransformer(model_name, device=device)


def _embed(encoder, sentences: list[str]) -> np.ndarray:
    return encoder.encode(sentences, normalize_embeddings=True, convert_to_numpy=True)


def score_trajectory(
    completion: str,
    reference_reasoning: str,
    reference_answer: str,
    encoder,
    score_fn: TrajectoryScore,
) -> float:
    """Score one completion against its reference trajectory; 0.0 if it cannot be scored."""
    if not is_well_formatted(completion):
        return 0.0
    reasoning = extract_think(completion)
    if not reasoning or not reference_reasoning:
        return 0.0

    generated = trajectory_steps(reasoning, extract_answer(completion))
    reference = trajectory_steps(reference_reasoning.strip(), (reference_answer or "").strip())
    if not generated or not reference:
        return 0.0
    return score_fn(_embed(encoder, generated), _embed(encoder, reference))


def make_trajectory_reward(name: str, encoder, score_fn: TrajectoryScore) -> Callable[..., list[float]]:
    """Wrap a trajectory score as a TRL-style reward function called ``name``.

    The trainer logs each reward under its function's ``__name__``.
    """

    def reward(completions, reasoning_answer, gt_answer, **kwargs) -> list[float]:
        rewards = []
        for text, ref_reasoning, ref_answer in zip(completion_texts(completions), reasoning_answer, gt_answer):
            try:
                value = score_trajectory(text, ref_reasoning, ref_answer, encoder, score_fn)
            except Exception as exc:  # never let one malformed sample crash a GRPO step
                if debug_enabled():
                    logger.warning("%s failed on a sample: %s", name, exc)
                value = 0.0
            rewards.append(float(value))
            debug_log(f"{name}: {value:.4f}", Content=text, Reference_reasoning=ref_reasoning,
                      Reference_answer=ref_answer)
        return rewards

    reward.__name__ = reward.__qualname__ = name
    return reward


def make_dtw_reward(encoder) -> Callable[..., list[float]]:
    """R_DTW = 1[format valid] * exp(-DTW(S_g, S_gt) / m)."""
    return make_trajectory_reward("dtw_embedding_reward", encoder, dtw_reward)


def make_nw_reward(
    encoder,
    gap_open: float = -0.5,
    gap_extend: float = -0.1,
    mismatch_penalty: float = -0.3,
    similarity_threshold: float = 0.3,
) -> Callable[..., list[float]]:
    """R_NW = 1[format valid] * normalised Needleman-Wunsch alignment score."""

    def score_fn(generated: np.ndarray, reference: np.ndarray) -> float:
        return nw_reward(generated, reference, gap_open, gap_extend, mismatch_penalty, similarity_threshold)

    return make_trajectory_reward("nw_alignment_reward", encoder, score_fn)
