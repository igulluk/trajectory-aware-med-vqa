"""Outcome rewards: output format (R_form) and exact-match answer accuracy (R_exact)."""

from __future__ import annotations

from .logging_utils import debug_log
from .parsing import ANSWER_PATTERN_SINGLE_LINE, completion_texts, is_well_formatted


def _answer_or_full_text(text: str) -> str:
    match = ANSWER_PATTERN_SINGLE_LINE.search(text)
    return match.group(1).strip() if match else text.strip()


def format_reward(completions, **kwargs) -> list[float]:
    """1.0 if the completion is exactly ``<think>...</think><answer>...</answer>``, else 0.0."""
    return [1.0 if is_well_formatted(text) else 0.0 for text in completion_texts(completions)]


def accuracy_reward(completions, gt_answer, **kwargs) -> list[float]:
    """1.0 if the predicted answer equals the reference answer (case-insensitive), else 0.0.

    The prediction is the content of the ``<answer>`` tags, or the whole
    completion if no tags are found.
    """
    rewards = []
    for text, reference in zip(completion_texts(completions), gt_answer):
        prediction = _answer_or_full_text(text)
        reward = 1.0 if prediction.lower() == reference.strip().lower() else 0.0
        rewards.append(reward)
        debug_log(f"Accuracy reward: {reward}", Content=text, Reference=reference)
    return rewards
