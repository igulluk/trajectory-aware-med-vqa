"""Parsing helpers for ``<think> ... </think><answer> ... </answer>`` completions."""

from __future__ import annotations

import re
from typing import Optional

# Leading/trailing whitespace is allowed because SFT targets start with "\n\n".
FORMAT_PATTERN = re.compile(r"\s*<think>.*?</think>\s*<answer>.*?</answer>\s*", re.DOTALL)
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# Single-line variant used by the exact-match reward: an answer that spans
# several lines is not extracted and therefore never counts as an exact match.
ANSWER_PATTERN_SINGLE_LINE = re.compile(r"<answer>(.*?)</answer>")


def completion_texts(completions) -> list[str]:
    """Extract the generated text from TRL conversational completions."""
    return [completion[0]["content"] for completion in completions]


def is_well_formatted(text: str) -> bool:
    return FORMAT_PATTERN.fullmatch(text) is not None


def _extract(pattern: re.Pattern, text: str) -> Optional[str]:
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def extract_think(text: str) -> str:
    return _extract(THINK_PATTERN, text) or ""


def extract_answer(text: str) -> str:
    return _extract(ANSWER_PATTERN, text) or ""


def split_sentences(text: str) -> list[str]:
    """Split a reasoning chain into non-empty sentences on ``'.'``.

    Case is preserved, which gives better sentence embeddings.
    """
    if not text:
        return []
    return [sentence.strip() for sentence in text.split(".") if sentence.strip()]


def trajectory_steps(reasoning: str, answer: str) -> list[str]:
    """Sentences of a reasoning chain followed by the final answer as the last step."""
    steps = split_sentences(reasoning)
    if answer:
        steps.append(answer)
    return steps
