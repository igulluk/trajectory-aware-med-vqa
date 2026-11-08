"""Prompt templates and chat-message construction shared by SFT, GRPO and inference."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT_REASONING = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

SYSTEM_PROMPT_DIRECT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant provides a direct answer to the question."
)

# The generated reasoning trajectories end with a sentence of the form
# "The final answer is: ...". It is removed from the SFT target because the
# answer is supplied separately inside the <answer> tags.
FINAL_ANSWER_MARKER = "\nThe final answer is"


def system_prompt(include_reasoning: bool = True) -> str:
    return SYSTEM_PROMPT_REASONING if include_reasoning else SYSTEM_PROMPT_DIRECT


def format_question(question: str) -> str:
    return f"Question: {question}"


def format_target(reasoning: str, answer: str) -> str:
    """Render a reasoning trajectory and answer in the ``<think>/<answer>`` format."""
    return f"<think> {reasoning} </think><answer> {answer} </answer>"


def build_messages(
    example: dict[str, Any],
    include_reasoning: bool = True,
    image_placeholder: bool = False,
) -> list[dict[str, Any]]:
    """Build the system + user turns for one VQA example.

    The question text precedes the image, matching the order used in training.

    Args:
        example: A dataset record with at least ``question`` and ``image``.
        include_reasoning: Choose the ``<think>/<answer>`` system prompt
            (``True``) or the direct-answer prompt (``False``).
        image_placeholder: If ``True``, emit ``{"type": "image"}`` without a path;
            the GRPO trainer attaches the image from the ``image`` column itself.
    """
    system_text = example.get("system") or system_prompt(include_reasoning)
    user_content: list[dict[str, Any]] = [{"type": "text", "text": format_question(example.get("question"))}]
    if image_placeholder:
        user_content.append({"type": "image"})
    elif example.get("image"):
        user_content.append({"type": "image", "image": example["image"]})

    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": user_content},
    ]
