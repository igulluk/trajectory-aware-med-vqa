import numpy as np
import pytest
from conftest import as_completions

from medvqa.rewards import accuracy_reward, build_reward_funcs, format_reward, make_dtw_reward, make_nw_reward
from medvqa.rewards.alignment import dtw_reward, needleman_wunsch_score, nw_reward
from medvqa.rewards.parsing import split_sentences, trajectory_steps

REASONING = (
    "The image is an axial CT scan of the chest.\n"
    "The lung fields appear clear without consolidation.\n"
    "There is no evidence of pleural effusion.\n"
    "The final answer is: no"
)
GOOD = "\n\n<think> The image is an axial CT scan of the chest. The lung fields appear clear without consolidation. " \
       "There is no evidence of pleural effusion. </think><answer> no </answer>"


# --------------------------------------------------------------------------- parsing


def test_split_sentences_drops_empty_pieces():
    assert split_sentences(" A. B..  C. ") == ["A", "B", "C"]
    assert split_sentences("") == []


def test_trajectory_appends_answer_as_last_step():
    assert trajectory_steps("A. B.", "yes") == ["A", "B", "yes"]
    assert trajectory_steps("A. B.", "") == ["A", "B"]


# --------------------------------------------------------------------------- outcome rewards


@pytest.mark.parametrize(
    "text, expected",
    [
        ("<think> r </think><answer> a </answer>", 1.0),
        ("\n\n<think> r\nmore </think>\n<answer> a </answer>\n", 1.0),
        ("<answer> a </answer>", 0.0),
        ("<think> r </think><answer> a </answer> trailing text", 0.0),
    ],
)
def test_format_reward(text, expected):
    assert format_reward(as_completions(text)) == [expected]


def test_accuracy_reward_is_case_insensitive_exact_match():
    completions = as_completions(
        "<think> r </think><answer> Yes </answer>",
        "<think> r </think><answer> yes, clearly </answer>",
        "yes",  # untagged output is compared as a whole
    )
    assert accuracy_reward(completions, gt_answer=["yes", "yes", "yes"]) == [1.0, 0.0, 1.0]


# --------------------------------------------------------------------------- alignment primitives


def _unit(rows):
    rows = np.asarray(rows, dtype=float)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_dtw_reward_is_one_for_identical_sequences():
    seq = _unit(np.eye(4))
    assert dtw_reward(seq, seq) == pytest.approx(1.0)


def test_dtw_reward_tolerates_repeated_steps_but_penalises_unrelated_ones():
    ref = _unit(np.eye(4)[:3])
    stretched = _unit(np.eye(4)[[0, 0, 1, 2]])  # same content, first step stated twice
    with_noise = _unit(np.eye(4)[[0, 3, 1, 2]])  # an unrelated step inserted
    assert dtw_reward(stretched, ref) == pytest.approx(1.0)
    assert dtw_reward(with_noise, ref) < 1.0


def test_nw_identical_sequences_score_m_and_normalise_to_one():
    seq = _unit(np.eye(3))
    assert needleman_wunsch_score(seq, seq, -0.5, -0.1, -0.3) == pytest.approx(3.0)
    assert nw_reward(seq, seq, -0.5, -0.1, -0.3) == pytest.approx(1.0)


def test_nw_penalises_a_missing_step_with_gap_open():
    ref = _unit(np.eye(3))
    missing = ref[[0, 2]]
    # two matches (1 + 1) plus one opened gap
    assert needleman_wunsch_score(missing, ref, -0.5, -0.1, -0.3) == pytest.approx(1.5)


def test_nw_reward_is_clamped_to_unit_interval():
    ref = _unit(np.eye(4)[:2])
    unrelated = -ref  # cosine similarity -1 everywhere
    assert nw_reward(unrelated, ref, -0.5, -0.1, -0.3) == 0.0


# --------------------------------------------------------------------------- trajectory rewards


def test_dtw_reward_function(encoder):
    reward = make_dtw_reward(encoder)
    assert reward.__name__ == "dtw_embedding_reward"

    off_topic = "<think> The sky is blue. Bananas are yellow. </think><answer> no </answer>"
    malformed = "The lungs are clear. no"
    scores = reward(as_completions(GOOD, off_topic, malformed), reasoning_answer=[REASONING] * 3, gt_answer=["no"] * 3)

    assert 0.0 < scores[1] < scores[0] <= 1.0
    assert scores[2] == 0.0


def test_nw_reward_function(encoder):
    reward = make_nw_reward(encoder)
    assert reward.__name__ == "nw_alignment_reward"

    shuffled = "<think> There is no evidence of pleural effusion. The image is an axial CT scan of the chest. " \
               "The lung fields appear clear without consolidation. </think><answer> no </answer>"
    good, reordered = reward(as_completions(GOOD, shuffled), reasoning_answer=[REASONING] * 2, gt_answer=["no"] * 2)
    assert 0.0 <= reordered < good <= 1.0


def test_build_reward_funcs(encoder):
    funcs = build_reward_funcs(["accuracy", "format", "dtw_embedding"], encoder=encoder)
    assert [f.__name__ for f in funcs] == ["accuracy_reward", "format_reward", "dtw_embedding_reward"]
    with pytest.raises(ValueError):
        build_reward_funcs(["bleu"], encoder=encoder)
