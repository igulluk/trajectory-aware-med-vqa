"""Sequence-alignment primitives over sentence-embedding trajectories.

Both functions take two arrays of L2-normalised sentence embeddings:
``generated`` with shape ``(n, d)`` and ``reference`` with shape ``(m, d)``.
"""

from __future__ import annotations

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def dtw_distance(generated: np.ndarray, reference: np.ndarray) -> float:
    """Dynamic Time Warping distance with Euclidean local cost (``fastdtw``, radius 1)."""
    distance, _ = fastdtw(generated, reference, dist=euclidean)
    return float(distance)


def dtw_reward(generated: np.ndarray, reference: np.ndarray) -> float:
    """``exp(-DTW / m)``: in ``(0, 1]``, where ``m`` is the reference length."""
    return float(np.exp(-dtw_distance(generated, reference) / len(reference)))


def needleman_wunsch_score(
    generated: np.ndarray,
    reference: np.ndarray,
    gap_open: float,
    gap_extend: float,
    mismatch_penalty: float,
    similarity_threshold: float = 0.3,
) -> float:
    """Global alignment score with affine gaps (Needleman-Wunsch / Gotoh).

    The match score of a pair is its cosine similarity, minus ``|mismatch_penalty|``
    when the similarity is below ``similarity_threshold``. Penalties are negative.

    The DP keeps three matrices: ``M`` (ending in a match), ``X`` (ending in a
    gap in ``generated``, i.e. a skipped reference step) and ``Y`` (ending in a
    gap in ``reference``, i.e. an extra generated step).
    """
    n, m = len(generated), len(reference)
    if n == 0 or m == 0:
        return 0.0

    similarity = generated @ reference.T
    score = np.where(similarity < similarity_threshold, similarity + mismatch_penalty, similarity)

    M = np.full((n + 1, m + 1), -np.inf)
    X = np.full((n + 1, m + 1), -np.inf)
    Y = np.full((n + 1, m + 1), -np.inf)
    M[0, 0] = 0.0
    X[0, 1:] = gap_open + np.arange(m) * gap_extend
    Y[1:, 0] = gap_open + np.arange(n) * gap_extend

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            M[i, j] = score[i - 1, j - 1] + max(M[i - 1, j - 1], X[i - 1, j - 1], Y[i - 1, j - 1])
            X[i, j] = max(M[i, j - 1] + gap_open, X[i, j - 1] + gap_extend, Y[i, j - 1] + gap_open)
            Y[i, j] = max(M[i - 1, j] + gap_open, Y[i - 1, j] + gap_extend, X[i - 1, j] + gap_open)

    return float(max(M[n, m], X[n, m], Y[n, m]))


def nw_reward(
    generated: np.ndarray,
    reference: np.ndarray,
    gap_open: float,
    gap_extend: float,
    mismatch_penalty: float,
    similarity_threshold: float = 0.3,
) -> float:
    """Needleman-Wunsch score min-max normalised to ``[0, 1]``.

    The upper bound is ``m`` (every reference step perfectly matched) and the
    lower bound is ``m * gap_open``.
    """
    m = len(reference)
    score = needleman_wunsch_score(
        generated, reference, gap_open, gap_extend, mismatch_penalty, similarity_threshold
    )
    best, worst = float(m), m * gap_open
    return float(min(1.0, max(0.0, (score - worst) / (best - worst))))
