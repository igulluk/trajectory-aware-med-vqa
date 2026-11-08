import hashlib
import re

import numpy as np
import pytest


class HashingEncoder:
    """Deterministic bag-of-words stand-in for a Sentence-Transformer (no downloads, CPU only)."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vector(self, sentence: str) -> np.ndarray:
        vec = np.zeros(self.dim)
        for token in re.findall(r"\w+", sentence.lower()):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        return vec

    def encode(self, sentences, normalize_embeddings=True, convert_to_numpy=True):
        out = np.stack([self._vector(s) for s in sentences])
        if normalize_embeddings:
            out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        return out


@pytest.fixture
def encoder():
    return HashingEncoder()


def as_completions(*texts):
    """Wrap strings in TRL's conversational completion format."""
    return [[{"role": "assistant", "content": t}] for t in texts]
