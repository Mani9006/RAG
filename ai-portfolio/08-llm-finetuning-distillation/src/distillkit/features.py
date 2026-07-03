"""Deterministic hashed text features (the "tokenizer + embedding" surrogate).

Real LLMs learn dense token embeddings. Here we use the classic *hashing trick*
to turn free-text instructions into a fixed-width numeric vector with **zero
learned vocabulary and zero external state** — so the whole pipeline is
reproducible and offline. Each token is hashed to a bucket with a signed count
(signed hashing reduces collision bias, Weinberger et al. 2009).

This is intentionally simple: the interesting techniques in this repo (LoRA,
distillation, eval gates) live *downstream* of featurization and are model- and
feature-agnostic.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric runs."""
    return _TOKEN_RE.findall(text.lower())


def _hash64(token: str) -> int:
    """Stable 64-bit hash (blake2b) — independent of PYTHONHASHSEED."""
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")


class HashingVectorizer:
    """Map an iterable of strings to a dense ``(n, dim)`` signed bag-of-hashes.

    Stateless and deterministic: ``transform`` never depends on the corpus, so
    train/val/test are featurized identically and new text is handled at
    inference with no vocabulary drift.
    """

    def __init__(self, dim: int = 1024) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)

    def transform(self, texts: Sequence[str] | Iterable[str]) -> np.ndarray:
        texts = list(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                h = _hash64(tok)
                idx = h % self.dim
                sign = 1.0 if (h >> 63) & 1 else -1.0
                out[i, idx] += sign
        return out

    # Sklearn-ish alias.
    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:  # noqa: D401
        return self.transform(texts)
