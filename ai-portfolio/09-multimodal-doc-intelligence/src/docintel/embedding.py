"""Offline content embeddings for search / dedup.

After extraction, each document has a bag of typed field values. We serialize
that to a normalized text string and embed it with TF-IDF + TruncatedSVD (LSA)
— L2-normalized so cosine similarity ranks semantically related documents and
surfaces near-duplicates. Deterministic and dependency-light; a real sentence
encoder drops in behind the same ``fit``/``encode`` interface.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class ContentEmbedder:
    def __init__(self, dim: int = 128, min_df: int = 2, seed: int = 42) -> None:
        self.dim = dim
        self.seed = seed
        self._vec = TfidfVectorizer(lowercase=True, min_df=min_df,
                                    ngram_range=(1, 2), sublinear_tf=True,
                                    max_features=40_000)
        self._svd: TruncatedSVD | None = None

    def fit(self, texts: Sequence[str]) -> "ContentEmbedder":
        tfidf = self._vec.fit_transform(texts)
        dim = min(self.dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        self._svd = TruncatedSVD(n_components=max(2, dim), random_state=self.seed)
        self._svd.fit(tfidf)
        self.dim = self._svd.n_components
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        assert self._svd is not None, "call fit first"
        return normalize(self._svd.transform(self._vec.transform(texts))).astype(np.float32)


def nearest_duplicates(emb: np.ndarray, threshold: float = 0.97) -> list[tuple[int, int, float]]:
    """Return (i, j, cosine) pairs above ``threshold`` (i<j). O(n^2), for demo scale."""
    sims = emb @ emb.T
    dups = []
    n = emb.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= threshold:
                dups.append((i, j, float(sims[i, j])))
    return dups
