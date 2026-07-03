from __future__ import annotations

import numpy as np

from docintel import DOC_TYPES
from docintel.embedding import ContentEmbedder, nearest_duplicates
from docintel.evaluation import accuracy
from docintel.generator import generate_document, iter_documents
from docintel.router import DocumentRouter, doc_features


def _build(n, seed):
    X, y = [], []
    for d in iter_documents(n, seed=seed):
        toks = [{"text": t.text, "x": t.x, "y": t.y, "w": t.w, "h": t.h}
                for t in d.tokens]
        X.append(doc_features(toks))
        y.append(d.doc_type)
    return np.array(X), y


def test_router_beats_baseline():
    Xtr, ytr = _build(1500, 1)
    Xte, yte = _build(600, 2)
    router = DocumentRouter(seed=1).fit(Xtr, ytr)
    acc = accuracy(yte, list(router.predict(Xte)))
    vals, counts = np.unique(ytr, return_counts=True)
    base = counts.max() / counts.sum()
    assert acc > base + 0.3
    assert acc > 0.9  # layout stats are highly separable


def test_embedding_dedup_finds_identical():
    docs = [generate_document(i, seed=5) for i in range(120)]
    texts = [" ".join(t.text for t in d.tokens) for d in docs]
    # inject an exact duplicate of doc 0
    texts.append(texts[0])
    emb = ContentEmbedder(dim=48, seed=1).fit(texts)
    vecs = emb.encode(texts)
    assert vecs.shape[0] == len(texts)
    # self-similarity normalized to ~1
    assert abs(np.linalg.norm(vecs[0]) - 1.0) < 1e-4
    dups = nearest_duplicates(vecs, threshold=0.999)
    pairs = {(i, j) for i, j, _ in dups}
    assert (0, len(texts) - 1) in pairs
