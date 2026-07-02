from __future__ import annotations

import numpy as np

from distillkit import data
from distillkit.features import HashingVectorizer, tokenize
from distillkit.metrics import softmax
from distillkit.model import MLPClassifier


def test_tokenize():
    assert tokenize("Hello, WORLD! 123 test-case") == ["hello", "world", "123", "test", "case"]


def test_vectorizer_deterministic_and_shaped():
    vec = HashingVectorizer(dim=64)
    a = vec.transform(["the quick brown fox", "another sentence"])
    b = vec.transform(["the quick brown fox", "another sentence"])
    assert a.shape == (2, 64)
    assert np.array_equal(a, b)  # stateless + stable hash => reproducible


def test_vectorizer_signal_present():
    vec = HashingVectorizer(dim=128)
    x = vec.transform(["excellent excellent excellent"])
    assert np.count_nonzero(x) >= 1
    assert np.abs(x).max() >= 3 - 1e-9  # repeated token accumulates


def test_softmax_normalizes():
    p = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p[1], 1 / 3)


def test_softmax_temperature_sharpens():
    z = np.array([[2.0, 1.0, 0.0]])
    cold = softmax(z, temperature=0.5)
    hot = softmax(z, temperature=5.0)
    # Higher temperature -> flatter distribution (larger entropy).
    ent = lambda p: -np.sum(p * np.log(p))
    assert ent(hot[0]) > ent(cold[0])


def test_mlp_learns_separable_task():
    vec = HashingVectorizer(dim=256)
    tr = data.generate(1200, seed=0, domain="source")
    te = data.generate(600, seed=1, domain="source")
    Xtr, ytr = vec.transform(tr.texts), tr.y
    Xte, yte = vec.transform(te.texts), te.y
    clf = MLPClassifier(256, 32, data.N_CLASSES, seed=0).fit(Xtr, ytr, epochs=40, lr=1e-2)
    acc = clf.score(Xte, yte)
    assert acc > 0.85, f"MLP failed to learn (acc={acc:.3f})"
    assert clf.predict(Xte).shape == (600,)
    assert np.allclose(clf.predict_proba(Xte).sum(axis=1), 1.0)


def test_mlp_param_count():
    clf = MLPClassifier(100, 16, 9, seed=0)
    # W1(100x16)+b1(16)+W2(16x9)+b2(9) = 1600+16+144+9
    assert clf.n_params() == 1600 + 16 + 144 + 9
