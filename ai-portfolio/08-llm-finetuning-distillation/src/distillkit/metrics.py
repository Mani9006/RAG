"""Numerically-stable losses and metrics shared across trainers."""
from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax with optional temperature ``T`` (softmax(z / T))."""
    z = logits / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood of the true classes."""
    n = y.shape[0]
    return float(-np.log(probs[np.arange(n), y] + eps).mean())


def accuracy(logits_or_probs: np.ndarray, y: np.ndarray) -> float:
    return float((logits_or_probs.argmax(axis=1) == y).mean())


def macro_f1(pred: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((pred == c) & (y == c)))
        fp = int(np.sum((pred == c) & (y != c)))
        fn = int(np.sum((pred != c) & (y == c)))
        if tp == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Mean KL(p || q) over rows."""
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=1).mean())
