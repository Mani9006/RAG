"""Evaluation metrics: per-field P/R/F1, table accuracy, doc-type accuracy.

Pure NumPy so the same code runs in tests and in the benchmark harness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    support: int


def prf_per_class(y_true: list[str], y_pred: list[str],
                  labels: list[str]) -> dict[str, PRF]:
    """Per-class precision/recall/F1 (multi-class, one-vs-rest)."""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    out: dict[str, PRF] = {}
    for lab in labels:
        tp = int(np.sum((yt == lab) & (yp == lab)))
        fp = int(np.sum((yt != lab) & (yp == lab)))
        fn = int(np.sum((yt == lab) & (yp != lab)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[lab] = PRF(prec, rec, f1, int(np.sum(yt == lab)))
    return out


def macro_f1(prf: dict[str, PRF], exclude: tuple[str, ...] = ("O",)) -> float:
    vals = [v.f1 for k, v in prf.items() if k not in exclude and v.support > 0]
    return float(np.mean(vals)) if vals else 0.0


def micro_f1(y_true: list[str], y_pred: list[str],
             exclude: tuple[str, ...] = ("O",)) -> float:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    keep = ~np.isin(yt, exclude)
    tp = int(np.sum((yt == yp) & keep))
    total_true = int(np.sum(keep))
    pred_pos = int(np.sum(~np.isin(yp, exclude)))
    prec = tp / pred_pos if pred_pos else 0.0
    rec = tp / total_true if total_true else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true: list[str], y_pred: list[str],
                     labels: list[str]) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[idx[t], idx[p]] += 1
    return m


def table_cell_accuracy(gold: list[list[str]], pred: list[list[str]]) -> float:
    """Fraction of gold cells whose text is exactly recovered at the same
    (row, col) position in the predicted grid. Shape mismatches count as misses.
    """
    total = sum(len(r) for r in gold)
    if total == 0:
        return 1.0 if not pred else 0.0
    hit = 0
    for i, row in enumerate(gold):
        for j, cell in enumerate(row):
            if i < len(pred) and j < len(pred[i]) and pred[i][j] == cell:
                hit += 1
    return hit / total


def table_shape_match(gold: list[list[str]], pred: list[list[str]]) -> bool:
    if len(gold) != len(pred):
        return False
    return all(len(g) == len(p) for g, p in zip(gold, pred))
