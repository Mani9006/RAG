"""Thin, real ML model layer used by the platform.

A scikit-learn pipeline (standardisation + logistic regression) is deliberately
simple and fast so the whole lifecycle runs in seconds, yet it is a genuine model
with genuine metrics — nothing is mocked. The public surface (``train_model``,
``evaluate``, ``save_model``, ``load_model``) is what the registry and
orchestrator depend on.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_PARAMS: dict[str, Any] = {"C": 1.0, "max_iter": 500, "solver": "lbfgs"}


def train_model(X: np.ndarray, y: np.ndarray, params: dict[str, Any] | None = None, seed: int = 42) -> Pipeline:
    """Fit a standardised logistic-regression pipeline."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=seed, **p)),
        ]
    )
    model.fit(X, y)
    return model


def evaluate(model: Pipeline, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Compute headline classification metrics on a holdout set."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    # Guard against a degenerate single-class holdout for AUC.
    auc = float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else 0.5
    return {
        "auc": round(auc, 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
    }


def batch_accuracy(model: Pipeline, X: np.ndarray, y: np.ndarray) -> float:
    """Accuracy on a single streamed window (used by the performance monitor)."""
    pred = model.predict(X)
    return float(accuracy_score(y, pred))


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)
