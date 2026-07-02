"""Document-type classifier / router.

Aggregates a document's tokens into a compact feature vector (layout stats +
printed-keyword presence) and classifies it into one of
:data:`docintel.DOC_TYPES`. Downstream, the predicted type selects the
extraction template to apply (invoices/POs get table + totals extraction, forms
get pure key/value), which is exactly how a production doc pipeline routes.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from . import DOC_TYPES

_KEYWORDS = ["invoice", "purchase", "order", "receipt", "form", "bill", "po",
             "total", "tax", "subtotal", "date", "employee", "description",
             "qty", "amount", "record"]

DOC_FEATURE_NAMES = [
    "n_tokens", "n_lines", "max_x", "max_y", "mean_y", "frac_numeric",
    "frac_dollar", "n_table_rows",
] + [f"kw_{k}" for k in _KEYWORDS]


def doc_features(tokens: list[dict]) -> np.ndarray:
    """Aggregate one document's tokens into a fixed-length feature vector."""
    n = len(tokens)
    if n == 0:
        return np.zeros(len(DOC_FEATURE_NAMES), dtype=np.float32)
    xs = np.array([t["x"] for t in tokens])
    ys = np.array([t["y"] for t in tokens])
    texts = [t["text"] for t in tokens]
    lows = [t.lower().rstrip(":") for t in texts]
    n_lines = len(np.unique(np.round(ys / 6)))
    frac_numeric = np.mean([any(c.isdigit() for c in t) for t in texts])
    frac_dollar = np.mean(["$" in t for t in texts])
    # crude table-row count: distinct y bands that contain a $ amount
    dollar_y = np.round(ys[[i for i, t in enumerate(texts) if "$" in t]] / 6) \
        if any("$" in t for t in texts) else np.array([])
    n_table_rows = len(np.unique(dollar_y))
    base = [n, n_lines, xs.max(), ys.max(), ys.mean(), frac_numeric,
            frac_dollar, n_table_rows]
    kw = [float(any(k == w for w in lows)) for k in _KEYWORDS]
    return np.array(base + kw, dtype=np.float32)


class DocumentRouter:
    """RandomForest document-type classifier over aggregate layout features."""

    def __init__(self, n_estimators: int = 120, seed: int = 42) -> None:
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=14, n_jobs=-1,
            random_state=seed, class_weight="balanced")
        self.classes_ = np.array(DOC_TYPES)

    def fit(self, X: np.ndarray, y: list[str]) -> "DocumentRouter":
        self.clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)
