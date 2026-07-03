"""Drift monitoring: data drift (PSI / KL per feature) and performance drift.

Two monitors run over the live-scoring stream:

* :class:`DriftMonitor` — unsupervised input drift. Bins each feature against a
  frozen reference distribution and reports the Population Stability Index (PSI)
  and symmetrised KL divergence per feature. PSI thresholds follow the standard
  rule of thumb (0.1 moderate, 0.25 major).
* :class:`PerformanceMonitor` — supervised concept drift. Tracks the model's
  rolling accuracy against its registered baseline and fires when the sustained
  relative drop crosses a threshold.

Both emit structured :class:`Alert` objects that the orchestrator consumes.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

_EPS = 1e-6


def _bin_edges(reference: np.ndarray, bins: int) -> np.ndarray:
    """Quantile bin edges from the reference sample, deduplicated and padded."""
    qs = np.linspace(0, 1, bins + 1)
    edges = np.quantile(reference, qs)
    edges = np.unique(edges)
    if edges.size < 2:  # degenerate constant feature
        edges = np.array([reference.min() - 1e-6, reference.max() + 1e-6])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _distribution(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(x, bins=edges)
    proportions = counts / max(counts.sum(), 1)
    return np.clip(proportions, _EPS, None)


def psi(reference: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between a reference and an actual sample.

    PSI = Σ (a_i - e_i) * ln(a_i / e_i). Symmetric, non-negative, 0 iff identical.
    """
    edges = _bin_edges(reference, bins)
    e = _distribution(reference, edges)
    a = _distribution(actual, edges)
    return float(np.sum((a - e) * np.log(a / e)))


def kl_divergence(reference: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """KL(actual || reference) over the reference's quantile bins (nats)."""
    edges = _bin_edges(reference, bins)
    e = _distribution(reference, edges)
    a = _distribution(actual, edges)
    return float(np.sum(a * np.log(a / e)))


@dataclass
class Alert:
    kind: str          # "data_drift" | "performance_drift"
    t: int             # window index
    metric: str
    value: float
    threshold: float
    detail: str = ""


class DriftMonitor:
    """Per-feature PSI/KL monitor against a frozen reference window."""

    def __init__(
        self,
        reference_X: np.ndarray,
        feature_names: Sequence[str],
        bins: int = 10,
        psi_threshold: float = 0.25,
    ) -> None:
        self.reference = np.asarray(reference_X, dtype=np.float64)
        self.feature_names = list(feature_names)
        self.bins = bins
        self.psi_threshold = psi_threshold
        # Freeze reference bin edges once for stable, comparable PSI over time.
        self._edges = [_bin_edges(self.reference[:, j], bins) for j in range(self.reference.shape[1])]
        self._ref_dist = [
            _distribution(self.reference[:, j], self._edges[j]) for j in range(self.reference.shape[1])
        ]

    def _psi_col(self, x: np.ndarray, j: int) -> float:
        a = _distribution(x, self._edges[j])
        e = self._ref_dist[j]
        return float(np.sum((a - e) * np.log(a / e)))

    def _kl_col(self, x: np.ndarray, j: int) -> float:
        a = _distribution(x, self._edges[j])
        e = self._ref_dist[j]
        return float(np.sum(a * np.log(a / e)))

    def psi_per_feature(self, window_X: np.ndarray) -> dict[str, float]:
        w = np.asarray(window_X, dtype=np.float64)
        return {self.feature_names[j]: self._psi_col(w[:, j], j) for j in range(w.shape[1])}

    def kl_per_feature(self, window_X: np.ndarray) -> dict[str, float]:
        w = np.asarray(window_X, dtype=np.float64)
        return {self.feature_names[j]: self._kl_col(w[:, j], j) for j in range(w.shape[1])}

    def check(self, window_X: np.ndarray, t: int) -> tuple[dict[str, float], Alert | None]:
        """Return per-feature PSI and an alert if the max crosses the threshold."""
        per_feat = self.psi_per_feature(window_X)
        worst_feat = max(per_feat, key=per_feat.get)
        worst = per_feat[worst_feat]
        alert = None
        if worst >= self.psi_threshold:
            alert = Alert(
                kind="data_drift",
                t=t,
                metric="psi",
                value=round(worst, 4),
                threshold=self.psi_threshold,
                detail=f"feature '{worst_feat}' PSI={worst:.3f}",
            )
        return per_feat, alert


class PerformanceMonitor:
    """Rolling accuracy monitor detecting sustained performance/concept drift."""

    def __init__(self, baseline_accuracy: float, rel_drop_threshold: float = 0.07, window: int = 3) -> None:
        self.baseline = baseline_accuracy
        self.rel_drop_threshold = rel_drop_threshold
        self._recent: deque[float] = deque(maxlen=window)
        self.history: list[float] = []

    def update(self, accuracy: float, t: int) -> tuple[float, Alert | None]:
        """Record a window's accuracy; alert on sustained relative degradation."""
        self._recent.append(accuracy)
        self.history.append(accuracy)
        rolling = float(np.mean(self._recent))
        rel_drop = (self.baseline - rolling) / max(self.baseline, _EPS)
        alert = None
        if len(self._recent) == self._recent.maxlen and rel_drop >= self.rel_drop_threshold:
            alert = Alert(
                kind="performance_drift",
                t=t,
                metric="rel_accuracy_drop",
                value=round(rel_drop, 4),
                threshold=self.rel_drop_threshold,
                detail=f"rolling acc={rolling:.3f} vs baseline={self.baseline:.3f}",
            )
        return rolling, alert
