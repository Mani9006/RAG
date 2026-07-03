"""Automated promotion gate — the CI check that guards model deployment.

Before any candidate model is promoted (base -> fine-tuned, or teacher ->
distilled student), it must clear a set of hard checks:

1. **Absolute floor** — val accuracy >= ``min_accuracy``.
2. **No regression** — val accuracy >= ``baseline_accuracy - max_regression``.
3. **Per-class floor** (optional) — worst-class recall >= ``min_class_recall``,
   so a model can't pass by ignoring a rare class.

The gate returns a structured, serializable verdict so it can be wired into CI
and rendered on a scorecard. A failing check blocks promotion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GateCheck:
    name: str
    passed: bool
    value: float
    threshold: float
    detail: str = ""


@dataclass
class GateResult:
    passed: bool
    checks: list[GateCheck] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"PROMOTION {'APPROVED' if self.passed else 'BLOCKED'}"]
        for c in self.checks:
            mark = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{mark}] {c.name}: {c.value:.4f} (threshold {c.threshold:.4f}) {c.detail}")
        return "\n".join(lines)


def _per_class_recall(pred: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    recs = np.ones(n_classes)
    for c in range(n_classes):
        mask = y == c
        if mask.any():
            recs[c] = float((pred[mask] == c).mean())
    return recs


def evaluate_gate(
    val_accuracy: float,
    *,
    min_accuracy: float = 0.80,
    baseline_accuracy: float | None = None,
    max_regression: float = 0.02,
    pred: np.ndarray | None = None,
    y: np.ndarray | None = None,
    n_classes: int | None = None,
    min_class_recall: float | None = None,
) -> GateResult:
    """Evaluate all configured checks and return a combined verdict."""
    checks: list[GateCheck] = []

    checks.append(
        GateCheck(
            name="absolute_accuracy_floor",
            passed=val_accuracy >= min_accuracy,
            value=float(val_accuracy),
            threshold=float(min_accuracy),
        )
    )

    if baseline_accuracy is not None:
        floor = baseline_accuracy - max_regression
        checks.append(
            GateCheck(
                name="no_regression_vs_baseline",
                passed=val_accuracy >= floor,
                value=float(val_accuracy),
                threshold=float(floor),
                detail=f"(baseline {baseline_accuracy:.4f}, tol {max_regression:.4f})",
            )
        )

    if min_class_recall is not None and pred is not None and y is not None and n_classes is not None:
        recs = _per_class_recall(pred, y, n_classes)
        worst = float(recs.min())
        checks.append(
            GateCheck(
                name="worst_class_recall_floor",
                passed=worst >= min_class_recall,
                value=worst,
                threshold=float(min_class_recall),
                detail=f"(worst class = {int(recs.argmin())})",
            )
        )

    return GateResult(passed=all(c.passed for c in checks), checks=checks)
