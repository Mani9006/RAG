"""CI/CD promotion gate.

A challenger model may only reach production if it passes every pre-promotion
check. The gate is a pure, deterministic function of the candidate's schema and
metrics plus the incumbent's metrics, so it is trivially testable and auditable:

1. **Schema compatibility** — candidate consumes the same feature contract.
2. **Quality bar** — candidate clears an absolute minimum metric.
3. **No regression** — candidate beats the incumbent on the holdout by a margin.

Each check contributes a :class:`GateCheck`; the gate passes only if all pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class GateCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class GateResult:
    passed: bool
    checks: list[GateCheck] = field(default_factory=list)

    def summary(self) -> str:
        return "; ".join(f"{c.name}={'PASS' if c.passed else 'FAIL'}" for c in self.checks)


class PromotionGate:
    """Deterministic pre-promotion policy for model promotion."""

    def __init__(self, min_auc: float = 0.70, promotion_margin: float = 0.02, metric: str = "auc") -> None:
        self.min_auc = min_auc
        self.promotion_margin = promotion_margin
        self.metric = metric

    def check_schema(self, candidate_features: Sequence[str], reference_features: Sequence[str]) -> GateCheck:
        ok = list(candidate_features) == list(reference_features)
        detail = "feature contract matches" if ok else (
            f"schema mismatch: {list(candidate_features)} != {list(reference_features)}"
        )
        return GateCheck("schema_compatibility", ok, detail)

    def check_quality_bar(self, candidate_metrics: dict[str, float]) -> GateCheck:
        val = candidate_metrics.get(self.metric, 0.0)
        ok = val >= self.min_auc
        return GateCheck(
            "quality_bar",
            ok,
            f"{self.metric}={val:.4f} vs min {self.min_auc:.2f}",
        )

    def check_no_regression(
        self, candidate_metrics: dict[str, float], incumbent_metrics: dict[str, float] | None
    ) -> GateCheck:
        cand = candidate_metrics.get(self.metric, 0.0)
        if incumbent_metrics is None:  # first ever model: nothing to regress against
            return GateCheck("no_regression", True, "no incumbent (bootstrap)")
        inc = incumbent_metrics.get(self.metric, 0.0)
        delta = cand - inc
        ok = delta >= self.promotion_margin
        return GateCheck(
            "no_regression",
            ok,
            f"Δ{self.metric}={delta:+.4f} (need ≥ {self.promotion_margin:+.4f})",
        )

    def evaluate(
        self,
        candidate_metrics: dict[str, float],
        candidate_features: Sequence[str],
        reference_features: Sequence[str],
        incumbent_metrics: dict[str, float] | None,
    ) -> GateResult:
        checks = [
            self.check_schema(candidate_features, reference_features),
            self.check_quality_bar(candidate_metrics),
            self.check_no_regression(candidate_metrics, incumbent_metrics),
        ]
        return GateResult(passed=all(c.passed for c in checks), checks=checks)
