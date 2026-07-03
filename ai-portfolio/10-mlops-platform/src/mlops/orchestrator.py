"""Automated retraining orchestrator.

Ties the registry, drift monitors, and promotion gate together into the closed
loop that keeps a production model healthy:

    detect drift  ->  retrain challenger  ->  A/B canary  ->  gate  ->  promote|hold

Key policies
------------
* A retrain is triggered only when data drift *or* performance drift persists for
  ``sustained_windows`` consecutive windows — transient spikes are ignored.
* A challenger is registered in ``staging`` with full lineage (parent = incumbent,
  data hash of its training set, params).
* Before promotion, a shadow/canary A/B compares challenger vs. incumbent on a
  fresh holdout. Promotion proceeds only if the :class:`PromotionGate` passes;
  otherwise the incumbent is kept and an alert is raised.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .data import data_hash
from .drift import Alert
from .gate import GateResult, PromotionGate
from .model import evaluate, save_model, train_model
from .registry import ModelRegistry, Stage
from .tracking import ExperimentTracker


@dataclass
class CanaryResult:
    """Outcome of a shadow/canary A/B comparison on a shared holdout."""

    incumbent_version: int
    challenger_version: int
    incumbent_metrics: dict[str, float]
    challenger_metrics: dict[str, float]
    winner: str  # "challenger" | "incumbent" | "tie"
    metric: str = "auc"

    @property
    def delta(self) -> float:
        return self.challenger_metrics[self.metric] - self.incumbent_metrics[self.metric]


@dataclass
class RetrainDecision:
    triggered: bool
    reason: str
    consecutive: int


class RetrainingOrchestrator:
    """Drives detection -> retrain -> canary -> gate -> promote/rollback."""

    def __init__(
        self,
        registry: ModelRegistry,
        tracker: ExperimentTracker,
        gate: PromotionGate,
        cfg: Config,
    ) -> None:
        self.registry = registry
        self.tracker = tracker
        self.gate = gate
        self.cfg = cfg
        self._consecutive = 0

    # -- trigger policy -----------------------------------------------------
    def observe(self, data_alert: Alert | None, perf_alert: Alert | None) -> RetrainDecision:
        """Feed one window's alerts; decide whether to trigger a retrain.

        Requires ``sustained_windows`` consecutive alerting windows so a single
        noisy batch cannot cause churn.
        """
        if data_alert is not None or perf_alert is not None:
            self._consecutive += 1
        else:
            self._consecutive = 0
        triggered = self._consecutive >= self.cfg.sustained_windows
        if triggered:
            drivers = []
            if data_alert:
                drivers.append(f"data_drift({data_alert.detail})")
            if perf_alert:
                drivers.append(f"perf_drift({perf_alert.detail})")
            reason = " & ".join(drivers) or "sustained drift"
        else:
            reason = "below sustained threshold"
        decision = RetrainDecision(triggered=triggered, reason=reason, consecutive=self._consecutive)
        if triggered:
            self._consecutive = 0  # reset after firing
        return decision

    # -- retrain ------------------------------------------------------------
    def retrain(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        params: dict[str, Any],
        parent_version: int | None,
        eval_X: np.ndarray,
        eval_y: np.ndarray,
    ) -> int:
        """Train a challenger on recent data, register it in staging with lineage."""
        model = train_model(X, y, params, seed=self.cfg.seed)
        metrics = evaluate(model, eval_X, eval_y)
        dh = data_hash(X, y)
        version = self.registry._next_version()
        artifact = str(self.cfg.artifacts_dir / f"v{version}.pkl")
        save_model(model, Path(artifact))
        registered = self.registry.register(
            name=name,
            algo="LogisticRegression",
            params=params,
            metrics=metrics,
            data_hash=dh,
            artifact_path=artifact,
            parent_version=parent_version,
            stage=Stage.STAGING,
        )
        self.tracker.log_run(
            experiment="auto_retrain",
            params=params,
            metrics=metrics,
            artifact_path=artifact,
            version=registered,
            tags={"trigger": "drift", "parent": str(parent_version)},
        )
        return registered

    # -- canary / A-B -------------------------------------------------------
    def canary(
        self,
        incumbent_model,
        incumbent_version: int,
        challenger_model,
        challenger_version: int,
        eval_X: np.ndarray,
        eval_y: np.ndarray,
        metric: str = "auc",
    ) -> CanaryResult:
        """Shadow both models on the same holdout and pick a winner."""
        inc = evaluate(incumbent_model, eval_X, eval_y)
        cha = evaluate(challenger_model, eval_X, eval_y)
        delta = cha[metric] - inc[metric]
        winner = "challenger" if delta > 0 else ("incumbent" if delta < 0 else "tie")
        return CanaryResult(
            incumbent_version=incumbent_version,
            challenger_version=challenger_version,
            incumbent_metrics=inc,
            challenger_metrics=cha,
            winner=winner,
            metric=metric,
        )

    # -- gate + promote -----------------------------------------------------
    def gate_and_promote(
        self,
        challenger_version: int,
        challenger_metrics: dict[str, float],
        incumbent_metrics: dict[str, float] | None,
    ) -> tuple[GateResult, bool]:
        """Run the promotion gate; promote the challenger iff it passes.

        Returns ``(gate_result, promoted)``. On failure the incumbent is retained
        (no stage change) and the caller can alert.
        """
        result = self.gate.evaluate(
            candidate_metrics=challenger_metrics,
            candidate_features=self.cfg.features,
            reference_features=self.cfg.features,
            incumbent_metrics=incumbent_metrics,
        )
        promoted = False
        if result.passed:
            self.registry.promote(challenger_version, reason="gate passed: auto-retrain challenger")
            promoted = True
        return result, promoted
