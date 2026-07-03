"""Promotion gate: blocks a worse model, passes a better one, enforces schema."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops.config import FEATURES
from mlops.gate import PromotionGate

GATE = PromotionGate(min_auc=0.70, promotion_margin=0.02)


def test_gate_passes_clearly_better_model():
    res = GATE.evaluate(
        candidate_metrics={"auc": 0.90},
        candidate_features=FEATURES,
        reference_features=FEATURES,
        incumbent_metrics={"auc": 0.80},
    )
    assert res.passed
    assert all(c.passed for c in res.checks)


def test_gate_blocks_regressing_model():
    # Candidate is worse than incumbent -> no_regression fails.
    res = GATE.evaluate(
        candidate_metrics={"auc": 0.78},
        candidate_features=FEATURES,
        reference_features=FEATURES,
        incumbent_metrics={"auc": 0.85},
    )
    assert not res.passed
    assert not next(c for c in res.checks if c.name == "no_regression").passed


def test_gate_blocks_marginal_improvement_below_margin():
    # Better, but not by the required margin (0.02).
    res = GATE.evaluate(
        candidate_metrics={"auc": 0.815},
        candidate_features=FEATURES,
        reference_features=FEATURES,
        incumbent_metrics={"auc": 0.80},
    )
    assert not res.passed


def test_gate_blocks_below_quality_bar_even_if_better():
    # Candidate beats a terrible incumbent but is still below the absolute bar.
    res = GATE.evaluate(
        candidate_metrics={"auc": 0.62},
        candidate_features=FEATURES,
        reference_features=FEATURES,
        incumbent_metrics={"auc": 0.50},
    )
    assert not res.passed
    assert not next(c for c in res.checks if c.name == "quality_bar").passed


def test_gate_blocks_schema_mismatch():
    res = GATE.evaluate(
        candidate_metrics={"auc": 0.95},
        candidate_features=list(FEATURES)[:-1],  # dropped a feature
        reference_features=FEATURES,
        incumbent_metrics={"auc": 0.80},
    )
    assert not res.passed
    assert not next(c for c in res.checks if c.name == "schema_compatibility").passed


def test_gate_bootstrap_no_incumbent():
    # First-ever model: no incumbent to regress against, still must clear the bar.
    res = GATE.evaluate(
        candidate_metrics={"auc": 0.88},
        candidate_features=FEATURES,
        reference_features=FEATURES,
        incumbent_metrics=None,
    )
    assert res.passed
