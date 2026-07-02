"""Orchestrator: trigger policy, retrain lineage, canary, and end-to-end path."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops.config import CONFIG, Config
from mlops.data import data_hash, make_data
from mlops.drift import Alert
from mlops.gate import PromotionGate
from mlops.model import evaluate, save_model, train_model
from mlops.orchestrator import RetrainingOrchestrator
from mlops.registry import ModelRegistry, Stage
from mlops.tracking import ExperimentTracker


@pytest.fixture()
def orch(tmp_path):
    reg = ModelRegistry(tmp_path / "reg.duckdb")
    trk = ExperimentTracker(tmp_path / "exp.duckdb")
    gate = PromotionGate(min_auc=CONFIG.min_auc, promotion_margin=CONFIG.promotion_margin)
    cfg = Config()
    object.__setattr__(cfg, "data_dir", tmp_path)
    o = RetrainingOrchestrator(reg, trk, gate, cfg)
    yield o
    reg.close()
    trk.close()


def _alert(kind="data_drift"):
    return Alert(kind=kind, t=0, metric="psi", value=0.5, threshold=0.25, detail="x")


def test_trigger_only_fires_after_sustained_windows(orch):
    # sustained_windows defaults to 3.
    d1 = orch.observe(_alert(), None)
    d2 = orch.observe(_alert(), None)
    assert not d1.triggered and not d2.triggered
    d3 = orch.observe(_alert(), None)
    assert d3.triggered


def test_trigger_resets_on_clean_window(orch):
    orch.observe(_alert(), None)
    orch.observe(_alert(), None)
    orch.observe(None, None)  # clean window resets the counter
    d = orch.observe(_alert(), None)
    assert not d.triggered  # only 1 consecutive again


def test_no_trigger_without_alerts(orch):
    for _ in range(5):
        d = orch.observe(None, None)
    assert not d.triggered


def test_retrain_registers_with_lineage(orch, tmp_path):
    # Seed an incumbent v1.
    X0, y0 = make_data(4000, seed=1, covariate_drift=0.0)
    m0 = train_model(X0, y0, seed=1)
    art0 = tmp_path / "v1.pkl"
    save_model(m0, art0)
    v1 = orch.registry.register(
        "m", "LogisticRegression", {"C": 1.0}, evaluate(m0, X0, y0),
        data_hash(X0, y0), str(art0),
    )
    orch.registry.promote(v1)

    # Retrain a challenger on drifted data.
    Xd, yd = make_data(6000, seed=2, covariate_drift=1.5, concept_drift=1.0)
    Xe, ye = make_data(3000, seed=3, covariate_drift=1.5, concept_drift=1.0)
    v2 = orch.retrain("m", Xd, yd, {"C": 1.0}, parent_version=v1, eval_X=Xe, eval_y=ye)

    mv = orch.registry.get(v2)
    assert mv.parent_version == v1
    assert mv.stage is Stage.STAGING
    assert mv.data_hash == data_hash(Xd, yd)
    # An experiment run was logged for the retrain.
    runs = orch.tracker.query("auto_retrain")
    assert len(runs) == 1


def test_gate_blocks_worse_and_promotes_better_end_to_end(orch, tmp_path):
    """Full path on real models: a stale incumbent, a worse candidate is held,
    a properly retrained candidate passes and is promoted (v1 archived)."""
    # v1 trained on the baseline regime.
    Xb, yb = make_data(8000, seed=10, covariate_drift=0.0, concept_drift=0.0)
    m1 = train_model(Xb, yb, seed=10)
    art1 = tmp_path / "v1.pkl"
    save_model(m1, art1)
    v1 = orch.registry.register(
        "m", "LogisticRegression", {"C": 1.0}, evaluate(m1, Xb, yb),
        data_hash(Xb, yb), str(art1),
    )
    orch.registry.promote(v1)

    # The world drifts. Holdout is the new regime.
    Xe, ye = make_data(6000, seed=11, covariate_drift=1.5, concept_drift=1.0)
    incumbent_now = evaluate(m1, Xe, ye)  # v1 is bad on the new regime

    # (a) A worse candidate (trained on the OLD regime) must be blocked.
    worse_metrics = incumbent_now
    res_bad, promoted_bad = orch.gate_and_promote(v1, worse_metrics, incumbent_now)
    assert not promoted_bad
    assert not res_bad.passed
    assert orch.registry.production_version().version == v1

    # (b) A proper challenger trained on the NEW regime must pass and promote.
    Xd, yd = make_data(12000, seed=12, covariate_drift=1.5, concept_drift=1.0)
    v2 = orch.retrain("m", Xd, yd, {"C": 1.0}, parent_version=v1, eval_X=Xe, eval_y=ye)
    challenger_metrics = orch.registry.get(v2).metrics
    res_good, promoted_good = orch.gate_and_promote(v2, challenger_metrics, incumbent_now)

    assert promoted_good
    assert res_good.passed
    assert orch.registry.production_version().version == v2
    assert orch.registry.get(v1).stage is Stage.ARCHIVED
    # Challenger genuinely beats the stale incumbent on the new regime.
    assert challenger_metrics["auc"] > incumbent_now["auc"] + CONFIG.promotion_margin


def test_canary_picks_better_model(orch, tmp_path):
    Xb, yb = make_data(6000, seed=20, covariate_drift=0.0)
    Xe, ye = make_data(4000, seed=21, covariate_drift=1.5, concept_drift=1.0)
    m1 = train_model(Xb, yb, seed=20)                       # stale
    m2 = train_model(*make_data(8000, seed=22, covariate_drift=1.5, concept_drift=1.0), seed=22)
    res = orch.canary(m1, 1, m2, 2, Xe, ye)
    assert res.winner == "challenger"
    assert res.delta > 0
