"""Registry behaviour: version lineage, stage state-machine, promote/rollback."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops.registry import InvalidTransition, ModelRegistry, Stage


@pytest.fixture()
def registry(tmp_path):
    reg = ModelRegistry(tmp_path / "reg.duckdb")
    yield reg
    reg.close()


def _register(reg, parent=None, auc=0.8):
    return reg.register(
        name="m",
        algo="LogisticRegression",
        params={"C": 1.0},
        metrics={"auc": auc, "accuracy": auc - 0.05},
        data_hash="deadbeef",
        artifact_path="/tmp/x.pkl",
        parent_version=parent,
    )


def test_register_autoincrements_versions(registry):
    v1 = _register(registry)
    v2 = _register(registry)
    assert (v1, v2) == (1, 2)
    assert registry.get(v1).stage is Stage.STAGING


def test_promote_sets_single_production_and_archives_incumbent(registry):
    v1 = _register(registry)
    v2 = _register(registry)
    registry.promote(v1)
    assert registry.production_version().version == v1
    registry.promote(v2)
    assert registry.production_version().version == v2
    assert registry.get(v1).stage is Stage.ARCHIVED
    # Exactly one production version at any time.
    prod = [m for m in registry.list_versions() if m.stage is Stage.PRODUCTION]
    assert len(prod) == 1


def test_invalid_transition_rejected(registry):
    v1 = _register(registry)
    registry.promote(v1)  # staging -> production
    # production -> staging is illegal
    with pytest.raises(InvalidTransition):
        registry.transition(v1, Stage.STAGING)


def test_valid_transition_matrix(registry):
    v1 = _register(registry)
    # staging -> archived allowed
    registry.transition(v1, Stage.ARCHIVED)
    assert registry.get(v1).stage is Stage.ARCHIVED
    # archived -> staging allowed (restore)
    registry.transition(v1, Stage.STAGING)
    assert registry.get(v1).stage is Stage.STAGING


def test_rollback_restores_previous_production(registry):
    v1 = _register(registry)
    v2 = _register(registry)
    registry.promote(v1)
    registry.promote(v2)
    assert registry.production_version().version == v2
    restored = registry.rollback()
    assert restored == v1
    assert registry.production_version().version == v1
    assert registry.get(v2).stage is Stage.ARCHIVED


def test_rollback_without_history_raises(registry):
    v1 = _register(registry)
    registry.promote(v1)
    with pytest.raises(InvalidTransition):
        registry.rollback()


def test_lineage_chain_recorded(registry):
    v1 = _register(registry)
    v2 = _register(registry, parent=v1)
    v3 = _register(registry, parent=v2)
    chain = [m.version for m in registry.lineage(v3)]
    assert chain == [v1, v2, v3]
    assert registry.get(v3).parent_version == v2
    assert registry.get(v1).parent_version is None


def test_metrics_and_hash_persisted(registry):
    v1 = _register(registry, auc=0.91)
    mv = registry.get(v1)
    assert mv.metrics["auc"] == 0.91
    assert mv.data_hash == "deadbeef"


def test_transition_log_is_audited(registry):
    v1 = _register(registry)
    v2 = _register(registry)
    registry.promote(v1)
    registry.promote(v2)
    log = registry.transition_log()
    to_prod = [e for e in log if e["to_stage"] == "production"]
    assert [e["version"] for e in to_prod] == [v1, v2]
