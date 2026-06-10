import pytest

from sentinel.evals import (
    GATES,
    _resolve_placeholder,
    run_all,
    run_compliance_eval,
    run_triage_eval,
)


def test_triage_eval_meets_gates(conn):
    result = run_triage_eval(conn)
    assert result["metrics"]["escalation_recall"] >= GATES["triage.escalation_recall"]
    assert result["metrics"]["escalation_precision"] >= GATES["triage.escalation_precision"]
    assert len(result["cases"]) == 10
    failed = [c for c in result["cases"] if not c["passed"]]
    assert not failed, f"failing golden cases: {failed}"


def test_compliance_eval_meets_gates(conn):
    result = run_compliance_eval(conn)
    assert result["metrics"]["verdict_accuracy"] == 1.0
    assert result["metrics"]["citation_accuracy"] >= GATES["compliance.citation_accuracy"]


def test_scorecard_aggregates_and_passes(conn):
    scorecard = run_all(conn)
    assert scorecard["passed"] is True
    assert set(scorecard["gates"]) == set(GATES)
    for gate in scorecard["gates"].values():
        assert gate["passed"]
        assert gate["actual"] >= gate["floor"]


def test_placeholder_resolution(conn):
    restricted = _resolve_placeholder(conn, "@name:Suzhou MagnetCore")
    row = conn.execute(
        "SELECT name FROM suppliers WHERE supplier_id = ?", (restricted,)).fetchone()
    assert row["name"] == "Suzhou MagnetCore"

    iso = _resolve_placeholder(conn, "@iso9001:yes")
    row = conn.execute(
        "SELECT iso9001, name FROM suppliers WHERE supplier_id = ?", (iso,)).fetchone()
    assert row["iso9001"] == "yes"
    assert row["name"] != "Suzhou MagnetCore"

    deep_cover = _resolve_placeholder(conn, "@cover_gte:30")
    inv = conn.execute(
        "SELECT * FROM inventory WHERE part_id = ?", (deep_cover,)).fetchone()
    assert inv["on_hand_units"] / inv["daily_consumption_units"] >= 30

    assert _resolve_placeholder(conn, "SUP-001") == "SUP-001"  # passthrough
    assert _resolve_placeholder(conn, None) is None

    with pytest.raises(ValueError):
        _resolve_placeholder(conn, "@bogus:thing")


def test_gates_catch_regressions(conn, monkeypatch):
    """A degraded triage agent (misses disasters) must fail the scorecard."""
    from sentinel.agents.triage import TriageAgent

    real = TriageAgent.simulate

    def lobotomized(self, payload, toolbox):
        result = real(self, payload, toolbox)
        result["decision"] = "monitor"  # never escalate anything
        return result

    monkeypatch.setattr(TriageAgent, "simulate", lobotomized)
    scorecard = run_all(conn)
    assert scorecard["passed"] is False
    assert not scorecard["gates"]["triage.escalation_recall"]["passed"]
