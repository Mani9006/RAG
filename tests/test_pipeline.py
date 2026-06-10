import pytest

from sentinel.orchestrator import Pipeline, decide_action


def test_full_pipeline_run(conn):
    summary = Pipeline(conn).run()
    assert summary["signals_processed"] == 12
    assert summary["incidents_opened"] >= 3   # quake, typhoon, fire at minimum
    assert summary["incidents_opened"] + summary["monitored"] + summary["dismissed"] == 12

    # Every escalated signal produced a persisted incident with a briefing.
    incidents = conn.execute("SELECT * FROM incidents").fetchall()
    assert len(incidents) == summary["incidents_opened"]
    for inc in incidents:
        assert inc["briefing_md"].startswith("# Situation Report")

    # The audit trail brackets the run.
    actions = [r["action"] for r in conn.execute(
        "SELECT action FROM audit_log WHERE run_id = ?", (summary["run_id"],)).fetchall()]
    assert actions[0] == "run_started"
    assert actions[-1] == "run_finished"


def test_runs_are_idempotent_over_processed_signals(conn):
    first = Pipeline(conn).run()
    second = Pipeline(conn).run()
    assert first["signals_processed"] == 12
    assert second["signals_processed"] == 0
    assert second["incidents_opened"] == 0


def test_approval_gate_enforces_spend_limit(conn):
    Pipeline(conn).run()
    rows = conn.execute("SELECT * FROM actions").fetchall()
    assert rows, "pipeline should register actions"
    for row in rows:
        if row["status"] == "auto_approved":
            assert row["value_usd"] <= 25_000
            assert row["compliance_verdict"] == "pass"
        if row["compliance_verdict"] == "fail":
            assert row["status"] == "blocked"


def test_human_decision_flow(conn):
    Pipeline(conn).run()
    pending = conn.execute(
        "SELECT action_id FROM actions WHERE status = 'pending_approval'").fetchone()
    assert pending is not None, "expected at least one action requiring human approval"

    decided = decide_action(conn, pending["action_id"], approve=True, approver="alice@vertex")
    assert decided["status"] == "approved"
    assert decided["approver"] == "alice@vertex"

    # Double-deciding is rejected.
    with pytest.raises(ValueError):
        decide_action(conn, pending["action_id"], approve=False, approver="bob@vertex")

    with pytest.raises(KeyError):
        decide_action(conn, "ACT-DOESNOTEXIST", approve=True, approver="alice@vertex")
