import json

from sentinel.ops import agreement_metrics, run_cycle, run_daemon
from sentinel.orchestrator import Pipeline, decide_action


def test_daemon_cycles_and_is_restart_safe(conn):
    # Two cycles in one "process": the second finds nothing new to do.
    cycles = run_daemon(conn, interval_seconds=0.0, max_cycles=2,
                        ingest_sources=None, sleep=lambda s: None)
    assert cycles == 2

    first_run_incidents = conn.execute("SELECT COUNT(*) AS n FROM incidents").fetchone()["n"]
    assert first_run_incidents >= 1

    # "Restart": a fresh daemon over the same store must not reprocess anything.
    cycles = run_daemon(conn, interval_seconds=0.0, max_cycles=1,
                        ingest_sources=None, sleep=lambda s: None)
    assert cycles == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM incidents").fetchone()["n"] \
        == first_run_incidents

    # Every cycle is audited.
    audited = conn.execute(
        "SELECT COUNT(*) AS n FROM audit_log WHERE action = 'daemon_cycle'").fetchone()["n"]
    assert audited == 3


def test_cycle_isolates_ingest_failures(conn, monkeypatch):
    import sentinel.connectors as connectors

    def exploding_ingest(c, source="all", window_hours=24):
        return {"usgs": {"error": "ConnectError: no network"}}

    monkeypatch.setattr(connectors, "run_ingest", exploding_ingest)
    outcome = run_cycle(conn, ingest_sources="all")
    # Ingest failed per-source, but the pipeline still ran to completion.
    assert "error" in outcome["ingest"]["usgs"]
    assert outcome["run"]["signals_processed"] == 12


def test_agreement_metrics(conn):
    Pipeline(conn).run()
    pending = [r["action_id"] for r in conn.execute(
        "SELECT action_id FROM actions WHERE status = 'pending_approval' LIMIT 3").fetchall()]
    assert len(pending) >= 2
    decide_action(conn, pending[0], approve=True, approver="alice@vertex")
    decide_action(conn, pending[1], approve=False, approver="alice@vertex")

    metrics = agreement_metrics(conn)
    assert metrics["decided_by_humans"] == 2
    assert metrics["human_agreement_rate"] == 0.5
    assert metrics["pending_approval"] >= 0
    assert sum(b["approved"] + b["rejected"] for b in metrics["by_kind"].values()) == 2


def test_precedent_raises_triage_score(conn):
    """The learning loop: a repeat event scores higher than its first occurrence."""
    from sentinel.agents import AgentContext, TriageAgent
    from sentinel.llm import SimulationBackend
    from sentinel.telemetry import UsageMeter
    from sentinel.tools import ToolBox

    ctx = AgentContext(
        backend=SimulationBackend(), toolbox=ToolBox(conn),
        meter=UsageMeter(model="claude-opus-4-8", budget_tokens=1), run_id="t")
    signal = dict(conn.execute(
        "SELECT * FROM signals WHERE event_id = 'EVT-2026-0602'").fetchone())

    agent = TriageAgent()
    fresh = agent.run({"signal": signal, "precedent": {
        "prior_incidents_same_type": 0, "prior_incidents_same_suppliers": []}}, ctx)
    repeat = agent.run({"signal": signal, "precedent": {
        "prior_incidents_same_type": 2,
        "prior_incidents_same_suppliers": [{"supplier": "Gumi PowerCells",
                                            "prior_incidents": 1}]}}, ctx)
    assert repeat["risk_score"] > fresh["risk_score"]
    assert "Institutional memory" in repeat["rationale"]

    # Precedent must NOT escalate pure noise (score capped below threshold).
    noise = dict(conn.execute(
        "SELECT * FROM signals WHERE event_id = 'EVT-2026-0611'").fetchone())
    noisy = agent.run({"signal": noise, "precedent": {
        "prior_incidents_same_type": 5, "prior_incidents_same_suppliers": []}}, ctx)
    assert noisy["decision"] in ("monitor", "dismiss")


def test_pipeline_gathers_precedent_across_runs(conn):
    """An incident in run 1 becomes precedent for a same-supplier signal later."""
    pipeline = Pipeline(conn)
    pipeline.run()

    # Inject a fresh signal about a supplier that already has an incident.
    conn.execute(
        "INSERT INTO signals (event_id, observed_at, source, type, severity_hint,"
        " headline, body, locations, suppliers_mentioned) VALUES (?,?,?,?,?,?,?,?,?)",
        ("EVT-REPEAT", "2026-06-11T00:00:00Z", "supplier_portal", "factory_fire",
         "high", "Second fire reported at Gumi PowerCells",
         "Another incident at the same site.", json.dumps(["South Korea"]),
         json.dumps(["Gumi PowerCells"])))
    conn.commit()

    precedent = pipeline._gather_precedent(dict(conn.execute(
        "SELECT * FROM signals WHERE event_id = 'EVT-REPEAT'").fetchone()))
    assert precedent["prior_incidents_same_type"] >= 1
    assert any(h["supplier"] == "Gumi PowerCells"
               for h in precedent["prior_incidents_same_suppliers"])


def test_agreement_endpoint(conn, monkeypatch):
    from fastapi.testclient import TestClient

    import sentinel.api as api_module
    import sentinel.db as db_module
    from sentinel.api import app

    class _NonClosing:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def close(self):
            pass

    wrapper = _NonClosing(conn)
    monkeypatch.setattr(api_module, "get_connection", lambda: wrapper)
    monkeypatch.setattr(db_module, "get_connection", lambda force_bootstrap=False: wrapper)
    with TestClient(app) as client:
        body = client.get("/ops/agreement").json()
        assert "human_agreement_rate" in body
        assert "pending_approval" in body
