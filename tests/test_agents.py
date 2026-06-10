import pytest

from sentinel.agents import (
    AgentContext,
    ComplianceAgent,
    ImpactAgent,
    MitigationAgent,
    TriageAgent,
)
from sentinel.llm import SimulationBackend, extract_json
from sentinel.telemetry import UsageMeter
from sentinel.tools import ToolBox


@pytest.fixture()
def ctx(conn):
    return AgentContext(
        backend=SimulationBackend(),
        toolbox=ToolBox(conn),
        meter=UsageMeter(model="claude-opus-4-8", budget_tokens=400_000),
        run_id="run-test",
    )


def _signal(conn, event_id):
    return dict(conn.execute("SELECT * FROM signals WHERE event_id = ?", (event_id,)).fetchone())


def test_extract_json_tolerates_fences_and_preamble():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go:\n{"a": {"b": 2}}') == {"a": {"b": 2}}


def test_triage_escalates_earthquake(conn, ctx):
    signal = _signal(conn, "EVT-2026-0612")  # M6.9 near Hsinchu, names two suppliers
    result = TriageAgent().run({"signal": signal}, ctx)
    assert result["decision"] == "investigate"
    assert result["severity"] in ("critical", "high")
    assert result["entities"]["suppliers"]  # named suppliers resolved to IDs


def test_triage_downgrades_signal_outside_footprint(conn, ctx):
    signal = _signal(conn, "EVT-2026-0611")  # cobalt price shock, no entities
    result = TriageAgent().run({"signal": signal}, ctx)
    assert result["decision"] in ("monitor", "dismiss")
    assert result["risk_score"] <= 0.35


def test_impact_quantifies_blast_radius(conn, ctx):
    signal = _signal(conn, "EVT-2026-0602")  # Gumi PowerCells fire
    triage = TriageAgent().run({"signal": signal}, ctx)
    impact = ImpactAgent().run({"signal": signal, "triage": triage}, ctx)
    assert impact["affected_suppliers"]
    assert impact["total_monthly_revenue_at_risk_usd"] >= 0
    assert isinstance(impact["affected_parts"], list)


def test_mitigation_produces_costed_options(conn, ctx):
    signal = _signal(conn, "EVT-2026-0602")
    triage = TriageAgent().run({"signal": signal}, ctx)
    impact = ImpactAgent().run({"signal": signal, "triage": triage}, ctx)
    plan = MitigationAgent().run({"signal": signal, "triage": triage, "impact": impact}, ctx)
    assert plan["options"]
    assert 0 <= plan["recommended_index"] < len(plan["options"])
    for opt in plan["options"]:
        assert opt["value_usd"] >= 0
        assert opt["kind"] in {
            "expedite_freight", "alternate_source_po", "inventory_reallocation",
            "supplier_quarantine", "monitor_only",
        }


def test_compliance_flags_over_limit_spend(conn, ctx):
    signal = _signal(conn, "EVT-2026-0602")
    mitigation = {
        "options": [{
            "kind": "alternate_source_po",
            "description": "huge bridge PO",
            "part_id": None,
            "supplier_id": "SUP-001",
            "value_usd": 900_000.0,
            "lead_time_days": 30,
            "tradeoffs": "",
        }],
        "recommended_index": 0,
        "rationale": "",
    }
    review = ComplianceAgent().run({"signal": signal, "mitigation": mitigation}, ctx)
    verdict = review["verdicts"][0]
    assert verdict["verdict"] in ("needs_review", "fail")
    assert verdict["cited_rules"]


def test_compliance_blocks_restricted_entity(conn, ctx):
    restricted_id = conn.execute(
        "SELECT supplier_id FROM suppliers WHERE name = 'Suzhou MagnetCore'"
    ).fetchone()["supplier_id"]
    signal = _signal(conn, "EVT-2026-0603")
    mitigation = {
        "options": [{
            "kind": "alternate_source_po",
            "description": "PO with restricted supplier",
            "part_id": None,
            "supplier_id": restricted_id,
            "value_usd": 5_000.0,
            "lead_time_days": 20,
            "tradeoffs": "",
        }],
        "recommended_index": 0,
        "rationale": "",
    }
    review = ComplianceAgent().run({"signal": signal, "mitigation": mitigation}, ctx)
    assert review["verdicts"][0]["verdict"] == "fail"
    assert "restricted_entities" in review["verdicts"][0]["cited_rules"]
