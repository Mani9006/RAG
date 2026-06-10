"""Agent evaluation harness.

The credibility layer: golden datasets with labeled expected outcomes, scored
on every CI run, with hard regression gates — a prompt or logic change that
degrades agent quality fails the build instead of silently degrading
production behavior.

Two suites ship today:
- **Triage**: escalation precision/recall (recall is the safety-critical
  metric — a missed disaster costs more than a false alarm), severity and
  score calibration, entity resolution.
- **Compliance**: verdict accuracy and rule-citation accuracy on labeled
  option sets, including restricted entities, spend authority, SR-002/SR-003.

Golden cases use `@selector` placeholders (e.g. `@name:Suzhou MagnetCore`,
`@cover_lt:18`) resolved against the live dataset at eval time, so cases stay
valid when the synthetic dataset is regenerated.

Both run against the simulation backend by default (free, deterministic, CI)
and against live Claude agents when SENTINEL_LLM_BACKEND=anthropic — the same
golden labels measure both.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import yaml

from sentinel.agents import AgentContext, ComplianceAgent, TriageAgent
from sentinel.config import get_settings
from sentinel.llm import build_backend
from sentinel.telemetry import UsageMeter
from sentinel.tools import ToolBox

# Regression gates — CI fails below these floors.
GATES = {
    "triage.escalation_recall": 1.0,    # never miss a disaster
    "triage.escalation_precision": 0.8,  # noise erodes operator trust
    "triage.expectation_pass_rate": 0.9,
    "compliance.verdict_accuracy": 1.0,  # the money gate must be exact
    "compliance.citation_accuracy": 0.9,
}


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _ctx(conn: sqlite3.Connection) -> AgentContext:
    settings = get_settings()
    return AgentContext(
        backend=build_backend(),
        toolbox=ToolBox(conn),
        meter=UsageMeter(model=settings.model, budget_tokens=settings.run_token_budget),
        run_id="eval",
    )


# ----------------------------------------------------------------- triage


def run_triage_eval(conn: sqlite3.Connection) -> dict:
    cases = _load_jsonl(get_settings().data_dir / "evals" / "triage_golden.jsonl")
    ctx = _ctx(conn)
    agent = TriageAgent()

    results = []
    tp = fp = fn = tn = 0
    for case in cases:
        signal = dict(case["signal"])
        signal["locations"] = json.dumps(signal["locations"])
        signal["suppliers_mentioned"] = json.dumps(signal["suppliers_mentioned"])
        prediction = agent.run({"signal": signal}, ctx)

        escalated = prediction["decision"] == "investigate"
        expected_escalate = case["escalate"]  # true | false | null (either ok)
        if expected_escalate is True:
            tp += escalated
            fn += not escalated
        elif expected_escalate is False:
            fp += escalated
            tn += not escalated

        expected = case["expected"]
        checks = {
            "severity": prediction["severity"] in expected["severity_any_of"],
            "min_score": prediction["risk_score"] >= expected.get("min_score", 0.0),
            "max_score": prediction["risk_score"] <= expected.get("max_score", 1.0),
            "suppliers_resolved": (
                bool(prediction["entities"]["suppliers"])
                if expected.get("must_resolve_suppliers") else True
            ),
        }
        if expected_escalate is True:
            checks["escalated"] = escalated
        elif expected_escalate is False:
            checks["not_escalated"] = not escalated
        results.append({
            "case_id": case["case_id"],
            "passed": all(checks.values()),
            "failed_checks": sorted(k for k, ok in checks.items() if not ok),
            "decision": prediction["decision"],
            "severity": prediction["severity"],
            "risk_score": prediction["risk_score"],
        })

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {
        "suite": "triage",
        "cases": results,
        "metrics": {
            "escalation_precision": round(precision, 3),
            "escalation_recall": round(recall, 3),
            "expectation_pass_rate": round(
                sum(r["passed"] for r in results) / len(results), 3),
        },
    }


# ------------------------------------------------------------- compliance


def _resolve_placeholder(conn: sqlite3.Connection, value: str | None) -> str | None:
    """Resolve `@selector` placeholders against the live dataset."""
    if not isinstance(value, str) or not value.startswith("@"):
        return value
    selector, _, arg = value[1:].partition(":")
    if selector == "name":
        row = conn.execute(
            "SELECT supplier_id FROM suppliers WHERE name = ?", (arg,)).fetchone()
    elif selector == "iso9001":
        policy = yaml.safe_load(
            (get_settings().data_dir / "policies" / "procurement_policy.yaml").read_text())
        restricted = [e["supplier_name"] for e in policy.get("restricted_entities", [])]
        placeholders = ",".join("?" * len(restricted)) or "''"
        row = conn.execute(
            f"SELECT supplier_id FROM suppliers WHERE iso9001 = ? AND name NOT IN"
            f" ({placeholders}) ORDER BY supplier_id LIMIT 1",
            (arg, *restricted)).fetchone()
    elif selector in ("cover_gte", "cover_lt"):
        op = ">=" if selector == "cover_gte" else "<"
        row = conn.execute(
            "SELECT part_id AS supplier_id FROM inventory WHERE daily_consumption_units > 0"
            f" AND CAST(on_hand_units AS REAL) / daily_consumption_units {op} ?"
            " ORDER BY part_id LIMIT 1", (float(arg),)).fetchone()
    else:
        raise ValueError(f"unknown placeholder selector: {value}")
    if row is None:
        raise ValueError(f"placeholder {value} matched nothing in the dataset")
    return row["supplier_id"]


def run_compliance_eval(conn: sqlite3.Connection) -> dict:
    cases = _load_jsonl(get_settings().data_dir / "evals" / "compliance_golden.jsonl")
    ctx = _ctx(conn)
    agent = ComplianceAgent()

    results = []
    verdict_hits = citation_hits = total = 0
    for case in cases:
        options = []
        for option in case["options"]:
            resolved = dict(option)
            resolved["supplier_id"] = _resolve_placeholder(conn, option.get("supplier_id"))
            resolved["part_id"] = _resolve_placeholder(conn, option.get("part_id"))
            options.append(resolved)

        prediction = agent.run({
            "signal": {"type": case["signal_type"]},
            "mitigation": {"options": options, "recommended_index": 0, "rationale": ""},
        }, ctx)
        verdicts = {v["option_index"]: v for v in prediction["verdicts"]}

        case_pass = True
        for expected in case["expected"]:
            total += 1
            verdict = verdicts.get(expected["option_index"], {})
            verdict_ok = verdict.get("verdict") == expected["verdict"]
            cited = set(verdict.get("cited_rules", []))
            citation_ok = set(expected["must_cite"]) <= cited
            verdict_hits += verdict_ok
            citation_hits += citation_ok
            case_pass = case_pass and verdict_ok and citation_ok
        results.append({"case_id": case["case_id"], "passed": case_pass,
                        "verdicts": prediction["verdicts"]})

    return {
        "suite": "compliance",
        "cases": results,
        "metrics": {
            "verdict_accuracy": round(verdict_hits / total, 3) if total else 1.0,
            "citation_accuracy": round(citation_hits / total, 3) if total else 1.0,
        },
    }


# -------------------------------------------------------------- scorecard


def run_all(conn: sqlite3.Connection) -> dict:
    triage = run_triage_eval(conn)
    compliance = run_compliance_eval(conn)

    flat = {
        f"{suite['suite']}.{metric}": value
        for suite in (triage, compliance)
        for metric, value in suite["metrics"].items()
    }
    gate_results = {
        name: {"floor": floor, "actual": flat.get(name, 0.0),
               "passed": flat.get(name, 0.0) >= floor}
        for name, floor in GATES.items()
    }
    return {
        "backend": get_settings().llm_backend,
        "suites": {"triage": triage, "compliance": compliance},
        "gates": gate_results,
        "passed": all(g["passed"] for g in gate_results.values()),
    }
