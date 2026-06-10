"""Pipeline orchestrator.

Drives the agentic pipeline over every unprocessed disruption signal:

    ingest -> triage -> [escalated?] -> impact -> mitigation -> compliance
           -> approval gate (human-in-the-loop) -> executive briefing

The orchestrator owns everything the agents must not: persistence, the audit
trail, the run token budget, and the rule that no spend ever executes without
passing compliance and the approval gate.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from sentinel.agents import (
    AgentContext,
    BriefingAgent,
    ComplianceAgent,
    ImpactAgent,
    MitigationAgent,
    TriageAgent,
)
from sentinel.config import get_settings
from sentinel.llm import build_backend
from sentinel.telemetry import UsageMeter, log, new_run_id
from sentinel.tools import ToolBox


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def audit(conn: sqlite3.Connection, run_id: str, actor: str, action: str, detail: dict) -> None:
    conn.execute(
        "INSERT INTO audit_log (run_id, ts, actor, action, detail_json) VALUES (?,?,?,?,?)",
        (run_id, _now(), actor, action, json.dumps(detail, default=str)),
    )
    conn.commit()


class Pipeline:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.settings = get_settings()
        self.run_id = new_run_id()
        self.ctx = AgentContext(
            backend=build_backend(),
            toolbox=ToolBox(conn),
            meter=UsageMeter(
                model=self.settings.model,
                budget_tokens=self.settings.run_token_budget,
            ),
            run_id=self.run_id,
        )
        self.triage = TriageAgent()
        self.impact = ImpactAgent()
        self.mitigation = MitigationAgent()
        self.compliance = ComplianceAgent()
        self.briefing = BriefingAgent()

    # ------------------------------------------------------------------ run

    def run(self, limit: int | None = None) -> dict:
        """Process all new signals. Returns a run summary."""
        signals = [dict(r) for r in self.conn.execute(
            "SELECT * FROM signals WHERE status = 'new' ORDER BY observed_at"
        ).fetchall()]
        if limit:
            signals = signals[:limit]

        audit(self.conn, self.run_id, "orchestrator", "run_started",
              {"backend": self.ctx.backend.name, "signals": len(signals)})

        incidents, dismissed, monitored = [], 0, 0
        for signal in signals:
            outcome = self._process_signal(signal)
            if outcome == "incident":
                incidents.append(signal["event_id"])
            elif outcome == "monitor":
                monitored += 1
            else:
                dismissed += 1

        summary = {
            "run_id": self.run_id,
            "backend": self.ctx.backend.name,
            "signals_processed": len(signals),
            "incidents_opened": len(incidents),
            "incident_events": incidents,
            "monitored": monitored,
            "dismissed": dismissed,
            "usage": self.ctx.meter.snapshot(),
        }
        audit(self.conn, self.run_id, "orchestrator", "run_finished", summary)
        return summary

    # --------------------------------------------------------- per-signal

    def _gather_precedent(self, signal: dict) -> dict:
        """Institutional memory: prior incidents of the same type or involving
        the same suppliers, surfaced to the triage agent as precedent."""
        same_type = self.conn.execute(
            "SELECT COUNT(*) AS n FROM incidents i JOIN signals s ON s.event_id = i.event_id"
            " WHERE s.type = ?", (signal["type"],)).fetchone()["n"]

        supplier_hits = []
        names = signal["suppliers_mentioned"]
        if isinstance(names, str):
            names = json.loads(names)
        for name in names:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM incidents i JOIN signals s ON s.event_id = i.event_id"
                " WHERE s.suppliers_mentioned LIKE ?", (f"%{name}%",)).fetchone()
            if row["n"]:
                supplier_hits.append({"supplier": name, "prior_incidents": row["n"]})

        return {
            "prior_incidents_same_type": same_type,
            "prior_incidents_same_suppliers": supplier_hits,
        }

    def _process_signal(self, signal: dict) -> str:
        run_id, conn = self.run_id, self.conn
        log("pipeline", "signal", run_id=run_id, event_id=signal["event_id"], type=signal["type"])

        precedent = self._gather_precedent(signal)
        triage = self.triage.run({"signal": signal, "precedent": precedent}, self.ctx)
        audit(conn, run_id, self.triage.name, "triage_completed",
              {"event_id": signal["event_id"], "result": triage})

        decision = triage.get("decision", "monitor")
        if decision != "investigate":
            conn.execute(
                "UPDATE signals SET status = ? WHERE event_id = ?",
                ("monitoring" if decision == "monitor" else "dismissed", signal["event_id"]),
            )
            conn.commit()
            return decision

        impact = self.impact.run({"signal": signal, "triage": triage}, self.ctx)
        audit(conn, run_id, self.impact.name, "impact_assessed",
              {"event_id": signal["event_id"],
               "revenue_at_risk": impact.get("total_monthly_revenue_at_risk_usd")})

        mitigation = self.mitigation.run(
            {"signal": signal, "triage": triage, "impact": impact}, self.ctx)
        audit(conn, run_id, self.mitigation.name, "mitigation_planned",
              {"event_id": signal["event_id"], "options": len(mitigation.get("options", []))})

        compliance = self.compliance.run(
            {"signal": signal, "mitigation": mitigation}, self.ctx)
        audit(conn, run_id, self.compliance.name, "compliance_reviewed",
              {"event_id": signal["event_id"], "summary": compliance.get("summary")})

        record = {
            "signal": signal, "triage": triage, "impact": impact,
            "mitigation": mitigation, "compliance": compliance,
        }
        briefing = self.briefing.run(record, self.ctx)

        incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        conn.execute(
            "INSERT INTO incidents (incident_id, run_id, event_id, severity, risk_score,"
            " summary, impact_json, mitigation_json, compliance_json, briefing_md, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                incident_id, run_id, signal["event_id"], triage["severity"],
                triage["risk_score"], triage["rationale"],
                json.dumps(impact, default=str), json.dumps(mitigation, default=str),
                json.dumps(compliance, default=str), briefing["briefing_md"], _now(),
            ),
        )
        self._register_actions(incident_id, mitigation, compliance)
        conn.execute("UPDATE signals SET status = 'incident' WHERE event_id = ?",
                     (signal["event_id"],))
        conn.commit()
        audit(conn, run_id, "orchestrator", "incident_opened",
              {"incident_id": incident_id, "event_id": signal["event_id"]})

        from sentinel.notify import notify_incident

        pending = conn.execute(
            "SELECT COUNT(*) AS n FROM actions WHERE incident_id = ?"
            " AND status = 'pending_approval'", (incident_id,)).fetchone()["n"]
        if notify_incident(
            incident_id, triage["severity"], signal["headline"],
            impact.get("total_monthly_revenue_at_risk_usd"), pending,
        ):
            audit(conn, run_id, "notifier", "notification_sent",
                  {"incident_id": incident_id})
        return "incident"

    # ------------------------------------------------------ approval gate

    def _register_actions(self, incident_id: str, mitigation: dict, compliance: dict) -> None:
        """Persist proposed actions with their gate status.

        Hard rules, regardless of what any agent said:
          - compliance 'fail' -> action blocked
          - value above the auto-approve limit -> pending human approval
          - compliance 'needs_review' -> pending human approval
          - otherwise -> auto-approved
        """
        verdicts = {v["option_index"]: v for v in compliance.get("verdicts", [])}
        limit = self.settings.auto_approve_limit_usd

        for i, opt in enumerate(mitigation.get("options", [])):
            if opt["kind"] == "monitor_only":
                continue
            verdict = verdicts.get(i, {}).get("verdict", "needs_review")
            if verdict == "fail":
                status = "blocked"
            elif verdict == "needs_review" or opt["value_usd"] > limit:
                status = "pending_approval"
            else:
                status = "auto_approved"

            action_id = f"ACT-{uuid.uuid4().hex[:8].upper()}"
            self.conn.execute(
                "INSERT INTO actions (action_id, incident_id, kind, description, supplier_id,"
                " value_usd, status, compliance_verdict, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    action_id, incident_id, opt["kind"], opt["description"],
                    opt.get("supplier_id"), opt["value_usd"], status, verdict, _now(),
                ),
            )
            audit(self.conn, self.run_id, "approval_gate", "action_registered",
                  {"action_id": action_id, "incident_id": incident_id,
                   "status": status, "value_usd": opt["value_usd"]})


# ---------------------------------------------------------------- approvals


def decide_action(
    conn: sqlite3.Connection, action_id: str, approve: bool, approver: str
) -> dict:
    """Resolve a pending human approval. Raises on invalid state transitions."""
    row = conn.execute("SELECT * FROM actions WHERE action_id = ?", (action_id,)).fetchone()
    if row is None:
        raise KeyError(f"unknown action: {action_id}")
    if row["status"] != "pending_approval":
        raise ValueError(f"action {action_id} is '{row['status']}', not pending_approval")

    status = "approved" if approve else "rejected"
    conn.execute(
        "UPDATE actions SET status = ?, approver = ?, decided_at = ? WHERE action_id = ?",
        (status, approver, _now(), action_id),
    )
    conn.commit()
    audit(conn, "manual", "human_approver", "action_decided",
          {"action_id": action_id, "status": status, "approver": approver})
    return dict(conn.execute("SELECT * FROM actions WHERE action_id = ?", (action_id,)).fetchone())
