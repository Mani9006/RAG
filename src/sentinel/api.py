"""Control-plane API.

Exposes the platform to dashboards, schedulers, and human approvers:
trigger runs, inspect signals/incidents/briefings, and work the approval queue.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sentinel import __version__
from sentinel.config import get_settings
from sentinel.db import get_connection
from sentinel.orchestrator import Pipeline, decide_action
from sentinel.telemetry import configure_logging


@asynccontextmanager
async def _lifespan(app: FastAPI):
    configure_logging(get_settings().log_level)
    get_connection().close()  # bootstrap the store on first boot
    yield


app = FastAPI(
    title="Sentinel SCM",
    version=__version__,
    description="Autonomous multi-agent supply chain disruption response platform.",
    lifespan=_lifespan,
)


class DecisionRequest(BaseModel):
    approve: bool
    approver: str


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {"status": "ok", "version": __version__, "backend": settings.llm_backend}


@app.post("/runs")
def trigger_run(limit: int | None = None) -> dict:
    conn = get_connection()
    try:
        return Pipeline(conn).run(limit=limit)
    finally:
        conn.close()


@app.get("/signals")
def list_signals() -> list[dict]:
    conn = get_connection()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT event_id, observed_at, type, severity_hint, headline, status"
            " FROM signals ORDER BY observed_at").fetchall()]
    finally:
        conn.close()


@app.get("/incidents")
def list_incidents() -> list[dict]:
    conn = get_connection()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT incident_id, run_id, event_id, severity, risk_score, summary, created_at"
            " FROM incidents ORDER BY created_at DESC").fetchall()]
    finally:
        conn.close()


@app.get("/incidents/{incident_id}")
def get_incident(incident_id: str) -> dict:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM incidents WHERE incident_id = ?", (incident_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="incident not found")
        incident = dict(row)
        for key in ("impact_json", "mitigation_json", "compliance_json"):
            incident[key.removesuffix("_json")] = json.loads(incident.pop(key))
        incident["actions"] = [dict(r) for r in conn.execute(
            "SELECT * FROM actions WHERE incident_id = ?", (incident_id,)).fetchall()]
        return incident
    finally:
        conn.close()


@app.get("/approvals")
def approval_queue() -> list[dict]:
    conn = get_connection()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM actions WHERE status = 'pending_approval' ORDER BY created_at"
        ).fetchall()]
    finally:
        conn.close()


@app.post("/approvals/{action_id}")
def decide(action_id: str, body: DecisionRequest) -> dict:
    conn = get_connection()
    try:
        return decide_action(conn, action_id, body.approve, body.approver)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        conn.close()


@app.get("/audit")
def audit_trail(run_id: str | None = None, limit: int = 200) -> list[dict]:
    conn = get_connection()
    try:
        if run_id:
            cur = conn.execute(
                "SELECT * FROM audit_log WHERE run_id = ? ORDER BY seq LIMIT ?", (run_id, limit))
        else:
            cur = conn.execute("SELECT * FROM audit_log ORDER BY seq DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
