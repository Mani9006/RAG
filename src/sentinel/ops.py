"""Continuous operations.

- `run_daemon`: the always-on loop — ingest live signals, run the pipeline,
  sleep, repeat. All state lives in SQLite, so the daemon is crash-safe and
  restart-safe by construction: a new process picks up exactly where the last
  one stopped (processed signals stay processed, pending approvals stay
  pending).
- `agreement_metrics`: the learning loop's measurement side — how often human
  approvers agree with what the agents proposed, sliced by action kind and
  compliance verdict. Falling agreement is the earliest signal that agent
  behavior has drifted from operator expectations.
"""
from __future__ import annotations

import sqlite3
import time

from sentinel.orchestrator import Pipeline, audit
from sentinel.telemetry import log


def agreement_metrics(conn: sqlite3.Connection) -> dict:
    rows = [dict(r) for r in conn.execute(
        "SELECT kind, compliance_verdict, status FROM actions"
        " WHERE status IN ('approved', 'rejected')")]
    decided = len(rows)
    approved = sum(1 for r in rows if r["status"] == "approved")

    by_kind: dict[str, dict] = {}
    for row in rows:
        bucket = by_kind.setdefault(row["kind"], {"approved": 0, "rejected": 0})
        bucket["approved" if row["status"] == "approved" else "rejected"] += 1

    by_verdict: dict[str, dict] = {}
    for row in rows:
        bucket = by_verdict.setdefault(
            row["compliance_verdict"], {"approved": 0, "rejected": 0})
        bucket["approved" if row["status"] == "approved" else "rejected"] += 1

    auto = conn.execute(
        "SELECT COUNT(*) AS n FROM actions WHERE status = 'auto_approved'").fetchone()["n"]
    blocked = conn.execute(
        "SELECT COUNT(*) AS n FROM actions WHERE status = 'blocked'").fetchone()["n"]
    pending = conn.execute(
        "SELECT COUNT(*) AS n FROM actions WHERE status = 'pending_approval'").fetchone()["n"]

    return {
        "decided_by_humans": decided,
        "human_agreement_rate": round(approved / decided, 3) if decided else None,
        "by_kind": by_kind,
        "by_compliance_verdict": by_verdict,
        "auto_approved": auto,
        "blocked_by_compliance": blocked,
        "pending_approval": pending,
    }


def run_cycle(conn: sqlite3.Connection, *, ingest_sources: str | None = "all") -> dict:
    """One daemon cycle: ingest live signals (optional), then run the pipeline."""
    ingest_report: dict = {}
    if ingest_sources:
        from sentinel.connectors import run_ingest

        ingest_report = run_ingest(conn, source=ingest_sources)
    summary = Pipeline(conn).run()
    audit(conn, summary["run_id"], "daemon", "daemon_cycle",
          {"ingest": ingest_report, "incidents_opened": summary["incidents_opened"]})
    return {"ingest": ingest_report, "run": summary}


def run_daemon(
    conn: sqlite3.Connection,
    *,
    interval_seconds: float = 900.0,
    max_cycles: int | None = None,
    ingest_sources: str | None = "all",
    sleep=time.sleep,
) -> int:
    """The continuous loop. Returns the number of completed cycles.

    `max_cycles=None` runs until interrupted (SIGINT exits cleanly after the
    in-flight cycle). State is entirely in SQLite — kill it anytime.
    """
    cycles = 0
    try:
        while max_cycles is None or cycles < max_cycles:
            started = time.monotonic()
            outcome = run_cycle(conn, ingest_sources=ingest_sources)
            cycles += 1
            log("daemon", "cycle_complete", cycle=cycles,
                incidents=outcome["run"]["incidents_opened"],
                signals=outcome["run"]["signals_processed"])
            if max_cycles is not None and cycles >= max_cycles:
                break
            elapsed = time.monotonic() - started
            sleep(max(0.0, interval_seconds - elapsed))
    except KeyboardInterrupt:
        log("daemon", "interrupted", cycles=cycles)
    return cycles
