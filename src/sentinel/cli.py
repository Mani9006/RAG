"""Sentinel SCM command-line interface.

    sentinel run                 # process all new signals through the pipeline
    sentinel signals             # show the signal inbox
    sentinel incidents           # list opened incidents
    sentinel brief INC-XXXXXXXX  # print an incident's executive briefing
    sentinel approvals           # show the human approval queue
    sentinel decide ACT-XXXXXXXX --approve --approver alice@vertex.example
    sentinel reset               # rebuild the local store from data/
    sentinel serve               # start the control-plane API
"""
from __future__ import annotations

import argparse
import json
import sys

from sentinel.config import get_settings
from sentinel.db import get_connection
from sentinel.orchestrator import Pipeline, decide_action
from sentinel.telemetry import configure_logging


def _print_table(rows: list[dict], columns: list[str]) -> None:
    if not rows:
        print("(none)")
        return
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    print("  ".join("-" * widths[c] for c in columns))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns))


def cmd_run(args) -> None:
    conn = get_connection()
    summary = Pipeline(conn).run(limit=args.limit)
    print(json.dumps(summary, indent=2))


def cmd_signals(_args) -> None:
    conn = get_connection()
    rows = [dict(r) for r in conn.execute(
        "SELECT event_id, type, severity_hint, status, headline FROM signals"
        " ORDER BY observed_at").fetchall()]
    _print_table(rows, ["event_id", "type", "severity_hint", "status", "headline"])


def cmd_incidents(_args) -> None:
    conn = get_connection()
    rows = [dict(r) for r in conn.execute(
        "SELECT incident_id, event_id, severity, risk_score, created_at FROM incidents"
        " ORDER BY created_at DESC").fetchall()]
    _print_table(rows, ["incident_id", "event_id", "severity", "risk_score", "created_at"])


def cmd_brief(args) -> None:
    conn = get_connection()
    row = conn.execute(
        "SELECT briefing_md FROM incidents WHERE incident_id = ?", (args.incident_id,)).fetchone()
    if row is None:
        print(f"unknown incident: {args.incident_id}", file=sys.stderr)
        sys.exit(1)
    print(row["briefing_md"])


def cmd_approvals(_args) -> None:
    conn = get_connection()
    rows = [dict(r) for r in conn.execute(
        "SELECT action_id, incident_id, kind, value_usd, compliance_verdict, description"
        " FROM actions WHERE status = 'pending_approval' ORDER BY created_at").fetchall()]
    _print_table(rows, ["action_id", "incident_id", "kind", "value_usd",
                        "compliance_verdict", "description"])


def cmd_decide(args) -> None:
    conn = get_connection()
    try:
        result = decide_action(conn, args.action_id, args.approve, args.approver)
    except (KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result, indent=2))


def cmd_reset(_args) -> None:
    settings = get_settings()
    if settings.db_path.exists():
        settings.db_path.unlink()
    get_connection()
    print(f"rebuilt {settings.db_path} from {settings.data_dir}")


def cmd_serve(_args) -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run("sentinel.api:app", host=settings.api_host, port=settings.api_port)


def main(argv: list[str] | None = None) -> None:
    configure_logging(get_settings().log_level)
    parser = argparse.ArgumentParser(prog="sentinel", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="process new signals through the agentic pipeline")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(fn=cmd_run)

    sub.add_parser("signals", help="show the signal inbox").set_defaults(fn=cmd_signals)
    sub.add_parser("incidents", help="list opened incidents").set_defaults(fn=cmd_incidents)

    p = sub.add_parser("brief", help="print an incident's executive briefing")
    p.add_argument("incident_id")
    p.set_defaults(fn=cmd_brief)

    sub.add_parser("approvals", help="show the human approval queue").set_defaults(fn=cmd_approvals)

    p = sub.add_parser("decide", help="approve or reject a pending action")
    p.add_argument("action_id")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--approve", action="store_true", dest="approve")
    group.add_argument("--reject", action="store_false", dest="approve")
    p.add_argument("--approver", required=True)
    p.set_defaults(fn=cmd_decide)

    sub.add_parser("reset", help="rebuild the local store from data/").set_defaults(fn=cmd_reset)
    sub.add_parser("serve", help="start the control-plane API").set_defaults(fn=cmd_serve)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
