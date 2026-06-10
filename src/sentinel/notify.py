"""Notification fabric.

Pushes incident notifications to a Slack-compatible incoming webhook
(`SENTINEL_WEBHOOK_URL`). Notification failures are logged and audited but
NEVER block the pipeline — alerting is best-effort; the system of record is
the database.
"""
from __future__ import annotations

from sentinel.config import get_settings
from sentinel.telemetry import log

_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def should_notify(severity: str) -> bool:
    settings = get_settings()
    if not settings.webhook_url:
        return False
    floor = _SEVERITY_RANK.get(settings.notify_min_severity, 2)
    return _SEVERITY_RANK.get(severity, 0) >= floor


def build_payload(incident_id: str, severity: str, headline: str,
                  revenue_at_risk_usd: float | None, pending_approvals: int) -> dict:
    """Slack-compatible payload (works with any webhook that accepts {"text"})."""
    icon = {"critical": "🔴", "high": "🟠"}.get(severity, "🟡")
    lines = [
        f"{icon} *[{severity.upper()}] {headline}*",
        f"Incident `{incident_id}` opened by Sentinel SCM.",
    ]
    if revenue_at_risk_usd is not None:
        lines.append(f"Monthly revenue at risk: *${revenue_at_risk_usd:,.0f}*")
    if pending_approvals:
        lines.append(f"⏳ {pending_approvals} action(s) awaiting human approval.")
    return {
        "text": "\n".join(lines),
        "sentinel": {
            "incident_id": incident_id,
            "severity": severity,
            "revenue_at_risk_usd": revenue_at_risk_usd,
            "pending_approvals": pending_approvals,
        },
    }


def notify_incident(incident_id: str, severity: str, headline: str,
                    revenue_at_risk_usd: float | None, pending_approvals: int) -> bool:
    """Send the webhook. Returns True only on confirmed delivery."""
    if not should_notify(severity):
        return False
    import httpx

    payload = build_payload(
        incident_id, severity, headline, revenue_at_risk_usd, pending_approvals)
    try:
        response = httpx.post(get_settings().webhook_url, json=payload, timeout=5.0)
        response.raise_for_status()
        log("notify", "sent", incident_id=incident_id, severity=severity)
        return True
    except Exception as exc:  # best-effort: never block the pipeline
        log("notify", "failed", incident_id=incident_id, error=f"{type(exc).__name__}: {exc}")
        return False
