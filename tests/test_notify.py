import httpx
import pytest

import sentinel.notify as notify_module
from sentinel.config import get_settings
from sentinel.notify import build_payload, notify_incident, should_notify


@pytest.fixture()
def webhook(monkeypatch):
    """Enable the webhook and capture outbound posts."""
    settings = get_settings()
    monkeypatch.setattr(settings, "webhook_url", "https://hooks.example/T000/B000")
    monkeypatch.setattr(settings, "notify_min_severity", "high")
    sent = []

    def fake_post(url, json=None, timeout=None):
        sent.append({"url": url, "json": json})
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    return sent


def test_disabled_without_url():
    assert get_settings().webhook_url == ""
    assert should_notify("critical") is False


def test_severity_floor(webhook):
    assert should_notify("critical") is True
    assert should_notify("high") is True
    assert should_notify("medium") is False
    assert should_notify("low") is False


def test_payload_is_slack_compatible():
    payload = build_payload("INC-TEST", "critical", "Factory fire", 1_500_000.0, 2)
    assert "text" in payload
    assert "INC-TEST" in payload["text"]
    assert "$1,500,000" in payload["text"]
    assert payload["sentinel"]["pending_approvals"] == 2


def test_notify_sends_and_reports(webhook):
    delivered = notify_incident("INC-1", "critical", "Quake", 2e6, 1)
    assert delivered is True
    assert len(webhook) == 1
    assert webhook[0]["url"].startswith("https://hooks.example/")

    # Below the floor: nothing sent.
    assert notify_incident("INC-2", "medium", "Minor", 0.0, 0) is False
    assert len(webhook) == 1


def test_notify_failure_never_raises(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "webhook_url", "https://hooks.example/down")

    def exploding_post(url, json=None, timeout=None):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", exploding_post)
    assert notify_incident("INC-3", "critical", "Anything", None, 0) is False


def test_pipeline_audits_notifications(conn, webhook):
    from sentinel.orchestrator import Pipeline

    summary = Pipeline(conn).run()
    assert summary["incidents_opened"] >= 1
    # Every high/critical incident produced a webhook + an audit row.
    notified = conn.execute(
        "SELECT COUNT(*) AS n FROM audit_log WHERE action = 'notification_sent'"
    ).fetchone()["n"]
    assert notified == len(webhook)
    assert notified >= 1


def test_notify_module_import_has_no_httpx_at_module_level():
    """httpx must be imported lazily so simulation-only installs never need it
    at import time (mirrors the anthropic lazy import)."""
    import inspect

    source = inspect.getsource(notify_module)
    module_level = source.split("def ")[0]
    assert "import httpx" not in module_level
