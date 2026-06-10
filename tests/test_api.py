import pytest
from fastapi.testclient import TestClient

import sentinel.api as api_module
import sentinel.db as db_module
from sentinel.api import app


@pytest.fixture()
def client(conn, monkeypatch):
    """API client wired to the test's in-memory store."""

    class _NonClosing:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def close(self):  # keep the fixture connection alive across requests
            pass

    wrapper = _NonClosing(conn)
    monkeypatch.setattr(api_module, "get_connection", lambda: wrapper)
    monkeypatch.setattr(db_module, "get_connection", lambda force_bootstrap=False: wrapper)
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["backend"] == "simulation"


def test_run_then_inspect_incident_and_approve(client):
    summary = client.post("/runs").json()
    assert summary["incidents_opened"] >= 1

    incidents = client.get("/incidents").json()
    incident_id = incidents[0]["incident_id"]

    detail = client.get(f"/incidents/{incident_id}").json()
    assert detail["incident_id"] == incident_id
    assert "impact" in detail and "mitigation" in detail and "compliance" in detail
    assert detail["briefing_md"].startswith("# Situation Report")

    queue = client.get("/approvals").json()
    assert queue, "expected pending approvals after a run"
    action_id = queue[0]["action_id"]

    decided = client.post(
        f"/approvals/{action_id}", json={"approve": True, "approver": "alice@vertex"}
    ).json()
    assert decided["status"] == "approved"

    conflict = client.post(
        f"/approvals/{action_id}", json={"approve": False, "approver": "bob@vertex"}
    )
    assert conflict.status_code == 409


def test_unknown_incident_404(client):
    assert client.get("/incidents/INC-NOPE").status_code == 404


def test_command_center_dashboard_served(client):
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    assert "SENTINEL" in html
    # The dashboard drives the same API surface this suite tests.
    for endpoint in ("/signals", "/incidents", "/approvals", "/network", "/simulate"):
        assert endpoint in html
    # Self-contained: no external CDNs, fonts, or trackers.
    assert "http://" not in html.replace("http://localhost", "")
    assert "cdn." not in html and "googleapis" not in html


def test_network_and_scenario_endpoints(client):
    network = client.get("/network").json()
    assert network["single_points_of_failure"]
    assert network["critical_suppliers"]

    scenarios = client.get("/scenarios").json()
    names = {s["name"] for s in scenarios}
    assert "taiwan-strait" in names

    sim = client.post("/simulate?scenario=taiwan-strait&trials=200").json()
    assert sim["loss_usd"]["p90"] >= sim["loss_usd"]["p50"]
    assert client.post("/simulate?scenario=nope").status_code == 404
