import json
from pathlib import Path

import pytest

from sentinel.connectors import GdeltConnector, NoaaConnector, UsgsConnector
from sentinel.connectors.base import Footprint, haversine_km, ingest_records

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture()
def footprint(conn):
    return Footprint.load(conn)


def test_haversine_known_distance():
    # Hsinchu -> Kaohsiung is ~250 km
    dist = haversine_km(24.80, 120.97, 22.62, 120.31)
    assert 230 < dist < 260


def test_footprint_includes_suppliers_and_ports(footprint):
    kinds = {site.kind for site in footprint.sites}
    assert kinds == {"supplier", "port"}
    assert len(footprint.supplier_names()) == 40


def test_usgs_normalize_geo_matches_and_filters(footprint):
    records = UsgsConnector().normalize(_fixture("usgs_sample.json"), footprint)
    by_id = {r.event_id: r for r in records}

    # Mid-Pacific quake has no footprint overlap -> dropped entirely.
    assert "USGS-us7000qaaa" not in by_id
    assert len(records) == 2

    taiwan = by_id["USGS-us7000qxyz"]
    assert taiwan.type == "earthquake"
    assert taiwan.severity_hint == "critical"  # M6.8 within 150km of Hsinchu
    assert "Taiwan" in taiwan.locations
    assert any("Hsinchu" in s for s in taiwan.suppliers_mentioned)

    hamburg = by_id["USGS-us7000qbbb"]
    assert hamburg.severity_hint == "low"  # M5.6: relevant (port nearby) but weak


def test_noaa_normalize_classification_and_geo(footprint):
    records = NoaaConnector().normalize(_fixture("noaa_sample.json"), footprint)
    # Thunderstorm warning is not an operationally relevant class -> dropped.
    assert len(records) == 2

    hurricane = next(r for r in records if r.type == "typhoon")
    assert hurricane.severity_hint == "high"
    assert "United States" in hurricane.locations
    # Long Beach polygon centroid is within 300km of the Tijuana supplier.
    assert any("Tijuana" in s for s in hurricane.suppliers_mentioned)

    flood = next(r for r in records if r.type == "flood")
    assert flood.locations == ["United States"]  # no geometry -> country-level


def test_gdelt_normalize_supplier_mention_escalates(footprint):
    records = GdeltConnector().normalize(_fixture("gdelt_sample.json"), footprint)
    # Bakery story has no disruption keyword -> dropped.
    assert len(records) == 2

    strike = next(r for r in records if r.type == "labor_strike")
    assert strike.suppliers_mentioned == ["Gdansk PCB House"]
    assert strike.severity_hint == "high"

    fire = next(r for r in records if r.type == "factory_fire")
    assert fire.suppliers_mentioned == []
    assert fire.severity_hint == "medium"


def test_ingest_is_idempotent(conn, footprint):
    records = UsgsConnector().normalize(_fixture("usgs_sample.json"), footprint)
    first = ingest_records(conn, records)
    assert first == {"received": 2, "inserted": 2, "duplicates": 0}

    second = ingest_records(conn, records)
    assert second == {"received": 2, "inserted": 0, "duplicates": 2}

    row = conn.execute(
        "SELECT * FROM signals WHERE event_id = 'USGS-us7000qxyz'").fetchone()
    assert row["status"] == "new"
    assert json.loads(row["locations"])


def test_ingested_live_signal_flows_through_pipeline(conn, footprint):
    from sentinel.orchestrator import Pipeline

    # Mark the seeded feed as handled so only the live quake is new.
    conn.execute("UPDATE signals SET status = 'monitoring'")
    conn.commit()
    records = UsgsConnector().normalize(_fixture("usgs_sample.json"), footprint)
    ingest_records(conn, records)

    summary = Pipeline(conn).run()
    assert summary["signals_processed"] == 2
    # The M6.8 near two Taiwan suppliers must escalate to an incident.
    assert "USGS-us7000qxyz" in summary["incident_events"]
