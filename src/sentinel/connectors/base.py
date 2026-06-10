"""Connector foundation for live disruption intelligence.

Commercial SCRM platforms (Resilinc EventWatch, Everstream) sell aggregated
feeds built largely on free primary sources. These connectors implement the
same strategy openly: each one fetches a public feed (USGS, NOAA/NWS, GDELT),
normalizes raw events into the signal schema, **geo-matches them against the
company's actual supplier/port footprint**, and dedupes into the signal inbox.
Downstream, the agentic pipeline is identical for live and replayed signals.

Design rules:
- `fetch()` is the only place that touches the network.
- `normalize(payload, footprint)` is pure → fully testable from committed
  fixtures, no network in tests or CI.
- Event IDs are deterministic → re-ingestion is idempotent (INSERT OR IGNORE).
"""
from __future__ import annotations

import csv
import json
import math
import sqlite3
from dataclasses import asdict, dataclass, field

from sentinel.config import get_settings

USER_AGENT = "sentinel-scm/0.1 (open-source supply chain risk platform)"
EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


@dataclass
class SignalRecord:
    """One normalized disruption signal, matching the `signals` table."""

    event_id: str
    observed_at: str
    source: str
    type: str
    severity_hint: str
    headline: str
    body: str
    locations: list[str] = field(default_factory=list)
    suppliers_mentioned: list[str] = field(default_factory=list)

    def as_row(self) -> tuple:
        return (
            self.event_id, self.observed_at, self.source, self.type, self.severity_hint,
            self.headline, self.body,
            json.dumps(self.locations), json.dumps(self.suppliers_mentioned),
        )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Site:
    name: str
    country: str
    lat: float
    lon: float
    kind: str  # "supplier" | "port"


class Footprint:
    """The company's geographic exposure: supplier sites + ports of origin.
    Built once per ingest from the system of record + the geo gazetteer."""

    def __init__(self, sites: list[Site]):
        self.sites = sites

    @classmethod
    def load(cls, conn: sqlite3.Connection) -> Footprint:
        sites = [
            Site(r["name"], r["country"], r["lat"], r["lon"], "supplier")
            for r in conn.execute("SELECT name, country, lat, lon FROM suppliers")
        ]
        gazetteer = get_settings().data_dir / "geo" / "locations.csv"
        with gazetteer.open() as f:
            for row in csv.DictReader(f):
                sites.append(Site(
                    row["name"], row["country"], float(row["lat"]), float(row["lon"]), row["kind"]
                ))
        return cls(sites)

    def near(self, lat: float, lon: float, radius_km: float) -> list[tuple[Site, float]]:
        """Sites within `radius_km` of a point, nearest first."""
        hits = []
        for site in self.sites:
            dist = haversine_km(lat, lon, site.lat, site.lon)
            if dist <= radius_km:
                hits.append((site, round(dist, 1)))
        hits.sort(key=lambda h: h[1])
        return hits

    def supplier_names(self) -> list[str]:
        return [s.name for s in self.sites if s.kind == "supplier"]


def ingest_records(conn: sqlite3.Connection, records: list[SignalRecord]) -> dict:
    """Idempotently insert normalized signals into the inbox."""
    inserted = 0
    for record in records:
        cur = conn.execute(
            "INSERT OR IGNORE INTO signals (event_id, observed_at, source, type,"
            " severity_hint, headline, body, locations, suppliers_mentioned)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            record.as_row(),
        )
        inserted += cur.rowcount
    conn.commit()
    return {"received": len(records), "inserted": inserted,
            "duplicates": len(records) - inserted}


def http_get_json(url: str, params: dict | None = None, timeout: float = 20.0) -> dict:
    """Single network entry point shared by all connectors."""
    import httpx

    response = httpx.get(
        url, params=params, timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        follow_redirects=True,
    )
    response.raise_for_status()
    return response.json()
