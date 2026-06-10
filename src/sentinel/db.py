"""Operational data store.

The repo-resident CSV/JSONL dataset is bootstrapped into a local SQLite
database on first use. SQLite keeps the platform self-contained (no cloud
services, no cost) while still giving the agents a real queryable system of
record. Swap this module for warehouse connectors in production.
"""
from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from sentinel.config import get_settings

SCHEMA = """
CREATE TABLE suppliers (
    supplier_id TEXT PRIMARY KEY, name TEXT, country TEXT, region TEXT,
    city TEXT, lat REAL, lon REAL,
    primary_port TEXT, tier_class TEXT, on_time_rate REAL,
    avg_lead_time_days INTEGER, annual_spend_musd REAL, geo_risk TEXT, iso9001 TEXT
);
CREATE TABLE parts (
    part_id TEXT PRIMARY KEY, part_number TEXT, description TEXT, category TEXT,
    primary_supplier_id TEXT, alternate_supplier_id TEXT,
    unit_cost_usd REAL, critical TEXT, replenish_days INTEGER
);
CREATE TABLE bom (product_id TEXT, part_id TEXT, qty_per_unit INTEGER);
CREATE TABLE products (
    product_id TEXT PRIMARY KEY, name TEXT, unit_price_usd REAL, monthly_demand_units INTEGER
);
CREATE TABLE inventory (
    part_id TEXT PRIMARY KEY, on_hand_units INTEGER,
    daily_consumption_units INTEGER, warehouse TEXT
);
CREATE TABLE shipments (
    shipment_id TEXT PRIMARY KEY, part_id TEXT, supplier_id TEXT, origin_port TEXT,
    mode TEXT, etd TEXT, eta TEXT, units INTEGER, status TEXT
);
CREATE TABLE purchase_orders (
    po_id TEXT PRIMARY KEY, part_id TEXT, supplier_id TEXT, units INTEGER,
    value_usd REAL, due_date TEXT, status TEXT
);
CREATE TABLE signals (
    event_id TEXT PRIMARY KEY, observed_at TEXT, source TEXT, type TEXT,
    severity_hint TEXT, headline TEXT, body TEXT, locations TEXT,
    suppliers_mentioned TEXT, status TEXT DEFAULT 'new'
);
CREATE TABLE incidents (
    incident_id TEXT PRIMARY KEY, run_id TEXT, event_id TEXT, severity TEXT,
    risk_score REAL, summary TEXT, impact_json TEXT, mitigation_json TEXT,
    compliance_json TEXT, briefing_md TEXT, created_at TEXT
);
CREATE TABLE actions (
    action_id TEXT PRIMARY KEY, incident_id TEXT, kind TEXT, description TEXT,
    supplier_id TEXT, value_usd REAL, status TEXT, compliance_verdict TEXT,
    approver TEXT, decided_at TEXT, created_at TEXT
);
CREATE TABLE audit_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts TEXT, actor TEXT,
    action TEXT, detail_json TEXT
);
"""

_CSV_TABLES = ["suppliers", "parts", "bom", "products", "inventory", "shipments", "purchase_orders"]


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    settings = get_settings()
    path = db_path or settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def is_bootstrapped(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT count(*) AS n FROM sqlite_master WHERE type='table' AND name='suppliers'"
    ).fetchone()
    return bool(row["n"])


def bootstrap(conn: sqlite3.Connection, data_dir: Path | None = None, force: bool = False) -> None:
    """(Re)build the SQLite store from the repo-resident dataset."""
    settings = get_settings()
    data_dir = data_dir or settings.data_dir

    if is_bootstrapped(conn):
        if not force:
            return
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            conn.execute(f"DROP TABLE IF EXISTS {row['name']}")

    conn.executescript(SCHEMA)

    for table in _CSV_TABLES:
        path = data_dir / f"{table}.csv"
        with path.open() as f:
            reader = csv.reader(f)
            header = next(reader)
            placeholders = ",".join("?" * len(header))
            conn.executemany(
                f"INSERT INTO {table} ({','.join(header)}) VALUES ({placeholders})",
                list(reader),
            )

    feed = data_dir / "disruption_feed.jsonl"
    with feed.open() as f:
        for line in f:
            ev = json.loads(line)
            conn.execute(
                "INSERT INTO signals (event_id, observed_at, source, type, severity_hint,"
                " headline, body, locations, suppliers_mentioned) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    ev["event_id"], ev["observed_at"], ev["source"], ev["type"],
                    ev["severity_hint"], ev["headline"], ev["body"],
                    json.dumps(ev["locations"]), json.dumps(ev["suppliers_mentioned"]),
                ),
            )
    conn.commit()


def get_connection(force_bootstrap: bool = False) -> sqlite3.Connection:
    conn = connect()
    bootstrap(conn, force=force_bootstrap)
    return conn
