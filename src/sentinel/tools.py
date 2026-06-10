"""Tool surface exposed to the agents.

Each tool is a pure function over the SQLite system of record, registered with
a JSON schema so it can be passed straight to the Anthropic Messages API. The
orchestrator executes tools client-side inside the agentic loop, which keeps a
typed, auditable hook on every action an agent takes.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

import yaml

from sentinel.config import get_settings

TODAY = date(2026, 6, 10)  # anchor date of the synthetic dataset


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., Any]

    def anthropic_schema(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


def _rows(cur) -> list[dict]:
    return [dict(r) for r in cur.fetchall()]


class ToolBox:
    """All tools bound to one DB connection. `registry` maps name -> Tool."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.registry: dict[str, Tool] = {}
        for tool in self._build():
            self.registry[tool.name] = tool

    def execute(self, name: str, tool_input: dict) -> str:
        """Run a tool and return a JSON string result (or a JSON error payload)."""
        tool = self.registry.get(name)
        if tool is None:
            return json.dumps({"error": f"unknown tool: {name}"})
        try:
            result = tool.fn(**tool_input)
            return json.dumps(result, default=str)
        except Exception as exc:  # surfaced to the model so it can adapt
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    def schemas(self, names: list[str] | None = None) -> list[dict]:
        tools = self.registry.values() if names is None else [self.registry[n] for n in names]
        return [t.anthropic_schema() for t in tools]

    # ------------------------------------------------------------------ tools

    def _build(self) -> list[Tool]:
        conn = self.conn

        def find_suppliers(query: str = "", country: str = "", region: str = "") -> list[dict]:
            sql = "SELECT * FROM suppliers WHERE 1=1"
            args: list = []
            if query:
                sql += " AND name LIKE ?"
                args.append(f"%{query}%")
            if country:
                sql += " AND country = ?"
                args.append(country)
            if region:
                sql += " AND region = ?"
                args.append(region)
            return _rows(conn.execute(sql + " LIMIT 50", args))

        def supplier_exposure(supplier_id: str) -> dict:
            """Everything we buy from a supplier: parts, open POs, in-transit units."""
            parts = _rows(conn.execute(
                "SELECT part_id, part_number, description, category, critical,"
                " (alternate_supplier_id = '') AS single_sourced"
                " FROM parts WHERE primary_supplier_id = ? OR alternate_supplier_id = ?",
                (supplier_id, supplier_id)))
            pos = _rows(conn.execute(
                "SELECT po_id, part_id, units, value_usd, due_date, status"
                " FROM purchase_orders WHERE supplier_id = ? AND status != 'closed'", (supplier_id,)))
            shipments = _rows(conn.execute(
                "SELECT shipment_id, part_id, eta, units, status FROM shipments"
                " WHERE supplier_id = ? AND status IN ('in_transit','customs_hold')", (supplier_id,)))
            return {
                "supplier_id": supplier_id,
                "parts": parts,
                "open_po_value_usd": round(sum(p["value_usd"] for p in pos), 2),
                "open_pos": pos,
                "in_transit_units": sum(s["units"] for s in shipments),
                "shipments": shipments,
            }

        def parts_blast_radius(part_ids: list[str]) -> list[dict]:
            """For each part: products that consume it, demand, days of inventory cover,
            and monthly revenue exposed if the part stocks out."""
            out = []
            for part_id in part_ids:
                inv = conn.execute("SELECT * FROM inventory WHERE part_id = ?", (part_id,)).fetchone()
                days_cover = (
                    round(inv["on_hand_units"] / inv["daily_consumption_units"], 1)
                    if inv and inv["daily_consumption_units"] else None
                )
                products = _rows(conn.execute(
                    "SELECT p.product_id, p.name, p.unit_price_usd, p.monthly_demand_units,"
                    " b.qty_per_unit FROM bom b JOIN products p ON p.product_id = b.product_id"
                    " WHERE b.part_id = ?", (part_id,)))
                revenue = round(sum(p["unit_price_usd"] * p["monthly_demand_units"] for p in products), 2)
                out.append({
                    "part_id": part_id,
                    "days_of_cover": days_cover,
                    "affected_products": products,
                    "monthly_revenue_at_risk_usd": revenue,
                })
            return out

        def find_alternate_sources(part_id: str) -> dict:
            """Qualified alternate for the part plus same-category suppliers as candidates."""
            part = conn.execute("SELECT * FROM parts WHERE part_id = ?", (part_id,)).fetchone()
            if part is None:
                return {"error": f"unknown part: {part_id}"}
            alt = None
            if part["alternate_supplier_id"]:
                alt = dict(conn.execute(
                    "SELECT * FROM suppliers WHERE supplier_id = ?",
                    (part["alternate_supplier_id"],)).fetchone())
            candidates = _rows(conn.execute(
                "SELECT DISTINCT s.* FROM suppliers s JOIN parts p"
                " ON p.primary_supplier_id = s.supplier_id"
                " WHERE p.category = ? AND s.supplier_id != ?"
                " ORDER BY s.on_time_rate DESC LIMIT 5",
                (part["category"], part["primary_supplier_id"])))
            return {
                "part": dict(part),
                "qualified_alternate": alt,
                "category_candidates": candidates,
            }

        def shipments_through_port(port: str) -> list[dict]:
            return _rows(conn.execute(
                "SELECT shipment_id, part_id, supplier_id, mode, eta, units, status"
                " FROM shipments WHERE origin_port = ? AND status IN ('in_transit','customs_hold')"
                " ORDER BY eta LIMIT 100", (port,)))

        def inventory_position(part_id: str) -> dict:
            inv = conn.execute("SELECT * FROM inventory WHERE part_id = ?", (part_id,)).fetchone()
            if inv is None:
                return {"error": f"no inventory record for {part_id}"}
            inbound = conn.execute(
                "SELECT COALESCE(SUM(units),0) AS units FROM shipments"
                " WHERE part_id = ? AND status = 'in_transit'", (part_id,)).fetchone()
            result = dict(inv)
            result["inbound_in_transit_units"] = inbound["units"]
            if inv["daily_consumption_units"]:
                result["days_of_cover"] = round(inv["on_hand_units"] / inv["daily_consumption_units"], 1)
            return result

        def get_policy() -> dict:
            path = get_settings().data_dir / "policies" / "procurement_policy.yaml"
            return yaml.safe_load(path.read_text())

        def estimate_expedite_cost(units: int, unit_cost_usd: float) -> dict:
            """Deterministic freight model: air premium ≈ 18% of goods value, min 8k USD."""
            goods_value = units * unit_cost_usd
            premium = max(round(goods_value * 0.18, 2), 8000.0)
            return {
                "units": units,
                "goods_value_usd": round(goods_value, 2),
                "air_freight_premium_usd": premium,
                "transit_days_air": 5,
                "transit_days_ocean": 28,
            }

        return [
            Tool("find_suppliers",
                 "Search the supplier master by name fragment, country, or region.",
                 {"type": "object", "properties": {
                     "query": {"type": "string", "description": "Supplier name fragment"},
                     "country": {"type": "string"},
                     "region": {"type": "string", "enum": ["APAC", "EMEA", "AMER"]}},
                  "required": []},
                 find_suppliers),
            Tool("supplier_exposure",
                 "Get full commercial exposure to one supplier: sourced parts (with single-source "
                 "flags), open purchase orders, and in-transit shipments. Call this first when a "
                 "disruption names a supplier.",
                 {"type": "object", "properties": {"supplier_id": {"type": "string"}},
                  "required": ["supplier_id"]},
                 supplier_exposure),
            Tool("parts_blast_radius",
                 "For a list of part IDs, compute downstream impact: affected products, days of "
                 "inventory cover, and monthly revenue at risk. Call this to size an incident.",
                 {"type": "object", "properties": {
                     "part_ids": {"type": "array", "items": {"type": "string"}}},
                  "required": ["part_ids"]},
                 parts_blast_radius),
            Tool("find_alternate_sources",
                 "Find the qualified alternate supplier and best same-category candidate suppliers "
                 "for a part. Call this when planning re-sourcing mitigations.",
                 {"type": "object", "properties": {"part_id": {"type": "string"}},
                  "required": ["part_id"]},
                 find_alternate_sources),
            Tool("shipments_through_port",
                 "List active shipments originating from a named port. Call this for port, weather, "
                 "or logistics disruptions.",
                 {"type": "object", "properties": {"port": {"type": "string"}},
                  "required": ["port"]},
                 shipments_through_port),
            Tool("inventory_position",
                 "Current on-hand inventory, daily consumption, inbound units, and days of cover "
                 "for one part.",
                 {"type": "object", "properties": {"part_id": {"type": "string"}},
                  "required": ["part_id"]},
                 inventory_position),
            Tool("get_policy",
                 "Read the procurement governance policy: spend authority limits, sourcing rules, "
                 "and restricted entities. Compliance checks MUST be grounded in this document.",
                 {"type": "object", "properties": {}, "required": []},
                 get_policy),
            Tool("estimate_expedite_cost",
                 "Estimate the cost and transit-time delta of expediting units by air freight.",
                 {"type": "object", "properties": {
                     "units": {"type": "integer"},
                     "unit_cost_usd": {"type": "number"}},
                  "required": ["units", "unit_cost_usd"]},
                 estimate_expedite_cost),
        ]
