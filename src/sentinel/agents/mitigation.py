"""Mitigation Planner agent — turns a quantified impact into a ranked set of
concrete, costed response actions."""
from __future__ import annotations

import json

from sentinel.agents.base import JSON_ONLY, Agent
from sentinel.tools import ToolBox


class MitigationAgent(Agent):
    name = "mitigation_planner"
    tools = [
        "find_alternate_sources", "estimate_expedite_cost", "inventory_position", "get_policy",
    ]
    system = (
        "You are the Mitigation Planner agent for Vertex Devices. Given a quantified "
        "disruption impact, produce a small set of concrete, costed mitigation options. "
        "Allowed action kinds: expedite_freight, alternate_source_po, inventory_reallocation, "
        "supplier_quarantine, monitor_only. Prefer actions on single-sourced critical parts "
        "with the fewest days of cover. Cost every option with the estimate_expedite_cost "
        "tool or unit economics from the data — no invented numbers. Read the policy first "
        "so options are plannable within governance (e.g. air freight only when stockout is "
        "near). Rank options by expected risk reduction per dollar. "
        + JSON_ONLY
        + ' Schema: {"options": [{"kind": str, "description": str, "part_id": str|null, '
        '"supplier_id": str|null, "value_usd": num, "lead_time_days": num, "tradeoffs": str}], '
        '"recommended_index": int, "rationale": str}'
    )

    def build_prompt(self, payload: dict) -> str:
        return (
            "Plan mitigations for this disruption.\n\nSignal:\n"
            + json.dumps(payload["signal"], indent=2)
            + "\n\nImpact assessment:\n"
            + json.dumps(payload["impact"], indent=2)
        )

    def simulate(self, payload: dict, toolbox: ToolBox) -> dict:
        impact = payload["impact"]
        options: list[dict] = []

        # Target the riskiest parts first: single-sourced/critical, least cover.
        parts = sorted(
            impact["affected_parts"],
            key=lambda p: (
                not p["single_sourced"], not p["critical"],
                p["days_of_cover"] if p["days_of_cover"] is not None else 999,
            ),
        )[:3]

        for part in parts:
            pos = self.tool_json(toolbox, "inventory_position", part_id=part["part_id"])
            if "error" in pos:
                continue
            sources = self.tool_json(toolbox, "find_alternate_sources", part_id=part["part_id"])
            unit_cost = sources["part"]["unit_cost_usd"]
            need_units = max(pos["daily_consumption_units"] * 30, 1)

            if part["days_of_cover"] is not None and part["days_of_cover"] < 21:
                expedite = self.tool_json(
                    toolbox, "estimate_expedite_cost",
                    units=need_units, unit_cost_usd=unit_cost,
                )
                options.append({
                    "kind": "expedite_freight",
                    "description": (
                        f"Air-freight {need_units:,} units of {part['part_id']} "
                        f"(~{part['days_of_cover']} days of cover remaining)."
                    ),
                    "part_id": part["part_id"],
                    "supplier_id": sources["part"]["primary_supplier_id"],
                    "value_usd": expedite["air_freight_premium_usd"],
                    "lead_time_days": expedite["transit_days_air"],
                    "tradeoffs": "Highest cost per unit of risk reduction; fastest relief.",
                })

            alt = sources.get("qualified_alternate") or (
                sources["category_candidates"][0] if sources["category_candidates"] else None
            )
            if alt:
                options.append({
                    "kind": "alternate_source_po",
                    "description": (
                        f"Place a bridge PO for {need_units:,} units of {part['part_id']} "
                        f"with {alt['name']} ({alt['supplier_id']})."
                    ),
                    "part_id": part["part_id"],
                    "supplier_id": alt["supplier_id"],
                    "value_usd": round(need_units * unit_cost, 2),
                    "lead_time_days": alt["avg_lead_time_days"],
                    "tradeoffs": (
                        "Slower than air freight but builds dual-source resilience; "
                        "requires incoming-quality checks on first lot."
                    ),
                })

        if not options:
            options.append({
                "kind": "monitor_only",
                "description": "No actionable exposure found; keep the signal on a 24h watch.",
                "part_id": None, "supplier_id": None,
                "value_usd": 0.0, "lead_time_days": 0,
                "tradeoffs": "Zero cost; accepts residual risk of late detection.",
            })

        # Cheapest meaningful action first is our deterministic ranking proxy.
        options.sort(key=lambda o: (o["kind"] == "monitor_only", o["value_usd"]))
        return {
            "options": options,
            "recommended_index": 0,
            "rationale": (
                "Options target single-sourced critical parts with the least inventory cover; "
                "ranked by cost subject to lead-time fit."
            ),
        }
