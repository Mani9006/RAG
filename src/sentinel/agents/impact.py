"""Impact Assessment agent — walks the supplier/BOM graph to quantify the
blast radius of an escalated disruption."""
from __future__ import annotations

import json

from sentinel.agents.base import JSON_ONLY, Agent
from sentinel.tools import ToolBox


class ImpactAgent(Agent):
    name = "impact_assessment"
    tools = [
        "supplier_exposure", "parts_blast_radius", "shipments_through_port",
        "inventory_position", "find_suppliers", "simulate_supplier_outage",
        "network_risk_profile", "run_disruption_scenario",
    ]
    system = (
        "You are the Impact Assessment agent for Vertex Devices. Given an escalated "
        "disruption and the entities the triage agent identified, quantify the blast "
        "radius end to end: which suppliers are exposed, which parts flow from them "
        "(flag single-sourced and critical parts), which shipments are in harm's way, "
        "which finished products depend on those parts, how many days of inventory "
        "cover remain, and the monthly revenue at risk. For supplier-level disruptions, "
        "run simulate_supplier_outage with a duration matching the event (e.g. 42 days "
        "for a 6-week fire restoration) and include its stockout timeline and bounded "
        "revenue-loss figure — it is more precise than raw revenue-at-risk because it "
        "accounts for inventory cover and alternate-supplier relief. Ground EVERY "
        "figure in tool results — never estimate what you can query. "
        + JSON_ONLY
        + ' Schema: {"affected_suppliers": [{"supplier_id": str, "name": str, '
        '"open_po_value_usd": num}], "affected_parts": [{"part_id": str, '
        '"single_sourced": bool, "critical": bool, "days_of_cover": num|null, '
        '"monthly_revenue_at_risk_usd": num}], "shipments_at_risk": int, '
        '"total_open_po_value_usd": num, "total_monthly_revenue_at_risk_usd": num, '
        '"min_days_of_cover": num|null, "outage_simulation": object|null, '
        '"loss_distribution": object|null, "narrative": str}'
    )

    def build_prompt(self, payload: dict) -> str:
        return (
            "Assess the impact of this escalated disruption.\n\nSignal:\n"
            + json.dumps(payload["signal"], indent=2)
            + "\n\nTriage result:\n"
            + json.dumps(payload["triage"], indent=2)
        )

    def simulate(self, payload: dict, toolbox: ToolBox) -> dict:
        entities = payload["triage"]["entities"]

        supplier_ids = list(entities.get("suppliers", []))
        for country in entities.get("countries", []):
            supplier_ids += [
                s["supplier_id"] for s in self.tool_json(toolbox, "find_suppliers", country=country)
            ]
        supplier_ids = sorted(set(supplier_ids))[:6]  # bound the walk

        affected_suppliers, part_ids = [], []
        total_po_value = 0.0
        for sid in supplier_ids:
            exposure = self.tool_json(toolbox, "supplier_exposure", supplier_id=sid)
            name_rows = self.tool_json(toolbox, "find_suppliers", query="")
            name = next((r["name"] for r in name_rows if r["supplier_id"] == sid), sid)
            affected_suppliers.append({
                "supplier_id": sid,
                "name": name,
                "open_po_value_usd": exposure["open_po_value_usd"],
            })
            total_po_value += exposure["open_po_value_usd"]
            # Prioritize the dangerous parts: single-sourced or critical first.
            ranked = sorted(
                exposure["parts"],
                key=lambda p: (p["single_sourced"], p["critical"] == "yes"),
                reverse=True,
            )
            part_ids += [p["part_id"] for p in ranked[:4]]

        shipments_at_risk = 0
        for port in entities.get("ports", []):
            shipments_at_risk += len(self.tool_json(toolbox, "shipments_through_port", port=port))

        part_ids = sorted(set(part_ids))[:10]
        blast = self.tool_json(toolbox, "parts_blast_radius", part_ids=part_ids) if part_ids else []

        affected_parts = []
        part_meta = {}
        for sid in supplier_ids:
            exposure = self.tool_json(toolbox, "supplier_exposure", supplier_id=sid)
            for p in exposure["parts"]:
                part_meta[p["part_id"]] = p
        for entry in blast:
            meta = part_meta.get(entry["part_id"], {})
            affected_parts.append({
                "part_id": entry["part_id"],
                "single_sourced": bool(meta.get("single_sourced")),
                "critical": meta.get("critical") == "yes",
                "days_of_cover": entry["days_of_cover"],
                "monthly_revenue_at_risk_usd": entry["monthly_revenue_at_risk_usd"],
            })

        covers = [p["days_of_cover"] for p in affected_parts if p["days_of_cover"] is not None]
        total_revenue = round(sum(p["monthly_revenue_at_risk_usd"] for p in affected_parts), 2)

        # Supplier-level events get a propagation simulation (30-day outage on
        # the most exposed named supplier) plus a Monte Carlo loss distribution
        # over duration/ramp/demand uncertainty.
        outage_simulation = None
        loss_distribution = None
        if entities.get("suppliers"):
            most_exposed = max(
                affected_suppliers, key=lambda s: s["open_po_value_usd"], default=None
            )
            if most_exposed:
                outage_simulation = self.tool_json(
                    toolbox, "simulate_supplier_outage",
                    supplier_id=most_exposed["supplier_id"], outage_days=30,
                )
                loss_distribution = self._loss_distribution(toolbox, entities["suppliers"])

        return {
            "affected_suppliers": affected_suppliers,
            "affected_parts": affected_parts,
            "shipments_at_risk": shipments_at_risk,
            "total_open_po_value_usd": round(total_po_value, 2),
            "total_monthly_revenue_at_risk_usd": total_revenue,
            "min_days_of_cover": min(covers) if covers else None,
            "outage_simulation": outage_simulation,
            "loss_distribution": loss_distribution,
            "narrative": (
                f"{len(affected_suppliers)} supplier(s) and {len(affected_parts)} part(s) are in "
                f"the blast radius; {shipments_at_risk} shipment(s) transit affected ports. "
                f"Open PO exposure ${total_po_value:,.0f}; up to ${total_revenue:,.0f}/month of "
                "product revenue depends on the affected parts."
            ),
        }

    @staticmethod
    def _loss_distribution(toolbox: ToolBox, supplier_ids: list[str]) -> dict:
        """Seeded Monte Carlo loss band for an ad-hoc outage of the named
        suppliers (duration 14/30/60 days triangular)."""
        from sentinel.graph import SupplyGraph
        from sentinel.montecarlo import MonteCarloEngine, ScenarioSpec, Triangular

        graph = SupplyGraph(toolbox.conn)
        names = [
            graph.suppliers[sid]["name"] for sid in supplier_ids if sid in graph.suppliers
        ]
        spec = ScenarioSpec(
            name="incident-adhoc",
            description="Ad-hoc uncertainty band for the incident's named suppliers",
            outage_days=Triangular(14, 30, 60),
            supplier_names=names,
        )
        result = MonteCarloEngine(graph).run_scenario(spec, trials=400)
        if "error" in result:
            return result
        return {"trials": result["trials"], "seed": result["seed"], **result["loss_usd"]}
