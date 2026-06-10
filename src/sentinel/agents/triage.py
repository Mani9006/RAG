"""Risk Triage agent — classifies raw disruption signals and decides which
deserve a full investigation."""
from __future__ import annotations

import json

from sentinel.agents.base import JSON_ONLY, Agent
from sentinel.tools import ToolBox

# Deterministic priors used by the simulation backend.
_TYPE_WEIGHT = {
    "typhoon": 0.9, "earthquake": 0.95, "factory_fire": 0.85, "cyber_incident": 0.75,
    "supplier_insolvency": 0.8, "export_control": 0.7, "labor_strike": 0.55,
    "port_congestion": 0.5, "quality_recall": 0.5, "price_shock": 0.45,
    "customs_delay": 0.3, "flood": 0.4,
}
_HINT_WEIGHT = {"critical": 1.0, "high": 0.8, "medium": 0.55, "low": 0.3}


class TriageAgent(Agent):
    name = "risk_triage"
    tools = ["find_suppliers", "shipments_through_port"]
    system = (
        "You are the Risk Triage agent for Vertex Devices' supply chain command center. "
        "You receive one raw disruption signal at a time. Score how threatening it is to "
        "Vertex's supply continuity, identify the entities involved (suppliers, ports, "
        "regions), and decide whether to escalate. Be calibrated: most signals are noise; "
        "escalating everything destroys the team's trust. Use find_suppliers to confirm "
        "whether mentioned companies or affected countries are actually in our supplier "
        "base, and shipments_through_port to check whether a named port carries our freight. "
        + JSON_ONLY
        + ' Schema: {"severity": "critical|high|medium|low", "risk_score": 0.0-1.0, '
        '"rationale": str, "entities": {"suppliers": [supplier_id], "ports": [str], '
        '"countries": [str]}, "decision": "investigate|monitor|dismiss"}'
    )

    def build_prompt(self, payload: dict) -> str:
        return "Triage this disruption signal:\n\n" + json.dumps(payload["signal"], indent=2)

    def simulate(self, payload: dict, toolbox: ToolBox) -> dict:
        signal = payload["signal"]
        score = round(
            0.6 * _TYPE_WEIGHT.get(signal["type"], 0.4)
            + 0.4 * _HINT_WEIGHT.get(signal["severity_hint"], 0.4),
            3,
        )

        supplier_ids: list[str] = []
        for name in json.loads(signal["suppliers_mentioned"]) if isinstance(
            signal["suppliers_mentioned"], str
        ) else signal["suppliers_mentioned"]:
            hits = self.tool_json(toolbox, "find_suppliers", query=name)
            supplier_ids += [h["supplier_id"] for h in hits]

        locations = json.loads(signal["locations"]) if isinstance(
            signal["locations"], str
        ) else signal["locations"]
        countries = [loc for loc in locations if self.tool_json(toolbox, "find_suppliers", country=loc)]
        ports = [
            loc for loc in locations
            if self.tool_json(toolbox, "shipments_through_port", port=loc)
        ]

        # No footprint overlap → the event can't touch us; cap the score.
        if not supplier_ids and not countries and not ports:
            score = min(score, 0.35)

        severity = (
            "critical" if score >= 0.8 else
            "high" if score >= 0.6 else
            "medium" if score >= 0.45 else "low"
        )
        decision = "investigate" if score >= 0.6 else ("monitor" if score >= 0.4 else "dismiss")
        return {
            "severity": severity,
            "risk_score": score,
            "rationale": (
                f"{signal['type']} signal with {signal['severity_hint']} source hint; "
                f"{len(supplier_ids)} known supplier(s), {len(ports)} active port(s) and "
                f"{len(countries)} sourcing countr(ies) in our network are implicated."
            ),
            "entities": {"suppliers": supplier_ids, "ports": ports, "countries": countries},
            "decision": decision,
        }
