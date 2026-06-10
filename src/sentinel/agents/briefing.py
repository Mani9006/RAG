"""Executive Briefing agent — compiles the incident into a situation report
for leadership."""
from __future__ import annotations

import json

from sentinel.agents.base import Agent
from sentinel.tools import ToolBox


class BriefingAgent(Agent):
    name = "executive_briefing"
    tools = []
    system = (
        "You are the Executive Briefing agent for Vertex Devices. Compile the incident "
        "record (signal, triage, impact, mitigation plan, compliance review) into a crisp "
        "situation report for the VP of Supply Chain. Lead with what happened and the "
        "single number that matters most. No hedging, no filler, no restating raw JSON. "
        'Respond with a single JSON object: {"briefing_md": "<markdown report>"} and '
        "nothing else."
    )

    def build_prompt(self, payload: dict) -> str:
        return "Write the situation report for this incident record:\n\n" + json.dumps(
            payload, indent=2, default=str
        )

    def simulate(self, payload: dict, toolbox: ToolBox) -> dict:
        signal, triage = payload["signal"], payload["triage"]
        impact, mitigation = payload["impact"], payload["mitigation"]
        compliance = payload["compliance"]

        lines = [
            f"# Situation Report — {signal['headline']}",
            "",
            f"**Severity:** {triage['severity'].upper()}  |  "
            f"**Risk score:** {triage['risk_score']}  |  "
            f"**Event:** {signal['type']} ({signal['event_id']})",
            "",
            "## What happened",
            signal["body"],
            "",
            "## Exposure",
            f"- Monthly revenue at risk: **${impact['total_monthly_revenue_at_risk_usd']:,.0f}**",
            f"- Open PO exposure: ${impact['total_open_po_value_usd']:,.0f}",
            f"- Suppliers affected: {len(impact['affected_suppliers'])}  |  "
            f"Parts affected: {len(impact['affected_parts'])}  |  "
            f"Shipments at risk: {impact['shipments_at_risk']}",
        ]
        if impact["min_days_of_cover"] is not None:
            lines.append(f"- Tightest inventory position: **{impact['min_days_of_cover']} days of cover**")
        band = impact.get("loss_distribution")
        if band and "p50" in band:
            lines.append(
                f"- Probabilistic loss (Monte Carlo, {band['trials']} trials): "
                f"P50 **${band['p50']:,.0f}** · P90 **${band['p90']:,.0f}** · "
                f"worst case ${band['max']:,.0f}"
            )
        lines += ["", "## Recommended actions"]
        verdict_by_index = {v["option_index"]: v for v in compliance["verdicts"]}
        for i, opt in enumerate(mitigation["options"]):
            verdict = verdict_by_index.get(i, {}).get("verdict", "needs_review")
            marker = {"pass": "✅", "needs_review": "🟡", "fail": "❌"}[verdict]
            rec = " **(recommended)**" if i == mitigation["recommended_index"] else ""
            lines.append(
                f"{i + 1}. {marker} [{opt['kind']}] {opt['description']} — "
                f"${opt['value_usd']:,.0f}, {opt['lead_time_days']}d lead time.{rec}"
            )
        lines += ["", "## Compliance", compliance["summary"]]
        return {"briefing_md": "\n".join(lines)}
