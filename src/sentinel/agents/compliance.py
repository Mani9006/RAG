"""Compliance Guardrail agent — validates every proposed action against the
procurement governance policy before it can reach an approver."""
from __future__ import annotations

import json

from sentinel.agents.base import JSON_ONLY, Agent
from sentinel.tools import ToolBox


class ComplianceAgent(Agent):
    name = "compliance_guardrail"
    tools = ["get_policy", "find_suppliers", "find_alternate_sources", "inventory_position"]
    system = (
        "You are the Compliance Guardrail agent for Vertex Devices. You are the last "
        "automated control before money moves. For EACH proposed mitigation option, read "
        "the procurement policy and return a verdict: 'pass' (compliant), 'fail' (violates "
        "a rule — cite it), or 'needs_review' (compliant but above auto-approval authority "
        "or judgement required). You must check at minimum: restricted entities, spend "
        "authority limits, the expedite-freight stockout window rule, ISO9001 requirements "
        "for alternates on critical parts, and insolvency restrictions. Cite rule IDs. "
        "You do not soften findings: a violation is a fail even if the action is operationally "
        "attractive. "
        + JSON_ONLY
        + ' Schema: {"verdicts": [{"option_index": int, "verdict": "pass|fail|needs_review", '
        '"cited_rules": [str], "notes": str}], "summary": str}'
    )

    def build_prompt(self, payload: dict) -> str:
        return (
            "Review these mitigation options for policy compliance.\n\nOptions:\n"
            + json.dumps(payload["mitigation"]["options"], indent=2)
            + "\n\nDisruption context:\n"
            + json.dumps(payload["signal"], indent=2)
        )

    def simulate(self, payload: dict, toolbox: ToolBox) -> dict:
        policy = self.tool_json(toolbox, "get_policy")
        restricted = {e["supplier_name"] for e in policy.get("restricted_entities", [])}
        auto_limit = policy["spend_authority"]["auto_approve_limit_usd"]
        insolvent_supplier_named = payload["signal"]["type"] == "supplier_insolvency"

        suppliers_by_id = {
            s["supplier_id"]: s for s in self.tool_json(toolbox, "find_suppliers", query="")
        }

        verdicts = []
        for i, opt in enumerate(payload["mitigation"]["options"]):
            cited, notes, verdict = [], [], "pass"
            supplier = suppliers_by_id.get(opt.get("supplier_id") or "")

            if supplier and supplier["name"] in restricted:
                verdict = "fail"
                cited.append("restricted_entities")
                notes.append(f"{supplier['name']} is on the restricted entities list.")

            if opt["kind"] == "alternate_source_po" and supplier and opt.get("part_id"):
                src = self.tool_json(toolbox, "find_alternate_sources", part_id=opt["part_id"])
                if src["part"]["critical"] == "yes" and supplier["iso9001"] != "yes":
                    verdict = "fail"
                    cited.append("SR-004")
                    notes.append("Alternate supplier lacks ISO9001 for a critical part.")

            if opt["kind"] == "expedite_freight" and opt.get("part_id"):
                pos = self.tool_json(toolbox, "inventory_position", part_id=opt["part_id"])
                cover = pos.get("days_of_cover")
                if cover is not None and cover >= 21:
                    verdict = "fail"
                    cited.append("SR-003")
                    notes.append(f"Air freight not permitted: {cover} days of cover (>21).")

            if insolvent_supplier_named and opt["kind"] == "alternate_source_po":
                # New spend triggered by an insolvency event still needs scrutiny of the
                # counterparty; flag for review rather than auto-pass.
                if verdict == "pass":
                    verdict = "needs_review"
                    cited.append("SR-002")
                    notes.append("Insolvency-driven re-sourcing; confirm counterparty solvency.")

            if verdict == "pass" and opt["value_usd"] > auto_limit:
                verdict = "needs_review"
                cited.append("spend_authority.auto_approve_limit_usd")
                notes.append(
                    f"Value ${opt['value_usd']:,.0f} exceeds auto-approval limit ${auto_limit:,.0f}."
                )

            verdicts.append({
                "option_index": i,
                "verdict": verdict,
                "cited_rules": cited,
                "notes": " ".join(notes) or "Compliant within current policy.",
            })

        n_fail = sum(1 for v in verdicts if v["verdict"] == "fail")
        n_review = sum(1 for v in verdicts if v["verdict"] == "needs_review")
        return {
            "verdicts": verdicts,
            "summary": (
                f"{len(verdicts)} option(s) reviewed: {n_fail} failed, {n_review} need human "
                f"review, {len(verdicts) - n_fail - n_review} passed."
            ),
        }
