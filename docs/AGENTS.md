# Agent Corps

Five specialist agents, each with a narrow charter, a restricted tool surface,
and a strict JSON output contract. In live mode each runs a full Claude
agentic loop (`claude-opus-4-8`, adaptive thinking); in simulation mode each
runs deterministic logic grounded in the *same* tools.

## 1. Risk Triage (`risk_triage`)
- **Input:** one raw disruption signal.
- **Tools:** `find_suppliers`, `shipments_through_port`.
- **Job:** score the threat (0–1), resolve mentioned entities against the
  supplier network, and decide `investigate | monitor | dismiss`. Signals with
  no footprint overlap are capped — the agent is explicitly prompted to be
  calibrated, because escalating noise destroys operator trust.

## 2. Impact Assessment (`impact_assessment`)
- **Input:** escalated signal + triage entities.
- **Tools:** `supplier_exposure`, `parts_blast_radius`, `shipments_through_port`,
  `inventory_position`, `find_suppliers`.
- **Job:** walk supplier → parts → BOM → products and quantify the blast
  radius: open-PO exposure, shipments in harm's way, days of inventory cover,
  monthly revenue at risk. Single-sourced critical parts are prioritized.
  Every number must come from a tool result.

## 3. Mitigation Planner (`mitigation_planner`)
- **Input:** signal + quantified impact.
- **Tools:** `find_alternate_sources`, `estimate_expedite_cost`,
  `inventory_position`, `get_policy`.
- **Job:** produce 2–5 costed options (`expedite_freight`,
  `alternate_source_po`, `inventory_reallocation`, `supplier_quarantine`,
  `monitor_only`) ranked by risk reduction per dollar, planned *within* policy
  (it reads the policy first).

## 4. Compliance Guardrail (`compliance_guardrail`)
- **Input:** the mitigation options.
- **Tools:** `get_policy`, `find_suppliers`, `find_alternate_sources`,
  `inventory_position`.
- **Job:** per-option verdict — `pass | fail | needs_review` — with rule
  citations. Checks at minimum: restricted entities, spend authority,
  the 21-day expedite window (SR-003), ISO9001 for critical-part alternates
  (SR-004), insolvency restrictions (SR-002). The prompt forbids softening
  findings.

## 5. Executive Briefing (`executive_briefing`)
- **Input:** the complete incident record.
- **Tools:** none.
- **Job:** a crisp markdown situation report — what happened, the one number
  that matters, the action list with compliance markers.

## Separation of duties

The pipeline deliberately mirrors enterprise control structure:

- The planner cannot judge its own compliance.
- The compliance agent cannot edit options — only verdict them.
- Neither can execute: the **orchestrator's approval gate** (code, not LLM)
  decides `blocked / pending_approval / auto_approved` from the persisted
  verdict and configured spend limits.
- Every stage emits audit events; `sentinel/api.py:/audit` exposes the trail.

## Adding an agent

1. Subclass `Agent` in `sentinel/agents/`, define `name`, `tools`, `system`.
2. Implement `build_prompt()` and a deterministic `simulate()` that honors the
   same output schema.
3. Wire it into `Pipeline` and add audit events.
4. Add behavioral tests against the simulation (see `tests/test_agents.py`).
