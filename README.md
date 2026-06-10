# Sentinel SCM

**Autonomous, multi-agent supply chain disruption response — an agentic pipeline, not RAG.**

Supply chain disruption is one of the most expensive unsolved problems in the
enterprise: signals (typhoons, factory fires, insolvencies, export controls,
ransomware) arrive faster than human teams can triage them, and the cost of a
missed single-sourced part is measured in millions per month. Sentinel SCM is
a command center where a corps of specialized Claude agents does that work
end-to-end — detect → quantify → plan → check → act — under hard, code-enforced
governance.

```
signal inbox ──▶ Risk Triage ──▶ Impact Assessment ──▶ Mitigation Planner
                    │                (BOM / supplier        │
              dismiss/monitor         graph walk)           ▼
                                                     Compliance Guardrail
                                                            │
                                            ┌───────────────┴───────────────┐
                                            ▼                               ▼
                                     Approval Gate                 Executive Briefing
                                  (human-in-the-loop)              (situation report)
```

## Why this is different

- **Agentic pipeline, not retrieval.** Agents *act* on an operational system of
  record through a typed tool surface (supplier exposure, BOM blast radius,
  inventory positions, freight cost models) — every number in every decision is
  grounded in a tool result, never estimated.
- **Governance is code, not prompts.** Agents propose; the orchestrator
  disposes. Compliance failures are blocked, spend above the auto-approval
  limit waits for a named human approver, and every stage writes to an
  immutable audit log. A hallucinating agent cannot move money.
- **Zero-cost by default.** A deterministic simulation backend implements every
  agent's behavior against the same tools and JSON contracts, so the demo, the
  full test suite, and CI run **offline with no API key and $0 spend**. Flip one
  env var to run the same pipeline on live Claude agents
  (`claude-opus-4-8`, adaptive thinking) with run-level token budgets and cost
  telemetry.
- **Everything in the repo.** A seeded synthetic enterprise dataset (40
  geo-located suppliers, 120 parts, BOM for 12 products, 220 shipments, 160
  POs, a governance policy, and a 12-event disruption feed) bootstraps into
  SQLite on demand. No cloud services, no credentials, clean seams for
  swapping in real warehouses and event feeds.
- **Watches the real world for free.** Live connectors for the same primary
  sources commercial risk platforms resell at six figures — USGS earthquakes,
  NOAA severe-weather alerts, GDELT global news — normalize raw events,
  **geo-match them against your actual supplier/port footprint** (haversine
  against the network), and dedupe into the same signal inbox. The pipeline
  downstream is identical for live and replayed events.
- **Network science, not just lists.** A supplier→part→product graph engine
  detects single points of failure, ranks suppliers by a composite criticality
  index, and simulates outage propagation (stockout timelines + revenue loss
  bounded per product, alternate-supplier relief modeled) — exposed to the
  agents as tools and to operators via `sentinel network`.
- **A probabilistic digital twin.** A seeded Monte Carlo engine turns point
  estimates into distributions: war-game scenarios like `taiwan-strait`
  (`sentinel simulate --scenario taiwan-strait`) return expected/P50/P90/P95
  loss and exceedance probabilities; incident briefings carry uncertainty
  bands; mitigation options are re-ranked by **expected loss reduction per
  dollar** using paired trials.
- **Agents with proof of quality.** Golden datasets + an eval harness score
  escalation precision/recall, severity calibration, and compliance verdict
  accuracy on every CI run, with hard regression gates — a prompt change that
  makes the agents worse fails the build (`make eval`).
- **Runs continuously and learns.** `sentinel daemon` loops ingest → pipeline
  with crash-safe SQLite state; approver decisions become agreement metrics
  (the earliest drift signal); and incident history feeds back into triage as
  institutional memory — repeat offenders score higher.

## Quickstart (60 seconds, no API key)

```bash
pip install -e ".[dev]"
make demo          # full pipeline run: 12 signals → incidents → approval queue
make test          # 20 tests, all offline
```

Inspect the results:

```bash
sentinel incidents                      # opened incidents with severity + risk score
sentinel brief INC-XXXXXXXX             # executive situation report (markdown)
sentinel approvals                      # actions waiting for a human
sentinel decide ACT-XXXXXXXX --approve --approver alice@vertex.example
sentinel network                        # SPOFs + supplier criticality index
sentinel ingest --source usgs           # pull real-world signals (USGS/NOAA/GDELT)
sentinel scenarios                      # list the what-if scenario library
sentinel simulate --scenario taiwan-strait   # Monte Carlo loss distribution
sentinel eval                           # agent quality scorecard + regression gates
sentinel daemon --interval 900          # continuous ingest + pipeline loop
sentinel agreement                      # human-vs-agent agreement metrics
```

Or run the **command center** (dashboard + API):

```bash
sentinel serve            # dashboard → http://localhost:8000  ·  API docs → /docs
# docker alternative:
docker compose up --build
```

The dashboard is a single self-contained HTML file (no build step, no CDNs):
overview KPIs with SPOF/criticality charts, signal inbox, incident board with
drill-down briefings, one-click approvals, a Monte Carlo war-game runner, and
the audit explorer. Set `SENTINEL_WEBHOOK_URL` to push high/critical incidents
to any Slack-compatible webhook.

## Live mode (real Claude agents)

```bash
cp .env.example .env
# set SENTINEL_LLM_BACKEND=anthropic and ANTHROPIC_API_KEY=sk-ant-...
sentinel reset && sentinel run --limit 1   # calibrate cost on one signal first
```

Live runs are protected by a run-level token budget
(`SENTINEL_RUN_TOKEN_BUDGET`), per-agent iteration caps, and per-run usage +
estimated-cost telemetry in every run summary.

## The agent corps

| Agent | Charter | Tools |
|---|---|---|
| **Risk Triage** | score each raw signal, resolve entities, escalate or dismiss | supplier search, port shipments |
| **Impact Assessment** | walk supplier → parts → BOM → products; quantify revenue at risk, days of cover, PO exposure | exposure, blast radius, inventory |
| **Mitigation Planner** | costed options: expedite, alternate-source PO, reallocation | alternates, freight cost model, policy |
| **Compliance Guardrail** | per-option verdict with rule citations (restricted entities, spend authority, SR-001…SR-004) | policy, supplier master |
| **Executive Briefing** | crisp situation report for leadership | — |

Full details: [docs/AGENTS.md](docs/AGENTS.md) ·
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ·
[docs/RESEARCH.md](docs/RESEARCH.md) ·
[docs/ROADMAP.md](docs/ROADMAP.md) ·
[docs/RUNBOOK.md](docs/RUNBOOK.md) ·
[docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md)

## Example output

From the seeded M6.9 Hsinchu earthquake event, fully offline:

> **Severity: CRITICAL | Risk score: 0.97**
> - Monthly revenue at risk: **$14,506,000** · Open PO exposure: $20,023,834
> - Tightest inventory position: **12.0 days of cover**
> 1. 🟡 [expedite_freight] Air-freight 56,910 units of PART-0019 — $35,034, 5d lead time **(recommended)**
> 2. 🟡 [alternate_source_po] Bridge PO with Taoyuan Sensor Labs — $194,632, 14d
> …
> Compliance: 5 option(s) reviewed: 0 failed, 5 need human review, 0 passed.

## Repository layout

```
data/                 committed enterprise dataset + governance policy
scripts/              deterministic dataset generator
src/sentinel/
  agents/             the five specialist agents (live + simulation behavior)
  llm.py              Anthropic agentic loop · budgets · simulation backend
  tools.py            typed tool registry executed client-side
  orchestrator.py     pipeline, approval gate, audit trail
  api.py / cli.py     control plane (FastAPI) and operator CLI
  db.py / config.py / telemetry.py
tests/                20 offline tests (tools, agents, pipeline, API)
docs/                 architecture, agents, runbook, data dictionary
```

## License

MIT — see [LICENSE](LICENSE).
