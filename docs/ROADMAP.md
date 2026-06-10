# Roadmap

Sentinel SCM ships in phases. Every phase lands as a working, tested,
documented increment — no big-bang releases. Acceptance criteria are
executable (tests/CI), not aspirational.

## ✅ Phase 1 — Core agentic pipeline & governance (shipped)

The foundation: 5-agent pipeline (triage → impact → mitigation → compliance →
briefing), typed tool surface over a SQLite system of record, code-enforced
approval gate, immutable audit log, dual backends (offline simulation / live
Claude with budgets + cost telemetry), FastAPI control plane, CLI, Docker, CI.

**Acceptance (met):** 20 offline tests green on 3 Python versions; `make demo`
runs the full pipeline at $0; dataset regeneration is byte-identical in CI.

## ✅ Phase 2 — Live signal intelligence & supply network science (shipped)

Close the gap between "replay a fixture feed" and "watch the real world", and
upgrade impact analysis from queries to graph analytics.

- **Connectors** for free primary sources — USGS earthquakes (GeoJSON), NOAA/NWS
  severe-weather alerts, GDELT 2.0 news events — normalizing raw events into
  the signal schema, geo-matched against the company's actual supplier/port
  footprint (haversine against `data/geo/locations.csv`), deduped by
  deterministic event IDs. Offline fixtures committed so tests never need
  network.
- **Network science engine** (`sentinel/graph.py`): supplier→part→product graph
  with single-point-of-failure detection, a composite supplier criticality
  index, and deterministic outage propagation simulation (stockout timelines +
  bounded revenue loss, alternate-supplier ramp modeled).
- **Two new agent tools** — `network_risk_profile`, `simulate_supplier_outage` —
  wired into the impact and mitigation agents.
- `sentinel ingest --source usgs|noaa|gdelt|all` CLI + `POST /ingest` API.

**Acceptance (met):** connector tests run from committed fixtures with zero
network; graph metrics are unit-tested against hand-computed values; ingested
live events flow through the *unchanged* pipeline.

## Phase 3 — Probabilistic digital twin & what-if simulation

From point estimates to distributions.

- Monte Carlo simulation over disruption scenarios (duration, severity,
  alternate-ramp uncertainty) → P50/P90 revenue-at-risk bands per incident.
- Scenario library ("Taiwan strait closure", "Suez blockage", "tier-2 chemical
  shortage") runnable on demand: `sentinel simulate --scenario taiwan-strait`.
- Mitigation options re-ranked by expected loss reduction per dollar under
  uncertainty, not deterministic cost.

**Acceptance:** simulations are seeded/reproducible; tests pin distribution
quantiles; briefings show uncertainty bands.

## Phase 4 — Agent evaluation harness & quality gates

The credibility phase: prove the agents are good, continuously.

- Golden dataset of signals with expert-labeled expected outcomes
  (severity, escalation decision, key impact figures).
- Eval runner scoring triage calibration (precision/recall on escalation),
  impact accuracy (figures vs. ground truth), compliance correctness
  (verdict + citation accuracy) — runs in CI against the simulation backend
  and on-demand against live Claude.
- Regression gates: a prompt change that degrades eval scores fails CI.

**Acceptance:** `make eval` produces a scorecard; CI blocks on score drops.

## Phase 5 — Command center UI & notification fabric

Where operators live.

- Web dashboard (served by the existing FastAPI app): signal inbox, incident
  board with blast-radius visualization of the supplier graph, approval queue
  with one-click decisions, audit explorer.
- Notification adapters: webhook-out (Slack-compatible) for critical-severity
  incidents and SLA breaches from the policy's escalation rules.

**Acceptance:** dashboard runs from the same zero-dependency stack; UI smoke
tests in CI.

## Phase 6 — Continuous operations & learning loop

- Scheduler for periodic ingest+run cycles (`sentinel daemon`).
- Outcome feedback: approvers' decisions recorded as labels; periodic eval
  reports on agent–human agreement rates.
- Memory: incident postmortems summarized and exposed to the triage agent as
  precedent ("we saw this supplier fail before").

**Acceptance:** a daemon run survives restarts (state in SQLite); agreement
metrics queryable via the API.

---

### Engineering invariants (all phases)

1. Offline-first: every feature must work and be tested at $0 with no keys.
2. Governance cannot be bypassed by any agent output, ever.
3. Every number traceable to a tool call; every action to an audit row.
4. The repo stays self-contained: data, fixtures, and docs live here.
