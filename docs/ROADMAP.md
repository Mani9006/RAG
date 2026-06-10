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

## ✅ Phase 3 — Probabilistic digital twin & what-if simulation (shipped)

From point estimates to distributions.

- Monte Carlo engine (`sentinel/montecarlo.py`) sampling outage duration,
  alternate-ramp time, and demand (triangular distributions) over the
  multi-supplier network propagation model → expected / P50 / P90 / P95 / max
  loss plus exceedance probabilities.
- Committed scenario library (`data/scenarios/*.yaml`): `taiwan-strait`,
  `korea-battery-fire`, `europe-port-strike`, `mexico-border-closure`.
  Run via `sentinel simulate --scenario taiwan-strait`, `GET /scenarios`,
  `POST /simulate`, or the `run_disruption_scenario` agent tool.
- Incident briefings carry a Monte Carlo loss band; mitigation options are
  re-ranked by **expected loss reduction per dollar** using *paired trials*
  (identical random draws with/without each option, isolating its effect).
- Multi-supplier outages model correlated failure: an alternate supplier
  inside the blast radius provides no relief.

**Acceptance (met):** simulations are seeded and byte-reproducible; tests pin
quantile ordering, stochastic dominance, and paired-ranking properties;
briefings show uncertainty bands.

## ✅ Phase 4 — Agent evaluation harness & quality gates (shipped)

The credibility phase: prove the agents are good, continuously.

- Golden datasets committed under `data/evals/`: 10 labeled triage cases
  (escalation ground truth, severity bands, score calibration, entity
  resolution) and 6 labeled compliance cases (verdicts + required rule
  citations) using `@selector` placeholders resolved against the live dataset
  so cases survive dataset regeneration.
- Eval runner (`sentinel/evals.py`) scoring escalation precision/recall
  (recall floor = 1.0 — a missed disaster is the unforgivable failure),
  expectation pass rate, compliance verdict accuracy (floor = 1.0 — the money
  gate must be exact) and citation accuracy. Runs against the simulation
  backend in CI and against live Claude with one env var.
- Hard regression gates: `sentinel eval` (and the CI job) exits non-zero when
  any metric drops below its floor.

**Acceptance (met):** `make eval` produces a scorecard; CI blocks on score
drops; a deliberately degraded agent fails the gates in the test suite.

## ✅ Phase 5 — Command center UI & notification fabric (shipped)

Where operators live.

- Web dashboard at `GET /` (single self-contained HTML file served by the
  existing FastAPI app — no build step, no CDNs, works fully offline):
  overview KPIs with SPOF/criticality visualizations, signal inbox, incident
  board with drill-down briefings, one-click approve/reject queue, Monte
  Carlo war-game runner, audit explorer, and a run-pipeline button.
- Notification fabric (`sentinel/notify.py`): Slack-compatible webhook-out
  for incidents at/above `SENTINEL_NOTIFY_MIN_SEVERITY`, carrying severity,
  revenue at risk, and pending-approval count. Best-effort by design —
  delivery failures are logged and audited but never block the pipeline.

**Acceptance (met):** dashboard runs from the same zero-dependency stack; UI
smoke tests assert it is served, self-contained, and wired to the live API;
notifier tests cover the severity floor, payload shape, failure isolation,
and the audit trail.

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
