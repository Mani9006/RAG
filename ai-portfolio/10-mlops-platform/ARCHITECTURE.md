# Architecture & Design Decisions

## 1. System overview

The platform is a **closed control loop** around a production model. Each arrow is
a real function call in this repo:

```
              ┌─────────────────────────────────────────────────────────┐
              │                    Model Registry (DuckDB)               │
              │   versions · stage machine · lineage · promote/rollback  │
              └───────▲───────────────────────────────────────▲─────────┘
                      │ register/promote                       │ query
   ┌──────────┐   ┌───┴────────┐   ┌───────────┐   ┌───────────┴───────┐
   │  train   │──▶│ orchestrator│◀─│   gate    │◀──│  canary A/B (shadow)│
   └──────────┘   └───▲────────┘   └───────────┘   └───────────────────┘
                      │ observe(alerts)
   ┌──────────────────┴───────────────┐
   │  DriftMonitor (PSI/KL)           │◀── live scoring stream (bounded batches)
   │  PerformanceMonitor (rolling acc)│
   └──────────────────────────────────┘
```

Experiment tracking sits alongside the registry: every train/retrain logs a run
(params, metrics, artifact path, tags) linked to the version it produced.

## 2. Key design decisions & trade-offs

### Registry as a state-machine, not a free-for-all
Stages are `staging → {production, archived}`, `production → archived`,
`archived → {staging, production}`. Illegal transitions raise `InvalidTransition`.
`promote()` is the only way into production and it **atomically archives the
incumbent**, guaranteeing the invariant *exactly one production version*. Every
transition is written to an append-only `transitions` audit table, which is what
makes `rollback()` a simple, correct lookup ("who was production before me?")
rather than bespoke bookkeeping.

**Trade-off:** DuckDB is single-writer. That is ideal for a control-plane registry
(low write rate, analytical reads) but is *not* the data-plane. High-volume
scoring never touches the registry (see §4).

### Two independent drift signals
- **PSI/KL (unsupervised)** reacts to *covariate* shift and needs no labels — it is
  the early-warning system on the raw input stream.
- **Rolling accuracy (supervised)** reacts to *concept* shift — the boundary moving
  — which PSI can miss and which is what actually costs money.

Separating them (the synthetic generator has independent `covariate_drift` and
`concept_drift` knobs) lets the tests assert each detector responds to the right
thing, and lets the orchestrator explain *why* it retrained.

### Trigger policy: sustained, not reactive
A retrain fires only after **`sustained_windows` consecutive** alerting windows, so
a single noisy batch cannot cause a retrain-storm. The counter resets on any clean
window and after firing.

### The gate is a pure function
`PromotionGate.evaluate()` depends only on the candidate's metrics + schema and the
incumbent's metrics. That makes it deterministic, unit-testable in isolation, and
identical whether it runs in the orchestrator or a CI job. Three checks:
1. **schema compatibility** — same feature contract (blocks silent breakage);
2. **quality bar** — absolute `min_auc` floor;
3. **no-regression** — must beat the incumbent by `promotion_margin` **on the same
   holdout** (the settled-regime holdout — the only fair basis).

A challenger that is merely *different* or *marginally better* is held, and the
incumbent stays live. This is exactly the behaviour proven in
`test_gate.py` and `test_orchestrator.py`.

### Canary / shadow before promotion
Before any stage change, both models score the **same** fresh holdout from the
current regime (`orchestrator.canary`). This is the shadow-deployment pattern
compressed into one comparison: no traffic is risked on an unproven model, and the
Δ metric is recorded to `canary.json` for the dashboard.

### Retrain on the current regime
When triggered, the challenger is trained on a fresh sample from the *current*
(drifted) distribution — the stand-in for "the last N logged, labelled production
rows". Evaluating and gating on an independent holdout from that same regime is
what lets v2 legitimately beat a v1 that was fit to a concept that no longer holds.

## 3. Why the numbers look the way they do

The post-drift concept is a **norm-preserving rotation** of the logistic weights
(same magnitudes, permuted + sign-flipped). So:
- v1's predictions become *anti-correlated* with the new labels → AUC **0.06**,
  accuracy **~0.09** (worse than random, as expected when a boundary inverts);
- the new regime is *equally learnable*, so v2 recovers to AUC **0.97** / accuracy
  **0.93**.

This is deliberately a stark, honest demonstration: covariate drift (PSI) and
concept drift (accuracy) both spike, the gate sees a Δ far above margin, and the
served-accuracy KPI visibly self-heals.

## 4. Scaling — to many models and 1B-row scoring

The system has two planes with very different scaling profiles.

### Control plane (registry + tracking) — scales by *count of models*
- DuckDB is embedded and analytical; the registry stores **metadata only**
  (versions, metrics, hashes, stage, lineage), not artifacts. Millions of versions
  fit comfortably and queries (`production_version`, `lineage`, `list_versions`)
  are indexed point/range scans.
- To go multi-team / high-availability, swap the DuckDB file for Postgres behind
  the same `ModelRegistry` interface — the schema and state-machine are unchanged.
  Artifacts live in object storage (S3/GCS); the registry holds the URI.
- The stage state-machine and audit log are O(1) per transition regardless of how
  many models exist.

### Data plane (scoring + drift monitoring) — scales by *volume of traffic*
This is the hot path, and it is built to stream:
- **Bounded memory.** Scoring and PSI run on fixed-size batches
  (`stream_batches`, `bench_scoring.py`); memory is O(batch), independent of total
  rows. Reference bin edges are frozen once, so per-window PSI is a histogram +
  vectorised sum.
- **Measured throughput** (this machine, single core):

  | Rows | Batch | Time | Throughput | Peak memory |
  |---:|---:|---:|---:|---:|
  | 100K | 50K | 0.15 s | 0.68M rows/s | O(batch) |
  | 1M | 50K | 0.79 s | 1.27M rows/s | O(batch) |
  | 10M | 50K | 6.46 s | 1.55M rows/s | O(batch) |

- **Extrapolation to 1B:** at the measured ~1.5M rows/s, 1B rows is **~11 minutes
  single-core**. The loop is embarrassingly parallel — partition the stream by
  key/shard and the wall-clock drops linearly with cores/workers (≈ **40 s across
  16 shards**). Drift statistics are additive over shards: accumulate per-bin
  counts per shard and combine, so PSI/KL over 1B rows needs only the bin-count
  vectors, never the rows.
- **Persisting monitoring at scale:** window aggregates (PSI per feature, rolling
  accuracy) are tiny and append to DuckDB/Parquet; DuckDB's out-of-core engine
  queries months of monitoring history without loading it into RAM. Polars `scan_*`
  (lazy) is the drop-in for feature-store reads feeding retraining.

### What stays constant as it scales
The registry API, the gate policy, the drift math, and the orchestrator loop are
identical from the 60K-row simulation to a 1B-row production stream — only the
*backends* (Postgres, object store, a sharded stream consumer) change. That
separation is the point.

## 5. Testing strategy

`make test` (27 tests) asserts behaviour, not smoke:
- **Registry:** version autoincrement, single-production invariant, illegal
  transitions rejected, valid matrix, rollback restores the prior production
  version, lineage chain, audit log.
- **Drift:** PSI = 0 for identical dists, rises with shift into the >0.25 regime,
  monotone on a shifted stream; performance monitor requires a *sustained* drop and
  never fires on a stable stream.
- **Gate:** passes a clearly-better model; blocks regressions, sub-margin
  improvements, below-bar quality, and schema mismatches; bootstraps with no
  incumbent.
- **Orchestrator (end-to-end):** trigger fires only after sustained windows and
  resets on a clean one; retrain registers with correct lineage & data hash; a
  worse candidate is held while a properly-retrained one is promoted and the old
  version archived; canary picks the better model.
