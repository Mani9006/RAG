# Portfolio Engineering Standards

Every project in this portfolio is **repo-ready** (can be split into its own GitHub repo) and
follows the same senior-engineer conventions. Consistency is part of what makes the portfolio
read as professional.

## Required folder layout (per project)

```
NN-project-name/
├── README.md              # Hero doc: problem, architecture, benchmarks, screenshots, how-to-run
├── ARCHITECTURE.md        # Design decisions, trade-offs, scaling notes to 1B
├── requirements.txt       # Pinned-ish deps (only what's needed)
├── Makefile               # make setup | make data | make run | make test | make bench | make screenshots
├── src/<pkg>/             # Library code, importable, typed, docstrings
├── tests/                 # pytest suite, real assertions (not smoke-only)
├── benchmarks/            # Scaling benchmarks + results tables (CSV/MD)
├── data/                  # .gitkeep; generated data lives here (gitignored)
├── assets/                # Generated PNG "screenshots" (charts/dashboards) — committed
├── scripts/               # generate_data.py, run_*.py, make_screenshots.py
└── .gitignore
```

## Non-negotiables

1. **Real, runnable code.** No pseudo-code, no `...`, no TODO stubs in core logic. If a heavy
   external service (real OpenAI, real GPU) isn't available, provide a clean local/offline
   implementation (deterministic mock embeddings, local models) behind an interface so the code
   runs end-to-end with zero paid API keys.
2. **Scale honestly.** Data generators must be parameterized by `--rows` and stream/chunk so they
   can produce up to 1B rows without OOM. Actually run at a feasible large scale (>=1M, ideally
   10M–100M) and record real numbers. Document the extrapolation to 1B with the architecture that
   supports it (DuckDB out-of-core, Polars lazy, chunked NumPy, sharding).
3. **Tests must assert behavior.** Cover core algorithms, edge cases, and a small end-to-end path.
   `make test` must pass.
4. **Screenshots are generated, not faked.** `make screenshots` produces PNGs into `assets/` from
   real runs (matplotlib). Dashboards can be matplotlib multi-panel figures styled to look like a
   product UI. Charts must reflect actual data the pipeline produced.
5. **README has a screenshot section** embedding the PNGs, a benchmark table with real numbers, and
   a one-command quickstart.
6. **No secrets, no real company names, no proprietary data.** Everything synthetic.

## Style
- Python 3.11, type hints, `from __future__ import annotations`.
- Standard lib + the shared stack: numpy, pandas, matplotlib, scikit-learn, duckdb, polars, pyarrow, scipy, pytest.
- Deterministic: seed everything. Reproducibility is a selling point.
- Charts: clean, professional palette. Dark or light but consistent within a project. Title, axis
  labels, units, legend. No default matplotlib ugliness — set figure DPI 130+, tight_layout.

## The "impressive but truthful" rule
Claims in READMEs must be backed by a number the code actually produced. "Processed 1B rows" is
only allowed if a benchmark script demonstrably streams 1B rows (generate-on-the-fly + aggregate,
bounded memory). Otherwise say "1M rows measured, architected for 1B (see ARCHITECTURE.md)".
