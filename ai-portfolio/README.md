# AI Engineering Portfolio

Ten production-grade projects across the modern AI stack — retrieval, agents, LLM
evaluation, real-time ML, big-data engineering, and MLOps. Every project is
**repo-ready** (self-contained, own tests + benchmarks + generated screenshots),
runs **fully offline with zero paid API keys**, is **seeded and reproducible**, and
**backs every scale claim with a measured number**.

> Built to demonstrate end-to-end capability: system design, from-scratch algorithms,
> large-scale data handling, evaluation rigor, and the operational discipline
> (tests, benchmarks, monitoring, CI-style gates) that separates a demo from a system.

---

## The billion-row backbone

The headline "1 billion records" is **proven, not asserted**. A streaming benchmark
(`_shared/billion_row_proof/`) processes **1,000,000,000 rows** through a realistic
`filter → transform → group-by` workload:

| Engine | Rows | Wall-clock | Throughput | Peak RSS |
|--------|-----:|-----------:|-----------:|---------:|
| DuckDB | 1,000,000,000 | **47 s** | 21 M rows/s | **382 MB** |
| NumPy (streaming) | 1,000,000,000 | **49 s** | 20 M rows/s | **382 MB** |

Two independent engines, **identical checksums at every scale**, memory bounded to
~0.4 GB. Runtime scales linearly. This streaming discipline is reused across the
data-heavy projects, several of which run their own pipelines at 5M–100M rows and
extrapolate to 1B from measured per-row cost.

---

## The projects

| # | Project | One-liner | Tests | Headline result |
|---|---------|-----------|:-----:|-----------------|
| [01](01-enterprise-rag-platform) | **Enterprise RAG Platform** | Hybrid retrieval (BM25 + dense) + reranking + eval harness | 28 | Hybrid+Rerank: Recall@1 **0.92**, nDCG@10 **0.96** |
| [02](02-billion-scale-vector-search) | **Billion-Scale Vector Search** | IVF-PQ ANN engine from scratch (NumPy) | 22 | recall@10 **0.75–0.92** vs exact; **~7× compression**; 1B = ~67 GiB |
| [03](03-multi-agent-orchestrator) | **Multi-Agent Orchestrator** | Planner/executor DAG, tools, self-correction | 69 | Self-correction lifts task success **41.9% → 100%** |
| [04](04-llm-observability-eval) | **LLM Observability & Eval** | Trace analytics + drift + hallucination detection | 21 | **5M traces / 1.79B tokens**; PSI flags injected drift |
| [05](05-realtime-fraud-detection) | **Real-Time Fraud Detection** | Streaming features + sub-ms online serving | 15 | **PR-AUC 0.974** (188× baseline); serving **e2e p99 1.19 ms** |
| [06](06-nl-to-sql-lakehouse) | **NL-to-SQL Lakehouse** | DuckDB star schema + semantic layer + text-to-SQL | 36 | **50M-row** lakehouse; **100%** NL→SQL execution accuracy |
| [07](07-two-tower-recommender) | **Two-Tower Recommender** | Retrieval + ranking, trained from scratch | 33 | beats popularity **+76% recall**; ranker **+7.8% nDCG** |
| [08](08-llm-finetuning-distillation) | **Fine-Tuning & Distillation** | LoRA + knowledge distillation, from scratch | 29 | LoRA **−91% params**; KD recovers **100%** of teacher |
| [09](09-multimodal-doc-intelligence) | **Document Intelligence** | Layout-aware field + table extraction | 18 | field macro-F1 **0.976**; 100k docs, doc-type **100%** |
| [10](10-mlops-platform) | **MLOps Platform** | Registry + drift + auto-retrain + promotion gates | 27 | full drift → retrain → A/B → promote lifecycle |

**298 passing tests across the ten projects**, all offline and seeded. Every
headline number above is reproduced by that project's `make run` / `make bench`.

---

## What's inside every folder

```
NN-project/
├── README.md          # problem, architecture, benchmark table, screenshots, quickstart
├── ARCHITECTURE.md    # design decisions, trade-offs, how it scales to 1B
├── src/               # typed, importable library code — no notebooks-as-code
├── tests/             # pytest suites that assert behavior (not smoke tests)
├── benchmarks/        # scaling results (CSV/MD) from real runs
├── assets/            # generated PNG dashboards/screenshots (committed)
├── scripts/           # generate_data · run · benchmark · make_screenshots
└── Makefile           # make setup | data | run | test | bench | screenshots
```

## Engineering principles

- **Real algorithms, not wrappers.** BM25, IVF-PQ, product quantization, in-batch
  softmax two-towers, LoRA, knowledge distillation, PSI drift — implemented from
  scratch in NumPy/scikit-learn so the understanding is demonstrable.
- **Offline & free.** No API keys, no GPUs, no paid services. Where a real LLM or
  vision model would sit, a deterministic local component implements the same
  interface, so everything runs end-to-end on a laptop — and a production model
  drops in at a documented seam.
- **Honest scale.** Numbers come from runs. Where 1B isn't materialized, the
  extrapolation is derived from measured per-row/per-vector cost and a
  bounded-memory streaming architecture.
- **Evaluation first.** Every project ships metrics and baselines, because a model
  or system you can't measure is a liability.

## Tech

Python 3.11 · NumPy · pandas · scikit-learn · **DuckDB** · **Polars** · PyArrow ·
SciPy · matplotlib · pytest. Chosen to be powerful, free, and portable.

## Run any project

```bash
cd 01-enterprise-rag-platform
make setup && make run && make test && make screenshots
```
