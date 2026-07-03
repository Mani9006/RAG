# Architecture & Design Decisions

## Goal
A truthful, offline, staff-level demonstration of a **lakehouse + semantic layer
+ text-to-SQL** stack: generate a large star-schema dataset, query it out-of-core
with DuckDB, and answer natural-language business questions as *validated* SQL —
with honest scaling numbers and a clear path to 1B rows.

## Components

### 1. Star-schema lakehouse (`generator.py`, `lakehouse.py`)
- **Model:** `fact_sales` (grain = order line item) + four conformed dimensions
  `dim_customer`, `dim_product`, `dim_store`, `dim_date`. Surrogate integer keys.
- **Storage:** Hive-partitioned Parquet, `fact_sales/year=YYYY/data.parquet`,
  snappy-compressed. DuckDB reads the files directly via `read_parquet(...,
  hive_partitioning=true)` exposed as SQL views — no load/copy step, genuinely
  out-of-core.
- **Memory-bounded generation:** rows are produced in `--chunk-size` batches
  (default 5M). Each chunk is a vectorised NumPy build → Arrow table, split by
  year and appended as a row-group to a long-lived `ParquetWriter` per year.
  Peak memory is bounded by one chunk (~250 MB for 5M rows), independent of
  `--rows`. This is the same code path for 1M and 1B rows.

### 2. Semantic layer (`semantic_model.yaml`, `semantic.py`)
Business concepts are declared once and compiled to SQL:
- **Metrics** → aggregate expressions (`revenue = SUM(net_revenue)`,
  `aov = SUM(net_revenue)/COUNT(DISTINCT order_id)`,
  `active_customers = COUNT(DISTINCT customer_key)`, `gross_margin`, …).
- **Dimensions** → physical columns, each tagged with its owning table.
- A declarative `Query` IR (`metrics`, `dimensions`, `filters`, `order_by`,
  `limit`) compiles to SQL with **joins resolved automatically** from the star
  schema: the required dimension tables are collected from the referenced
  dimensions/filters and joined to the fact on their surrogate keys (deduped).
- Filter literals are SQL-escaped (single-quote doubling) — no string injection.

Keeping metric semantics in one place means "revenue" is defined once and is
identical whether asked by the NL engine, a notebook, or a future LLM.

### 3. NL-to-SQL engine (`schema_linker.py`, `nl2sql.py`)
Deterministic and offline, in two halves behind the `NLToSQLEngine` interface:
- **Schema linking:** longest-phrase-first synonym matching maps NL to metrics
  and dimensions; categorical filter values (`Electronics`, `West`, …) and years
  are recognised from a shared domain vocabulary (`domains.py`) so generator and
  parser can never drift. Group-by is only inferred from explicit grouping
  phrasing (`by/per/each`, grain adverbs, `X by <metric>`, or a superlative
  context), so a filter value never becomes a GROUP BY by accident.
- **Intent/slots:** ordering direction (`highest`/`lowest`…), top-N limit,
  singular-superlative → `LIMIT 1`, default metric when none stated.
- **Drop-in LLM seam:** `LLMNLToSQL` documents exactly where a model plugs in — it
  would emit a `Query` (or raw SQL) and route through the *same* validator and
  execution path. The offline build keeps it inert by design.

### 4. SQL validation (`validator.py`)
Three independent guards, run before any execution:
1. **Shape** — exactly one statement; must start `SELECT`/`WITH`; every forbidden
   verb (`DROP/DELETE/UPDATE/INSERT/ALTER/CREATE/TRUNCATE/COPY/ATTACH/PRAGMA/…`)
   rejected as a whole word, defeating CTE-smuggled mutations like
   `WITH x AS (...) DELETE ...`.
2. **Columns** — qualified `table.column` refs checked against the semantic
   model's physical catalog, so a hallucinated column fails fast.
3. **Live bind** — optional DuckDB `EXPLAIN` binds against the real schema
   without executing.

This is the load-bearing safety story for text-to-SQL: swap the deterministic
generator for an LLM and the guardrail is unchanged.

## Key finding: file count dominates latency, not compression
The first build wrote 360 small Parquet files (partition by `year`×`month`, one
file per chunk per partition). On this environment each file open cost ~0.55 s of
metadata latency, so a full-scan `SUM` over 50M rows took **~15 s** — yet the
identical data in a *single* file scanned in **61 ms**. Fix: one file per year
partition via streaming `ParquetWriter`s (**3 files**). Results:

| | 360 files (year×month) | 3 files (year) |
|---|---:|---:|
| Generation (50M) | 265 s | **46 s** |
| Full-scan SUM (50M, warm) | ~15 000 ms | **196 ms** |
| `COUNT(DISTINCT)` (50M) | ~19 000 ms | **2 307 ms** |

Takeaway: on lakehouse layouts, **row-group and file sizing** is a first-order
performance lever. Partition for pruning, but coalesce to few, large files.

## Scaling to 1B rows — the honest path
**Measured:** 50M rows → 1.42 GB fact Parquet → **28.3 bytes/row**, generated at
1.09M rows/s (bounded memory).

**Extrapolated to 1B rows:**
- Storage: 28.3 B/row × 1e9 ≈ **~28 GB** Parquet (snappy).
- Generation: ~1e9 / 1.09e6 ≈ **~15 min** single-process, bounded memory;
  embarrassingly parallel across year/month shards.
- Query: DuckDB scans out-of-core, so working set is bounded by the aggregation
  state, not the data. Full-scan cost grows ~linearly with bytes read; the
  cold-scan curve (17 ms @1M → 195 ms @50M, sub-linear thanks to parallel
  row-group reads) extrapolates to a few seconds per full scan at 1B, and much
  less for partition-pruned / column-projected queries.

**What would change for a real 1B+ deployment:**
- Partition by `year`×`month` *and* coalesce to few large files per partition
  (target ~128–512 MB/file) to get pruning without the small-file penalty.
- Sort within partitions by common filter columns for row-group min/max pruning.
- Shard generation across processes; DuckDB `PRAGMA threads` + `memory_limit` for
  spill-to-disk on constrained boxes.
- Object storage (S3/GCS) via DuckDB httpfs; the query layer is unchanged.

## Trade-offs & non-goals
- **Deterministic parser, not an LLM.** It solves this schema's question space at
  100% and is instant/free; it is not open-domain. The interface is the point —
  an LLM backend is a drop-in that inherits validation for free.
- **Fact `cost` vs `dim_product.unit_cost`.** The fact carries its own cost basis
  (≈60% of list price) so `gross_margin` is self-consistent on the fact alone;
  the dimension's `unit_cost` is illustrative, not joined for margin.
- **Synthetic data.** No real companies/PII; categorical domains are fixed and
  shared between generator and linker.

## Reproducibility
Single seed (`GenSpec.seed`, default 7) drives all randomness. `MPLBACKEND=Agg`
for headless charts. Every README/benchmark number is emitted by the scripts in
`scripts/` and stored under `benchmarks/`.
