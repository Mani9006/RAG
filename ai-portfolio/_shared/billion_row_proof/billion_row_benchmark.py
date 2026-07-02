"""Billion-Row Proof — a genuine, reproducible large-scale processing benchmark.

This is the honest backbone of the portfolio's "1B records" claim. It does NOT
materialize a billion rows in memory. Instead it *streams* a synthetic fact
table generated on the fly and runs a realistic analytical workload
(filter -> transform -> group-by aggregation) with **bounded memory**, timing
the whole pass. That is exactly how billion-row jobs run in production
(out-of-core / vectorized execution), and it is measured here, not asserted.

Two independent engines are benchmarked so the result isn't a single-tool fluke:
  1. DuckDB   — vectorized, out-of-core SQL over a generated range.
  2. NumPy    — hand-rolled chunked streaming aggregation (bounded chunk buffer).

Outputs:
  results.csv        — rows, engine, seconds, throughput, peak RSS MB
  assets/throughput.png, assets/scaling.png — publication-grade charts

Run:  python billion_row_benchmark.py --max-rows 1000000000
"""
from __future__ import annotations

import argparse
import csv
import os
import resource
import sys
import time
from pathlib import Path

import duckdb
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # for viztheme
MIX = 2654435761  # Knuth multiplicative hash constant for deterministic pseudo-data


def peak_rss_mb() -> float:
    # ru_maxrss is KB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def duckdb_pass(n: int, threads: int = 4) -> dict:
    """Filter + transform + 8-way group-by over n generated rows, streamed."""
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute("PRAGMA memory_limit='2GB'")  # prove bounded memory
    sql = f"""
        SELECT bucket,
               count(*)      AS n,
               avg(amount)   AS avg_amount,
               sum(amount)   AS gmv,
               max(amount)   AS max_amount
        FROM (
            SELECT (i * {MIX}) % 8                       AS bucket,
                   ((i * {MIX}) % 100000) / 100.0        AS amount
            FROM range({n}) t(i)
        )
        WHERE amount > 5.0          -- filter stage
        GROUP BY bucket
        ORDER BY bucket
    """
    t0 = time.perf_counter()
    rows = con.execute(sql).fetchall()
    dt = time.perf_counter() - t0
    con.close()
    total = sum(r[1] for r in rows)
    return {"seconds": dt, "checksum": total, "groups": len(rows)}


def numpy_streaming_pass(n: int, chunk: int = 5_000_000) -> dict:
    """Chunked streaming aggregation with a bounded (chunk-sized) buffer.

    Peak memory is O(chunk), independent of n — the defining property of a
    real out-of-core pipeline.
    """
    n_buckets = 8
    counts = np.zeros(n_buckets, dtype=np.int64)
    sums = np.zeros(n_buckets, dtype=np.float64)
    t0 = time.perf_counter()
    start = 0
    kept = 0
    while start < n:
        end = min(start + chunk, n)
        i = np.arange(start, end, dtype=np.uint64)
        h = i * np.uint64(MIX)
        bucket = (h % np.uint64(n_buckets)).astype(np.int64)
        amount = (h % np.uint64(100000)).astype(np.float64) / 100.0
        mask = amount > 5.0
        b = bucket[mask]
        a = amount[mask]
        counts += np.bincount(b, minlength=n_buckets)
        sums += np.bincount(b, weights=a, minlength=n_buckets)
        kept += int(mask.sum())
        start = end
    dt = time.perf_counter() - t0
    return {"seconds": dt, "checksum": int(counts.sum()), "groups": n_buckets, "kept": kept}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=1_000_000_000)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    scales = [s for s in (1_000_000, 10_000_000, 100_000_000, 1_000_000_000)
              if s <= args.max_rows]
    rows_out: list[dict] = []

    print(f"{'engine':<10}{'rows':>16}{'seconds':>10}{'M rows/s':>12}{'peakRSS MB':>12}")
    print("-" * 60)
    for n in scales:
        for engine, fn in (("duckdb", lambda n=n: duckdb_pass(n, args.threads)),
                           ("numpy", lambda n=n: numpy_streaming_pass(n))):
            res = fn()
            rss = peak_rss_mb()
            thr = n / res["seconds"] / 1e6
            rows_out.append({
                "engine": engine, "rows": n, "seconds": round(res["seconds"], 3),
                "m_rows_per_s": round(thr, 1), "peak_rss_mb": round(rss, 1),
                "checksum": res["checksum"],
            })
            print(f"{engine:<10}{n:>16,}{res['seconds']:>10.2f}{thr:>12.1f}{rss:>12.1f}")

    out_csv = HERE / "results.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nwrote {out_csv}")
    make_charts(rows_out)


def make_charts(rows: list[dict]) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    from viztheme import apply_theme, ACCENT, GOOD, MUTED, save_panel
    apply_theme()

    engines = sorted({r["engine"] for r in rows})
    colors = {"duckdb": ACCENT, "numpy": GOOD}

    # Throughput at the largest (billion) scale
    top = max(r["rows"] for r in rows)
    top_rows = [r for r in rows if r["rows"] == top]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([r["engine"] for r in top_rows],
                  [r["m_rows_per_s"] for r in top_rows],
                  color=[colors[r["engine"]] for r in top_rows], width=0.55)
    for b, r in zip(bars, top_rows):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{r['m_rows_per_s']:.0f}M/s\n{r['seconds']:.1f}s",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("throughput (million rows / sec)")
    ax.set_title(f"Processing {top:,} rows — bounded memory\n"
                 f"peak RSS ≤ {max(r['peak_rss_mb'] for r in top_rows):.0f} MB",
                 fontsize=11)
    ax.margins(y=0.18)
    save_panel(fig, str(HERE / "assets" / "throughput.png"))

    # Scaling: seconds vs rows (log-log), one line per engine
    fig, ax = plt.subplots(figsize=(7, 4))
    for e in engines:
        pts = sorted([r for r in rows if r["engine"] == e], key=lambda r: r["rows"])
        ax.plot([p["rows"] for p in pts], [p["seconds"] for p in pts],
                "o-", color=colors[e], label=e, linewidth=2, markersize=6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("rows processed"); ax.set_ylabel("wall-clock seconds")
    ax.set_title("Linear scaling to one billion rows")
    ax.legend()
    save_panel(fig, str(HERE / "assets" / "scaling.png"))
    print("wrote charts to assets/")


if __name__ == "__main__":
    main()
