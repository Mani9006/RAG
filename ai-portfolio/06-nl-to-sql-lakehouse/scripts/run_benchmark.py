#!/usr/bin/env python
"""Query-performance benchmark across data scales.

For each scale it builds (or reuses) a partitioned-Parquet lakehouse, then times
a fixed set of representative analytical queries with DuckDB scanning the files
out-of-core. Reports cold vs warm latency and parquet size per scale, and writes
a CSV consumed by the latency-vs-scale screenshot.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlsql import connect  # noqa: E402
from nlsql.generator import GenSpec, generate  # noqa: E402

# Representative analytical workload (mix of full-scan, join, filter, group, topn)
QUERIES = {
    "full_scan_sum": "SELECT SUM(net_revenue) FROM fact_sales",
    "groupby_region":
        "SELECT s.region, SUM(f.net_revenue) FROM fact_sales f "
        "JOIN dim_store s ON f.store_key=s.store_key GROUP BY s.region",
    "filter_year_month":
        "SELECT d.month, SUM(f.net_revenue) FROM fact_sales f "
        "JOIN dim_date d ON f.date_key=d.date_key WHERE d.year=2023 GROUP BY d.month",
    "topn_brand":
        "SELECT p.brand, SUM(f.net_revenue) m FROM fact_sales f "
        "JOIN dim_product p ON f.product_key=p.product_key "
        "GROUP BY p.brand ORDER BY m DESC LIMIT 5",
    "distinct_customers": "SELECT COUNT(DISTINCT customer_key) FROM fact_sales",
}


def time_query(con, sql: str, repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        con.execute(sql).fetchall()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def bench_scale(rows: int, data_dir: Path, main_dir: Path, threads: int) -> dict:
    reused = False
    if main_dir.exists() and _row_count(main_dir) == rows:
        target = main_dir
        reused = True
    else:
        spec = GenSpec(rows=rows, customers=min(2_000_000, max(rows // 25, 1000)),
                       products=50_000, stores=1_000, chunk_size=5_000_000)
        generate(spec, data_dir)
        target = data_dir

    fact_bytes = sum(p.stat().st_size for p in (target / "fact_sales").rglob("*.parquet"))

    con = connect(target, threads=threads or None)
    # cold: first touch of the full-scan query
    t0 = time.perf_counter()
    con.execute(QUERIES["full_scan_sum"]).fetchall()
    cold_ms = (time.perf_counter() - t0) * 1000

    warm = {name: time_query(con, sql) for name, sql in QUERIES.items()}
    con.close()

    return {
        "rows": rows,
        "parquet_gb": round(fact_bytes / 1e9, 4),
        "cold_ms": round(cold_ms, 2),
        "warm_median_ms": round(statistics.median(warm.values()), 2),
        **{f"q_{k}_ms": round(v, 2) for k, v in warm.items()},
        "reused_main": reused,
    }


def _row_count(data_dir: Path) -> int:
    try:
        con = connect(data_dir)
        n = con.execute("SELECT COUNT(*) FROM fact_sales").fetchone()[0]
        con.close()
        return int(n)
    except Exception:
        return -1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scales", type=int, nargs="+",
                    default=[1_000_000, 5_000_000, 20_000_000, 50_000_000])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--main", type=str, default=str(ROOT / "data" / "lakehouse"))
    ap.add_argument("--keep", action="store_true", help="keep temp bench datasets")
    args = ap.parse_args()

    main_dir = Path(args.main)
    results = []
    for rows in args.scales:
        bench_dir = ROOT / "data" / f"_bench_{rows}"
        print(f"benchmarking {rows:,} rows ...")
        r = bench_scale(rows, bench_dir, main_dir, args.threads)
        print("  ", json.dumps(r))
        results.append(r)
        if not args.keep and bench_dir.exists() and not r["reused_main"]:
            import shutil
            shutil.rmtree(bench_dir)

    fields = sorted({k for r in results for k in r})
    fields = ["rows", "parquet_gb", "cold_ms", "warm_median_ms"] + \
             [f for f in fields if f not in
              {"rows", "parquet_gb", "cold_ms", "warm_median_ms"}]
    out = ROOT / "benchmarks" / "scaling_results.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
