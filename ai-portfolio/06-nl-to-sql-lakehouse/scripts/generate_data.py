#!/usr/bin/env python
"""Generate the star-schema lakehouse to partitioned Parquet.

    python scripts/generate_data.py --rows 50000000 --out data/lakehouse

Memory-bounded: rows are produced in --chunk-size batches, so --rows can be set
to 1_000_000_000 without changing peak RAM (only wall-clock and disk grow).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlsql.generator import GenSpec, generate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=50_000_000)
    ap.add_argument("--customers", type=int, default=2_000_000)
    ap.add_argument("--products", type=int, default=50_000)
    ap.add_argument("--stores", type=int, default=1_000)
    ap.add_argument("--chunk-size", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default=str(ROOT / "data" / "lakehouse"))
    args = ap.parse_args()

    spec = GenSpec(rows=args.rows, customers=args.customers, products=args.products,
                   stores=args.stores, chunk_size=args.chunk_size, seed=args.seed)
    print(f"Generating {args.rows:,} fact rows -> {args.out}")
    rep = generate(spec, args.out)

    fact_gb = rep.fact_bytes / 1e9
    dim_gb = rep.dim_bytes / 1e9
    extrap_1b_gb = rep.bytes_per_row * 1_000_000_000 / 1e9
    summary = {
        "rows": rep.rows,
        "fact_gb": round(fact_gb, 3),
        "dim_gb": round(dim_gb, 4),
        "total_gb": round(fact_gb + dim_gb, 3),
        "num_partitions": rep.num_partitions,
        "num_files": rep.num_files,
        "bytes_per_row": round(rep.bytes_per_row, 3),
        "gen_seconds": round(rep.seconds, 2),
        "gen_rows_per_sec": int(rep.rows / rep.seconds),
        "extrapolated_1B_parquet_gb": round(extrap_1b_gb, 1),
    }
    out = ROOT / "benchmarks" / "generation_report.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
