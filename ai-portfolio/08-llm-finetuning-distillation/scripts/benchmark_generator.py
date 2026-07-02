"""Bounded-memory scaling benchmark for the instruction-data generator.

Streams ``--rows`` examples in chunks, aggregating a label histogram on the fly
without ever holding more than one chunk in memory — the same pattern used to
feed a real fine-tuning run from a data pipeline. Demonstrates that the generator
scales to very large corpora on a laptop.

    python scripts/benchmark_generator.py --rows 1000000
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import csv
import os
import time

import numpy as np

from distillkit import data

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(os.path.dirname(HERE), "benchmarks")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--chunk", type=int, default=100_000)
    args = ap.parse_args()
    os.makedirs(BENCH, exist_ok=True)

    hist = np.zeros(data.N_CLASSES, dtype=np.int64)
    total = 0
    t0 = time.time()
    for texts, y in data.stream(args.rows, seed=0, chunk_size=args.chunk):
        hist += np.bincount(y, minlength=data.N_CLASSES)
        total += len(texts)
    dt = time.time() - t0

    rate = total / dt
    print(f"generated {total:,} examples in {dt:.1f}s  ({rate:,.0f} rows/s)")
    print(f"peak resident chunk: {args.chunk:,} rows (memory bounded, independent of --rows)")
    print("label histogram:", hist.tolist())
    hours_for_1b = 1_000_000_000 / rate / 3600
    print(f"extrapolation: 1e9 rows in ~{hours_for_1b:.1f}h single-process "
          f"(embarrassingly parallel across shards/seeds)")

    with open(os.path.join(BENCH, "generator_throughput.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rows", "seconds", "rows_per_sec", "chunk_size"])
        w.writerow([total, f"{dt:.3f}", f"{rate:.1f}", args.chunk])


if __name__ == "__main__":
    main()
