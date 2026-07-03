"""Scaling benchmark for the monitoring / scoring hot path.

Streams up to N rows through the production model + PSI drift monitor in bounded
memory (fixed batch size), measuring sustained scoring+monitoring throughput.
This is the loop that must keep up with high-volume live traffic, so it is the
component whose scaling matters most.

Usage:
    python benchmarks/bench_scoring.py --rows 10_000_000 --batch 50_000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops.config import CONFIG, FEATURES, seed_everything
from mlops.data import make_data
from mlops.drift import DriftMonitor
from mlops.model import train_model


def run(rows: int, batch: int) -> dict:
    seed_everything(CONFIG.seed)
    # Train a small production model + freeze a drift reference once.
    Xtr, ytr = make_data(20_000, seed=CONFIG.seed)
    model = train_model(Xtr, ytr, seed=CONFIG.seed)
    ref_X, _ = make_data(20_000, seed=CONFIG.seed + 1)
    mon = DriftMonitor(ref_X, FEATURES, bins=CONFIG.psi_bins, psi_threshold=CONFIG.psi_alert)

    scored = 0
    windows = 0
    max_psi = 0.0
    t0 = time.perf_counter()
    remaining = rows
    seed = 10_000
    # Generate-on-the-fly and aggregate: memory is O(batch), never O(rows).
    while remaining > 0:
        n = min(batch, remaining)
        X, _y = make_data(n, seed=seed, covariate_drift=1.2 * (windows % 7) / 7.0)
        seed += 1
        _proba = model.predict_proba(X)[:, 1]        # real scoring
        per_feat = mon.psi_per_feature(X)            # real drift monitoring
        max_psi = max(max_psi, max(per_feat.values()))
        scored += n
        remaining -= n
        windows += 1
    dt = time.perf_counter() - t0
    return {
        "rows": rows,
        "batch": batch,
        "windows": windows,
        "seconds": round(dt, 3),
        "rows_per_sec": int(scored / dt),
        "max_psi_seen": round(max_psi, 4),
        "peak_batch_rows": batch,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--batch", type=int, default=50_000)
    ap.add_argument("--sweep", action="store_true", help="run a small scaling sweep")
    args = ap.parse_args()

    results = []
    if args.sweep:
        for r in (100_000, 1_000_000, 10_000_000):
            res = run(r, args.batch)
            print(res)
            results.append(res)
    else:
        res = run(args.rows, args.batch)
        print(res)
        results.append(res)

    out = Path(__file__).resolve().parent / "results.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
