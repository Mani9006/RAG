"""Generate synthetic training/reference datasets to ``data/`` (Parquet).

Parameterised by ``--rows`` and streamed in chunks so it scales to very large
sizes without OOM. Two regimes are emitted: the baseline (pre-drift) distribution
used to train v1, and the settled (post-drift) regime used to retrain v2.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops.config import CONFIG, FEATURES, seed_everything
from mlops.data import make_data


def _write_regime(path: Path, rows: int, chunk: int, seed: int, cov: float, con: float) -> None:
    writer = None
    remaining, s = rows, seed
    while remaining > 0:
        n = min(chunk, remaining)
        X, y = make_data(n, seed=s, covariate_drift=cov, concept_drift=con)
        s += 1
        remaining -= n
        table = pa.table(
            {**{f: X[:, i] for i, f in enumerate(FEATURES)}, "label": y}
        )
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema)
        writer.write_table(table)
    if writer is not None:
        writer.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50_000)
    ap.add_argument("--chunk", type=int, default=100_000)
    args = ap.parse_args()

    seed_everything(CONFIG.seed)
    CONFIG.ensure_dirs()
    base = CONFIG.data_dir / "baseline.parquet"
    drifted = CONFIG.data_dir / "drifted.parquet"
    _write_regime(base, args.rows, args.chunk, seed=CONFIG.seed, cov=0.0, con=0.0)
    _write_regime(drifted, args.rows, args.chunk, seed=CONFIG.seed + 777, cov=1.5, con=1.0)
    print(f"Wrote {args.rows:,} rows/regime -> {base.name}, {drifted.name}")


if __name__ == "__main__":
    main()
