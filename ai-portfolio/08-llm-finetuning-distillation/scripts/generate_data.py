"""Materialize the synthetic instruction dataset to disk (parquet).

    python scripts/generate_data.py --rows 20000 --domain source

Writes train/val/test splits to ``data/<domain>_{train,val,test}.parquet``.
The generator streams in chunks, so ``--rows`` can be arbitrarily large without
OOM (see ``benchmark_generator.py`` for a bounded-memory scale demonstration).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import os

import pandas as pd

from distillkit import data

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--domain", choices=["source", "target"], default="source")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)

    ds = data.generate(args.rows, seed=args.seed, domain=args.domain)
    tr, va, te = data.split(ds, seed=0)
    for name, split in [("train", tr), ("val", va), ("test", te)]:
        df = pd.DataFrame({"text": split.texts, "label": split.y,
                           "label_name": [data.LABELS[i] for i in split.y]})
        path = os.path.join(DATA, f"{args.domain}_{name}.parquet")
        df.to_parquet(path, index=False)
        print(f"wrote {len(df):>7,} rows -> {os.path.relpath(path, os.path.dirname(HERE))}")


if __name__ == "__main__":
    main()
