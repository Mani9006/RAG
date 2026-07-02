"""Stream synthetic documents to a partitioned Parquet dataset.

Documents are generated on the fly and flushed in chunks, so memory stays
bounded no matter how many are requested — this is the mechanism behind the
"scales to 1B tokens" claim. Each row is one token.

Usage:
    python scripts/generate_data.py --docs 100000 --out data/tokens.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docintel import SEED  # noqa: E402
from docintel.generator import documents_to_table, iter_documents  # noqa: E402

_SCHEMA = pa.schema([
    ("doc_id", pa.int64()), ("doc_type", pa.string()), ("token_idx", pa.int32()),
    ("text", pa.string()), ("x", pa.float32()), ("y", pa.float32()),
    ("w", pa.float32()), ("h", pa.float32()), ("label", pa.string()),
    ("row_group", pa.int32()),
])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=100_000)
    ap.add_argument("--chunk", type=int, default=5_000, help="docs per row group")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=str, default=str(ROOT / "data" / "tokens.parquet"))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_tokens = 0
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(out, _SCHEMA, compression="zstd")
        for start in range(0, args.docs, args.chunk):
            n = min(args.chunk, args.docs - start)
            cols = documents_to_table(iter_documents(n, seed=args.seed, start=start))
            table = pa.table({k: cols[k] for k in _SCHEMA.names}, schema=_SCHEMA)
            writer.write_table(table)
            n_tokens += table.num_rows
            done = start + n
            rate = done / (time.time() - t0)
            print(f"\r  {done:>8,}/{args.docs:,} docs  "
                  f"{n_tokens:>11,} tokens  {rate:,.0f} docs/s", end="", flush=True)
    finally:
        if writer is not None:
            writer.close()

    dt = time.time() - t0
    size_mb = out.stat().st_size / 1e6
    print(f"\nWrote {args.docs:,} docs / {n_tokens:,} tokens to {out} "
          f"({size_mb:.1f} MB) in {dt:.1f}s = {args.docs / dt:,.0f} docs/s, "
          f"{n_tokens / dt:,.0f} tokens/s")


if __name__ == "__main__":
    main()
