"""Scaling benchmark: streaming generation + extraction throughput.

Two measurements, both on real runs:

1. **Streaming generation** at increasing document counts. Documents are
   generated on the fly and only counted/aggregated (bounded memory), which is
   exactly the mechanism that extrapolates to 1B tokens — we report the
   sustained docs/sec and tokens/sec and the projected wall-clock for 1B tokens.
2. **Inference throughput** (feature extraction + token classification) in
   docs/sec, so the extraction stage's scaling is grounded in a number.

Writes ``benchmarks/results.csv`` and ``benchmarks/results.md``.

Usage:
    python scripts/benchmark.py --scales 2000 10000 50000
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docintel import SEED  # noqa: E402
from docintel.extraction import TokenClassifier  # noqa: E402
from docintel.generator import (  # noqa: E402
    documents_to_table, iter_documents)


def bench_generation(n: int, seed: int) -> tuple[float, int]:
    """Stream ``n`` docs, counting tokens only (bounded memory). Returns (s, tokens)."""
    t0 = time.time()
    n_tokens = 0
    for d in iter_documents(n, seed=seed):
        n_tokens += len(d.tokens)
    return time.time() - t0, n_tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=int, nargs="+",
                    default=[2000, 10000, 50000, 100000])
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    # Train one classifier for the inference-throughput column.
    train = documents_to_table(iter_documents(4000, seed=args.seed))
    clf = TokenClassifier(seed=args.seed).fit(train)

    rows = []
    peak_docs_s = 0.0
    peak_tok_s = 0.0
    for n in args.scales:
        gen_s, n_tokens = bench_generation(n, args.seed + 1)
        gen_docs_s = n / gen_s
        gen_tok_s = n_tokens / gen_s
        # inference throughput on a fixed 2k-doc probe (keeps it quick)
        probe = documents_to_table(iter_documents(2000, seed=args.seed + 2))
        t0 = time.time(); clf.predict(probe); inf_s = time.time() - t0
        inf_docs_s = 2000 / inf_s
        inf_tok_s = len(probe["text"]) / inf_s
        peak_docs_s = max(peak_docs_s, gen_docs_s)
        peak_tok_s = max(peak_tok_s, gen_tok_s)
        rows.append({
            "docs": n, "tokens": n_tokens, "gen_s": round(gen_s, 3),
            "gen_docs_per_s": round(gen_docs_s), "gen_tokens_per_s": round(gen_tok_s),
            "infer_docs_per_s": round(inf_docs_s), "infer_tokens_per_s": round(inf_tok_s),
        })
        print(f"  {n:>8,} docs / {n_tokens:>11,} tokens | gen {gen_docs_s:>7,.0f} docs/s "
              f"{gen_tok_s:>9,.0f} tok/s | infer {inf_docs_s:>6,.0f} docs/s")

    bdir = ROOT / "benchmarks"
    bdir.mkdir(exist_ok=True)
    with (bdir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # 1B extrapolation from the peak sustained generation rate.
    proj_1b_tokens_h = 1e9 / peak_tok_s / 3600
    with (bdir / "results.md").open("w") as f:
        f.write("# Scaling benchmark\n\n")
        f.write("Streaming generation (bounded memory) + extraction inference.\n\n")
        f.write("| Docs | Tokens | Gen docs/s | Gen tokens/s | Infer docs/s | Infer tokens/s |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['docs']:,} | {r['tokens']:,} | {r['gen_docs_per_s']:,} | "
                    f"{r['gen_tokens_per_s']:,} | {r['infer_docs_per_s']:,} | "
                    f"{r['infer_tokens_per_s']:,} |\n")
        f.write(f"\nPeak sustained generation: **{peak_tok_s:,.0f} tokens/s** "
                f"({peak_docs_s:,.0f} docs/s), single process, bounded memory.\n\n")
        f.write(f"Projected wall-clock to stream **1B tokens**: "
                f"**{proj_1b_tokens_h:.1f} h** single-process; near-linear with shards "
                f"(generation is a pure function of `(seed, doc_id)`).\n")
    print(f"\nWrote benchmarks/results.csv + results.md  "
          f"(1B tokens ~= {proj_1b_tokens_h:.1f}h single-process)")


if __name__ == "__main__":
    main()
