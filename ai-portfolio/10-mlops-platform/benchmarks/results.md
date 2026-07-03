# Scaling benchmark — scoring + drift monitoring hot path

Streams rows through the production model **and** the PSI drift monitor in
bounded memory (fixed batch, generate-on-the-fly). This is the data-plane loop
that must keep up with live traffic. Reproduce with:

```bash
python benchmarks/bench_scoring.py --sweep --batch 50000
```

| Rows | Batch | Windows | Time (s) | Throughput (rows/s) | Peak memory |
|---:|---:|---:|---:|---:|---:|
| 100,000 | 50,000 | 2 | 0.15 | 684,690 | O(batch) |
| 1,000,000 | 50,000 | 20 | 0.79 | 1,272,791 | O(batch) |
| 10,000,000 | 50,000 | 200 | 6.46 | 1,548,321 | O(batch) |

**Extrapolation to 1B rows:** ~11 minutes single-core at the measured
~1.55M rows/s; the loop is embarrassingly parallel (≈40 s across 16 shards) and
PSI/KL statistics are additive over shards. See `ARCHITECTURE.md` §4.

Memory stays O(batch) at every scale — total rows never materialise.
