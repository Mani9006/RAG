# Scaling benchmark

Streaming generation (bounded memory) + extraction inference.

| Docs | Tokens | Gen docs/s | Gen tokens/s | Infer docs/s | Infer tokens/s |
|---|---|---|---|---|---|
| 2,000 | 99,071 | 728 | 36,065 | 1,838 | 90,840 |
| 10,000 | 497,136 | 1,027 | 51,037 | 2,262 | 111,818 |
| 50,000 | 2,482,852 | 774 | 38,452 | 2,108 | 104,170 |
| 100,000 | 4,964,026 | 1,009 | 50,080 | 2,405 | 118,876 |

Peak sustained generation: **51,037 tokens/s** (1,027 docs/s), single process, bounded memory.

Projected wall-clock to stream **1B tokens**: **5.4 h** single-process; near-linear with shards (generation is a pure function of `(seed, doc_id)`).
