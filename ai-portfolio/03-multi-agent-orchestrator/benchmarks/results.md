# Benchmark Results

Task suite: **31** multi-step tasks. Deterministic, fully offline.

| Condition | Success rate | Solved | Avg steps | Avg attempts | Avg sim. cost | Avg tokens |
|---|---|---|---|---|---|---|
| Without self-correction | 41.9% | 13/31 | 1.35 | 1.00 | $0.16524 | 238 |
| With self-correction | 100.0% | 31/31 | 1.35 | 1.58 | $0.25834 | 381 |

**Self-correction lift: +58.1 percentage points** (41.9% -> 100.0%).

## Tool-call distribution (with self-correction)

| Tool | Calls |
|---|---|
| `calculator` | 20 |
| `unit_convert` | 14 |
| `corpus_search` | 9 |
| `code_check` | 3 |

## Per-task detail

| Task | Category | Without SC | With SC | Attempts (SC) | Steps |
|---|---|---|---|---|---|
| T01 | calc | PASS | PASS | 1 | 1 |
| T02 | calc | PASS | PASS | 1 | 1 |
| T03 | calc | PASS | PASS | 1 | 1 |
| T04 | calc | PASS | PASS | 1 | 1 |
| T05 | calc_pow | FAIL | PASS | 2 | 1 |
| T06 | calc_pow | FAIL | PASS | 2 | 1 |
| T07 | calc_pow | FAIL | PASS | 2 | 1 |
| T08 | convert | PASS | PASS | 1 | 1 |
| T09 | convert | PASS | PASS | 1 | 1 |
| T10 | convert | FAIL | PASS | 2 | 1 |
| T11 | convert | FAIL | PASS | 2 | 1 |
| T12 | convert | FAIL | PASS | 2 | 1 |
| T13 | convert | FAIL | PASS | 2 | 1 |
| T14 | convert_calc | FAIL | PASS | 2 | 2 |
| T15 | convert_calc | FAIL | PASS | 2 | 2 |
| T16 | percent | FAIL | PASS | 2 | 1 |
| T17 | percent | FAIL | PASS | 2 | 1 |
| T18 | percent | FAIL | PASS | 2 | 1 |
| T19 | corpus | PASS | PASS | 1 | 1 |
| T20 | corpus | PASS | PASS | 1 | 1 |
| T21 | corpus | PASS | PASS | 1 | 1 |
| T22 | corpus | PASS | PASS | 1 | 1 |
| T23 | corpus | PASS | PASS | 1 | 1 |
| T24 | corpus_calc | PASS | PASS | 1 | 2 |
| T25 | corpus_calc | PASS | PASS | 1 | 2 |
| T26 | code | FAIL | PASS | 2 | 1 |
| T27 | code | FAIL | PASS | 2 | 1 |
| T28 | code | FAIL | PASS | 2 | 1 |
| T29 | parallel | FAIL | PASS | 2 | 3 |
| T30 | parallel | FAIL | PASS | 2 | 3 |
| T31 | showcase | FAIL | PASS | 2 | 4 |
