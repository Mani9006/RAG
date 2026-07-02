# Architecture & Design Decisions

## 1. Problem

Enterprise document AI (accounts-payable, KYC, claims) has to turn a page image
into **structured, typed fields**: which tokens are the invoice number, the
date, the vendor, the totals, and the line-item table — then route the document
and make its content searchable. The hard part is that the signal is **spatial**:
`"$1,240.00"` is a `TOTAL` in the bottom-right totals block and an `LI_AMOUNT`
inside the table. A layout-blind text model cannot tell them apart.

## 2. Why synthetic *rendered layouts*

There is no offline OCR/vision stack here, so we generate documents as
**structured layouts**: every document is a list of word-level tokens, each with
`(text, x, y, w, h)` and a gold label. This is a standard, legitimate way to
prototype document AI (FUNSD/CORD/DocILE all expose exactly this token+box+label
schema) and it buys two things a scanned corpus cannot:

* **Perfect, controllable ground truth** at arbitrary scale — labels, table row
  groups, and key→value structure are known by construction, so every metric is
  exact and every failure is attributable.
* **A real image** — `render.py` rasterizes the layout to PNG with matplotlib,
  so the "money-shot" screenshot is an actual rendered document with detected
  boxes overlaid, not a mockup.

The pipeline consumes a generic `(text, x, y, w, h)` token stream, so a real
backend (Tesseract words, or LayoutLM/Donut token embeddings) drops in at the
`documents_to_table` boundary **without touching** feature extraction, the
classifier, KV pairing, table recovery, routing, or evaluation. See README §
"Swapping in a real OCR/vision model".

## 3. Pipeline

```
 generator ──► token table ──► features ──► TokenClassifier ──► field labels
 (text,box,      (columnar)     (pos+text+     (SGD softmax)          │
  label)                         neighbour)                           ├─► group_entities
                                                                      │      │
                     router (RandomForest) ──► doc type / route       │      ├─► pair_key_values (geometry)
                                                                      │      └─► extract_table (row/col clustering)
                     embedding (TF-IDF+SVD) ──► search / dedup ◄───── extracted content
```

### 3.1 Feature extraction (`features.py`)
Every token becomes a dense vector from three families:
* **Positional** — normalized box geometry, page quadrant flags, reading-order
  rank. This is what makes the model layout-aware.
* **Textual** — glyph-pattern signals (is-amount, is-date, is-integer, casing,
  digit fraction) and printed-label keyword groups (money / id / date / table /
  party). These are *patterns*, not the label — `is_amount` fires for TOTAL, TAX
  and LI_AMOUNT alike, so position must disambiguate.
* **Neighbour** — reading-order context: does the previous token end in `:` or
  look like a printed key? A 2-back context flag (`Invoice` vs `PO`) resolves the
  otherwise-identical `INVOICE_NO`/`PO_NO` numeric values — lifting their F1 from
  ~0.5 to ~1.0. Fully vectorized: O(n) over the token stream.

### 3.2 Token classifier (`extraction.py`)
`StandardScaler` + multinomial `SGDClassifier` (log-loss) — a linear softmax head
trained by SGD. Chosen deliberately over a heavier model:
* O(n) per epoch and `partial_fit`-capable → **streams to millions of tokens**,
* calibrated per-class scores,
* the engineered layout features carry the accuracy (macro-F1 ≈ 0.98), so a
  linear head is enough. `class_weight="balanced"` handles the long tail (rare
  id fields vs. common `O`/`LI_*`).

### 3.3 Key→value pairing (`pair_key_values`)
Pure geometry over grouped entities: for each printed `KEY`, link the value
entity nearest **to the right on the same line**, falling back to **directly
below**. This mirrors heuristic KV linkers applied on top of LayoutLM entity
spans and needs no training.

### 3.4 Table extraction (`extract_table`)
Line-item cells are recovered by **1-D clustering**: token `y`-centers → rows,
`x`-centers → columns (gap-based clustering). The 2-D grid falls out of the two
1-D partitions with no supervision on the grid itself — robust to variable row
counts and multi-word description cells.

### 3.5 Router (`router.py`)
A `RandomForestClassifier` over aggregate document features (token/line counts,
`$`-density, table-row estimate, printed-keyword presence) selects the doc type
and hence the extraction template. Layout statistics are highly separable across
invoice / receipt / PO / form, so accuracy is ~1.0.

### 3.6 Content embedding (`embedding.py`)
Extracted field values are serialized and embedded with TF-IDF + TruncatedSVD
(LSA), L2-normalized → cosine similarity powers semantic search and
near-duplicate detection. A real sentence encoder fits the same `fit`/`encode`
interface.

## 4. Scaling to 1B

**Generation is the scale story.** A document is a pure function of
`(seed, doc_id)`:
`generate_document(doc_id, seed)`. Therefore:
* **Bounded memory** — `iter_documents` streams; `generate_data.py` flushes
  Parquet row-groups every `--chunk` docs. Producing 100k docs (≈5M tokens)
  never holds more than one chunk in RAM.
* **Embarrassingly parallel / shardable** — shard *k* just generates
  `doc_id % K == k`; no coordination, no shuffle. N machines ⇒ ~N× throughput.
* **Out-of-core analytics** — the Parquet token table is DuckDB/Polars-lazy
  friendly; per-field aggregates and joins run without loading the corpus.

`benchmark.py` measures sustained single-process generation throughput and
extrapolates the wall-clock for 1B tokens (see `benchmarks/results.md`). We
report **measured** numbers at 100k docs and the **architected** path to 1B; we
do not claim to have persisted 1B rows.

Training also scales: SGD is O(n·epochs) with `partial_fit`, so the classifier
can consume a streamed token corpus instead of an in-memory matrix.

## 5. Trade-offs & honest limitations

* **Synthetic layouts, not scans.** No real OCR noise, skew, or handwriting.
  The value is the *architecture and evaluation harness*; realism arrives by
  swapping the token source (§README). Vendor/party name disambiguation
  (`VENDOR` vs `BILL_TO`, F1 ≈ 0.85) is the hardest field precisely because it is
  pure position — the same class of error a real model shows.
* **Linear classifier.** A tree/transformer head would squeeze the last few
  points but costs the streaming/`partial_fit` property; the engineered features
  make the trade worthwhile.
* **Router near-perfect** because synthetic type templates are cleanly
  separable; on real mixed corpora expect confusion (receipt↔invoice) — the
  confusion-matrix panel is built to surface it.
* **KV pairing is geometric**, so unusual layouts (value far below key, multi-
  column forms) can mispair; the ~3.5% linkage error is dominated by upstream
  token-typing mistakes.
