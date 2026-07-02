"""End-to-end training + evaluation of the document-intelligence pipeline.

Trains the token classifier and the document router, then evaluates:
  * per-field precision / recall / F1 (token classification),
  * key→value geometric pairing linkage accuracy,
  * line-item TABLE extraction accuracy (cell-level, end-to-end),
  * doc-type classification accuracy + confusion matrix,
  * inference throughput (docs/sec, tokens/sec),
  * a near-duplicate detection demo over content embeddings.

Writes ``data/metrics.json`` and ``data/samples.json`` (consumed by
``make_screenshots.py``). Everything is seeded and deterministic.

Usage:
    python scripts/run_pipeline.py --train-docs 8000 --eval-docs 2000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docintel import DOC_TYPES, LABELS, LI_COLUMNS, SEED  # noqa: E402
from docintel.embedding import ContentEmbedder, nearest_duplicates  # noqa: E402
from docintel.evaluation import (  # noqa: E402
    accuracy, confusion_matrix, macro_f1, micro_f1, prf_per_class,
    table_cell_accuracy, table_shape_match)
from docintel.extraction import (  # noqa: E402
    TokenClassifier, extract_table, group_entities, majority_baseline,
    pair_key_values)
from docintel.generator import documents_to_table, iter_documents  # noqa: E402
from docintel.router import DocumentRouter, doc_features  # noqa: E402


def _doc_slices(table: dict[str, list]):
    """Yield (doc_id, doc_type, start, stop) for each contiguous doc block."""
    doc_ids = table["doc_id"]
    n = len(doc_ids)
    i = 0
    while i < n:
        j = i
        while j < n and doc_ids[j] == doc_ids[i]:
            j += 1
        yield doc_ids[i], table["doc_type"][i], i, j
        i = j


def _tokens_slice(table, s, e):
    return [{"text": table["text"][k], "x": table["x"][k], "y": table["y"][k],
             "w": table["w"][k], "h": table["h"][k]} for k in range(s, e)]


def _gold_grid(tokens, labels, row_groups):
    """Reconstruct the true line-item grid from gold labels + row_group."""
    rows: dict[int, dict[str, list[tuple[float, str]]]] = {}
    for t, lab, rg in zip(tokens, labels, row_groups):
        if not lab.startswith("LI_") or rg < 0:
            continue
        rows.setdefault(rg, {}).setdefault(lab, []).append((t["x"], t["text"]))
    grid = []
    for rg in sorted(rows):
        row = []
        for col in LI_COLUMNS:
            cells = sorted(rows[rg].get(col, []))
            row.append(" ".join(w for _, w in cells))
        grid.append(row)
    return grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-docs", type=int, default=8000)
    ap.add_argument("--eval-docs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    print(f"[1/6] generating {args.train_docs:,} train + {args.eval_docs:,} eval docs")
    train = documents_to_table(iter_documents(args.train_docs, seed=args.seed))
    test = documents_to_table(iter_documents(args.eval_docs, seed=args.seed + 777))
    print(f"      train tokens={len(train['text']):,}  eval tokens={len(test['text']):,}")

    # ---- token classifier ----
    print("[2/6] training token classifier (SGD)")
    t0 = time.time()
    clf = TokenClassifier(seed=args.seed).fit(train)
    train_time = time.time() - t0
    t0 = time.time()
    pred_ids = clf.predict(test)
    infer_time = time.time() - t0
    from docintel import ID_TO_LABEL
    pred = [ID_TO_LABEL[int(i)] for i in pred_ids]
    prf = prf_per_class(test["label"], pred, LABELS)
    field_macro = macro_f1(prf)
    field_micro = micro_f1(test["label"], pred)
    tok_acc = accuracy(test["label"], pred)
    base = majority_baseline(train["label"], test["label"])
    n_tokens = len(test["text"])
    tok_per_s = n_tokens / infer_time
    docs_per_s = args.eval_docs / infer_time
    print(f"      macro-F1={field_macro:.4f} micro-F1={field_micro:.4f} "
          f"acc={tok_acc:.4f} baseline={base:.4f}")
    print(f"      train={train_time:.1f}s  infer={tok_per_s:,.0f} tok/s  "
          f"{docs_per_s:,.0f} docs/s")

    # ---- doc router ----
    print("[3/6] training document router (RandomForest)")
    Xtr, ytr, Xte, yte = [], [], [], []
    for _id, dt, s, e in _doc_slices(train):
        Xtr.append(doc_features(_tokens_slice(train, s, e))); ytr.append(dt)
    for _id, dt, s, e in _doc_slices(test):
        Xte.append(doc_features(_tokens_slice(test, s, e))); yte.append(dt)
    router = DocumentRouter(seed=args.seed).fit(np.array(Xtr), ytr)
    ypr = list(router.predict(np.array(Xte)))
    route_acc = accuracy(yte, ypr)
    cm = confusion_matrix(yte, ypr, DOC_TYPES)
    print(f"      doc-type accuracy={route_acc:.4f}")

    # ---- table extraction + KV pairing (per doc, end-to-end w/ predicted labels) ----
    print("[4/6] evaluating table extraction + key/value pairing")
    # align predicted labels back to the test table's doc order
    pred_arr = pred
    tbl_cell_accs, tbl_shape_ok = [], []
    kv_total, kv_correct = 0, 0
    n_tables = 0
    for _id, dt, s, e in _doc_slices(test):
        toks = _tokens_slice(test, s, e)
        gold_labels = test["label"][s:e]
        pred_labels = pred_arr[s:e]
        rgs = test["row_group"][s:e]
        gold_grid = _gold_grid(toks, gold_labels, rgs)
        if gold_grid:
            pred_grid = extract_table(toks, pred_labels)
            tbl_cell_accs.append(table_cell_accuracy(gold_grid, pred_grid))
            tbl_shape_ok.append(table_shape_match(gold_grid, pred_grid))
            n_tables += 1
        # KV linkage: compare gold-label pairing vs predicted-label pairing
        gold_pairs = {p.key_text: p.value_text
                      for p in pair_key_values(group_entities(toks, gold_labels))}
        pred_pairs = {p.key_text: p.value_text
                      for p in pair_key_values(group_entities(toks, pred_labels))}
        for k, v in gold_pairs.items():
            kv_total += 1
            if pred_pairs.get(k) == v:
                kv_correct += 1
    table_acc = float(np.mean(tbl_cell_accs)) if tbl_cell_accs else 0.0
    table_shape = float(np.mean(tbl_shape_ok)) if tbl_shape_ok else 0.0
    kv_acc = kv_correct / kv_total if kv_total else 0.0
    print(f"      table cell acc={table_acc:.4f}  shape match={table_shape:.4f} "
          f"({n_tables} tables)  KV linkage acc={kv_acc:.4f} (n={kv_total})")

    # ---- content embeddings + dedup demo ----
    print("[5/6] content embeddings + near-duplicate detection")
    contents = []
    for _id, dt, s, e in _doc_slices(test):
        contents.append(" ".join(test["text"][s:e]))
    emb = ContentEmbedder(dim=96, seed=args.seed).fit(contents)
    vecs = emb.encode(contents[: min(600, len(contents))])
    dups = nearest_duplicates(vecs, threshold=0.995)
    print(f"      embedding dim={emb.dim}  candidate dup pairs (>0.995)={len(dups)}")

    # ---- persist metrics ----
    print("[6/6] writing artifacts")
    metrics = {
        "config": {"train_docs": args.train_docs, "eval_docs": args.eval_docs,
                   "train_tokens": len(train["text"]), "eval_tokens": n_tokens,
                   "seed": args.seed},
        "token_classifier": {
            "macro_f1": field_macro, "micro_f1": field_micro, "accuracy": tok_acc,
            "majority_baseline": base, "train_time_s": train_time,
            "tokens_per_s": tok_per_s, "docs_per_s": docs_per_s,
            "per_field": {k: {"precision": v.precision, "recall": v.recall,
                              "f1": v.f1, "support": v.support}
                          for k, v in prf.items()},
        },
        "router": {"accuracy": route_acc,
                   "confusion_matrix": cm.tolist(), "labels": DOC_TYPES,
                   "distribution": {t: int(np.sum(np.array(yte) == t)) for t in DOC_TYPES}},
        "table": {"cell_accuracy": table_acc, "shape_match": table_shape,
                  "n_tables": n_tables},
        "kv_pairing": {"linkage_accuracy": kv_acc, "n_pairs": kv_total},
        "embedding": {"dim": emb.dim, "dup_pairs": len(dups)},
    }
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- sample docs for the money-shot screenshot ----
    samples = []
    want = {"INVOICE", "RECEIPT", "FORM"}
    for _id, dt, s, e in _doc_slices(test):
        if dt in want:
            samples.append({"doc_id": int(_id), "doc_type": dt,
                            "tokens": _tokens_slice(test, s, e),
                            "gold": list(test["label"][s:e]),
                            "pred": list(pred_arr[s:e])})
            want.discard(dt)
        if not want:
            break
    (ROOT / "data" / "samples.json").write_text(json.dumps(samples))
    print(f"      wrote data/metrics.json and data/samples.json ({len(samples)} samples)")
    print("\nDONE. Headline: field macro-F1 %.3f | doc-type acc %.3f | %.0f docs/s"
          % (field_macro, route_acc, docs_per_s))


if __name__ == "__main__":
    main()
