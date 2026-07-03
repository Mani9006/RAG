#!/usr/bin/env python
"""Run the NL-to-SQL suite end-to-end against the DuckDB lakehouse.

Generates SQL for each of the ~26 gold questions, validates it, executes it, and
compares the result to the gold SQL. Reports overall + per-category execution
accuracy and writes CSVs used by the screenshots.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlsql import DeterministicNLToSQL, SemanticModel, connect  # noqa: E402
from nlsql.eval_suite import CASES, categories, results_match  # noqa: E402
from nlsql.validator import SQLValidationError, validate_sql  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "lakehouse"))
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    model = SemanticModel.load()
    con = connect(args.data, threads=args.threads or None)
    engine = DeterministicNLToSQL(model, con=con)
    catalog = model.column_catalog()

    rows = []
    per_cat: dict[str, list[int]] = {c: [] for c in categories()}
    n_valid = 0
    for case in CASES:
        res = engine.generate(case.question)
        # guardrail
        valid = True
        try:
            validate_sql(res.sql, catalog=catalog, con=con)
        except SQLValidationError:
            valid = False
        n_valid += valid

        t0 = time.perf_counter()
        gen = con.execute(res.sql).fetchall()
        gen_ms = (time.perf_counter() - t0) * 1000
        gold = con.execute(case.gold_sql).fetchall()
        correct = results_match(gen, gold)
        per_cat[case.category].append(int(correct))
        rows.append({
            "id": case.id, "category": case.category, "question": case.question,
            "valid": valid, "correct": correct, "exec_ms": round(gen_ms, 2),
            "sql": res.sql.replace("\n", " "),
            "answer": str(gen[:5]),
        })

    total = len(CASES)
    correct_n = sum(r["correct"] for r in rows)
    acc = correct_n / total

    # write per-question detail
    det = ROOT / "benchmarks" / "suite_results.csv"
    with open(det, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # write per-category accuracy
    cat_rows = []
    for cat, hits in per_cat.items():
        cat_rows.append({"category": cat, "n": len(hits),
                         "correct": sum(hits),
                         "accuracy": round(sum(hits) / len(hits), 4)})
    catcsv = ROOT / "benchmarks" / "accuracy_by_category.csv"
    with open(catcsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["category", "n", "correct", "accuracy"])
        w.writeheader()
        w.writerows(cat_rows)

    summary = {
        "questions": total,
        "valid_sql": n_valid,
        "correct": correct_n,
        "execution_accuracy": round(acc, 4),
        "validation_rate": round(n_valid / total, 4),
        "by_category": {r["category"]: r["accuracy"] for r in cat_rows},
    }
    (ROOT / "benchmarks" / "suite_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nExecution accuracy: {correct_n}/{total} = {acc:.1%}")
    print(f"SQL validated: {n_valid}/{total}")
    print(f"wrote {det.name}, {catcsv.name}, suite_summary.json")
    con.close()


if __name__ == "__main__":
    main()
