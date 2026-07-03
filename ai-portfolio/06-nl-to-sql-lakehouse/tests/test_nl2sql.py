from __future__ import annotations

import pytest

from nlsql.eval_suite import CASES, categories, results_match
from nlsql.validator import validate_sql


def test_generated_sql_is_valid_and_executes(engine, con, model):
    for case in CASES:
        res = engine.generate(case.question)
        # generated SQL passes the guardrail (shape + columns + live bind)
        validate_sql(res.sql, catalog=model.column_catalog(), con=con)
        con.execute(res.sql).fetchall()  # executes without error


def test_execution_accuracy_full_suite(engine, con):
    correct = 0
    for case in CASES:
        res = engine.generate(case.question)
        gen = con.execute(res.sql).fetchall()
        gold = con.execute(case.gold_sql).fetchall()
        if results_match(gen, gold):
            correct += 1
    acc = correct / len(CASES)
    # deterministic engine should solve the whole suite on the fixture
    assert acc == 1.0, f"execution accuracy {acc:.2%} ({correct}/{len(CASES)})"


def test_every_category_covered(engine, con):
    for cat in categories():
        cases = [c for c in CASES if c.category == cat]
        for case in cases:
            res = engine.generate(case.question)
            gen = con.execute(res.sql).fetchall()
            gold = con.execute(case.gold_sql).fetchall()
            assert results_match(gen, gold), f"{cat}:{case.id} mismatch"


def test_topn_applies_limit(engine):
    res = engine.generate("Top 3 product categories by revenue")
    assert res.query.limit == 3
    assert "LIMIT 3" in res.sql


def test_default_metric_when_none_detected(engine):
    res = engine.generate("show me numbers by region")
    assert "revenue" in res.query.metrics
    assert any("defaulted" in n for n in res.notes)


def test_llm_backend_is_a_seam(model):
    from nlsql import LLMNLToSQL
    eng = LLMNLToSQL(model)
    with pytest.raises(NotImplementedError):
        eng.generate("anything")
