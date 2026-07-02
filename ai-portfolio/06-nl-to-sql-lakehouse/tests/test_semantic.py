from __future__ import annotations

import pytest

from nlsql import Filter, Query


def test_metric_only_no_join(model):
    sql = model.compile(Query(metrics=["revenue"]))
    assert "SUM(fact_sales.net_revenue)" in sql
    assert "JOIN" not in sql


def test_groupby_resolves_join(model):
    sql = model.compile(Query(metrics=["revenue"], dimensions=["store_region"]))
    assert "JOIN dim_store ON fact_sales.store_key = dim_store.store_key" in sql
    assert "GROUP BY 1" in sql


def test_filter_adds_join_and_where(model):
    q = Query(metrics=["revenue"],
              filters=[Filter("product_category", "=", "Electronics")])
    sql = model.compile(q)
    assert "JOIN dim_product" in sql
    assert "dim_product.category = 'Electronics'" in sql


def test_multi_dimension_join_dedup(model):
    q = Query(metrics=["revenue"], dimensions=["store_region"],
              filters=[Filter("store_type", "=", "Online")])
    sql = model.compile(q)
    # dim_store referenced twice but joined once
    assert sql.count("JOIN dim_store") == 1


def test_sql_injection_escaped(model):
    q = Query(metrics=["revenue"],
              filters=[Filter("product_category", "=", "O'Brien'; DROP")])
    sql = model.compile(q)
    assert "O''Brien" in sql  # single quote doubled


def test_unknown_metric_raises(model):
    with pytest.raises(KeyError):
        model.compile(Query(metrics=["not_a_metric"]))


def test_metrics_compute_on_fixture(con, model):
    """Semantic metric SQL must equal a hand-written aggregate on the fixture."""
    revenue = con.execute(model.compile(Query(metrics=["revenue"]))).fetchone()[0]
    gold = con.execute("SELECT SUM(net_revenue) FROM fact_sales").fetchone()[0]
    assert revenue == pytest.approx(gold, rel=1e-9)

    aov = con.execute(model.compile(Query(metrics=["aov"]))).fetchone()[0]
    gold_aov = con.execute(
        "SELECT SUM(net_revenue)/COUNT(DISTINCT order_id) FROM fact_sales"
    ).fetchone()[0]
    assert aov == pytest.approx(gold_aov, rel=1e-9)

    margin = con.execute(model.compile(Query(metrics=["gross_margin"]))).fetchone()[0]
    gold_m = con.execute("SELECT SUM(net_revenue-cost) FROM fact_sales").fetchone()[0]
    assert margin == pytest.approx(gold_m, rel=1e-9)
