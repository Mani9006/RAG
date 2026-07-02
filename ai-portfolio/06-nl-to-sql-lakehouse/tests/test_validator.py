from __future__ import annotations

import pytest

from nlsql.validator import SQLValidationError, is_safe, validate_sql


@pytest.mark.parametrize("stmt", [
    "DROP TABLE fact_sales",
    "DELETE FROM fact_sales",
    "UPDATE fact_sales SET cost = 0",
    "INSERT INTO fact_sales VALUES (1)",
    "ALTER TABLE fact_sales ADD COLUMN x INT",
    "CREATE TABLE evil AS SELECT 1",
    "TRUNCATE fact_sales",
    "COPY fact_sales TO 'out.csv'",
    "PRAGMA database_list",
    "ATTACH 'x.db' AS y",
])
def test_blocks_non_select(stmt):
    assert not is_safe(stmt)
    with pytest.raises(SQLValidationError):
        validate_sql(stmt)


def test_blocks_stacked_statements():
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT 1; DROP TABLE fact_sales")


def test_blocks_cte_smuggled_mutation():
    # WITH ... then a DELETE must not slip through
    with pytest.raises(SQLValidationError):
        validate_sql("WITH x AS (SELECT 1) DELETE FROM fact_sales")


def test_allows_plain_select(model):
    validate_sql("SELECT SUM(net_revenue) FROM fact_sales",
                 catalog=model.column_catalog())


def test_allows_with_cte():
    rep = validate_sql("WITH t AS (SELECT 1 AS a) SELECT a FROM t")
    assert rep.ok


def test_unknown_column_rejected(model):
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT dim_store.nonexistent FROM dim_store",
                     catalog=model.column_catalog())


def test_known_column_accepted(model):
    rep = validate_sql("SELECT dim_store.region FROM dim_store",
                       catalog=model.column_catalog())
    assert rep.ok


def test_live_bind_catches_bad_sql(con):
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT no_such_col FROM fact_sales", con=con)
