"""DuckDB lakehouse access layer.

Registers the Parquet star schema as SQL views so queries run out-of-core
straight over the files -- no load/copy step. ``fact_sales`` is a Hive-
partitioned scan (``year=``/``month=`` pruning); the dimensions are single
Parquet files. A tiny in-memory builder is also provided for tests/fixtures.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

_DIMS = ["dim_customer", "dim_product", "dim_store", "dim_date"]


def connect(data_dir: str | Path, threads: int | None = None,
            memory_limit: str | None = None) -> duckdb.DuckDBPyConnection:
    """Open a connection with the lakehouse registered as views over Parquet."""
    data = Path(data_dir)
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA disable_progress_bar")
    if threads:
        con.execute(f"PRAGMA threads={int(threads)}")
    if memory_limit:
        con.execute(f"PRAGMA memory_limit='{memory_limit}'")

    fact_glob = (data / "fact_sales" / "**" / "*.parquet").as_posix()
    con.execute(
        f"""CREATE VIEW fact_sales AS
            SELECT * FROM read_parquet('{fact_glob}', hive_partitioning=true)"""
    )
    for dim in _DIMS:
        path = (data / f"{dim}.parquet").as_posix()
        con.execute(f"CREATE VIEW {dim} AS SELECT * FROM read_parquet('{path}')")
    return con


def build_memory_fixture(seed: int = 3, rows: int = 5_000,
                         customers: int = 300, products: int = 120,
                         stores: int = 12) -> duckdb.DuckDBPyConnection:
    """Build a small in-memory star schema as real DuckDB tables (for tests)."""
    import numpy as np
    import pyarrow as pa

    from .generator import (GenSpec, _build_dim_customer, _build_dim_date,
                            _build_dim_product, _build_dim_store,
                            _seasonal_day_weights, _unit_cost_arr)

    spec = GenSpec(rows=rows, customers=customers, products=products, stores=stores,
                   seed=seed, start_year=2022, end_year=2023)
    rng = np.random.default_rng(spec.seed)
    dim_customer = _build_dim_customer(rng, customers)
    dim_product, _uc, list_price = _build_dim_product(rng, products)
    dim_store = _build_dim_store(rng, stores)
    dim_date, date_keys, day_year, day_month = _build_dim_date(
        spec.start_year, spec.end_year)
    n_days = len(date_keys)
    day_p = _seasonal_day_weights(day_month.astype(np.float64))

    day_idx = rng.choice(n_days, size=rows, p=day_p)
    product_key = rng.integers(0, products, size=rows, dtype=np.int32)
    quantity = rng.integers(1, 10, size=rows).astype(np.int16)
    discount = rng.choice([0.0, 0.1, 0.2], size=rows).astype(np.float32)
    unit_price = np.round(list_price[product_key] * rng.uniform(0.9, 1.15, size=rows),
                          2).astype(np.float32)
    net_revenue = np.round(quantity.astype(np.float32) * unit_price * (1 - discount),
                           2).astype(np.float32)
    cost = np.round(_unit_cost_arr(list_price, product_key)
                    * quantity.astype(np.float32), 2).astype(np.float32)
    new_order = rng.random(rows) < 0.4
    new_order[0] = True
    order_id = np.cumsum(new_order).astype(np.int64)

    fact = pa.table({
        "sale_id": np.arange(rows, dtype=np.int64),
        "order_id": order_id,
        "date_key": date_keys[day_idx],
        "customer_key": rng.integers(0, customers, size=rows, dtype=np.int32),
        "product_key": product_key,
        "store_key": rng.integers(0, stores, size=rows, dtype=np.int32),
        "quantity": quantity,
        "unit_price": unit_price,
        "discount": discount,
        "net_revenue": net_revenue,
        "cost": cost,
        "year": day_year[day_idx],
        "month": day_month[day_idx],
    })

    con = duckdb.connect(":memory:")
    con.register("_fact", fact)
    con.execute("CREATE TABLE fact_sales AS SELECT * FROM _fact")
    con.unregister("_fact")
    for name, tbl in [("dim_customer", dim_customer), ("dim_product", dim_product),
                      ("dim_store", dim_store), ("dim_date", dim_date)]:
        con.register("_t", tbl)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _t")
        con.unregister("_t")
    return con
