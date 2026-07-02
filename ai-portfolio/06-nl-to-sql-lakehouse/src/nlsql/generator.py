"""Star-schema data generator: memory-bounded, streamed to partitioned Parquet.

The fact table is produced in fixed-size chunks so peak memory stays flat no
matter how many rows are requested -- the same code path that writes 1M rows
writes 1B. Each chunk is a vectorised NumPy build converted to an Arrow table
and appended to a Hive-partitioned dataset (``fact_sales/year=YYYY/month=MM/``),
which DuckDB can then scan out-of-core with partition pruning.

Dimensions are small and written once as single Parquet files.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import domains


@dataclass
class GenSpec:
    rows: int = 1_000_000
    customers: int = 50_000
    products: int = 5_000
    stores: int = 200
    seed: int = 7
    chunk_size: int = 2_000_000
    start_year: int = 2021
    end_year: int = 2023


@dataclass
class GenReport:
    rows: int
    fact_bytes: int
    dim_bytes: int
    num_partitions: int
    num_files: int
    bytes_per_row: float
    seconds: float


# ---------------------------------------------------------------------------
# dimensions
# ---------------------------------------------------------------------------
def _build_dim_customer(rng: np.random.Generator, n: int) -> pa.Table:
    seg = rng.choice(len(domains.SEGMENTS), size=n, p=[0.6, 0.25, 0.15])
    ctry = rng.integers(0, len(domains.CUSTOMER_COUNTRIES), size=n)
    return pa.table({
        "customer_key": np.arange(n, dtype=np.int32),
        "customer_name": [f"customer_{i:07d}" for i in range(n)],
        "segment": [domains.SEGMENTS[i] for i in seg],
        "city": [f"city_{c:04d}" for c in rng.integers(0, 500, size=n)],
        "country": [domains.CUSTOMER_COUNTRIES[i] for i in ctry],
        "signup_date": rng.integers(20180101, 20221231, size=n).astype(np.int32),
    })


def _build_dim_product(rng: np.random.Generator, n: int) -> tuple[pa.Table, np.ndarray, np.ndarray]:
    cat_idx = rng.integers(0, len(domains.CATEGORIES), size=n)
    categories = [domains.CATEGORIES[i] for i in cat_idx]
    subcats = [rng.choice(domains.SUBCATEGORIES[c]) for c in categories]
    brand_idx = rng.integers(0, len(domains.BRANDS), size=n)
    unit_cost = np.round(rng.uniform(2, 400, size=n), 2).astype(np.float32)
    list_price = np.round(unit_cost * rng.uniform(1.2, 2.5, size=n), 2).astype(np.float32)
    tbl = pa.table({
        "product_key": np.arange(n, dtype=np.int32),
        "product_name": [f"product_{i:06d}" for i in range(n)],
        "category": categories,
        "subcategory": subcats,
        "brand": [domains.BRANDS[i] for i in brand_idx],
        "unit_cost": unit_cost,
        "list_price": list_price,
    })
    return tbl, unit_cost, list_price


def _build_dim_store(rng: np.random.Generator, n: int) -> pa.Table:
    region = rng.integers(0, len(domains.REGIONS), size=n)
    stype = rng.integers(0, len(domains.STORE_TYPES), size=n)
    ctry = rng.integers(0, len(domains.STORE_COUNTRIES), size=n)
    return pa.table({
        "store_key": np.arange(n, dtype=np.int32),
        "store_name": [f"store_{i:04d}" for i in range(n)],
        "region": [domains.REGIONS[i] for i in region],
        "country": [domains.STORE_COUNTRIES[i] for i in ctry],
        "store_type": [domains.STORE_TYPES[i] for i in stype],
    })


def _build_dim_date(start_year: int, end_year: int) -> tuple[pa.Table, np.ndarray, np.ndarray, np.ndarray]:
    dates = np.arange(f"{start_year}-01-01", f"{end_year + 1}-01-01",
                      dtype="datetime64[D]")
    y = dates.astype("datetime64[Y]").astype(int) + 1970
    m = dates.astype("datetime64[M]").astype(int) % 12 + 1
    d = (dates - dates.astype("datetime64[M]")).astype(int) + 1
    weekday = (dates.astype("datetime64[D]").astype(int) + 4) % 7  # 0=Mon
    quarter = (m - 1) // 3 + 1
    date_key = (y * 10000 + m * 100 + d).astype(np.int32)
    tbl = pa.table({
        "date_key": date_key,
        "date": [str(x) for x in dates],
        "year": y.astype(np.int16),
        "quarter": quarter.astype(np.int8),
        "month": m.astype(np.int8),
        "day": d.astype(np.int8),
        "weekday": weekday.astype(np.int8),
        "is_weekend": (weekday >= 5),
    })
    return tbl, date_key, y.astype(np.int16), m.astype(np.int8)


# ---------------------------------------------------------------------------
# fact
# ---------------------------------------------------------------------------
def _seasonal_day_weights(month_of_day: np.ndarray) -> np.ndarray:
    # mild Q4 uplift so monthly trends look realistic
    w = 1.0 + 0.30 * np.sin((month_of_day - 3) / 12.0 * 2 * np.pi) + \
        0.25 * (month_of_day >= 11)
    return w / w.sum()


def generate(spec: GenSpec, out_dir: str | Path) -> GenReport:
    import time

    out = Path(out_dir)
    fact_dir = out / "fact_sales"
    if fact_dir.exists():
        shutil.rmtree(fact_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(spec.seed)
    t0 = time.perf_counter()

    # --- dims ---
    dim_customer = _build_dim_customer(rng, spec.customers)
    dim_product, _unit_cost, list_price = _build_dim_product(rng, spec.products)
    dim_store = _build_dim_store(rng, spec.stores)
    dim_date, date_keys, day_year, day_month = _build_dim_date(
        spec.start_year, spec.end_year)
    n_days = len(date_keys)
    day_p = _seasonal_day_weights(day_month.astype(np.float64))

    dim_bytes = 0
    for name, tbl in [("dim_customer", dim_customer), ("dim_product", dim_product),
                      ("dim_store", dim_store), ("dim_date", dim_date)]:
        path = out / f"{name}.parquet"
        pq.write_table(tbl, path, compression="snappy")
        dim_bytes += path.stat().st_size

    # --- fact (streamed) ---
    # One Parquet file per year partition (fact_sales/year=YYYY/data.parquet),
    # written incrementally via a long-lived ParquetWriter per year so the file
    # count stays tiny (= number of years) regardless of --rows. This matters a
    # lot for scan latency: opening many small files dominates otherwise.
    writers: dict[int, pq.ParquetWriter] = {}
    order_offset = np.int64(0)
    written = 0
    remaining = spec.rows
    while remaining > 0:
        n = int(min(spec.chunk_size, remaining))
        day_idx = rng.choice(n_days, size=n, p=day_p)
        product_key = rng.integers(0, spec.products, size=n, dtype=np.int32)
        quantity = rng.integers(1, 10, size=n).astype(np.int16)
        discount = rng.choice([0.0, 0.05, 0.10, 0.15, 0.20], size=n,
                              p=[0.55, 0.2, 0.13, 0.08, 0.04]).astype(np.float32)
        base_price = list_price[product_key]
        unit_price = np.round(base_price * rng.uniform(0.9, 1.15, size=n), 2
                              ).astype(np.float32)
        gross = quantity.astype(np.float32) * unit_price
        net_revenue = np.round(gross * (1.0 - discount), 2).astype(np.float32)
        unit_cost = _unit_cost_arr(list_price, product_key)
        cost = np.round(unit_cost * quantity.astype(np.float32), 2).astype(np.float32)

        # baskets: ~2.5 line items per order, vectorised via cumulative "new order"
        new_order = rng.random(n) < 0.4
        new_order[0] = True
        order_local = np.cumsum(new_order).astype(np.int64)
        order_id = order_offset + order_local
        order_offset = int(order_id[-1])

        year = day_year[day_idx]
        month = day_month[day_idx]

        # `year` is encoded in the Hive path (year=YYYY/), not stored in-file.
        table = pa.table({
            "sale_id": np.arange(written, written + n, dtype=np.int64),
            "order_id": order_id,
            "date_key": date_keys[day_idx],
            "customer_key": rng.integers(0, spec.customers, size=n, dtype=np.int32),
            "product_key": product_key,
            "store_key": rng.integers(0, spec.stores, size=n, dtype=np.int32),
            "quantity": quantity,
            "unit_price": unit_price,
            "discount": discount,
            "net_revenue": net_revenue,
            "cost": cost,
            "month": month,
        })
        for y in np.unique(year):
            sub = table.filter(pa.array(year == y))
            w = writers.get(int(y))
            if w is None:
                part_dir = fact_dir / f"year={int(y)}"
                part_dir.mkdir(parents=True, exist_ok=True)
                w = pq.ParquetWriter(part_dir / "data.parquet", sub.schema,
                                     compression="snappy")
                writers[int(y)] = w
            w.write_table(sub)

        written += n
        remaining -= n

    for w in writers.values():
        w.close()

    seconds = time.perf_counter() - t0

    fact_bytes = 0
    num_files = 0
    partitions: set[str] = set()
    for p in fact_dir.rglob("*.parquet"):
        fact_bytes += p.stat().st_size
        num_files += 1
        partitions.add(str(p.parent))

    return GenReport(
        rows=written,
        fact_bytes=fact_bytes,
        dim_bytes=dim_bytes,
        num_partitions=len(partitions),
        num_files=num_files,
        bytes_per_row=fact_bytes / max(written, 1),
        seconds=seconds,
    )


def _unit_cost_arr(list_price: np.ndarray, product_key: np.ndarray) -> np.ndarray:
    # cost basis ~ 55-70% of list price, deterministic per product via its price
    return (list_price[product_key] * 0.6).astype(np.float32)
