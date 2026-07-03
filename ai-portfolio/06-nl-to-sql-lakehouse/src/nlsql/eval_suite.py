"""Gold NL-to-SQL evaluation suite.

~26 natural-language questions, each paired with independently hand-written
*gold* SQL. Execution accuracy is measured by running the engine's generated SQL
and the gold SQL against the same DuckDB lakehouse and comparing result sets
(order-insensitive, rounded). Grouped by category so we can report per-category
accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    question: str
    gold_sql: str


CASES: list[Case] = [
    # ---- aggregate (single number) ------------------------------------
    Case("agg_revenue", "aggregate", "What is the total revenue?",
         "SELECT SUM(net_revenue) FROM fact_sales"),
    Case("agg_units", "aggregate", "How many units were sold in total?",
         "SELECT SUM(quantity) FROM fact_sales"),
    Case("agg_orders", "aggregate", "How many orders are there?",
         "SELECT COUNT(DISTINCT order_id) FROM fact_sales"),
    Case("agg_margin", "aggregate", "What is the total gross margin?",
         "SELECT SUM(net_revenue - cost) FROM fact_sales"),

    # ---- distinct count -----------------------------------------------
    Case("dc_customers", "distinct_count", "How many active customers do we have?",
         "SELECT COUNT(DISTINCT customer_key) FROM fact_sales"),
    Case("dc_customers_2023", "distinct_count",
         "How many active customers were there in 2023?",
         "SELECT COUNT(DISTINCT f.customer_key) FROM fact_sales f "
         "JOIN dim_date d ON f.date_key=d.date_key WHERE d.year=2023"),

    # ---- ratio metrics -------------------------------------------------
    Case("ratio_aov", "ratio", "What is the average order value?",
         "SELECT SUM(net_revenue)/COUNT(DISTINCT order_id) FROM fact_sales"),
    Case("ratio_aov_2023", "ratio", "What was the AOV in 2023?",
         "SELECT SUM(f.net_revenue)/COUNT(DISTINCT f.order_id) FROM fact_sales f "
         "JOIN dim_date d ON f.date_key=d.date_key WHERE d.year=2023"),

    # ---- filter (single number, filtered) -----------------------------
    Case("flt_electronics", "filter", "What is the revenue for Electronics?",
         "SELECT SUM(f.net_revenue) FROM fact_sales f "
         "JOIN dim_product p ON f.product_key=p.product_key "
         "WHERE p.category='Electronics'"),
    Case("flt_west", "filter", "What is the total revenue in the West region?",
         "SELECT SUM(f.net_revenue) FROM fact_sales f "
         "JOIN dim_store s ON f.store_key=s.store_key WHERE s.region='West'"),
    Case("flt_corporate", "filter", "What is revenue for the Corporate segment?",
         "SELECT SUM(f.net_revenue) FROM fact_sales f "
         "JOIN dim_customer c ON f.customer_key=c.customer_key "
         "WHERE c.segment='Corporate'"),
    Case("flt_online_units", "filter", "How many units were sold Online?",
         "SELECT SUM(f.quantity) FROM fact_sales f "
         "JOIN dim_store s ON f.store_key=s.store_key WHERE s.store_type='Online'"),
    Case("flt_2022_rev", "filter", "What was the revenue in 2022?",
         "SELECT SUM(f.net_revenue) FROM fact_sales f "
         "JOIN dim_date d ON f.date_key=d.date_key WHERE d.year=2022"),

    # ---- group_by ------------------------------------------------------
    Case("grp_region", "group_by", "Show revenue by region.",
         "SELECT s.region, SUM(f.net_revenue) FROM fact_sales f "
         "JOIN dim_store s ON f.store_key=s.store_key GROUP BY s.region"),
    Case("grp_category", "group_by", "What is revenue by product category?",
         "SELECT p.category, SUM(f.net_revenue) FROM fact_sales f "
         "JOIN dim_product p ON f.product_key=p.product_key GROUP BY p.category"),
    Case("grp_segment", "group_by", "Show active customers by segment.",
         "SELECT c.segment, COUNT(DISTINCT f.customer_key) FROM fact_sales f "
         "JOIN dim_customer c ON f.customer_key=c.customer_key GROUP BY c.segment"),
    Case("grp_storetype", "group_by", "What is the AOV by store type?",
         "SELECT s.store_type, SUM(f.net_revenue)/COUNT(DISTINCT f.order_id) "
         "FROM fact_sales f JOIN dim_store s ON f.store_key=s.store_key "
         "GROUP BY s.store_type"),
    Case("grp_orders_region", "group_by", "How many orders by region?",
         "SELECT s.region, COUNT(DISTINCT f.order_id) FROM fact_sales f "
         "JOIN dim_store s ON f.store_key=s.store_key GROUP BY s.region"),

    # ---- timeseries ----------------------------------------------------
    Case("ts_month", "timeseries", "Show monthly revenue.",
         "SELECT d.month, SUM(f.net_revenue) FROM fact_sales f "
         "JOIN dim_date d ON f.date_key=d.date_key GROUP BY d.month"),
    Case("ts_quarter", "timeseries", "What is revenue by quarter?",
         "SELECT d.quarter, SUM(f.net_revenue) FROM fact_sales f "
         "JOIN dim_date d ON f.date_key=d.date_key GROUP BY d.quarter"),
    Case("ts_year", "timeseries", "Show revenue by year.",
         "SELECT d.year, SUM(f.net_revenue) FROM fact_sales f "
         "JOIN dim_date d ON f.date_key=d.date_key GROUP BY d.year"),
    Case("ts_month_2023", "timeseries", "Show monthly revenue in 2023.",
         "SELECT d.month, SUM(f.net_revenue) FROM fact_sales f "
         "JOIN dim_date d ON f.date_key=d.date_key WHERE d.year=2023 GROUP BY d.month"),

    # ---- topn / ranking -----------------------------------------------
    Case("top_cat", "topn", "Top 3 product categories by revenue.",
         "SELECT p.category, SUM(f.net_revenue) AS m FROM fact_sales f "
         "JOIN dim_product p ON f.product_key=p.product_key "
         "GROUP BY p.category ORDER BY m DESC LIMIT 3"),
    Case("top_brand", "topn", "What are the top 5 brands by revenue?",
         "SELECT p.brand, SUM(f.net_revenue) AS m FROM fact_sales f "
         "JOIN dim_product p ON f.product_key=p.product_key "
         "GROUP BY p.brand ORDER BY m DESC LIMIT 5"),
    Case("top_region_low", "topn", "Which region has the lowest revenue?",
         "SELECT s.region, SUM(f.net_revenue) AS m FROM fact_sales f "
         "JOIN dim_store s ON f.store_key=s.store_key "
         "GROUP BY s.region ORDER BY m ASC LIMIT 1"),
    Case("top_cat_margin", "topn",
         "Which product category has the highest gross margin?",
         "SELECT p.category, SUM(f.net_revenue - f.cost) AS m FROM fact_sales f "
         "JOIN dim_product p ON f.product_key=p.product_key "
         "GROUP BY p.category ORDER BY m DESC LIMIT 1"),
]


def categories() -> list[str]:
    seen: list[str] = []
    for c in CASES:
        if c.category not in seen:
            seen.append(c.category)
    return seen


def _norm_row(row: tuple, ndigits: int = 2) -> tuple:
    out = []
    for v in row:
        if isinstance(v, float):
            out.append(round(v, ndigits))
        elif isinstance(v, int):
            out.append(float(v))  # 5 and 5.0 compare equal across engines
        else:
            out.append(v)
    return tuple(out)


def results_match(a: list[tuple], b: list[tuple], ndigits: int = 2) -> bool:
    """Order-insensitive comparison of two result sets (rounded floats)."""
    na = sorted(_norm_row(r, ndigits) for r in a)
    nb = sorted(_norm_row(r, ndigits) for r in b)
    return na == nb
