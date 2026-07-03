"""Semantic layer: business metrics/dimensions -> SQL, with auto-joins.

The semantic layer is the single source of truth for *what a metric means*. The
NL-to-SQL engine (or a human, or a future LLM) declares intent as a small
``Query`` object -- a list of metric names, a list of group-by dimension names,
filters, ordering and a limit -- and this module compiles it to a validated
DuckDB SQL string. Crucially, joins are resolved automatically from the star
schema: asking for ``revenue`` grouped by ``store_region`` knows it must join
``fact_sales`` to ``dim_store`` on ``store_key``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MODEL_PATH = Path(__file__).with_name("semantic_model.yaml")

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    sql: str
    fmt: str
    synonyms: tuple[str, ...]


@dataclass(frozen=True)
class Dimension:
    name: str
    label: str
    table: str
    sql: str
    synonyms: tuple[str, ...]


@dataclass(frozen=True)
class Filter:
    """A single WHERE predicate on a dimension.

    ``op`` is one of ``=``, ``!=``, ``>``, ``>=``, ``<``, ``<=``.
    """

    dimension: str
    op: str
    value: Any

    def render(self, model: "SemanticModel") -> str:
        dim = model.dimension(self.dimension)
        if self.op not in {"=", "!=", ">", ">=", "<", "<="}:
            raise ValueError(f"unsupported filter op: {self.op}")
        return f"{dim.sql} {self.op} {_sql_literal(self.value)}"


@dataclass
class Query:
    """Declarative, semantic-layer query. Compiles to SQL via ``SemanticModel``."""

    metrics: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)
    filters: list[Filter] = field(default_factory=list)
    order_by: str | None = None          # a metric or dimension name
    order_desc: bool = True
    limit: int | None = None


def _sql_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    # string -> single-quote and escape embedded quotes
    return "'" + str(value).replace("'", "''") + "'"


class SemanticModel:
    """Loads the YAML semantic model and compiles ``Query`` objects to SQL."""

    def __init__(self, spec: dict[str, Any]):
        self.fact_name: str = spec["fact"]["name"]
        self.fact_pk: str = spec["fact"]["primary_key"]
        self._dim_tables: dict[str, dict[str, str]] = spec["dimensions_tables"]

        self.metrics: dict[str, Metric] = {}
        for name, m in spec["metrics"].items():
            self.metrics[name] = Metric(
                name=name,
                label=m["label"],
                sql=m["sql"],
                fmt=m.get("format", "number"),
                synonyms=tuple(s.lower() for s in m.get("synonyms", [])),
            )

        self.dimensions: dict[str, Dimension] = {}
        for name, d in spec["dimensions"].items():
            self.dimensions[name] = Dimension(
                name=name,
                label=d["label"],
                table=d["table"],
                sql=d["sql"],
                synonyms=tuple(s.lower() for s in d.get("synonyms", [])),
            )

    # ---- loading -------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "SemanticModel":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    # ---- catalog helpers ----------------------------------------------
    def metric(self, name: str) -> Metric:
        if name not in self.metrics:
            raise KeyError(f"unknown metric: {name}")
        return self.metrics[name]

    def dimension(self, name: str) -> Dimension:
        if name not in self.dimensions:
            raise KeyError(f"unknown dimension: {name}")
        return self.dimensions[name]

    def column_catalog(self) -> dict[str, set[str]]:
        """Physical ``table -> {columns}`` map used by the SQL validator."""
        cat: dict[str, set[str]] = {
            self.fact_name: {
                "sale_id", "order_id", "date_key", "customer_key", "product_key",
                "store_key", "quantity", "unit_price", "discount", "net_revenue",
                "cost", "year", "month",
            },
            "dim_customer": {"customer_key", "customer_name", "segment", "city",
                             "country", "signup_date"},
            "dim_product": {"product_key", "product_name", "category", "subcategory",
                            "brand", "unit_cost", "list_price"},
            "dim_store": {"store_key", "store_name", "region", "country", "store_type"},
            "dim_date": {"date_key", "date", "year", "quarter", "month", "day",
                         "weekday", "is_weekend"},
        }
        return cat

    # ---- compilation ---------------------------------------------------
    def _required_dim_tables(self, query: Query) -> list[str]:
        tables: list[str] = []
        names = list(query.dimensions) + [f.dimension for f in query.filters]
        if query.order_by and query.order_by in self.dimensions:
            names.append(query.order_by)
        for dname in names:
            tbl = self.dimension(dname).table
            if tbl not in tables:
                tables.append(tbl)
        return tables

    def compile(self, query: Query) -> str:
        """Compile a ``Query`` into a single validated-shape SELECT string."""
        if not query.metrics and not query.dimensions:
            raise ValueError("query must reference at least one metric or dimension")

        select_parts: list[str] = []
        for dname in query.dimensions:
            dim = self.dimension(dname)
            select_parts.append(f"{dim.sql} AS {dname}")
        for mname in query.metrics:
            metric = self.metric(mname)
            select_parts.append(f"{metric.sql} AS {mname}")

        joins: list[str] = []
        for tbl in self._required_dim_tables(query):
            spec = self._dim_tables[tbl]
            joins.append(
                f"JOIN {tbl} ON {self.fact_name}.{spec['fact_key']} = "
                f"{tbl}.{spec['key']}"
            )

        where = ""
        if query.filters:
            preds = " AND ".join(f.render(self) for f in query.filters)
            where = f"\nWHERE {preds}"

        group_by = ""
        if query.dimensions and query.metrics:
            group_by = "\nGROUP BY " + ", ".join(
                str(i + 1) for i in range(len(query.dimensions))
            )

        order = ""
        if query.order_by is not None:
            if query.order_by not in self.metrics and query.order_by not in self.dimensions:
                raise KeyError(f"cannot order by unknown field: {query.order_by}")
            direction = "DESC" if query.order_desc else "ASC"
            order = f"\nORDER BY {query.order_by} {direction}"

        limit = f"\nLIMIT {int(query.limit)}" if query.limit else ""

        sql = (
            "SELECT " + ", ".join(select_parts)
            + f"\nFROM {self.fact_name}"
            + ("\n" + "\n".join(joins) if joins else "")
            + where + group_by + order + limit
        )
        return sql
