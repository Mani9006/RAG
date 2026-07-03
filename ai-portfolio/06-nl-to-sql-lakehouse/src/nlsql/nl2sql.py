"""NL-to-SQL engine (deterministic, offline) behind a pluggable interface.

``NLToSQLEngine`` is the interface a real LLM backend would implement. The
shipped ``DeterministicNLToSQL`` needs no model and no API key: it links schema
entities (:mod:`nlsql.schema_linker`), parses intent/slots (ordering, top-N,
limit), assembles a :class:`~nlsql.semantic.Query`, compiles it through the
semantic layer, and *validates* the SQL before returning it. Swapping in an LLM
means implementing ``generate`` to emit a ``Query`` (or raw SQL) -- the
validation guardrail and execution path are unchanged.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .schema_linker import LinkResult, SchemaLinker
from .semantic import Query, SemanticModel
from .validator import ValidationReport, validate_sql

_DESC_WORDS = ["top", "highest", "most", "largest", "best", "biggest", "leading"]
_ASC_WORDS = ["bottom", "lowest", "least", "smallest", "worst", "fewest"]


@dataclass
class GenerationResult:
    question: str
    sql: str
    query: Query
    links: LinkResult
    report: ValidationReport | None = None
    notes: list[str] = field(default_factory=list)


class NLToSQLEngine(ABC):
    """Interface for any NL-to-SQL backend (deterministic today, LLM tomorrow)."""

    @abstractmethod
    def generate(self, question: str, validate: bool = True) -> GenerationResult:
        ...


class DeterministicNLToSQL(NLToSQLEngine):
    def __init__(self, model: SemanticModel, con=None, default_metric: str = "revenue"):
        self.model = model
        self.linker = SchemaLinker(model)
        self.con = con
        self.default_metric = default_metric

    # ---- intent / slot parsing ----------------------------------------
    @staticmethod
    def _parse_limit(norm: str) -> int | None:
        m = re.search(r"\b(?:top|bottom|first|last)\s+(\d{1,4})\b", norm)
        if m:
            return int(m.group(1))
        return None

    @staticmethod
    def _parse_direction(norm: str) -> bool | None:
        """Return True for DESC, False for ASC, None if unspecified."""
        for w in _DESC_WORDS:
            if re.search(rf"\b{w}\b", norm):
                return True
        for w in _ASC_WORDS:
            if re.search(rf"\b{w}\b", norm):
                return False
        return None

    # ---- generation ----------------------------------------------------
    def generate(self, question: str, validate: bool = True) -> GenerationResult:
        norm = re.sub(r"[^a-z0-9\s]", " ", question.lower())
        links = self.linker.link(question)
        notes: list[str] = []

        metrics = list(dict.fromkeys(links.metrics))
        if not metrics:
            metrics = [self.default_metric]
            notes.append(f"no metric detected; defaulted to '{self.default_metric}'")

        dimensions = list(dict.fromkeys(links.dimensions))

        limit = self._parse_limit(norm)
        direction = self._parse_direction(norm)
        # singular superlative ("which region has the highest revenue") -> one row
        if (limit is None and direction is not None and dimensions
                and re.search(r"\b(which|what)\b", norm)):
            limit = 1

        order_by = None
        order_desc = True
        if dimensions and (direction is not None or limit is not None):
            # rank the groups by the primary metric
            order_by = metrics[0]
            order_desc = True if direction is None else direction
        elif dimensions:
            # deterministic output ordering by the first group-by dimension
            order_by = dimensions[0]
            order_desc = False

        query = Query(
            metrics=metrics,
            dimensions=dimensions,
            filters=links.filters,
            order_by=order_by,
            order_desc=order_desc,
            limit=limit,
        )
        sql = self.model.compile(query)

        report = None
        if validate:
            report = validate_sql(
                sql, catalog=self.model.column_catalog(), con=self.con
            )

        return GenerationResult(
            question=question, sql=sql, query=query, links=links,
            report=report, notes=notes,
        )


class LLMNLToSQL(NLToSQLEngine):
    """Placeholder showing the drop-in seam for a real LLM backend.

    A production build would prompt an LLM with the semantic-model catalog and
    ask it to emit a ``Query`` (or raw SQL), then route the result through the
    exact same :func:`~nlsql.validator.validate_sql` guardrail used above. It is
    intentionally inert here because this project is strictly offline.
    """

    def __init__(self, model: SemanticModel, client=None):
        self.model = model
        self.client = client

    def generate(self, question: str, validate: bool = True) -> GenerationResult:
        raise NotImplementedError(
            "LLM backend not wired in this offline build; use DeterministicNLToSQL. "
            "Implement this by calling self.client, parsing its Query/SQL, then "
            "reusing nlsql.validator.validate_sql before execution."
        )
