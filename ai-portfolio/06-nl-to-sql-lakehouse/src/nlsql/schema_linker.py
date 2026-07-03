"""Schema linking: natural-language phrases -> semantic-model entities.

The linker is the "understanding" half of the offline NL-to-SQL engine. Given a
question it recognises:

* **metrics** -- via each metric's synonym list ("turnover" -> ``revenue``),
* **group-by dimensions** -- from ``by <dim>`` / ``per <dim>`` phrasing and bare
  dimension synonyms ("monthly" -> ``month``),
* **filters** -- literal categorical values ("Electronics" ->
  ``product_category = 'Electronics'``) and year mentions ("in 2023").

Matching is longest-phrase-first so "average order value" beats "value", and is
fully deterministic. It returns structured links that the generator half
(:mod:`nlsql.nl2sql`) turns into a :class:`~nlsql.semantic.Query`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .domains import VALUE_TO_DIMENSION, YEARS
from .semantic import Filter, SemanticModel


@dataclass
class LinkResult:
    metrics: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)   # group-by dimensions
    filters: list[Filter] = field(default_factory=list)
    matched_spans: list[tuple[str, str]] = field(default_factory=list)  # (phrase, entity)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())


class SchemaLinker:
    def __init__(self, model: SemanticModel):
        self.model = model
        # Build (phrase, kind, name) triples, longest phrase first.
        self._metric_syns: list[tuple[str, str]] = []
        for m in model.metrics.values():
            for syn in m.synonyms:
                self._metric_syns.append((syn, m.name))
        self._metric_syns.sort(key=lambda t: len(t[0]), reverse=True)

        self._dim_syns: list[tuple[str, str]] = []
        for d in model.dimensions.values():
            for syn in d.synonyms:
                self._dim_syns.append((syn, d.name))
        self._dim_syns.sort(key=lambda t: len(t[0]), reverse=True)

    # ---- individual linkers -------------------------------------------
    def link_metrics(self, norm: str) -> list[tuple[str, str]]:
        found: list[tuple[str, str]] = []
        used = norm
        seen: set[str] = set()
        for syn, name in self._metric_syns:
            if name in seen:
                continue
            if re.search(rf"\b{re.escape(syn)}\b", used):
                found.append((syn, name))
                seen.add(name)
        return found

    def link_dimensions(self, norm: str) -> list[tuple[str, str]]:
        """Detect *group-by* dimensions.

        A dimension is a group-by only when introduced by ``by``/``per``/``each``
        or a grain adverb ("monthly"), so that a filter value like "in the West
        region" does not accidentally become a GROUP BY.
        """
        found: list[tuple[str, str]] = []
        seen: set[str] = set()
        grain_adverbs = {"monthly", "quarterly", "annually", "by year",
                         "by month", "by quarter"}
        # "which region has the highest ..." -> the dimension is a group-by even
        # though it is stated bare, because a superlative implies ranking groups.
        ranking = bool(re.search(
            r"\b(top|bottom|highest|lowest|most|least|best|worst|"
            r"largest|smallest|leading)\b", norm))
        for syn, name in self._dim_syns:
            if name in seen:
                continue
            esc = re.escape(syn)
            patterns = [
                # explicit grouping: "by region", "per category", "for each store type"
                rf"\b(?:by|per|each|across|for\s+each)\s+{esc}\b",
                # ranking phrasing: "top 5 categories by revenue"
                rf"\b{esc}\s+by\b",
            ]
            # bare grain adverbs are group-bys on their own ("monthly revenue")
            if syn in grain_adverbs:
                patterns.append(rf"\b{esc}\b")
            # bare dimension noun under a ranking/superlative context
            if ranking:
                patterns.append(rf"\b{esc}\b")
            for pat in patterns:
                if re.search(pat, norm):
                    found.append((syn, name))
                    seen.add(name)
                    break
        return found

    def link_filters(self, norm: str) -> list[Filter]:
        filters: list[Filter] = []
        seen_dims: set[str] = set()
        # categorical literal values
        for value_lc, (dim_name, canonical) in VALUE_TO_DIMENSION.items():
            if re.search(rf"\b{re.escape(value_lc)}\b", norm) and dim_name not in seen_dims:
                filters.append(Filter(dimension=dim_name, op="=", value=canonical))
                seen_dims.add(dim_name)
        # year mentions -> equality filter on dim_date.year
        for y in YEARS:
            if re.search(rf"\b{y}\b", norm) and "year_filter" not in seen_dims:
                filters.append(Filter(dimension="year", op="=", value=y))
                seen_dims.add("year_filter")
        return filters

    # ---- combined -----------------------------------------------------
    def link(self, question: str) -> LinkResult:
        norm = _normalize(question)
        res = LinkResult()
        for phrase, name in self.link_metrics(norm):
            res.metrics.append(name)
            res.matched_spans.append((phrase, f"metric:{name}"))
        for phrase, name in self.link_dimensions(norm):
            res.dimensions.append(name)
            res.matched_spans.append((phrase, f"dimension:{name}"))
        for f in self.link_filters(norm):
            res.filters.append(f)
            res.matched_spans.append((str(f.value), f"filter:{f.dimension}"))
        return res
