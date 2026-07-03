from __future__ import annotations

from nlsql import SchemaLinker


def test_maps_metric_synonyms(model):
    linker = SchemaLinker(model)
    assert "revenue" in linker.link("what is total turnover").metrics
    assert "aov" in linker.link("show me the average order value").metrics
    assert "active_customers" in linker.link("how many unique customers").metrics


def test_maps_groupby_dimensions(model):
    linker = SchemaLinker(model)
    assert "store_region" in linker.link("revenue by region").dimensions
    assert "product_category" in linker.link("sales per category").dimensions
    assert "month" in linker.link("monthly revenue").dimensions


def test_filter_value_linking(model):
    linker = SchemaLinker(model)
    res = linker.link("revenue for Electronics in the West region")
    dims = {f.dimension for f in res.filters}
    assert dims == {"product_category", "store_region"}
    vals = {f.value for f in res.filters}
    assert vals == {"Electronics", "West"}


def test_year_filter_linking(model):
    linker = SchemaLinker(model)
    res = linker.link("revenue in 2023")
    assert any(f.dimension == "year" and f.value == 2023 for f in res.filters)


def test_bare_dimension_only_under_superlative(model):
    linker = SchemaLinker(model)
    # plain "what is revenue in the West region" must NOT group by region
    assert "store_region" not in linker.link(
        "what is total revenue in the West region").dimensions
    # but a superlative makes it a group-by
    assert "store_region" in linker.link(
        "which region has the highest revenue").dimensions


def test_unknown_phrase_links_nothing(model):
    linker = SchemaLinker(model)
    res = linker.link("tell me a joke about penguins")
    assert res.metrics == [] and res.dimensions == [] and res.filters == []
