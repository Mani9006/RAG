import pytest

from sentinel.graph import SupplyGraph
from sentinel.montecarlo import (
    MonteCarloEngine,
    ScenarioSpec,
    Triangular,
    load_scenarios,
)


@pytest.fixture()
def graph(conn):
    return SupplyGraph(conn)


@pytest.fixture()
def engine(graph):
    return MonteCarloEngine(graph, seed=90061)


def test_scenario_library_loads_and_selects(graph):
    library = load_scenarios()
    assert {"taiwan-strait", "korea-battery-fire",
            "europe-port-strike", "mexico-border-closure"} <= set(library)

    taiwan = library["taiwan-strait"].resolve_suppliers(graph)
    assert taiwan, "taiwan-strait must select suppliers"
    for sid in taiwan:
        supplier = graph.suppliers[sid]
        assert supplier["country"] == "Taiwan" or supplier["primary_port"] == "Kaohsiung"

    korea = library["korea-battery-fire"].resolve_suppliers(graph)
    assert len(korea) == 1
    assert graph.suppliers[korea[0]]["name"] == "Gumi PowerCells"


def test_runs_are_seeded_and_reproducible(engine, graph):
    spec = load_scenarios()["taiwan-strait"]
    first = engine.run_scenario(spec, trials=300)
    second = MonteCarloEngine(graph, seed=90061).run_scenario(spec, trials=300)
    assert first == second

    different_seed = MonteCarloEngine(graph, seed=7).run_scenario(spec, trials=300)
    assert different_seed["loss_usd"] != first["loss_usd"]


def test_quantiles_are_ordered(engine):
    spec = load_scenarios()["taiwan-strait"]
    result = engine.run_scenario(spec, trials=500)
    loss = result["loss_usd"]
    assert 0 <= loss["p50"] <= loss["p90"] <= loss["p95"] <= loss["max"]
    assert loss["expected"] > 0
    probs = result["probability_loss_exceeds"]
    assert probs["1m_usd"] >= probs["10m_usd"] >= probs["50m_usd"]


def test_longer_outages_dominate(engine, graph):
    suppliers = [graph.suppliers[next(iter(graph.suppliers))]["name"]]
    short = ScenarioSpec(
        name="short", description="", outage_days=Triangular(5, 10, 15),
        supplier_names=suppliers)
    long = ScenarioSpec(
        name="long", description="", outage_days=Triangular(60, 90, 150),
        supplier_names=suppliers)
    short_result = engine.run_scenario(short, trials=300)
    long_result = engine.run_scenario(long, trials=300)
    assert (long_result["loss_usd"]["expected"]
            >= short_result["loss_usd"]["expected"])


def test_alternates_inside_blast_radius_give_no_relief(graph):
    """A multi-supplier outage must not let an affected alternate 'rescue' a part."""
    part = next(
        node for node in graph.parts.values()
        if not node.single_sourced and node.alternate_supplier_id in graph.suppliers
    )
    both = graph.simulate_multi_outage(
        [part.primary_supplier_id, part.alternate_supplier_id], outage_days=120)
    entry = next(p for p in both["affected_parts"] if p["part_id"] == part.part_id)
    assert entry["relief_day"] == 120.0  # alternate is also dark


def test_rank_options_paired_trials(engine, graph):
    spec = load_scenarios()["korea-battery-fire"]
    supplier_ids = spec.resolve_suppliers(graph)
    part = next(
        node for node in graph.parts.values()
        if node.primary_supplier_id == supplier_ids[0] and node.products
    )
    options = [
        {"kind": "expedite_freight", "description": "air freight",
         "part_id": part.part_id, "supplier_id": supplier_ids[0],
         "value_usd": 50_000.0, "lead_time_days": 5, "tradeoffs": ""},
        {"kind": "monitor_only", "description": "watch", "part_id": None,
         "supplier_id": None, "value_usd": 0.0, "lead_time_days": 0, "tradeoffs": ""},
    ]
    ranked = engine.rank_options(supplier_ids, options, spec, trials=100)
    by_kind = {o["kind"]: o for o in ranked}
    assert by_kind["monitor_only"]["expected_loss_reduction_usd"] == 0.0
    assert by_kind["expedite_freight"]["expected_loss_reduction_usd"] >= 0.0
    # Ranking is by reduction per dollar, descending.
    scores = [o["reduction_per_dollar"] for o in ranked]
    assert scores == sorted(scores, reverse=True)


def test_pipeline_briefing_includes_uncertainty_band(conn):
    from sentinel.orchestrator import Pipeline

    Pipeline(conn).run()
    row = conn.execute(
        "SELECT briefing_md FROM incidents WHERE event_id = 'EVT-2026-0602'"
    ).fetchone()
    assert row is not None
    assert "Probabilistic loss" in row["briefing_md"]
    assert "P90" in row["briefing_md"]


def test_scenario_tool_registered(conn):
    import json

    from sentinel.tools import ToolBox

    box = ToolBox(conn)
    result = json.loads(box.execute(
        "run_disruption_scenario", {"scenario_name": "korea-battery-fire", "trials": 200}))
    assert result["loss_usd"]["p90"] >= result["loss_usd"]["p50"]

    unknown = json.loads(box.execute(
        "run_disruption_scenario", {"scenario_name": "nope"}))
    assert "error" in unknown and "taiwan-strait" in unknown["available"]
