import pytest

from sentinel.graph import SupplyGraph


@pytest.fixture()
def graph(conn):
    return SupplyGraph(conn)


def test_spofs_are_critical_single_sourced_and_ranked(conn, graph):
    spofs = graph.single_points_of_failure()
    assert spofs, "the seeded dataset is designed to contain SPOFs"
    revenues = [s["monthly_revenue_dependent_usd"] for s in spofs]
    assert revenues == sorted(revenues, reverse=True)
    for spof in spofs:
        row = conn.execute(
            "SELECT critical, alternate_supplier_id FROM parts WHERE part_id = ?",
            (spof["part_id"],)).fetchone()
        assert row["critical"] == "yes"
        assert row["alternate_supplier_id"] == ""


def test_criticality_scores_bounded_and_ranked(graph):
    ranking = graph.supplier_criticality(top=40)
    assert ranking
    scores = [s["criticality_score"] for s in ranking]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_outage_zero_shortage_when_cover_exceeds_outage(graph):
    result = graph.simulate_outage(next(iter(graph.suppliers)), outage_days=1)
    # One-day outage can never out-run inventory cover in this dataset.
    assert result["parts_in_shortage"] == 0
    assert result["total_estimated_revenue_loss_usd"] == 0.0


def test_outage_revenue_math_matches_model(conn, graph):
    supplier_id = next(iter(graph.suppliers))
    result = graph.simulate_outage(supplier_id, outage_days=120)

    # Recompute one product's loss by hand from the model definition.
    if result["product_losses"]:
        entry = result["product_losses"][0]
        product = conn.execute(
            "SELECT * FROM products WHERE product_id = ?", (entry["product_id"],)).fetchone()
        expected = round(
            product["unit_price_usd"] * product["monthly_demand_units"]
            * entry["shortage_days"] / 30.0, 2)
        assert entry["estimated_revenue_loss_usd"] == expected

    # Single-sourced parts get no relief: shortage = outage - cover (when positive).
    for part in result["affected_parts"]:
        if part["single_sourced"] and part["shortage_days"] > 0:
            assert part["relief_day"] == 120.0
            assert part["shortage_days"] == round(120.0 - part["days_of_cover"], 1)


def test_outage_longer_outage_never_costs_less(graph):
    supplier_id = max(
        graph.suppliers,
        key=lambda sid: sum(
            1 for n in graph.parts.values() if n.primary_supplier_id == sid),
    )
    short = graph.simulate_outage(supplier_id, outage_days=30)
    long = graph.simulate_outage(supplier_id, outage_days=90)
    assert (long["total_estimated_revenue_loss_usd"]
            >= short["total_estimated_revenue_loss_usd"])


def test_outage_unknown_supplier(graph):
    assert "error" in graph.simulate_outage("SUP-999", 30)


def test_new_tools_registered(conn):
    from sentinel.tools import ToolBox

    box = ToolBox(conn)
    assert "network_risk_profile" in box.registry
    assert "simulate_supplier_outage" in box.registry
    import json as _json

    profile = _json.loads(box.execute("network_risk_profile", {}))
    assert profile["single_points_of_failure"]
    assert profile["critical_suppliers"]
