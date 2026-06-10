import json

from sentinel.tools import ToolBox


def test_registry_exposes_anthropic_schemas(conn):
    box = ToolBox(conn)
    schemas = box.schemas()
    assert len(schemas) == 11
    for schema in schemas:
        assert set(schema) == {"name", "description", "input_schema"}
        assert schema["input_schema"]["type"] == "object"


def test_find_suppliers_by_name_and_country(conn):
    box = ToolBox(conn)
    hits = json.loads(box.execute("find_suppliers", {"query": "Gumi"}))
    assert len(hits) == 1
    assert hits[0]["name"] == "Gumi PowerCells"

    taiwan = json.loads(box.execute("find_suppliers", {"country": "Taiwan"}))
    assert taiwan and all(s["country"] == "Taiwan" for s in taiwan)


def test_supplier_exposure_aggregates(conn):
    box = ToolBox(conn)
    sid = json.loads(box.execute("find_suppliers", {"query": "Gumi"}))[0]["supplier_id"]
    exposure = json.loads(box.execute("supplier_exposure", {"supplier_id": sid}))
    assert exposure["supplier_id"] == sid
    assert exposure["open_po_value_usd"] == round(
        sum(p["value_usd"] for p in exposure["open_pos"]), 2
    )


def test_parts_blast_radius_revenue_math(conn):
    box = ToolBox(conn)
    part_id = conn.execute("SELECT part_id FROM bom LIMIT 1").fetchone()["part_id"]
    result = json.loads(box.execute("parts_blast_radius", {"part_ids": [part_id]}))
    entry = result[0]
    expected = round(
        sum(p["unit_price_usd"] * p["monthly_demand_units"] for p in entry["affected_products"]), 2
    )
    assert entry["monthly_revenue_at_risk_usd"] == expected


def test_expedite_cost_floor(conn):
    box = ToolBox(conn)
    small = json.loads(box.execute("estimate_expedite_cost", {"units": 10, "unit_cost_usd": 1.0}))
    assert small["air_freight_premium_usd"] == 8000.0
    big = json.loads(box.execute("estimate_expedite_cost", {"units": 100000, "unit_cost_usd": 10.0}))
    assert big["air_freight_premium_usd"] == round(100000 * 10.0 * 0.18, 2)


def test_unknown_tool_and_tool_error_are_json(conn):
    box = ToolBox(conn)
    assert "error" in json.loads(box.execute("nope", {}))
    assert "error" in json.loads(box.execute("inventory_position", {"part_id": "PART-9999"}))
