"""DAG topological correctness and cycle detection."""
from __future__ import annotations

import pytest

from orchestrator.dag import DAG, CycleError, Ref, Step


def test_levels_respect_dependencies():
    steps = [
        Step("s1", "calculator", {"expression": "1 + 1"}),
        Step("s2", "calculator", {"expression": "2 + 2"}),
        Step("s3", "calculator", {"expression": "$s1 + $s2"}),
        Step("s4", "calculator", {"expression": "$s3 * 10"}),
    ]
    dag = DAG(steps)
    order = dag.topo_order()
    pos = {sid: i for i, sid in enumerate(order)}
    # Every dependency must appear before its dependent.
    for s in steps:
        for dep in s.all_deps():
            assert pos[dep] < pos[s.id]
    # s1 and s2 are independent -> same (first) level; parallel width 2.
    assert set(dag.levels[0]) == {"s1", "s2"}
    assert dag.levels[1] == ["s3"]
    assert dag.levels[2] == ["s4"]
    assert dag.width() == 2


def test_string_refs_create_edges():
    steps = [
        Step("s1", "corpus_search", {"query": "gravity"}),
        Step("s2", "calculator", {"expression": "$s1.number / 2"}),
    ]
    dag = DAG(steps)
    assert dag.steps["s2"].all_deps() == {"s1"}
    assert dag.topo_order() == ["s1", "s2"]


def test_ref_objects_create_edges():
    steps = [
        Step("a", "calculator", {"expression": "3 * 3"}),
        Step("b", "unit_convert", {"value": Ref("a"), "from_unit": "m", "to_unit": "km"}),
    ]
    dag = DAG(steps)
    assert dag.sinks() == ["b"]


def test_cycle_is_rejected():
    steps = [
        Step("s1", "calculator", {"expression": "$s2 + 1"}),
        Step("s2", "calculator", {"expression": "$s1 + 1"}),
    ]
    with pytest.raises(CycleError):
        DAG(steps)


def test_self_dependency_is_rejected():
    steps = [Step("s1", "calculator", {"expression": "$s1 + 1"})]
    with pytest.raises(CycleError):
        DAG(steps)


def test_unknown_dependency_is_rejected():
    steps = [Step("s1", "calculator", {"expression": "$sX + 1"})]
    with pytest.raises(ValueError):
        DAG(steps)


def test_duplicate_id_rejected():
    with pytest.raises(ValueError):
        DAG([Step("s1", "calculator", {}), Step("s1", "calculator", {})])


def test_three_way_parallel_width():
    steps = [
        Step("s1", "unit_convert", {"value": 1.0, "from_unit": "km", "to_unit": "m"}),
        Step("s2", "unit_convert", {"value": 2.0, "from_unit": "kg", "to_unit": "lb"}),
        Step("s3", "corpus_search", {"query": "sound"}),
        Step("s4", "calculator", {"expression": "$s1 + $s2 + $s3"}),
    ]
    dag = DAG(steps)
    assert dag.width() == 3
    assert dag.sinks() == ["s4"]
