"""Unit converter, corpus search, and code-check tool behavior."""
from __future__ import annotations

import math

import pytest

from orchestrator.tools import (
    CodeCheckTool,
    CorpusSearchTool,
    ToolError,
    UnitConverterTool,
    default_toolbox,
)


def test_default_toolbox_has_all_tools():
    box = default_toolbox()
    assert set(box) == {"calculator", "unit_convert", "corpus_search", "code_check"}


@pytest.mark.parametrize("v,a,b,expected", [
    (5, "km", "mi", 5 * 1000.0 / 1609.344),
    (100, "cm", "m", 1.0),
    (1, "kg", "g", 1000.0),
    (0, "c", "k", 273.15),
    (100, "c", "f", 212.0),
    (32, "f", "c", 0.0),
])
def test_unit_conversions(v, a, b, expected):
    out = UnitConverterTool().run({"value": v, "from_unit": a, "to_unit": b})
    assert math.isclose(out["value"], expected, rel_tol=1e-9, abs_tol=1e-9)


def test_unknown_unit_raises():
    with pytest.raises(ToolError):
        UnitConverterTool().run({"value": 5, "from_unit": "miles", "to_unit": "km"})


def test_incompatible_units_raise():
    with pytest.raises(ToolError):
        UnitConverterTool().run({"value": 5, "from_unit": "km", "to_unit": "kg"})


def test_corpus_search_retrieves_fact():
    out = CorpusSearchTool().run({"query": "speed of light in a vacuum", "top_k": 3})
    assert out["number"] == 299792458.0
    assert len(out["hits"]) == 3
    assert out["hits"][0]["score"] >= out["hits"][1]["score"]


def test_corpus_empty_query_raises():
    with pytest.raises(ToolError):
        CorpusSearchTool().run({"query": "   "})


def test_code_check_flags_eval():
    out = CodeCheckTool().run({"code": "def f(x):\n    return eval(x)\n"})
    assert out["value"] == 1
    assert out["ok"] is False


def test_code_check_flags_multiple():
    code = "def g(items=[]):\n    try:\n        return items[0]\n    except:\n        return None\n"
    out = CodeCheckTool().run({"code": code})
    assert out["value"] == 2  # mutable default + bare except


def test_code_check_clean_code():
    out = CodeCheckTool().run({"code": "def add(a, b):\n    return a + b\n"})
    assert out["ok"] is True
    assert out["value"] == 0


def test_code_check_none_comparison():
    out = CodeCheckTool().run({"code": "def h(x):\n    return x == None\n"})
    assert out["value"] == 1


def test_code_check_syntax_error():
    out = CodeCheckTool().run({"code": "def broken(:\n"})
    assert out["ok"] is False
