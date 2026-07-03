"""Calculator correctness and AST-safety (rejects unsafe/bad input)."""
from __future__ import annotations

import math

import pytest

from orchestrator.tools import CalculatorTool, SafeCalculator, ToolError


@pytest.fixture
def calc():
    return SafeCalculator()


@pytest.mark.parametrize("expr,expected", [
    ("12 * (3 + 4) / 2", 42.0),
    ("2 ** 10 + 5", 1029.0),
    ("sqrt(144) + 8", 20.0),
    ("max(3, 9, 5) * 2", 18.0),
    ("-5 + 10", 5.0),
    ("10 % 3", 1.0),
    ("abs(-7)", 7.0),
    ("round(3.14159, 2)", 3.14),
    ("pi * 0", 0.0),
])
def test_valid_expressions(calc, expr, expected):
    assert math.isclose(calc.eval(expr), expected, rel_tol=1e-9)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('ls')",   # attribute/call injection
    "os.system('rm -rf /')",           # unknown name
    "eval('2+2')",                      # function not allowed
    "open('/etc/passwd')",             # function not allowed
    "[i for i in range(3)]",           # comprehension
    "lambda: 1",                        # lambda
    "1; 2",                             # multiple statements (syntax)
    "x + 1",                            # unknown name
    "2 & 3",                            # bitwise-and not allowed
    "(1,2,3)",                          # tuple literal
])
def test_unsafe_expressions_rejected(calc, expr):
    with pytest.raises(ToolError):
        calc.eval(expr)


def test_caret_is_rejected_with_helpful_message(calc):
    with pytest.raises(ToolError) as exc:
        calc.eval("2 ^ 10")
    assert "**" in str(exc.value)  # tells the caller to use ** for power


def test_no_eval_used_in_source():
    import ast
    import inspect

    import orchestrator.tools as tools_mod

    # Parse the tool source and assert the builtins eval()/exec() are never
    # *called* (calls to a bare Name 'eval'/'exec'). Method names are fine.
    tree = ast.parse(inspect.getsource(tools_mod))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"eval", "exec"}


def test_division_by_zero_raises():
    tool = CalculatorTool()
    with pytest.raises((ToolError, ZeroDivisionError)):
        tool.run({"expression": "1 / 0"})


def test_exponent_guard():
    calc = SafeCalculator()
    with pytest.raises(ToolError):
        calc.eval("10 ** 100000")


def test_tool_returns_value():
    out = CalculatorTool().run({"expression": "3 + 4"})
    assert out["value"] == 7.0
