"""The tool layer.

A :class:`Tool` has a name, a human description, a typed argument ``schema``,
and a ``run`` method.  Schema validation lives in :mod:`guardrails`; tools
themselves raise :class:`ToolError` for domain errors (bad unit, unsafe
expression, ...) which the executor treats as a self-correction trigger.

Four concrete tools are provided:

  * :class:`CalculatorTool`   — AST-based safe arithmetic (never uses ``eval``)
  * :class:`UnitConverterTool`— length / mass / temperature conversion
  * :class:`CorpusSearchTool` — TF-IDF retrieval over a synthetic knowledge base
  * :class:`CodeCheckTool`    — AST-based static lint of a Python snippet
"""
from __future__ import annotations

import ast
import math
import operator
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .knowledge import KNOWLEDGE_BASE, KBDoc


class ToolError(RuntimeError):
    """Raised by a tool for a recoverable, domain-level failure."""


@dataclass
class Tool:
    """Base tool.  Subclasses set ``name``/``description``/``schema`` and
    override :meth:`run`.

    ``schema`` maps argument name -> spec dict with keys ``type`` (a Python
    type) and optional ``required`` (default ``True``) and ``default``.
    """

    name: str = "tool"
    description: str = ""
    schema: dict[str, dict[str, Any]] = None  # type: ignore[assignment]

    def run(self, args: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 1. Safe calculator (AST, no eval)                                           #
# --------------------------------------------------------------------------- #

_BIN_OPS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_CONSTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}
_FUNCS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": lambda x, n=0: round(x, int(n)),
    "min": min,
    "max": max,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}
_MAX_POW = 1_000  # guard against 10**10**10 style blowups


class SafeCalculator:
    """Evaluate a numeric expression by walking a restricted AST.

    Only literals, the whitelisted binary/unary operators, whitelisted function
    calls and named constants are permitted.  Anything else — attribute access,
    names, comprehensions, the bitwise-xor ``^`` operator (a common "power"
    mistake), etc. — raises :class:`ToolError`.
    """

    def eval(self, expr: str) -> float:
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise ToolError(f"syntax error in expression: {exc.msg}") from exc
        return self._eval(tree.body)

    def _eval(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                node.value, (int, float)
            ):
                raise ToolError(f"non-numeric constant: {node.value!r}")
            return float(node.value)
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type is ast.BitXor:
                raise ToolError(
                    "'^' is bitwise-xor, not power; use '**' for exponentiation"
                )
            if op_type not in _BIN_OPS:
                raise ToolError(f"operator not allowed: {op_type.__name__}")
            left, right = self._eval(node.left), self._eval(node.right)
            if op_type is ast.Pow and abs(right) > _MAX_POW:
                raise ToolError("exponent too large")
            return _BIN_OPS[op_type](left, right)
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _UNARY_OPS:
                raise ToolError(f"unary operator not allowed: {op_type.__name__}")
            return _UNARY_OPS[op_type](self._eval(node.operand))
        if isinstance(node, ast.Name):
            if node.id in _CONSTS:
                return _CONSTS[node.id]
            raise ToolError(f"unknown name: {node.id!r}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
                fname = getattr(node.func, "id", type(node.func).__name__)
                raise ToolError(f"function not allowed: {fname!r}")
            if node.keywords:
                raise ToolError("keyword arguments are not allowed")
            fargs = [self._eval(a) for a in node.args]
            return float(_FUNCS[node.func.id](*fargs))
        raise ToolError(f"disallowed syntax: {type(node).__name__}")


class CalculatorTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="calculator",
            description="Evaluate an arithmetic expression safely (AST, no eval).",
            schema={"expression": {"type": str, "required": True}},
        )
        self._calc = SafeCalculator()

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        value = self._calc.eval(str(args["expression"]))
        if not math.isfinite(value):
            raise ToolError("result is not finite")
        return {"value": value, "expression": args["expression"]}


# --------------------------------------------------------------------------- #
# 2. Unit converter                                                           #
# --------------------------------------------------------------------------- #

# Canonical linear units expressed in a base unit.
_LENGTH_M = {"m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001,
             "mi": 1609.344, "ft": 0.3048, "in": 0.0254, "yd": 0.9144}
_MASS_KG = {"kg": 1.0, "g": 0.001, "mg": 1e-6, "lb": 0.45359237, "oz": 0.0283495231}


class UnitConverterTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="unit_convert",
            description="Convert a value between units (length, mass, temperature).",
            schema={
                "value": {"type": (int, float), "required": True},
                "from_unit": {"type": str, "required": True},
                "to_unit": {"type": str, "required": True},
            },
        )

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        value = float(args["value"])
        src = str(args["from_unit"]).strip()
        dst = str(args["to_unit"]).strip()
        result = self._convert(value, src, dst)
        return {"value": result, "from_unit": src, "to_unit": dst}

    def _convert(self, value: float, src: str, dst: str) -> float:
        if src in _LENGTH_M and dst in _LENGTH_M:
            return value * _LENGTH_M[src] / _LENGTH_M[dst]
        if src in _MASS_KG and dst in _MASS_KG:
            return value * _MASS_KG[src] / _MASS_KG[dst]
        temps = {"c", "f", "k"}
        if src in temps and dst in temps:
            return self._temp(value, src, dst)
        for token in (src, dst):
            known = set(_LENGTH_M) | set(_MASS_KG) | temps
            if token not in known:
                raise ToolError(f"unknown unit: {token!r}")
        raise ToolError(f"incompatible units: {src!r} -> {dst!r}")

    @staticmethod
    def _temp(value: float, src: str, dst: str) -> float:
        # to Celsius
        c = {"c": value, "f": (value - 32.0) * 5.0 / 9.0, "k": value - 273.15}[src]
        return {"c": c, "f": c * 9.0 / 5.0 + 32.0, "k": c + 273.15}[dst]


# --------------------------------------------------------------------------- #
# 3. Corpus search (TF-IDF over the synthetic KB)                             #
# --------------------------------------------------------------------------- #

class CorpusSearchTool(Tool):
    def __init__(self, docs: list[KBDoc] | None = None) -> None:
        super().__init__(
            name="corpus_search",
            description="Retrieve the best-matching fact from the knowledge base.",
            schema={
                "query": {"type": str, "required": True},
                "top_k": {"type": int, "required": False, "default": 3},
            },
        )
        self._docs = list(docs if docs is not None else KNOWLEDGE_BASE)
        self._corpus = [
            f"{d.text} {' '.join(d.keywords)}" for d in self._docs
        ]
        self._vec = TfidfVectorizer(stop_words="english")
        self._matrix = self._vec.fit_transform(self._corpus)

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args["query"]).strip()
        if not query:
            raise ToolError("empty query")
        top_k = int(args.get("top_k", 3) or 3)
        q = self._vec.transform([query])
        sims = cosine_similarity(q, self._matrix).ravel()
        order = np.argsort(-sims)[: max(1, top_k)]
        hits = [
            {
                "doc_id": self._docs[i].doc_id,
                "text": self._docs[i].text,
                "number": self._docs[i].number,
                "unit": self._docs[i].unit,
                "score": float(sims[i]),
            }
            for i in order
        ]
        best = hits[0]
        if best["score"] <= 0.0:
            raise ToolError(f"no knowledge-base match for query: {query!r}")
        return {
            "value": best["number"],
            "number": best["number"],
            "text": best["text"],
            "unit": best["unit"],
            "hits": hits,
        }


# --------------------------------------------------------------------------- #
# 4. Code check (AST static lint)                                             #
# --------------------------------------------------------------------------- #

class _LintVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.issues: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
            self.issues.append(f"use of `{node.func.id}` is unsafe")
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.issues.append("bare `except:` swallows all errors")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for op, comp in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Eq, ast.NotEq)) and _is_none(comp):
                self.issues.append("comparison to None should use `is`/`is not`")
        self.generic_visit(node)

    def _check_defaults(self, node: ast.AST) -> None:
        for default in getattr(node.args, "defaults", []):  # type: ignore[attr-defined]
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.issues.append("mutable default argument")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_defaults(node)
        self.generic_visit(node)


def _is_none(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


class CodeCheckTool(Tool):
    def __init__(self) -> None:
        super().__init__(
            name="code_check",
            description="Statically lint a Python snippet for common defects.",
            schema={"code": {"type": str, "required": True}},
        )

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        code = str(args["code"])
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return {"value": 1, "ok": False, "issues": [f"syntax error: {exc.msg}"]}
        visitor = _LintVisitor()
        visitor.visit(tree)
        issues = visitor.issues
        return {"value": len(issues), "ok": len(issues) == 0, "issues": issues}


# --------------------------------------------------------------------------- #

def default_toolbox() -> dict[str, Tool]:
    """Return a fresh, name-keyed registry of the four standard tools."""
    tools = [CalculatorTool(), UnitConverterTool(), CorpusSearchTool(), CodeCheckTool()]
    return {t.name: t for t in tools}
