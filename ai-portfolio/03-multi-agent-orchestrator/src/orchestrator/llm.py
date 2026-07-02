"""The offline "LLM" engine.

:class:`LLMEngine` is the interface an agent talks to.  A real model could
implement it; here :class:`OfflineRuleEngine` provides a deterministic,
pattern-based policy that decomposes a natural-language task into a DAG of tool
calls and repairs a plan from execution feedback.

The engine is intentionally *fast and imperfect* on the first pass — mirroring
how a language model produces plausible-but-sometimes-wrong tool calls: it may
render exponentiation with ``^`` (bitwise-xor), leave units as English words the
converter does not know, pass a code snippet under the wrong argument key, or
return a bare fraction where a percentage was asked for.  The self-correction
loop uses real execution feedback to fix each of these.  It never sees the task
ground truth, so any improvement is earned.
"""
from __future__ import annotations

import copy
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .dag import Step


@dataclass
class Task:
    """A benchmark/task specification.

    ``expected`` and ``output_contract`` are used for *scoring and guarding* only;
    the engine is given the ``prompt`` alone.
    """

    id: str
    prompt: str
    expected: Any = None
    output_contract: str | None = None
    tolerance: float = 1e-6
    category: str = ""


# Non-canonical unit words a model might emit, mapped to the converter's tokens.
UNIT_SYNONYMS: dict[str, str] = {
    "kilometers": "km", "kilometres": "km", "kilometer": "km", "kilometre": "km", "kms": "km",
    "meters": "m", "metres": "m", "meter": "m", "metre": "m",
    "centimeters": "cm", "centimetres": "cm", "centimeter": "cm", "centimetre": "cm",
    "millimeters": "mm", "millimetres": "mm", "millimeter": "mm", "millimetre": "mm",
    "miles": "mi", "mile": "mi",
    "feet": "ft", "foot": "ft",
    "inches": "in", "inch": "in",
    "yards": "yd", "yard": "yd",
    "kilograms": "kg", "kilogram": "kg", "kgs": "kg",
    "grams": "g", "gram": "g",
    "milligrams": "mg", "milligram": "mg",
    "pounds": "lb", "pound": "lb", "lbs": "lb",
    "ounces": "oz", "ounce": "oz",
    "celsius": "c", "fahrenheit": "f", "kelvin": "k",
}


class LLMEngine(ABC):
    """The planning/repair interface an agent depends on (a real LLM slots in)."""

    @abstractmethod
    def plan(self, prompt: str) -> list[Step]:
        """Decompose a task prompt into an ordered list of DAG steps."""

    @abstractmethod
    def repair(self, prompt: str, plan: list[Step], error: dict[str, Any]) -> list[Step]:
        """Return a revised plan given structured execution feedback."""


# --------------------------------------------------------------------------- #
# Clause parsers                                                              #
# --------------------------------------------------------------------------- #

_CONVERT_RE = re.compile(
    r"convert\s+(-?\d+(?:\.\d+)?)\s+([a-zA-Z/]+)\s+(?:to|into)\s+([a-zA-Z/]+)",
    re.IGNORECASE,
)
_CORPUS_RE = re.compile(
    r"(?:search the knowledge base(?:\s+for|:)?|look up)\s+(.+)",
    re.IGNORECASE,
)
_CALC_RE = re.compile(
    r"(?:evaluate the expression|evaluate|compute|calculate)\s*:?\s*(.+)",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(
    r"what percentage of\s+(-?\d+(?:\.\d+)?)\s+is\s+(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_COMBINE_RE = re.compile(
    r",?\s*then\s+(?:(multiply|divide)\s+by\s+(-?\d+(?:\.\d+)?)"
    r"|(add|sum)(?:\s+the\s+two\s+results|\s+all\s+(?:three\s+)?results|\s+them|\s+all)?)"
    r"\s*\.?\s*$",
    re.IGNORECASE,
)
_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)


class PlanningError(ValueError):
    """The engine could not parse a prompt into any known plan."""


class OfflineRuleEngine(LLMEngine):
    """A deterministic rule/pattern policy behind the LLMEngine interface."""

    def plan(self, prompt: str) -> list[Step]:
        p = prompt.strip()

        # 1. Code review (fenced block).
        m = _CODE_FENCE_RE.search(p)
        if m:
            code = m.group(1)
            # First-draft mistake: snippet placed under the wrong argument key.
            return [Step("s1", "code_check", {"snippet": code}, note="code review")]

        # 2. Percentage.
        m = _PERCENT_RE.search(p)
        if m:
            b, a = m.group(1), m.group(2)
            # First-draft mistake: bare fraction, forgetting the x100 scaling.
            return [Step("s1", "calculator", {"expression": f"{a} / {b}"}, note="percent")]

        # 3. Optional trailing combine clause.
        combine = _COMBINE_RE.search(p)
        head = p[: combine.start()] if combine else p

        if combine and (combine.group(3)):  # add / sum -> parallel branches
            segments = re.split(r"\s+and\s+", head.strip(), flags=re.IGNORECASE)
        else:
            segments = [head.strip()]

        steps: list[Step] = []
        for i, seg in enumerate(segments, start=1):
            steps.append(self._parse_segment(f"s{i}", seg))

        if combine:
            op = (combine.group(1) or "").lower()
            k = combine.group(2)
            final_id = f"s{len(steps) + 1}"
            if op in {"multiply", "divide"}:
                symbol = "*" if op == "multiply" else "/"
                expr = f"$s1 {symbol} {k}"
            else:  # add / sum across branches
                expr = " + ".join(f"${s.id}" for s in steps)
            steps.append(Step(final_id, "calculator", {"expression": expr}, note="combine"))

        return steps

    def _parse_segment(self, step_id: str, text: str) -> Step:
        text = text.strip().rstrip("?.")
        m = _CONVERT_RE.search(text)
        if m:
            value, u_from, u_to = m.group(1), m.group(2), m.group(3)
            # First-draft mistake: keep the raw (possibly non-canonical) unit words.
            return Step(
                step_id,
                "unit_convert",
                {"value": float(value), "from_unit": u_from, "to_unit": u_to},
                note="convert",
            )
        m = _CALC_RE.match(text)
        if m and any(c in text for c in "+-*/^0123456789"):
            return Step(step_id, "calculator", {"expression": m.group(1).strip()}, note="calc")
        m = _CORPUS_RE.search(text)
        if m:
            query = m.group(1).strip().rstrip("?.").strip()
            return Step(step_id, "corpus_search", {"query": query, "top_k": 3}, note="lookup")
        raise PlanningError(f"could not parse segment: {text!r}")

    # ------------------------------------------------------------------ #
    # Self-correction                                                    #
    # ------------------------------------------------------------------ #

    def repair(self, prompt: str, plan: list[Step], error: dict[str, Any]) -> list[Step]:
        stage = error.get("stage")
        message = str(error.get("message", ""))
        contract = error.get("contract")
        failed_id = error.get("step_id")
        new_plan = copy.deepcopy(plan)
        by_id = {s.id: s for s in new_plan}

        if stage == "schema" and "code" in message.lower():
            # Move the snippet from the wrong key to the required 'code' key.
            for s in new_plan:
                if s.tool == "code_check":
                    for wrong in ("snippet", "source", "src", "text"):
                        if wrong in s.args:
                            s.args["code"] = s.args.pop(wrong)
            return new_plan

        if stage == "tool" and ("bitwise-xor" in message or "'^'" in message or "^" in message):
            for s in new_plan:
                if s.tool == "calculator" and "^" in str(s.args.get("expression", "")):
                    s.args["expression"] = str(s.args["expression"]).replace("^", "**")
            return new_plan

        if stage == "tool" and "unknown unit" in message:
            for s in new_plan:
                if s.tool == "unit_convert":
                    for key in ("from_unit", "to_unit"):
                        tok = str(s.args.get(key, "")).lower()
                        s.args[key] = UNIT_SYNONYMS.get(tok, tok)
            return new_plan

        if stage == "output" and contract == "percentage":
            s = by_id.get(failed_id)
            if s is not None and s.tool == "calculator":
                s.args["expression"] = f"({s.args['expression']}) * 100"
            return new_plan

        # No known remedy: return the plan unchanged (retries will exhaust).
        return new_plan
