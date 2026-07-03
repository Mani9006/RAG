"""The benchmark task suite (~30 multi-step tasks).

Every task carries an *independently computed* ground-truth ``expected`` value.
Ground truth is computed here with plain Python reference functions — never via
the orchestrator's tools — so the success measurement is honest.

The suite deliberately spans:

  * clean single-step tasks that succeed on the first plan,
  * tasks whose first plan contains a realistic model-style mistake that only
    the self-correction loop can recover (``^`` for power, English unit words,
    a code snippet under the wrong arg key, a bare fraction for a percentage),
  * multi-step tasks with dependencies and independent (parallelisable) branches.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orchestrator.knowledge import KNOWLEDGE_BASE  # noqa: E402
from orchestrator.llm import Task  # noqa: E402

# Reference (oracle) unit factors, independent of the tool implementation.
_LEN = {"m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001,
        "mi": 1609.344, "ft": 0.3048, "in": 0.0254, "yd": 0.9144}
_MASS = {"kg": 1.0, "g": 0.001, "mg": 1e-6, "lb": 0.45359237, "oz": 0.0283495231}
_KB = {d.doc_id: d.number for d in KNOWLEDGE_BASE}


def _len(v, a, b):
    return v * _LEN[a] / _LEN[b]


def _mass(v, a, b):
    return v * _MASS[a] / _MASS[b]


def build_suite() -> list[Task]:
    t: list[Task] = []
    n = 0

    def add(prompt, expected, contract=None, category="", tol=1e-6):
        nonlocal n
        n += 1
        t.append(Task(id=f"T{n:02d}", prompt=prompt, expected=expected,
                      output_contract=contract, category=category, tolerance=tol))

    # --- clean calculator (succeed first try) --------------------------------
    add("Evaluate the expression: 12 * (3 + 4) / 2", 42.0, category="calc")
    add("Evaluate the expression: (100 - 55) / 9", 5.0, category="calc")
    add("Evaluate the expression: sqrt(144) + 8", 20.0, category="calc")
    add("Evaluate the expression: max(3, 9, 5) * 2", 18.0, category="calc")

    # --- calculator with the '^' power mistake (needs self-correction) -------
    add("Evaluate the expression: 2 ^ 10 + 5", 2 ** 10 + 5, category="calc_pow")
    add("Evaluate the expression: 3 ^ 4 - 1", 3 ** 4 - 1, category="calc_pow")
    add("Evaluate the expression: 10 ^ 3 / 8", 10 ** 3 / 8, category="calc_pow")

    # --- unit conversion, canonical tokens (succeed first try) ---------------
    add("Convert 5 km to mi.", _len(5, "km", "mi"), category="convert")
    add("Convert 2500 g to kg.", _mass(2500, "g", "kg"), category="convert")

    # --- unit conversion, English words (needs self-correction) --------------
    add("Convert 10 kilometers to miles.", _len(10, "km", "mi"), category="convert")
    add("Convert 3 pounds to grams.", _mass(3, "lb", "g"), category="convert")
    add("Convert 100 centimeters to meters.", _len(100, "cm", "m"), category="convert")
    add("Convert 6 feet to inches.", _len(6, "ft", "in"), category="convert")

    # --- convert then arithmetic (dependency; word units -> self-correction) -
    add("Convert 10 kilometers to miles, then multiply by 3",
        _len(10, "km", "mi") * 3, category="convert_calc")
    add("Convert 500 grams to kilograms, then multiply by 4",
        _mass(500, "g", "kg") * 4, category="convert_calc")

    # --- percentage (bare-fraction mistake -> output guard -> self-correction)
    add("What percentage of 8 is 3?", 3 / 8 * 100, contract="percentage",
        category="percent")
    add("What percentage of 200 is 50?", 50 / 200 * 100, contract="percentage",
        category="percent")
    add("What percentage of 40 is 9?", 9 / 40 * 100, contract="percentage",
        category="percent")

    # --- corpus lookup (succeed first try) -----------------------------------
    add("Search the knowledge base: boiling point of water?",
        _KB["kb-boil"], category="corpus")
    add("Search the knowledge base: speed of light in a vacuum?",
        _KB["kb-light"], category="corpus")
    add("Search the knowledge base: acceleration due to gravity on Earth?",
        _KB["kb-gravity"], category="corpus")
    add("Search the knowledge base: mean radius of the Earth?",
        _KB["kb-radius"], category="corpus")
    add("Search the knowledge base: speed of sound in air?",
        _KB["kb-sound"], category="corpus")

    # --- corpus then arithmetic (dependency) ---------------------------------
    add("Search the knowledge base for mean radius of the Earth, then divide by 2",
        _KB["kb-radius"] / 2, category="corpus_calc")
    add("Search the knowledge base for days in a calendar year, then divide by 5",
        _KB["kb-year"] / 5, category="corpus_calc")

    # --- code review (wrong arg key -> schema guard -> self-correction) -------
    add("Review this Python code:\n```\n"
        "def f(x):\n    return eval(x)\n```", 1, category="code")
    add("Review this Python code:\n```\n"
        "def g(items=[]):\n    try:\n        return items[0]\n    except:\n"
        "        return None\n```", 2, category="code")
    add("Review this Python code:\n```\n"
        "def h(x):\n    if x == None:\n        return 0\n    return x\n```",
        1, category="code")

    # --- parallel branches (independent steps; word units -> self-correction) -
    add("Convert 100 centimeters to meters and convert 2 kilograms to pounds, "
        "then add the two results",
        _len(100, "cm", "m") + _mass(2, "kg", "lb"), category="parallel")
    add("Convert 5 kilometers to meters and convert 3 miles to kilometers, "
        "then add the two results",
        _len(5, "km", "m") + _len(3, "mi", "km"), category="parallel")

    # --- showcase: 3 independent branches + a combine (rich DAG for the Gantt) -
    add("Convert 12 kilometers to miles and convert 4 kilograms to pounds and "
        "search the knowledge base for speed of sound in air, then add all three results",
        _len(12, "km", "mi") + _mass(4, "kg", "lb") + _KB["kb-sound"],
        category="showcase")

    return t


SUITE = build_suite()

if __name__ == "__main__":
    print(f"{len(SUITE)} tasks")
    for task in SUITE:
        print(f"  {task.id}  [{task.category:12s}] {task.prompt[:60]!r}")
