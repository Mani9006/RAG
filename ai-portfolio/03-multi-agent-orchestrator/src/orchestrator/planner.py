"""Planner: turns an engine's step list into a validated, schedulable DAG.

Thin by design — the intelligence lives in the :class:`LLMEngine`.  The planner
constructs a :class:`DAG` (which validates references and acyclicity and derives
the parallel execution schedule) and forwards repair requests to the engine.
"""
from __future__ import annotations

from typing import Any

from .dag import DAG, Step
from .llm import LLMEngine


class Planner:
    def __init__(self, engine: LLMEngine):
        self.engine = engine

    def plan(self, prompt: str) -> tuple[list[Step], DAG]:
        steps = self.engine.plan(prompt)
        return steps, DAG(steps)

    def repair(self, prompt: str, steps: list[Step],
               error: dict[str, Any]) -> tuple[list[Step], DAG]:
        steps = self.engine.repair(prompt, steps, error)
        return steps, DAG(steps)
