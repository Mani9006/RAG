"""Benchmark harness: run the task suite and aggregate honest metrics.

Shared by ``scripts/run_benchmark.py`` (writes tables) and
``scripts/make_screenshots.py`` (renders charts) so both report identical
numbers from the same deterministic run.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orchestrator.agent import Agent, AgentConfig  # noqa: E402
from orchestrator.executor import ExecutionTrace  # noqa: E402

from tasks import SUITE  # noqa: E402


def scored(trace: ExecutionTrace, task) -> bool:
    """True iff the task executed successfully AND matched ground truth."""
    if not trace.success:
        return False
    exp, got = task.expected, trace.final_answer
    if isinstance(exp, (int, float)) and isinstance(got, (int, float)):
        return abs(got - exp) <= max(task.tolerance, abs(exp) * 1e-6)
    return got == exp


@dataclass
class Condition:
    name: str
    self_correct: bool
    traces: list[ExecutionTrace] = field(default_factory=list)
    correct: list[bool] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.traces)

    @property
    def solved(self) -> int:
        return sum(self.correct)

    @property
    def success_rate(self) -> float:
        return self.solved / self.n if self.n else 0.0

    @property
    def avg_steps(self) -> float:
        return sum(t.n_steps for t in self.traces) / self.n if self.n else 0.0

    @property
    def avg_attempts(self) -> float:
        return sum(t.attempts for t in self.traces) / self.n if self.n else 0.0

    @property
    def avg_cost(self) -> float:
        return sum(t.total_cost for t in self.traces) / self.n if self.n else 0.0

    @property
    def avg_tokens(self) -> float:
        return sum(t.total_tokens for t in self.traces) / self.n if self.n else 0.0

    def tool_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.traces:
            for tool, c in t.tool_counts().items():
                counts[tool] = counts.get(tool, 0) + c
        return counts


def run_condition(name: str, self_correct: bool,
                  suite: list | None = None) -> Condition:
    suite = suite if suite is not None else SUITE
    cond = Condition(name=name, self_correct=self_correct)
    agent = Agent(config=AgentConfig(self_correct=self_correct, max_retries=3))
    for task in suite:
        tr = agent.solve(task)
        cond.traces.append(tr)
        cond.correct.append(scored(tr, task))
    return cond


def run_all(suite: list | None = None) -> dict[str, Condition]:
    return {
        "without": run_condition("Without self-correction", False, suite),
        "with": run_condition("With self-correction", True, suite),
    }


def showcase_trace(suite: list | None = None) -> ExecutionTrace:
    """Return the trace of the richest multi-branch task (for the Gantt view)."""
    suite = suite if suite is not None else SUITE
    task = next(t for t in suite if t.category == "showcase")
    agent = Agent(config=AgentConfig(self_correct=True))
    return agent.solve(task)
