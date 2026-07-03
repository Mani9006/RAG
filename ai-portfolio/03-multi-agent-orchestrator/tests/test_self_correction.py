"""End-to-end: self-correction improves success; individual repair paths work."""
from __future__ import annotations

from orchestrator.agent import Agent, AgentConfig
from orchestrator.llm import Task

from runner import run_condition
from tasks import SUITE


def _score(trace, task):
    if not trace.success:
        return False
    e, g = task.expected, trace.final_answer
    if isinstance(e, (int, float)) and isinstance(g, (int, float)):
        return abs(g - e) <= max(task.tolerance, abs(e) * 1e-6)
    return g == e


def test_self_correction_improves_success_rate():
    without = run_condition("without", False, SUITE)
    with_ = run_condition("with", True, SUITE)
    # The loop must strictly improve outcomes, and materially so.
    assert with_.success_rate > without.success_rate
    assert with_.success_rate - without.success_rate >= 0.25
    assert with_.success_rate == 1.0  # every task solved with correction


def test_caret_power_recovered():
    task = Task("p", "Evaluate the expression: 2 ^ 10 + 5", expected=1029.0)
    on = Agent(config=AgentConfig(self_correct=True)).solve(task)
    off = Agent(config=AgentConfig(self_correct=False)).solve(task)
    assert on.success and _score(on, task)
    assert on.attempts >= 2 and on.self_corrected
    assert not off.success  # fails without correction (calculator rejects '^')


def test_english_units_recovered():
    task = Task("u", "Convert 10 kilometers to miles.",
                expected=10 * 1000.0 / 1609.344)
    on = Agent(config=AgentConfig(self_correct=True)).solve(task)
    off = Agent(config=AgentConfig(self_correct=False)).solve(task)
    assert on.success and _score(on, task)
    assert not off.success


def test_schema_error_recovered_for_code_task():
    task = Task("c", "Review this Python code:\n```\ndef f(x):\n    return eval(x)\n```",
                expected=1)
    on = Agent(config=AgentConfig(self_correct=True)).solve(task)
    off = Agent(config=AgentConfig(self_correct=False)).solve(task)
    # First plan puts the snippet under the wrong key -> schema guard fires.
    assert off.success is False
    assert any("code" in e for e in off.errors)
    assert on.success and on.final_answer == 1


def test_percentage_output_guard_recovered():
    task = Task("pct", "What percentage of 8 is 3?", expected=37.5,
                output_contract="percentage")
    on = Agent(config=AgentConfig(self_correct=True)).solve(task)
    off = Agent(config=AgentConfig(self_correct=False)).solve(task)
    assert off.success is False              # 0.375 tripped the output guard
    assert on.success and abs(on.final_answer - 37.5) < 1e-9


def test_clean_task_needs_no_correction():
    task = Task("clean", "Evaluate the expression: 12 * (3 + 4) / 2", expected=42.0)
    tr = Agent(config=AgentConfig(self_correct=True)).solve(task)
    assert tr.success and tr.attempts == 1 and not tr.self_corrected


def test_input_validation_rejects_empty_task():
    tr = Agent().solve(Task("bad", "   ", expected=None))
    assert tr.success is False and tr.attempts == 0


def test_trace_has_cost_and_tokens():
    tr = Agent().solve(next(t for t in SUITE if t.category == "showcase"))
    assert tr.total_tokens > 0
    assert tr.total_cost > 0
    assert tr.width >= 3  # showcase has 3 parallel branches
