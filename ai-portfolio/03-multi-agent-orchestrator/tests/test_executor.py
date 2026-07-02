"""Executor mechanics: parallel scheduling, ref resolution, blackboard/bus."""
from __future__ import annotations

from orchestrator.agent import Agent, AgentConfig
from orchestrator.blackboard import Blackboard, MessageBus
from orchestrator.executor import Executor
from orchestrator.llm import OfflineRuleEngine, Task
from orchestrator.tools import default_toolbox


def test_dependency_chain_resolves_refs():
    task = Task("d", "Convert 10 kilometers to miles, then multiply by 3",
                expected=10 * 1000.0 / 1609.344 * 3)
    tr = Agent(config=AgentConfig(self_correct=True)).solve(task)
    assert tr.success
    assert abs(tr.final_answer - task.expected) < 1e-6
    # Two steps: a convert then a calculator that references it.
    final = tr.final_attempt_steps()
    assert len(final) == 2


def test_parallel_branches_scheduled_together():
    task = next(iter([Task(
        "p",
        "Convert 100 centimeters to meters and convert 2 kilograms to pounds, "
        "then add the two results",
        expected=1.0 + 2 / 0.45359237,
    )]))
    tr = Agent(config=AgentConfig(self_correct=True)).solve(task)
    assert tr.success
    final = tr.final_attempt_steps()
    # Two independent converts share level 0 -> identical simulated start time.
    convs = [s for s in final if s.tool == "unit_convert"]
    assert len(convs) == 2
    assert abs(convs[0].sim_start - convs[1].sim_start) < 1e-9
    # The combine step starts only after the branches finish.
    combine = [s for s in final if s.tool == "calculator"][0]
    assert combine.sim_start >= max(c.sim_end for c in convs) - 1e-9


def test_blackboard_and_bus_populated():
    bb, bus = Blackboard(), MessageBus()
    ex = Executor(OfflineRuleEngine(), default_toolbox())
    task = Task("bb", "Evaluate the expression: 2 + 2", expected=4.0)
    tr = ex.run(task, blackboard=bb, bus=bus)
    assert tr.success
    assert bb.has("s1")                       # step result recorded in memory
    topics = {m.topic for m in bus.log}
    assert "task.done" in topics and "step.done" in topics


def test_unknown_tool_reported_as_error():
    from orchestrator.dag import Step

    class BadEngine(OfflineRuleEngine):
        def plan(self, prompt):
            return [Step("s1", "does_not_exist", {"x": 1})]

    ex = Executor(BadEngine(), default_toolbox())
    tr = ex.run(Task("bad", "anything", expected=None), self_correct=False)
    assert tr.success is False


def test_simulated_cost_monotonic_in_steps():
    a = Agent()
    one = a.solve(Task("a", "Evaluate the expression: 1 + 1", expected=2.0))
    many = a.solve(next(iter([Task(
        "b",
        "Convert 12 kilometers to miles and convert 4 kilograms to pounds and "
        "search the knowledge base for speed of sound in air, then add all three results",
        expected=0.0)])))
    # A richer DAG costs more simulated tokens than a trivial one.
    assert many.total_tokens > one.total_tokens
