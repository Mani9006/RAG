"""The Executor: runs a plan's DAG with parallelism and self-correction.

Responsibilities:

  * validate the task input (guardrail)
  * for each attempt, build the DAG, run it level-by-level (independent steps in
    a level execute concurrently on a thread pool), resolving ``Ref`` sentinels
    and ``$step`` template variables from prior results
  * validate every tool call's arguments against its schema (guardrail) and the
    final answer against the task's output contract (guardrail)
  * on any failure, ask the engine to repair the plan and retry, up to
    ``max_retries`` times
  * record a full execution trace with simulated token/cost accounting and a
    simulated per-step latency timeline (for the Gantt view)
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .accounting import CostModel, count_tokens
from .blackboard import Blackboard, MessageBus
from .dag import DAG, CycleError, Ref, Step
from .guardrails import (
    OutputValidationError,
    SchemaError,
    validate_args,
    validate_input,
    validate_output,
)
from .llm import LLMEngine
from .tools import Tool, ToolError

_STR_REF = re.compile(r"\$([A-Za-z]\w*)(?:\.(\w+))?")


@dataclass
class StepTrace:
    step_id: str
    tool: str
    attempt: int
    level: int
    args: dict[str, Any]
    status: str            # "ok" | "error" | "skipped"
    value: Any = None
    error: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    sim_start: float = 0.0
    sim_end: float = 0.0


@dataclass
class ExecutionTrace:
    task_id: str
    prompt: str
    success: bool = False
    attempts: int = 0
    self_corrected: bool = False
    final_answer: Any = None
    steps: list[StepTrace] = field(default_factory=list)   # across all attempts
    errors: list[str] = field(default_factory=list)         # one per failed attempt
    total_tokens: int = 0
    total_cost: float = 0.0
    plan_tokens: int = 0
    n_steps: int = 0        # steps in the final (last) plan
    width: int = 1          # max parallelism of the final plan
    wall_ms: float = 0.0

    def final_attempt_steps(self) -> list[StepTrace]:
        if not self.steps:
            return []
        last = max(s.attempt for s in self.steps)
        return [s for s in self.steps if s.attempt == last]

    def tool_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.steps:
            if s.status == "ok":
                counts[s.tool] = counts.get(s.tool, 0) + 1
        return counts


def _resolve(value: Any, results: dict[str, dict[str, Any]]) -> Any:
    """Resolve Ref objects and ``$step[.field]`` template vars against results."""
    if isinstance(value, Ref):
        return results[value.step_id][value.field]
    if isinstance(value, str):
        def _sub(m: "re.Match[str]") -> str:
            sid, fld = m.group(1), m.group(2) or "value"
            return _fmt(results[sid][fld])
        return _STR_REF.sub(_sub, value)
    if isinstance(value, list):
        return [_resolve(v, results) for v in value]
    if isinstance(value, dict):
        return {k: _resolve(v, results) for k, v in value.items()}
    return value


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return repr(v)
    return str(v)


class Executor:
    def __init__(self, engine: LLMEngine, tools: dict[str, Tool],
                 cost_model: CostModel | None = None, max_workers: int = 4):
        self.engine = engine
        self.tools = tools
        self.cost = cost_model or CostModel()
        self.max_workers = max_workers

    def run(self, task: Any, *, self_correct: bool = True, max_retries: int = 3,
            blackboard: Blackboard | None = None,
            bus: MessageBus | None = None) -> ExecutionTrace:
        trace = ExecutionTrace(task_id=task.id, prompt=task.prompt)
        blackboard = blackboard or Blackboard()
        bus = bus or MessageBus()
        t0 = time.perf_counter()

        try:
            validate_input(task.prompt)
        except Exception as exc:  # InputValidationError
            trace.errors.append(f"input: {exc}")
            trace.attempts = 0
            trace.wall_ms = (time.perf_counter() - t0) * 1000
            bus.publish("task.rejected", {"task": task.id, "error": str(exc)})
            return trace

        steps = self.engine.plan(task.prompt)
        attempt = 0
        while True:
            trace.plan_tokens += self.cost.plan_tokens
            error_info: dict[str, Any] | None = None
            try:
                dag = DAG(steps)
            except (CycleError, ValueError) as exc:
                error_info = {"stage": "plan", "message": str(exc), "step_id": None}
                trace.errors.append(f"plan: {exc}")
                ok = False
            else:
                ok, error_info, final = self._run_dag(
                    dag, task, attempt, trace, blackboard, bus
                )
                if not ok and error_info:
                    trace.errors.append(
                        f"{error_info.get('stage')}: {error_info.get('message')}"
                    )
                if ok:
                    trace.final_answer = final
                    if task.output_contract:
                        try:
                            validate_output(final, task.output_contract)
                        except OutputValidationError as exc:
                            ok = False
                            error_info = {
                                "stage": "output",
                                "message": str(exc),
                                "contract": task.output_contract,
                                "step_id": dag.sinks()[-1] if dag.sinks() else None,
                            }
                            trace.errors.append(f"output: {exc}")

            if ok:
                trace.success = True
                trace.attempts = attempt + 1
                trace.n_steps = len(steps)
                trace.width = DAG(steps).width()
                break

            if not self_correct or attempt >= max_retries:
                trace.attempts = attempt + 1
                trace.n_steps = len(steps)
                try:
                    trace.width = DAG(steps).width()
                except Exception:
                    trace.width = 1
                break

            bus.publish("task.repair", {"task": task.id, "attempt": attempt,
                                        "error": error_info})
            steps = self.engine.repair(task.prompt, steps, error_info or {})
            attempt += 1

        trace.self_corrected = trace.success and trace.attempts > 1
        trace.total_tokens = trace.plan_tokens + sum(
            s.tokens_in + s.tokens_out for s in trace.steps
        )
        trace.total_cost = self.cost.cost(trace.plan_tokens, 0) + sum(
            s.cost for s in trace.steps
        )
        trace.wall_ms = (time.perf_counter() - t0) * 1000
        bus.publish("task.done", {"task": task.id, "success": trace.success,
                                  "attempts": trace.attempts})
        return trace

    # ------------------------------------------------------------------ #

    def _run_dag(self, dag: DAG, task: Any, attempt: int, trace: ExecutionTrace,
                 blackboard: Blackboard, bus: MessageBus):
        results: dict[str, dict[str, Any]] = {}
        sim_end: dict[str, float] = {}
        failed = False
        error_info: dict[str, Any] | None = None

        for level_idx, level in enumerate(dag.levels):
            if failed:
                for sid in level:
                    trace.steps.append(StepTrace(sid, dag.steps[sid].tool, attempt,
                                                 level_idx, {}, "skipped"))
                continue

            def _do(sid: str):
                return sid, self._run_step(dag.steps[sid], results, sim_end,
                                           level_idx, attempt)

            if len(level) == 1:
                outputs = [_do(level[0])]
            else:
                with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    outputs = list(pool.map(_do, level))

            # Deterministic merge order (by step id).
            for sid, st in sorted(outputs, key=lambda x: x[0]):
                trace.steps.append(st)
                if st.status == "ok":
                    results[sid] = st.value
                    sim_end[sid] = st.sim_end
                    blackboard.write(sid, st.value, author=st.tool)
                    bus.publish("step.done", {"task": task.id, "step": sid,
                                              "tool": st.tool})
                else:
                    if not failed:
                        failed = True
                        error_info = st.error_info  # type: ignore[attr-defined]

        if failed:
            return False, error_info, None

        final_id = dag.sinks()[-1]
        return True, None, results[final_id].get("value")

    def _run_step(self, step: Step, results: dict[str, dict[str, Any]],
                  sim_end: dict[str, float], level_idx: int, attempt: int) -> StepTrace:
        tool = self.tools.get(step.tool)
        start = max((sim_end[d] for d in step.all_deps() if d in sim_end), default=0.0)
        st = StepTrace(step.id, step.tool, attempt, level_idx, {}, "error",
                       sim_start=start, sim_end=start)
        try:
            if tool is None:
                raise ToolError(f"unknown tool: {step.tool!r}")
            args = _resolve(step.args, results)
            st.args = args
            validate_args(tool.name, tool.schema, args)  # SchemaError -> stage schema
            result = tool.run(args)
        except SchemaError as exc:
            st.error = str(exc)
            st.error_info = {"stage": "schema", "message": str(exc), "step_id": step.id}  # type: ignore[attr-defined]
            return st
        except (ToolError, ValueError, ZeroDivisionError, ArithmeticError) as exc:
            st.error = str(exc)
            st.error_info = {"stage": "tool", "message": str(exc), "step_id": step.id}  # type: ignore[attr-defined]
            return st

        st.status = "ok"
        st.value = result
        st.tokens_in = count_tokens(str(args)) + 8
        st.tokens_out = count_tokens(str(result)) + 4
        st.cost = self.cost.cost(st.tokens_in, st.tokens_out)
        latency = self.cost.latency_ms(step.tool, st.tokens_out)
        st.sim_end = start + latency
        return st
