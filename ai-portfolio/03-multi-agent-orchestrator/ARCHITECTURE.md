# Architecture

## 1. Design goals

1. **Real components, not a demo script.** Every part — planner, DAG scheduler,
   tools, guardrails, accounting — is a standalone, tested unit.
2. **Offline and deterministic.** Zero external API. The "LLM" is a rule/pattern
   policy behind an interface, so results are reproducible and CI-friendly, and a
   real model can be swapped in without touching the executor or tools.
3. **Measure recovery, honestly.** The self-correction loop must *earn* its lift:
   it may only use execution feedback (errors, failed validators), never the
   task's ground-truth answer.

## 2. Component model

```
Agent ── owns ──▶ LLMEngine (plan/repair)   Toolbox {name → Tool}
             └──▶ Blackboard (shared memory) MessageBus (events)
             └──▶ Executor (schedule + guardrails + accounting)
```

### Tool
`Tool` = `name` + human `description` + typed `schema` + `run(args) -> dict`.
The schema is `{arg: {type, required, default}}`. Tools raise `ToolError` for
recoverable domain failures; guardrails (not tools) enforce the schema, so the
same validator protects every tool and the executor can attribute a failure to
the correct stage.

The four tools are deliberately heterogeneous to exercise the framework:
- **`calculator`** — evaluates arithmetic by walking a *restricted AST*
  (`ast.parse(mode="eval")`). Only literals, whitelisted binary/unary operators,
  whitelisted functions (`sqrt`, `abs`, `round`, `min/max`, `log`, ...) and named
  constants (`pi`, `e`) are allowed. Names, attribute access, calls to arbitrary
  builtins, comprehensions, lambdas and the bitwise operators are rejected.
  **`eval`/`exec` are never used** (asserted by a test that AST-scans the module).
- **`unit_convert`** — length / mass via linear base-unit factors, temperature via
  affine C/F/K formulas; unknown or incompatible units raise `ToolError`.
- **`corpus_search`** — TF-IDF + cosine over a small synthetic knowledge base
  (`scikit-learn`); returns the top fact's numeric value plus ranked hits.
- **`code_check`** — AST lint flagging `eval`/`exec`, bare `except:`, `== None`,
  and mutable default arguments; syntax errors are reported, not raised.

### Planner and the DAG
The `LLMEngine.plan(prompt)` returns an ordered `list[Step]`. A `Step` names a
tool, an args dict, and optional explicit deps. Dependencies are also **derived**
automatically from two forms of reference embedded in args:
- **`Ref(step_id, field)`** objects, and
- **`$step[.field]`** template variables inside string args (e.g. an expression
  `"$s1 + $s2 + $s3"`).

`DAG` runs **Kahn's algorithm** to produce *levels*: level *k* contains steps that
depend only on levels `< k`. This simultaneously (a) validates acyclicity —
`CycleError` if not all nodes drain, (b) validates that every reference resolves,
and (c) yields the parallel schedule. `width()` = the widest level = the maximum
achievable parallelism.

### Executor
For each attempt the executor builds the DAG and walks it level by level.
Independent steps within a level run on a `ThreadPoolExecutor`; results merge in a
deterministic (id-sorted) order so runs are reproducible despite threading. Per
step it:
1. resolves `Ref`/`$var` references from prior results,
2. runs `validate_args` (→ `SchemaError`, stage `schema`),
3. calls `tool.run` (`ToolError`/arithmetic errors → stage `tool`),
4. records a `StepTrace` with simulated tokens, cost, and a simulated
   start/end latency for the Gantt, and writes the result to the `Blackboard`
   and a `step.done` event to the `MessageBus`.

After a clean run the *task-level* `output_contract` is checked
(`validate_output` → stage `output`). Any failure produces a structured
`error_info = {stage, message, step_id, contract}`.

### Self-correction loop
On failure with `self_correct=True` and attempts remaining, the executor calls
`engine.repair(prompt, plan, error_info)`. The offline engine matches the error
stage/message and applies a targeted, deterministic transform (see the table in
the README). It returns a new plan and the executor retries, up to
`max_retries` (default 3). Because repair consumes only feedback, the same
mechanism generalizes to a real model prompted with the error trace.

### Accounting and tracing
`CostModel` estimates tokens (~4 chars/token, the usual heuristic) for planning
and per-step I/O, prices them against a simulated sheet, and models per-tool
latency (base + per-token). Every attempt appends to an `ExecutionTrace`
(all `StepTrace`s across attempts, per-attempt errors, totals, width). The
timeline in the Gantt uses each level's `sim_start = max(dep.sim_end)`, so
independent steps visibly overlap.

## 3. Key trade-offs

- **Deterministic offline engine vs. a real LLM.** We trade open-domain language
  understanding for reproducibility and zero cost. The `LLMEngine` seam is the
  whole point: `plan`/`repair` are the only two methods a real model must
  implement; nothing downstream changes.
- **Level-synchronous scheduling vs. a work-stealing scheduler.** Level batches
  are simple, deterministic, and easy to reason about, at the cost of a small
  amount of parallelism when levels are uneven. A future `asyncio`/work-stealing
  executor could start a step the instant its deps finish; the DAG already
  exposes per-node dependencies to make that a drop-in change.
- **Simulated latency for the timeline vs. wall-clock.** Tool calls here take
  microseconds, which makes a real-time Gantt meaningless. We record real
  `wall_ms` for information but draw the timeline from the (deterministic) cost
  model so the visualization is stable and legible.
- **Guardrails outside tools.** Slightly more plumbing, but it means one schema
  validator, one output-contract validator, and clean failure attribution that
  drives targeted repairs.

## 4. Scaling to a fleet (the "1B" story)

This system's scale axis is **task-executions and agent fan-out**, not rows in a
table. The design extends to a billion executions as follows:

- **Stateless executors, externalized state.** An `Executor` holds no global
  state; `Blackboard` and `MessageBus` are interfaces. Swap the in-process
  Blackboard for Redis/KeyDB and the MessageBus for Kafka/NATS, and executors
  become horizontally shardable workers behind a queue. A billion tasks is then a
  throughput problem: at a sustained 10k tasks/s, 1B tasks complete in ~28 hours
  on a modest worker pool; the workload is embarrassingly parallel across tasks.
- **DAG-level parallelism is already modeled.** Within a task, independent
  branches run concurrently; across tasks there is no shared mutable state, so
  scheduling is a matter of queue depth and worker count. Sub-DAGs can be farmed
  to specialist agents (map/reduce over the message bus) for deep task trees.
- **Bounded memory.** Traces stream to durable storage per attempt rather than
  accumulating; the benchmark harness itself processes tasks one at a time.
  Nothing in the hot path is O(total tasks) in memory.
- **Cost governance at scale.** The token/cost accounting that is *simulated*
  here is exactly the control surface you need in production: per-step budgets,
  a global spend ceiling, and back-pressure when a repair loop is not converging
  (cap `max_retries`, then dead-letter the task).
- **Determinism as a test asset.** Because the offline engine is reproducible,
  regression tests can pin exact plans and success rates. With a real model,
  the same benchmark harness becomes an **eval suite**: measure success rate,
  average attempts, and cost per task per model/prompt version.

## 5. Testing strategy

`make test` runs 69 assertions covering: DAG topological correctness, parallel
width, and cycle/self-dependency/unknown-ref rejection; calculator correctness
and AST-safety (injection, comprehensions, `^`, arbitrary builtins all rejected;
no `eval` in source); schema/input/output guardrails; each tool's behavior and
error paths; and end-to-end that **self-correction strictly and materially
improves success** (and reaches 100% on the suite) while clean tasks need no
correction and bad input is rejected before any step runs.
