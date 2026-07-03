#!/usr/bin/env python3
"""Run the full task suite and write benchmark tables + a trace log.

Outputs:
  benchmarks/results.csv      per-task results (both conditions)
  benchmarks/results.md       summary + per-task Markdown tables
  data/trace_showcase.json    full execution trace of the showcase task
"""
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import asdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

from runner import run_all, scored, showcase_trace  # noqa: E402
from tasks import SUITE  # noqa: E402


def main() -> None:
    conds = run_all(SUITE)
    without, with_ = conds["without"], conds["with"]

    os.makedirs(os.path.join(ROOT, "benchmarks"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)

    # --- per-task CSV ---
    csv_path = os.path.join(ROOT, "benchmarks", "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "task_id", "category", "prompt",
            "off_success", "off_attempts", "off_cost",
            "on_success", "on_attempts", "on_steps", "on_cost", "on_tokens",
        ])
        for i, task in enumerate(SUITE):
            o, n = without.traces[i], with_.traces[i]
            w.writerow([
                task.id, task.category, task.prompt.replace("\n", " ")[:80],
                int(without.correct[i]), o.attempts, round(o.total_cost, 6),
                int(with_.correct[i]), n.attempts, n.n_steps,
                round(n.total_cost, 6), n.total_tokens,
            ])

    # --- summary Markdown ---
    md_path = os.path.join(ROOT, "benchmarks", "results.md")
    lift = with_.success_rate - without.success_rate
    with open(md_path, "w") as f:
        f.write("# Benchmark Results\n\n")
        f.write(f"Task suite: **{len(SUITE)}** multi-step tasks. "
                "Deterministic, fully offline.\n\n")
        f.write("| Condition | Success rate | Solved | Avg steps | "
                "Avg attempts | Avg sim. cost | Avg tokens |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for c in (without, with_):
            f.write(f"| {c.name} | {c.success_rate*100:.1f}% | "
                    f"{c.solved}/{c.n} | {c.avg_steps:.2f} | "
                    f"{c.avg_attempts:.2f} | ${c.avg_cost:.5f} | "
                    f"{c.avg_tokens:.0f} |\n")
        f.write(f"\n**Self-correction lift: +{lift*100:.1f} percentage points** "
                f"({without.success_rate*100:.1f}% -> "
                f"{with_.success_rate*100:.1f}%).\n\n")

        f.write("## Tool-call distribution (with self-correction)\n\n")
        f.write("| Tool | Calls |\n|---|---|\n")
        for tool, cnt in sorted(with_.tool_counts().items(), key=lambda x: -x[1]):
            f.write(f"| `{tool}` | {cnt} |\n")

        f.write("\n## Per-task detail\n\n")
        f.write("| Task | Category | Without SC | With SC | Attempts (SC) | Steps |\n")
        f.write("|---|---|---|---|---|---|\n")
        for i, task in enumerate(SUITE):
            n = with_.traces[i]
            off = "PASS" if without.correct[i] else "FAIL"
            on = "PASS" if with_.correct[i] else "FAIL"
            f.write(f"| {task.id} | {task.category} | {off} | {on} | "
                    f"{n.attempts} | {n.n_steps} |\n")

    # --- showcase trace JSON ---
    tr = showcase_trace(SUITE)
    trace_path = os.path.join(ROOT, "data", "trace_showcase.json")
    with open(trace_path, "w") as f:
        json.dump({
            "task_id": tr.task_id,
            "prompt": tr.prompt,
            "success": tr.success,
            "attempts": tr.attempts,
            "width": tr.width,
            "total_tokens": tr.total_tokens,
            "total_cost": tr.total_cost,
            "steps": [asdict(s) for s in tr.final_attempt_steps()],
        }, f, indent=2, default=str)

    # --- console summary ---
    print(f"Tasks: {len(SUITE)}")
    print(f"  Without self-correction: {without.solved}/{without.n} "
          f"({without.success_rate*100:.1f}%)")
    print(f"  With self-correction:    {with_.solved}/{with_.n} "
          f"({with_.success_rate*100:.1f}%)")
    print(f"  Lift: +{lift*100:.1f} pts")
    print(f"  Avg steps (SC): {with_.avg_steps:.2f}, "
          f"avg attempts: {with_.avg_attempts:.2f}, "
          f"avg sim. cost: ${with_.avg_cost:.5f}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {trace_path}")


if __name__ == "__main__":
    main()
