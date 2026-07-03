#!/usr/bin/env python3
"""Render the portfolio screenshots into assets/ from a real benchmark run.

Produces four PNGs:
  1. assets/execution_gantt.png     — simulated DAG execution timeline (Gantt)
  2. assets/success_rate.png        — success rate with vs without self-correction
  3. assets/tool_distribution.png   — tool-call distribution
  4. assets/kpi_dashboard.png       — KPI overview dashboard
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

from orchestrator import viztheme as vt  # noqa: E402
from runner import run_all, showcase_trace  # noqa: E402
from tasks import SUITE  # noqa: E402

ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)

TOOL_COLOR = {
    "unit_convert": vt.PALETTE[0],
    "calculator": vt.PALETTE[1],
    "corpus_search": vt.PALETTE[2],
    "code_check": vt.PALETTE[3],
}


def gantt(trace) -> None:
    steps = sorted(trace.final_attempt_steps(), key=lambda s: (s.sim_start, s.step_id))
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ylabels = []
    for i, st in enumerate(reversed(steps)):
        y = i
        color = TOOL_COLOR.get(st.tool, vt.ACCENT)
        dur = max(st.sim_end - st.sim_start, 0.4)
        ax.barh(y, dur, left=st.sim_start, height=0.58, color=color,
                edgecolor=vt.INK, linewidth=1.2, zorder=3)
        ax.text(st.sim_start + dur / 2, y, f"{st.tool}",
                ha="center", va="center", color=vt.INK, fontsize=8.5,
                fontweight="bold", zorder=4)
        ylabels.append(f"{st.step_id}  (L{st.level})")
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Simulated latency (ms) — independent steps overlap in a level")
    ax.set_title(f"Execution timeline · task {trace.task_id} · "
                 f"{len(steps)} steps · width {trace.width}")
    ax.grid(axis="y", visible=False)
    legend = [Patch(facecolor=c, label=t) for t, c in TOOL_COLOR.items()]
    ax.legend(handles=legend, loc="upper right", fontsize=8, ncol=2,
              framealpha=0.0)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.02)
    fig.tight_layout()
    vt.save_panel(fig, os.path.join(ASSETS, "execution_gantt.png"))


def success_bars(without, with_) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    names = ["Without\nself-correction", "With\nself-correction"]
    vals = [without.success_rate * 100, with_.success_rate * 100]
    colors = [vt.BAD, vt.GOOD]
    bars = ax.bar(names, vals, color=colors, width=0.55, zorder=3)
    for b, v, c in zip(bars, vals, colors):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%",
                ha="center", color=c, fontweight="bold", fontsize=13)
    lift = (with_.success_rate - without.success_rate) * 100
    ax.set_ylim(0, 112)
    ax.set_ylabel("Task success rate (%)")
    ax.set_title(f"Self-correction lift: +{lift:.1f} pts on {without.n} tasks")
    ax.annotate("", xy=(1, vals[1]), xytext=(0, vals[0]),
                arrowprops=dict(arrowstyle="->", color=vt.MUTED, lw=1.4,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(0.5, (vals[0] + vals[1]) / 2 + 6, f"+{lift:.0f} pts",
            ha="center", color=vt.TEXT, fontsize=11, fontweight="bold")
    fig.tight_layout()
    vt.save_panel(fig, os.path.join(ASSETS, "success_rate.png"))


def tool_distribution(with_) -> None:
    counts = with_.tool_counts()
    order = sorted(counts.items(), key=lambda x: -x[1])
    labels = [k for k, _ in order]
    vals = [v for _, v in order]
    colors = [TOOL_COLOR.get(k, vt.ACCENT) for k in labels]
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1], zorder=3, height=0.6)
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + max(vals) * 0.01, b.get_y() + b.get_height() / 2,
                str(v), va="center", color=vt.TEXT, fontsize=10, fontweight="bold")
    ax.set_xlabel("Successful tool invocations across the suite")
    ax.set_title("Tool-call distribution")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    vt.save_panel(fig, os.path.join(ASSETS, "tool_distribution.png"))


def kpi_dashboard(without, with_) -> None:
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.35], hspace=0.35, wspace=0.25)

    lift = (with_.success_rate - without.success_rate) * 100
    tiles = [
        ("Tasks solved", f"{with_.solved}/{with_.n}", "with self-correction", vt.GOOD),
        ("Success rate", f"{with_.success_rate*100:.0f}%", f"+{lift:.0f} pts vs baseline", vt.ACCENT),
        ("Avg steps / task", f"{with_.avg_steps:.2f}", "DAG nodes executed", vt.PALETTE[4]),
        ("Avg sim. cost", f"${with_.avg_cost:.3f}", f"{with_.avg_tokens:.0f} tokens/task", vt.WARN),
    ]
    for i, (label, value, sub, color) in enumerate(tiles):
        ax = fig.add_subplot(gs[0, i])
        vt.kpi(ax, label, value, sub, color=color)

    # Bottom-left: success comparison.
    axl = fig.add_subplot(gs[1, :2])
    names = ["without SC", "with SC"]
    vals = [without.success_rate * 100, with_.success_rate * 100]
    bars = axl.bar(names, vals, color=[vt.BAD, vt.GOOD], width=0.5, zorder=3)
    for b, v in zip(bars, vals):
        axl.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%",
                 ha="center", color=vt.TEXT, fontweight="bold")
    axl.set_ylim(0, 115)
    axl.set_ylabel("Success rate (%)")
    axl.set_title("Success rate: self-correction on/off")

    # Bottom-right: tool distribution.
    axr = fig.add_subplot(gs[1, 2:])
    counts = with_.tool_counts()
    order = sorted(counts.items(), key=lambda x: -x[1])
    labels = [k for k, _ in order]
    vals2 = [v for _, v in order]
    colors = [TOOL_COLOR.get(k, vt.ACCENT) for k in labels]
    axr.barh(labels[::-1], vals2[::-1], color=colors[::-1], zorder=3, height=0.6)
    axr.set_title("Tool-call distribution")
    axr.set_xlabel("calls")
    axr.grid(axis="y", visible=False)

    vt.save_panel(fig, os.path.join(ASSETS, "kpi_dashboard.png"),
                  suptitle="Multi-Agent Orchestrator · Benchmark Dashboard")


def main() -> None:
    vt.apply_theme()
    conds = run_all(SUITE)
    without, with_ = conds["without"], conds["with"]
    trace = showcase_trace(SUITE)

    gantt(trace)
    success_bars(without, with_)
    tool_distribution(with_)
    kpi_dashboard(without, with_)
    print("Wrote 4 screenshots to", ASSETS)
    for name in ("execution_gantt", "success_rate", "tool_distribution", "kpi_dashboard"):
        print("  -", os.path.join("assets", name + ".png"))


if __name__ == "__main__":
    main()
