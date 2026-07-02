"""Render the four portfolio screenshots from ``benchmarks/results.json``.

Run ``scripts/run_pipeline.py`` first (or with ``--run`` here) to produce the
results. Every number plotted comes from a real training run.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from distillkit import viztheme as vt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "assets")
RESULTS = os.path.join(ROOT, "benchmarks", "results.json")


def _load() -> dict:
    with open(RESULTS) as f:
        return json.load(f)


# --- 1. KD learning curves ---------------------------------------------------
def fig_kd_curves(r: dict) -> None:
    d = r["distill"]
    kd, nokd = np.array(d["kd_curve"]), np.array(d["nokd_curve"])
    epochs = np.arange(1, len(kd) + 1)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.axhline(d["teacher_acc"], color=vt.WARN, ls="--", lw=2,
               label=f"Teacher ensemble ({d['teacher_acc']:.3f})")
    ax.plot(epochs, kd, color=vt.GOOD, lw=2.4,
            label=f"Student + KD ({d['student_kd_acc']:.3f})")
    ax.plot(epochs, nokd, color=vt.BAD, lw=2.4,
            label=f"Student, no KD ({d['student_nokd_acc']:.3f})")
    ax.fill_between(epochs, nokd, kd, where=(kd >= nokd), color=vt.GOOD, alpha=0.12)

    gain = d["student_kd_acc"] - d["student_nokd_acc"]
    ax.annotate(f"KD lift\n{gain:+.3f}",
                xy=(epochs[-1], (kd[-1] + nokd[-1]) / 2),
                xytext=(epochs[-1] * 0.62, (kd[-1] + nokd[-1]) / 2 - 0.02),
                color=vt.TEXT, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=vt.MUTED))

    ax.set_xlabel("Student training epoch")
    ax.set_ylabel("Held-out accuracy")
    ax.set_title(f"Knowledge distillation: dark knowledge beats hard labels "
                 f"(T={d['temperature']:g}, alpha={d['alpha']:g})")
    ax.set_ylim(min(nokd.min(), 0.6) - 0.03, 1.02)
    ax.legend(loc="lower right")
    vt.save_panel(fig, os.path.join(ASSETS, "01_kd_curves.png"))


# --- 2. Params vs accuracy (distillation efficiency / Pareto) ----------------
def fig_param_pareto(r: dict) -> None:
    d = r["distill"]
    points = [
        ("Teacher ensemble", d["teacher_params"], d["teacher_acc"], vt.WARN, "s"),
        ("Student + KD", d["student_params"], d["student_kd_acc"], vt.GOOD, "o"),
        ("Student, no KD", d["student_params"], d["student_nokd_acc"], vt.BAD, "^"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ps = [p for _, p, _, _, _ in points]
    for name, p, a, c, m in points:
        ax.scatter(p, a, s=240, color=c, marker=m, edgecolor=vt.INK, linewidth=1.2, zorder=3, label=name)
        txt = f"{name}\n{p:,} params\n{a:.3f}"
        if m == "^":  # no-KD student: annotate to the RIGHT to avoid the KD label
            ax.annotate(txt, xy=(p, a), xytext=(34, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8, color=vt.TEXT)
        else:  # ceiling points: annotate below, clear of the title
            ax.annotate(txt, xy=(p, a), xytext=(0, -34), textcoords="offset points",
                        ha="center", va="top", fontsize=8, color=vt.TEXT)
    # Guide line from teacher down to the KD student (the efficient frontier).
    ax.plot([d["teacher_params"], d["student_params"]],
            [d["teacher_acc"], d["student_kd_acc"]],
            color=vt.MUTED, ls=":", lw=1.3, zorder=1)

    ax.set_xscale("log")
    ax.set_xlim(min(ps) / 1.8, max(ps) * 1.8)
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Held-out accuracy")
    ax.set_title(f"Distillation efficiency: {d['compression']:.1f}x smaller, "
                 f"{100 * d['kd_recovery']:.0f}% of teacher accuracy")
    ax.set_ylim(min(d["student_nokd_acc"], 0.7) - 0.06, 1.04)
    ax.legend(loc="lower right")
    vt.save_panel(fig, os.path.join(ASSETS, "02_param_pareto.png"))


# --- 3. Data-scaling curve ---------------------------------------------------
def fig_scaling(r: dict) -> None:
    s = r["scaling"]
    sizes, accs = np.array(s["sizes"]), np.array(s["accuracies"])
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(sizes, accs, "-o", color=vt.ACCENT, lw=2.4, markersize=7)
    for x, y in zip(sizes, accs):
        ax.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8, color=vt.MUTED)
    ax.set_xscale("log")
    ax.set_xlabel("Synthetic training examples (log scale)")
    ax.set_ylabel("Held-out accuracy")
    ax.set_title("Synthetic-data scaling: accuracy vs training-set size")
    ax.set_ylim(accs.min() - 0.05, min(accs.max() + 0.04, 1.02))
    vt.save_panel(fig, os.path.join(ASSETS, "03_data_scaling.png"))


# --- 4. LoRA param reduction + eval-gate scorecard ---------------------------
def fig_lora_and_gate(r: dict) -> None:
    lo, ga = r["lora"], r["gate"]
    fig = plt.figure(figsize=(11.5, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.28)

    # Left: trainable-parameter reduction (log bars).
    axb = fig.add_subplot(gs[0, 0])
    labels = ["Full\nfine-tune", f"LoRA\n(r={lo['rank']})"]
    vals = [lo["full_trainable"], lo["lora_trainable"]]
    colors = [vt.MUTED, vt.GOOD]
    bars = axb.bar(labels, vals, color=colors, width=0.6)
    axb.set_yscale("log")
    axb.set_ylabel("Trainable parameters (log)")
    axb.set_title(f"LoRA trains {lo['reduction_pct']:.1f}% fewer parameters")
    for b, v, acc in zip(bars, vals, [lo["full_target_acc"], lo["lora_target_acc"]]):
        axb.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:,}\nacc {acc:.3f}",
                 ha="center", va="bottom", fontsize=9, color=vt.TEXT)
    axb.text(0.5, 0.55, f"-{lo['reduction_pct']:.1f}% trainable\nparams at\nmatching accuracy",
             transform=axb.transAxes, ha="center", va="center", fontsize=9.5,
             color=vt.GOOD, fontweight="bold")
    axb.set_ylim(top=vals[0] * 3)

    # Right: eval-gate scorecard.
    axg = fig.add_subplot(gs[0, 1])
    axg.axis("off")
    axg.set_title("Promotion gate scorecard", loc="left")

    def render_card(x0, title, acc, checks, passed):
        head_col = vt.GOOD if passed else vt.BAD
        axg.text(x0, 0.94, title, fontsize=10, fontweight="bold", color=vt.TEXT)
        axg.text(x0, 0.86, f"val acc {acc:.3f}", fontsize=9, color=vt.MUTED)
        verdict = "APPROVED" if passed else "BLOCKED"
        axg.text(x0, 0.78, verdict, fontsize=11, fontweight="bold", color=head_col)
        y = 0.66
        for c in checks:
            mark = "PASS" if c["passed"] else "FAIL"
            col = vt.GOOD if c["passed"] else vt.BAD
            axg.text(x0, y, mark, fontsize=8.5, fontweight="bold", color=col)
            axg.text(x0 + 0.11, y, c["name"].replace("_", " "), fontsize=8, color=vt.TEXT)
            axg.text(x0 + 0.11, y - 0.045, f"{c['value']:.3f}  (>= {c['threshold']:.3f})",
                     fontsize=7.5, color=vt.MUTED)
            y -= 0.135

    render_card(0.02, "Candidate A (trained)", ga["good_acc"], ga["good_checks"], ga["good_passed"])
    render_card(0.53, "Candidate B (corrupt labels)", ga["bad_acc"], ga["bad_checks"], ga["bad_passed"])
    axg.axvline(0.5, 0.05, 0.9, color=vt.GRID, lw=1)

    vt.save_panel(fig, os.path.join(ASSETS, "04_lora_and_gate.png"),
                  suptitle="LoRA parameter efficiency  +  automated promotion gate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="run the pipeline first")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    os.makedirs(ASSETS, exist_ok=True)

    if args.run or not os.path.exists(RESULTS):
        import subprocess
        cmd = ["python3", os.path.join(HERE, "run_pipeline.py")]
        if args.fast:
            cmd.append("--fast")
        subprocess.check_call(cmd)

    r = _load()
    vt.apply_theme()
    fig_kd_curves(r)
    fig_param_pareto(r)
    fig_scaling(r)
    fig_lora_and_gate(r)
    print(f"Wrote 4 screenshots to {ASSETS}/")


if __name__ == "__main__":
    main()
