"""Generate the portfolio PNG screenshots from real pipeline artifacts.

Reads ``data/metrics.json``, ``data/samples.json`` and ``benchmarks/results.csv``
(produced by ``run_pipeline.py`` / ``benchmark.py``) and renders four panels
into ``assets/``:

  1. ``detected_document.png`` — a rendered document with detected field boxes,
     labels and key→value links (the product money-shot).
  2. ``field_f1.png``          — per-field precision/recall/F1 bar chart.
  3. ``doc_type.png``          — doc-type confusion matrix + class distribution.
  4. ``dashboard.png``         — KPI + throughput scaling dashboard.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docintel import LABELS  # noqa: E402
from docintel.extraction import group_entities, pair_key_values  # noqa: E402
from docintel.render import render_with_detections  # noqa: E402
from docintel.viztheme import (  # noqa: E402
    ACCENT, BAD, GOOD, GRID, MUTED, PALETTE, PANEL, TEXT, WARN, apply_theme,
    kpi, save_panel)

ASSETS = ROOT / "assets"


def load(name: str):
    return json.loads((ROOT / "data" / name).read_text())


# ---------------------------------------------------------------------------
def shot_document(samples) -> None:
    inv = next((s for s in samples if s["doc_type"] == "INVOICE"), samples[0])
    toks = inv["tokens"]
    pred = inv["pred"]
    ents = group_entities(toks, pred)
    pairs = pair_key_values(ents)
    render_with_detections(
        toks, ents, pairs, str(ASSETS / "detected_document.png"),
        title=f"Document AI  ·  {inv['doc_type']}  ·  fields auto-detected")


# ---------------------------------------------------------------------------
def shot_field_f1(metrics) -> None:
    apply_theme()
    per = metrics["token_classifier"]["per_field"]
    labs = [l for l in LABELS if l != "O" and per[l]["support"] > 0]
    f1 = [per[l]["f1"] for l in labs]
    prec = [per[l]["precision"] for l in labs]
    rec = [per[l]["recall"] for l in labs]
    order = np.argsort(f1)
    labs = [labs[i] for i in order]
    f1 = [f1[i] for i in order]; prec = [prec[i] for i in order]; rec = [rec[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 6.4))
    y = np.arange(len(labs))
    ax.barh(y, f1, color=ACCENT, height=0.7, zorder=3, label="F1")
    ax.scatter(prec, y, color=WARN, s=26, zorder=4, label="precision")
    ax.scatter(rec, y, color=GOOD, s=26, zorder=4, label="recall")
    for yi, v in zip(y, f1):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", ha="left",
                color=TEXT, fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=9)
    ax.set_xlim(0, 1.08); ax.set_xlabel("score")
    ax.axvline(metrics["token_classifier"]["majority_baseline"], color=BAD,
               ls="--", lw=1.2, zorder=2)
    ax.text(metrics["token_classifier"]["majority_baseline"], len(labs) - 0.3,
            "  majority baseline", color=BAD, fontsize=8, va="top")
    ax.legend(loc="lower right", fontsize=9)
    macro = metrics["token_classifier"]["macro_f1"]
    save_panel(fig, str(ASSETS / "field_f1.png"),
               suptitle=f"Per-field extraction quality   (macro-F1 = {macro:.3f})")


# ---------------------------------------------------------------------------
def shot_doc_type(metrics) -> None:
    apply_theme()
    cm = np.array(metrics["router"]["confusion_matrix"])
    labs = metrics["router"]["labels"]
    dist = metrics["router"]["distribution"]
    fig = plt.figure(figsize=(11, 4.8))
    gs = GridSpec(1, 2, width_ratios=[1.15, 1], wspace=0.32)

    ax = fig.add_subplot(gs[0])
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cmn, cmap="mako" if "mako" in plt.colormaps() else "viridis",
                   vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labs))); ax.set_yticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=25, ha="right", fontsize=8)
    ax.set_yticklabels(labs, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.grid(False)
    for i in range(len(labs)):
        for j in range(len(labs)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9,
                    color="white" if cmn[i, j] < 0.6 else "#0d1117", fontweight="bold")
    acc = metrics["router"]["accuracy"]
    ax.set_title(f"Doc-type confusion matrix  (acc = {acc:.3f})")

    ax2 = fig.add_subplot(gs[1])
    vals = [dist[l] for l in labs]
    ax2.bar(range(len(labs)), vals, color=PALETTE[:len(labs)], zorder=3)
    for i, v in enumerate(vals):
        ax2.text(i, v, f"{v}", ha="center", va="bottom", color=TEXT, fontsize=9)
    ax2.set_xticks(range(len(labs)))
    ax2.set_xticklabels(labs, rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel("eval documents"); ax2.set_title("Routed document mix")
    save_panel(fig, str(ASSETS / "doc_type.png"))


# ---------------------------------------------------------------------------
def shot_dashboard(metrics) -> None:
    apply_theme()
    tc = metrics["token_classifier"]
    fig = plt.figure(figsize=(12, 6.6))
    gs = GridSpec(3, 4, height_ratios=[1, 1, 1.5], hspace=0.55, wspace=0.28)

    tiles = [
        ("Field macro-F1", f"{tc['macro_f1']:.3f}", "token typing", GOOD),
        ("Doc-type acc", f"{metrics['router']['accuracy']:.3f}", "router", ACCENT),
        ("Table cell acc", f"{metrics['table']['cell_accuracy']:.3f}",
         f"{metrics['table']['n_tables']} tables", ACCENT),
        ("KV linkage", f"{metrics['kv_pairing']['linkage_accuracy']:.3f}",
         f"{metrics['kv_pairing']['n_pairs']:,} pairs", GOOD),
        ("Throughput", f"{tc['docs_per_s']:,.0f}", "docs / sec", WARN),
        ("Token rate", f"{tc['tokens_per_s'] / 1000:,.0f}k", "tokens / sec", WARN),
        ("Eval tokens", f"{metrics['config']['eval_tokens'] / 1000:,.0f}k",
         "held-out", TEXT),
        ("vs baseline", f"{tc['macro_f1'] / max(tc['majority_baseline'], 1e-6):.1f}x",
         "majority-class", GOOD),
    ]
    for i, (lab, val, sub, col) in enumerate(tiles):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        kpi(ax, lab, val, sub, color=col)

    # throughput scaling line (from benchmark csv if present)
    axb = fig.add_subplot(gs[2, :2])
    csv_path = ROOT / "benchmarks" / "results.csv"
    if csv_path.exists():
        docs, gen, inf = [], [], []
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                docs.append(int(r["docs"]))
                gen.append(float(r["gen_tokens_per_s"]) / 1000)
                inf.append(float(r["infer_tokens_per_s"]) / 1000)
        axb.plot(docs, gen, "-o", color=ACCENT, label="generation", zorder=3)
        axb.plot(docs, inf, "-o", color=GOOD, label="inference", zorder=3)
        axb.set_xscale("log")
        axb.set_xlabel("documents streamed (log)")
        axb.set_ylabel("k tokens / sec")
        axb.set_title("Sustained throughput vs scale")
        axb.legend(fontsize=8)
    else:
        axb.axis("off")
        axb.text(0.5, 0.5, "run benchmark.py for scaling curve",
                 ha="center", color=MUTED)

    # per-field F1 mini-bars
    axf = fig.add_subplot(gs[2, 2:])
    per = tc["per_field"]
    labs = [l for l in LABELS if l != "O" and per[l]["support"] > 0]
    f1 = [per[l]["f1"] for l in labs]
    order = np.argsort(f1)[::-1]
    labs = [labs[i] for i in order]; f1 = [f1[i] for i in order]
    cols = [GOOD if v >= 0.9 else (WARN if v >= 0.75 else BAD) for v in f1]
    axf.bar(range(len(labs)), f1, color=cols, zorder=3)
    axf.set_xticks(range(len(labs)))
    axf.set_xticklabels(labs, rotation=55, ha="right", fontsize=6.5)
    axf.set_ylim(0, 1.05); axf.set_ylabel("F1")
    axf.set_title("Per-field F1")
    save_panel(fig, str(ASSETS / "dashboard.png"),
               suptitle="Multimodal Document Intelligence — extraction KPIs")


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    metrics = load("metrics.json")
    samples = load("samples.json")
    shot_document(samples)
    shot_field_f1(metrics)
    shot_doc_type(metrics)
    shot_dashboard(metrics)
    print("wrote:", ", ".join(p.name for p in sorted(ASSETS.glob("*.png"))))


if __name__ == "__main__":
    main()
