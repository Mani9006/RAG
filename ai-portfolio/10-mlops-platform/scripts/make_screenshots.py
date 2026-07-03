"""Render the four product-grade PNG dashboards from real lifecycle output.

Reads ``data/monitoring.csv``, ``data/timeline.json``, ``data/canary.json`` and
``data/registry_snapshot.json`` (produced by ``run_lifecycle.py``) and renders:

    1. drift_timeline.png     PSI over the stream, retrain + promotion annotated
    2. version_performance.png v1 degrading vs v2 recovering (window accuracy)
    3. registry_table.png     the registry rendered as a styled table view
    4. ab_canary_dashboard.png A/B canary comparison + KPI dashboard

Charts follow the shared viztheme; no dual axes, fixed-order palette, reserved
status colours (green=good, amber=warn, red=bad).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops.config import CONFIG
from mlops.viztheme import (
    ACCENT, BAD, GOOD, GRID, MUTED, PALETTE, PANEL, TEXT, WARN,
    apply_theme, kpi, save_panel,
)

DATA = CONFIG.data_dir
ASSETS = CONFIG.assets_dir


def _load():
    df = pd.read_csv(DATA / "monitoring.csv")
    timeline = json.load(open(DATA / "timeline.json"))
    canary = json.load(open(DATA / "canary.json"))
    snapshot = json.load(open(DATA / "registry_snapshot.json"))
    return df, timeline, canary, snapshot


def _events(timeline):
    trig = next((e["t"] for e in timeline if e["kind"] == "trigger"), None)
    promo2 = next((e["t"] for e in timeline if e["kind"] == "promote" and e.get("version") == 2), None)
    return trig, promo2


# --- 1. drift timeline ----------------------------------------------------
def chart_drift_timeline(df, timeline):
    trig, promo = _events(timeline)
    psi_alert = CONFIG.psi_alert
    feat_cols = [c for c in df.columns if c.startswith("psi_")]
    # Rank features by peak PSI, show the top 4 in fixed palette order.
    top = sorted(feat_cols, key=lambda c: df[c].max(), reverse=True)[:4]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, c in enumerate(top):
        name = c.replace("psi_", "")
        ax.plot(df["t"], df[c], color=PALETTE[i], lw=1.6, alpha=0.85, label=name)
    ax.plot(df["t"], df["max_psi"], color=TEXT, lw=2.6, label="max PSI (monitored)")

    ax.axhline(psi_alert, color=WARN, ls="--", lw=1.4)
    ax.text(0.4, psi_alert + 0.03, f"alert threshold {psi_alert}", color=WARN, fontsize=8.5)

    # Drift ramp shading (first non-zero drift to first plateau).
    ramp = df[df["drift_level"] > 0]["t"]
    if len(ramp):
        ax.axvspan(ramp.min() - 0.5, df["t"].max() + 0.5, color=WARN, alpha=0.05)
        ax.text(ramp.min() + 0.2, ax.get_ylim()[1] * 0.94, "distribution shift", color=MUTED, fontsize=8.5)

    if trig is not None:
        ax.axvline(trig, color=BAD, lw=1.8)
        ax.annotate("retrain\ntriggered", xy=(trig, df["max_psi"].max() * 0.72),
                    xytext=(trig - 4.6, df["max_psi"].max() * 0.80),
                    color=BAD, fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=BAD, lw=1.4))
    if promo is not None:
        ax.axvline(promo, color=GOOD, lw=1.8, ls=":")
        ax.annotate("v2 promoted", xy=(promo, df["max_psi"].max() * 0.42),
                    xytext=(promo + 0.6, df["max_psi"].max() * 0.5),
                    color=GOOD, fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=GOOD, lw=1.4))

    ax.set_xlabel("stream window (t)")
    ax.set_ylabel("Population Stability Index (PSI)")
    ax.set_title("Data drift monitoring — PSI per feature with automated response", loc="left")
    ax.legend(loc="upper left", ncol=3, fontsize=8.5, bbox_to_anchor=(0.0, -0.12))
    ax.margins(x=0.01)
    fig.subplots_adjust(bottom=0.24)
    save_panel(fig, str(ASSETS / "drift_timeline.png"))


# --- 2. version performance ----------------------------------------------
def chart_version_performance(df, timeline):
    trig, promo = _events(timeline)
    fig, ax = plt.subplots(figsize=(11, 5.2))

    ax.plot(df["t"], df["acc_v1"], color=PALETTE[5], lw=2.4, label="v1 (baseline model)")
    v2 = df["acc_v2"].astype(float)
    mask = v2.notna()
    ax.plot(df["t"][mask], v2[mask], color=GOOD, lw=2.6, label="v2 (retrained model)")

    ramp = df[df["drift_level"] > 0]["t"]
    if len(ramp):
        ax.axvspan(ramp.min() - 0.5, df["t"].max() + 0.5, color=WARN, alpha=0.05)

    if promo is not None:
        ax.axvline(promo, color=GOOD, lw=1.6, ls=":")
        ax.text(promo + 0.3, 0.12, "v2 promoted\nto production", color=GOOD, fontsize=8.5, fontweight="bold")

    # Direct end-labels (selective, not per-point).
    ax.text(df["t"].iloc[-1] + 0.3, df["acc_v1"].iloc[-1], f"{df['acc_v1'].iloc[-1]:.2f}",
            color=PALETTE[5], fontsize=9, va="center", fontweight="bold")
    ax.text(df["t"].iloc[-1] + 0.3, v2.iloc[-1], f"{v2.iloc[-1]:.2f}",
            color=GOOD, fontsize=9, va="center", fontweight="bold")

    ax.axhline(0.5, color=GRID, lw=1.0, ls="--")
    ax.text(df["t"].max() * 0.72, 0.52, "random baseline (acc = 0.5)", color=MUTED, fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("stream window (t)")
    ax.set_ylabel("per-window accuracy on live traffic")
    ax.set_title("Model performance across versions — v1 degrades under drift, v2 recovers", loc="left")
    ax.legend(loc="lower left", fontsize=9)
    ax.margins(x=0.02)
    save_panel(fig, str(ASSETS / "version_performance.png"))


# --- 3. registry table ----------------------------------------------------
def chart_registry_table(snapshot):
    versions = snapshot["versions"]
    stage_color = {"production": GOOD, "archived": MUTED, "staging": WARN}

    cols = ["version", "stage", "algo", "auc", "accuracy", "f1", "parent", "data_hash", "created"]
    header = ["Version", "Stage", "Algorithm", "AUC", "Accuracy", "F1", "Parent", "Data hash", "Registered"]
    cells = []
    for v in versions:
        cells.append([
            f"v{v['version']}", v["stage"], v["algo"],
            f"{v['auc']:.4f}", f"{v['accuracy']:.4f}", f"{v['f1']:.4f}",
            f"v{v['parent']}" if v["parent"] else "—",
            v["data_hash"], v["created"][:16],  # trim seconds
        ])

    fig, ax = plt.subplots(figsize=(14.5, 2.4 + 0.5 * len(cells)))
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(col=list(range(len(header))))
    tbl.scale(1, 2.0)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color=TEXT, fontweight="bold")
        else:
            v = versions[r - 1]
            cell.set_facecolor(PANEL if r % 2 else "#12171e")
            cell.set_text_props(color=TEXT)
            if header[c] == "Stage":
                cell.set_text_props(color=stage_color.get(v["stage"], TEXT), fontweight="bold")
            if header[c] == "Version":
                cell.set_text_props(color=ACCENT, fontweight="bold")

    lineage = " → ".join(f"v{x}" for x in snapshot["lineage_of_prod"])
    fig.suptitle("Model Registry — version table with stage, metrics & lineage",
                 fontsize=15, fontweight="bold", color=TEXT, x=0.5, y=0.98)
    fig.text(0.5, 0.04, f"production = v{snapshot['production']}    •    lineage: {lineage}"
                        f"    •    {len(versions)} versions tracked",
             ha="center", color=MUTED, fontsize=10)
    fig.savefig(ASSETS / "registry_table.png", bbox_inches="tight")
    plt.close(fig)


# --- 4. A/B canary + KPI dashboard ----------------------------------------
def chart_ab_dashboard(df, canary, snapshot):
    inc, cha = canary["incumbent_metrics"], canary["challenger_metrics"]
    v_inc, v_cha = canary["incumbent_version"], canary["challenger_version"]
    peak_psi = df["max_psi"].max()
    v1_end = float(df["acc_v1"].iloc[-1])
    v2_end = float(df["acc_v2"].astype(float).iloc[-1])

    fig = plt.figure(figsize=(13, 7.4))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.9, 1.5, 1.3], hspace=0.55, wspace=0.35)

    # KPI row
    kpi(fig.add_subplot(gs[0, 0]), "production model", f"v{snapshot['production']}", "gate-approved", GOOD)
    kpi(fig.add_subplot(gs[0, 1]), "peak PSI", f"{peak_psi:.2f}", f"alert @ {CONFIG.psi_alert}", WARN)
    kpi(fig.add_subplot(gs[0, 2]), "canary Δ AUC", f"+{canary['delta_auc']:.2f}",
        f"v{v_cha} vs v{v_inc}", GOOD)
    kpi(fig.add_subplot(gs[0, 3]), "accuracy recovered", f"+{v2_end - v1_end:.2f}",
        f"{v1_end:.2f} → {v2_end:.2f}", ACCENT)

    # A/B grouped bars (one axis, fixed palette by entity)
    ax_ab = fig.add_subplot(gs[1, :2])
    metrics = ["auc", "accuracy", "f1"]
    x = np.arange(len(metrics))
    w = 0.38
    b1 = ax_ab.bar(x - w / 2, [inc[m] for m in metrics], w, color=PALETTE[5], label=f"v{v_inc} (incumbent)")
    b2 = ax_ab.bar(x + w / 2, [cha[m] for m in metrics], w, color=GOOD, label=f"v{v_cha} (challenger)")
    for bars in (b1, b2):
        for b in bars:
            ax_ab.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                       f"{b.get_height():.2f}", ha="center", color=TEXT, fontsize=8.5)
    ax_ab.set_xticks(x)
    ax_ab.set_xticklabels([m.upper() for m in metrics])
    ax_ab.set_ylim(0, 1.22)
    ax_ab.set_ylabel("score on shared holdout")
    ax_ab.set_title("A/B canary — challenger vs incumbent (settled regime)", loc="left", fontsize=11)
    ax_ab.legend(fontsize=8.5, loc="upper right")

    # Gate checks panel
    ax_g = fig.add_subplot(gs[1, 2:])
    ax_g.axis("off")
    ax_g.set_title("Promotion gate", loc="left", fontsize=11, color=TEXT)
    for i, ck in enumerate(canary["gate_checks"]):
        y = 0.82 - i * 0.26
        ok = ck["passed"]
        ax_g.text(0.02, y, "PASS" if ok else "FAIL", color=GOOD if ok else BAD,
                  fontsize=11, fontweight="bold", family="DejaVu Sans")
        ax_g.text(0.20, y, ck["name"].replace("_", " "), color=TEXT, fontsize=10, fontweight="bold")
        ax_g.text(0.20, y - 0.09, ck["detail"], color=MUTED, fontsize=8.5)
    verdict = "PROMOTED" if canary["gate_passed"] else "BLOCKED"
    ax_g.text(0.02, 0.03, f"→ {verdict}", color=GOOD if canary["gate_passed"] else BAD,
              fontsize=12, fontweight="bold")

    # Rolling accuracy KPI trend across the whole stream
    ax_tr = fig.add_subplot(gs[2, :])
    ax_tr.plot(df["t"], df["acc_prod"], color=ACCENT, lw=2.2, label="production accuracy (served)")
    ax_tr.fill_between(df["t"], 0, df["acc_prod"], color=ACCENT, alpha=0.08)
    trig = df[df["trigger"]]["t"]
    if len(trig):
        ax_tr.axvline(trig.iloc[0], color=BAD, lw=1.6, ls="--")
        ax_tr.text(trig.iloc[0] + 0.3, 0.1, "retrain + promote", color=BAD, fontsize=8.5, fontweight="bold")
    ax_tr.set_ylim(0, 1.02)
    ax_tr.set_xlabel("stream window (t)")
    ax_tr.set_ylabel("served accuracy")
    ax_tr.set_title("Production KPI — accuracy served to live traffic (self-healed by auto-retrain)",
                    loc="left", fontsize=11)
    ax_tr.legend(fontsize=8.5, loc="lower right")
    ax_tr.margins(x=0.01)

    save_panel(fig, str(ASSETS / "ab_canary_dashboard.png"),
               suptitle="MLOps Platform — Automated Retraining & Canary Promotion Dashboard")


def main():
    apply_theme()
    ASSETS.mkdir(parents=True, exist_ok=True)
    df, timeline, canary, snapshot = _load()
    chart_drift_timeline(df, timeline)
    chart_version_performance(df, timeline)
    chart_registry_table(snapshot)
    chart_ab_dashboard(df, canary, snapshot)
    print(f"Wrote 4 screenshots to {ASSETS}")


if __name__ == "__main__":
    main()
