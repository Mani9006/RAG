#!/usr/bin/env python
"""Generate the portfolio PNG screenshots from real pipeline outputs.

Panels:
  1. query latency vs data scale (log-log)          -> latency_vs_scale.png
  2. NL->SQL execution accuracy by category (bars)   -> accuracy_by_category.png
  3. star-schema data model diagram (boxes+arrows)   -> star_schema.png
  4. analytics result panel (questions -> answers)   -> analytics_dashboard.png
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlsql import DeterministicNLToSQL, SemanticModel, connect  # noqa: E402
from nlsql.viztheme import (ACCENT, BAD, GOOD, GRID, MUTED, PALETTE, PANEL,  # noqa: E402
                            TEXT, WARN, apply_theme, kpi, save_panel)

ASSETS = ROOT / "assets"
BENCH = ROOT / "benchmarks"


def _read_csv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------
def panel_latency() -> None:
    rows = _read_csv(BENCH / "scaling_results.csv")
    rows.sort(key=lambda r: int(r["rows"]))
    x = np.array([int(r["rows"]) for r in rows], dtype=float)
    warm = np.array([float(r["warm_median_ms"]) for r in rows])
    cold = np.array([float(r["cold_ms"]) for r in rows])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, cold, "o-", color=WARN, label="cold (first scan)", linewidth=2, markersize=7)
    ax.plot(x, warm, "o-", color=ACCENT, label="warm median", linewidth=2, markersize=7)
    # linear-scaling reference anchored at the largest measured point
    ref = warm[-1] * (x / x[-1])
    ax.plot(x, ref, "--", color=MUTED, linewidth=1.2, label="linear reference")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("fact rows (log scale)")
    ax.set_ylabel("query latency ms (log scale)")
    ax.set_title("DuckDB out-of-core query latency vs data scale")
    for xi, yi, r in zip(x, warm, rows):
        ax.annotate(f"{float(r['parquet_gb']):.2f} GB", (xi, yi),
                    textcoords="offset points", xytext=(6, -12),
                    fontsize=8, color=MUTED)
    ax.legend(loc="upper left")
    save_panel(fig, str(ASSETS / "latency_vs_scale.png"))


def panel_accuracy() -> None:
    rows = _read_csv(BENCH / "accuracy_by_category.csv")
    summ = json.loads((BENCH / "suite_summary.json").read_text())
    rows.sort(key=lambda r: float(r["accuracy"]))
    cats = [r["category"] for r in rows]
    acc = [float(r["accuracy"]) * 100 for r in rows]
    ns = [int(r["n"]) for r in rows]
    colors = [GOOD if a >= 99.9 else (WARN if a >= 70 else BAD) for a in acc]

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(cats))
    ax.barh(y, acc, color=colors, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}  (n={n})" for c, n in zip(cats, ns)])
    ax.set_xlim(0, 105)
    ax.set_xlabel("execution accuracy (%)")
    ax.set_title(
        f"NL->SQL execution accuracy by category "
        f"(overall {summ['execution_accuracy']*100:.0f}%, "
        f"{summ['correct']}/{summ['questions']})")
    for yi, a in zip(y, acc):
        ax.text(a + 1.5, yi, f"{a:.0f}%", va="center", fontsize=9, color=TEXT)
    ax.grid(axis="y", visible=False)
    save_panel(fig, str(ASSETS / "accuracy_by_category.png"))


def panel_star_schema() -> None:
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Star schema  ·  fact_sales + 4 conformed dimensions",
                 color=TEXT, fontweight="bold", fontsize=13, pad=6)

    def box(cx, cy, w, h, title, lines, color):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                     boxstyle="round,pad=0.05,rounding_size=0.12",
                     linewidth=1.6, edgecolor=color, facecolor=PANEL))
        ax.text(cx, cy + h / 2 - 0.28, title, ha="center", va="center",
                color=color, fontweight="bold", fontsize=10)
        ax.text(cx, cy - 0.15, "\n".join(lines), ha="center", va="center",
                color=MUTED, fontsize=7.6, linespacing=1.35)

    fx, fy = 5, 5
    box(fx, fy, 2.7, 2.1, "fact_sales  (50M rows)",
        ["order_id · date_key", "customer_key · product_key",
         "store_key", "quantity · unit_price", "net_revenue · cost"], ACCENT)

    dims = [
        (5, 8.4, "dim_date", ["date_key (PK)", "year · quarter", "month · weekday"], PALETTE[1]),
        (8.4, 5, "dim_product", ["product_key (PK)", "category · brand", "unit_cost · list_price"], PALETTE[2]),
        (5, 1.6, "dim_store", ["store_key (PK)", "region · country", "store_type"], PALETTE[3]),
        (1.6, 5, "dim_customer", ["customer_key (PK)", "segment · country", "city"], PALETTE[4]),
    ]
    for cx, cy, name, lines, color in dims:
        box(cx, cy, 2.3, 1.7, name, lines, color)
        arr = FancyArrowPatch((cx, cy), (fx, fy),
                              arrowstyle="-|>", mutation_scale=13,
                              color=GRID, linewidth=1.4, shrinkA=42, shrinkB=52)
        ax.add_patch(arr)
        mx, my = (cx + fx) / 2, (cy + fy) / 2
        ax.text(mx, my, "N:1", ha="center", va="center", fontsize=7.5,
                color=MUTED, bbox=dict(boxstyle="round,pad=0.15", fc=PANEL, ec="none"))

    ax.text(5, 0.15, "Hive-partitioned Parquet (year=YYYY/) · queried out-of-core by DuckDB",
            ha="center", color=MUTED, fontsize=8.5)
    save_panel(fig, str(ASSETS / "star_schema.png"))


def _fmt(v: float, currency: bool = True) -> str:
    if currency:
        if v >= 1e9:
            return f"${v/1e9:.2f}B"
        if v >= 1e6:
            return f"${v/1e6:.1f}M"
        if v >= 1e3:
            return f"${v/1e3:.0f}K"
        return f"${v:,.0f}"
    if v >= 1e6:
        return f"{v/1e6:.1f}M"
    if v >= 1e3:
        return f"{v/1e3:.0f}K"
    return f"{v:,.0f}"


def panel_dashboard() -> None:
    model = SemanticModel.load()
    con = connect(ROOT / "data" / "lakehouse")
    eng = DeterministicNLToSQL(model, con=con)

    rev = con.execute("SELECT SUM(net_revenue) FROM fact_sales").fetchone()[0]
    orders = con.execute("SELECT COUNT(DISTINCT order_id) FROM fact_sales").fetchone()[0]
    aov = rev / orders
    custs = con.execute("SELECT COUNT(DISTINCT customer_key) FROM fact_sales").fetchone()[0]

    reg = con.execute(eng.generate("revenue by region").sql).fetchall()
    cat = con.execute(eng.generate("top 5 product categories by revenue").sql).fetchall()
    monthly = con.execute(eng.generate("monthly revenue in 2023").sql).fetchall()
    monthly.sort(key=lambda r: r[0])

    fig = plt.figure(figsize=(11.5, 7))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.8, 1.25, 1.25],
                          hspace=0.55, wspace=0.35)

    for i, (label, val, sub) in enumerate([
        ("Total Revenue", _fmt(rev), "50M line items"),
        ("Orders", _fmt(orders, False), "distinct baskets"),
        ("Avg Order Value", _fmt(aov), "revenue / orders"),
        ("Active Customers", _fmt(custs, False), "distinct buyers"),
    ]):
        kpi(fig.add_subplot(gs[0, i]), label, val, sub,
            color=PALETTE[i % len(PALETTE)])

    ax1 = fig.add_subplot(gs[1, :2])
    reg.sort(key=lambda r: r[1])
    ax1.barh([r[0] for r in reg], [r[1] / 1e6 for r in reg], color=ACCENT)
    ax1.set_title("Revenue by region", fontsize=11)
    ax1.set_xlabel("$M")

    ax2 = fig.add_subplot(gs[1, 2:])
    ax2.bar([r[0] for r in cat], [r[1] / 1e6 for r in cat], color=PALETTE[2])
    ax2.set_title("Top product categories by revenue", fontsize=11)
    ax2.set_ylabel("$M")
    ax2.tick_params(axis="x", rotation=20)

    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot([r[0] for r in monthly], [r[1] / 1e6 for r in monthly],
             "o-", color=GOOD, linewidth=2)
    ax3.set_title("Monthly revenue, 2023 (seasonal demand)", fontsize=11)
    ax3.set_xlabel("month"); ax3.set_ylabel("$M")
    ax3.set_xticks(range(1, 13))

    save_panel(fig, str(ASSETS / "analytics_dashboard.png"),
               suptitle="NL-to-SQL Lakehouse — analytics answered from natural language")
    con.close()


def main() -> None:
    apply_theme()
    ASSETS.mkdir(exist_ok=True)
    panel_latency()
    panel_accuracy()
    panel_star_schema()
    panel_dashboard()
    print("wrote 4 PNGs to", ASSETS)


if __name__ == "__main__":
    main()
