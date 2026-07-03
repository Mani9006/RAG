"""Render a token layout to a PNG that looks like a real scanned document.

``render_document`` draws the actual token text at its bounding box on a
paper-white page. ``render_with_detections`` additionally overlays predicted
field boxes + labels (color-coded per field family) and detected key→value
links — the "doc-AI product" screenshot.
"""
from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from . import PAGE_H, PAGE_W
from .extraction import Entity, KVPair

# Field-family colors for overlays (readable on white paper).
_FIELD_COLOR = {
    "INVOICE_NO": "#1f6feb", "PO_NO": "#1f6feb",
    "DATE": "#8250df",
    "VENDOR": "#0f7b6c", "BILL_TO": "#0f7b6c",
    "SUBTOTAL": "#bf8700", "TAX": "#bf8700", "TOTAL": "#cf222e",
    "LI_DESC": "#57606a", "LI_QTY": "#57606a",
    "LI_PRICE": "#57606a", "LI_AMOUNT": "#57606a",
    "KEY": "#8b949e", "HEADER": "#24292f",
}
_PAPER = "#fbfbf8"


def _draw_tokens(ax, tokens: list[dict]) -> None:
    for t in tokens:
        fs = max(5.0, t["h"] / 1.15 * 0.72)
        ax.text(t["x"], PAGE_H - t["y"], t["text"], fontsize=fs,
                ha="left", va="top", color="#1a1a1a", family="DejaVu Sans")


def _content_bottom(tokens: list[dict]) -> float:
    """Lowest content edge (top-origin y). Used to crop empty page space."""
    if not tokens:
        return PAGE_H
    return max(t["y"] + t["h"] for t in tokens)


def _setup_page(ax, title: str, tokens: list[dict] | None = None) -> None:
    bottom = _content_bottom(tokens) + 28 if tokens else PAGE_H
    bottom = min(bottom, PAGE_H)
    page_h = bottom
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(PAGE_H - page_h, PAGE_H)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(mpatches.FancyBboxPatch(
        (6, PAGE_H - page_h + 6), PAGE_W - 12, page_h - 12,
        boxstyle="round,pad=2", linewidth=1.2,
        edgecolor="#d0d7de", facecolor=_PAPER, zorder=0))
    if title:
        ax.set_title(title, color="#e6edf3", fontsize=12, fontweight="bold", pad=8)


def render_document(tokens: list[dict], path: str, title: str = "") -> None:
    from .viztheme import INK
    fig, ax = plt.subplots(figsize=(6.4, 8.3))
    fig.patch.set_facecolor(INK)
    _setup_page(ax, title, tokens)
    _draw_tokens(ax, tokens)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=INK)
    plt.close(fig)


def render_with_detections(tokens: list[dict], entities: list[Entity],
                           pairs: list[KVPair], path: str,
                           title: str = "Detected fields") -> None:
    from .viztheme import INK
    fig, ax = plt.subplots(figsize=(6.6, 8.6))
    fig.patch.set_facecolor(INK)
    _setup_page(ax, title, tokens)
    _draw_tokens(ax, tokens)

    # KV links first (under boxes)
    for p in pairs:
        kx, ky, kw, kh = p.key_box
        vx, vy, vw, vh = p.value_box
        ax.annotate("", xy=(vx, PAGE_H - vy - vh / 2),
                    xytext=(kx + kw, PAGE_H - ky - kh / 2),
                    arrowprops=dict(arrowstyle="->", color="#58a6ff",
                                    lw=1.0, alpha=0.7))

    for e in entities:
        if e.label in ("O",):
            continue
        color = _FIELD_COLOR.get(e.label, "#57606a")
        y = PAGE_H - e.y - e.h
        ax.add_patch(mpatches.Rectangle(
            (e.x - 2, y - 2), e.w + 4, e.h + 4, linewidth=1.3,
            edgecolor=color, facecolor=color, alpha=0.12, zorder=2))
        ax.add_patch(mpatches.Rectangle(
            (e.x - 2, y - 2), e.w + 4, e.h + 4, linewidth=1.3,
            edgecolor=color, facecolor="none", zorder=3))
        if e.label not in ("KEY", "HEADER") and not e.label.startswith("LI_"):
            ax.text(e.x - 2, y + e.h + 3, e.label, fontsize=5.2,
                    color="white", ha="left", va="bottom", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.12", fc=color, ec="none"))

    # legend
    fams = [("Identifier", "#1f6feb"), ("Date", "#8250df"),
            ("Party", "#0f7b6c"), ("Money", "#cf222e"),
            ("Line item", "#57606a"), ("Key", "#8b949e")]
    handles = [mpatches.Patch(color=c, label=n) for n, c in fams]
    leg = ax.legend(handles=handles, loc="lower center", ncol=3, fontsize=7,
                    bbox_to_anchor=(0.5, -0.06), frameon=False)
    for txt in leg.get_texts():
        txt.set_color("#e6edf3")
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=INK)
    plt.close(fig)
