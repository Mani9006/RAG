"""Layout-aware feature extraction per token.

Turns a columnar token table (see :func:`docintel.generator.documents_to_table`)
into a dense ``float32`` feature matrix combining three signal families that a
real document-AI model also relies on:

* **Positional** — normalized box geometry, page quadrant, reading-order rank.
  This is what makes the model *layout*-aware: a ``$1,240.00`` in the totals
  block (bottom-right) is a TOTAL, the same string inside the table is an
  LI_AMOUNT.
* **Textual** — glyph-pattern signals (is-amount, is-date, is-integer, casing)
  and printed-label keyword groups (money / id / date / table / party).
* **Neighbour** — reading-order context (does the previous token end in ``:`` or
  look like a printed key?), which is decisive for key→value typing.

The extractor is stateless and fully vectorized, so it streams to millions of
tokens. A real OCR/vision backend can feed the exact same columns.
"""
from __future__ import annotations

import re

import numpy as np

from . import PAGE_H, PAGE_W

_RE_AMOUNT = re.compile(r"^\$?\d{1,3}(,\d{3})*(\.\d{2})?$|^\$?\d+\.\d{2}$")
_RE_INT = re.compile(r"^\d+$")
_RE_DATE = re.compile(r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$")

_KW_MONEY = {"total", "subtotal", "tax", "amount", "balance", "due", "sub-total"}
_KW_ID = {"invoice", "no", "no.", "po", "number", "#", "ref", "id", "employee"}
_KW_DATE = {"date", "dated", "birth", "hire"}
_KW_TABLE = {"description", "qty", "quantity", "price", "unit", "amount", "item"}
_KW_PARTY = {"bill", "ship", "to", "vendor", "from", "sold", "name", "manager",
             "department", "location", "cost", "center"}

FEATURE_NAMES = [
    # positional
    "cx", "cy", "x0", "y0", "w_n", "h_n", "area", "aspect",
    "left_half", "right_third", "top_quarter", "bottom_quarter", "mid_band",
    "rank_in_doc",
    # textual
    "n_chars", "has_digit", "has_alpha", "has_dollar", "ends_colon",
    "is_amount", "is_int", "is_date", "is_upper", "is_title", "frac_digits",
    "kw_money", "kw_id", "kw_date", "kw_table", "kw_party", "kw_invoice", "kw_po",
    # neighbour (reading order)
    "prev_ends_colon", "prev_kw_money", "prev_kw_id", "prev_kw_date",
    "prev_kw_party", "prev_is_key_like",
    # 2-back context (disambiguates INVOICE_NO vs PO_NO from the printed key)
    "ctx_invoice", "ctx_po",
]


def _text_flags(text: str) -> tuple[float, ...]:
    low = text.lower()
    stripped = low.rstrip(":")
    n = len(text)
    digits = sum(c.isdigit() for c in text)
    has_digit = float(digits > 0)
    has_alpha = float(any(c.isalpha() for c in text))
    has_dollar = float("$" in text)
    ends_colon = float(text.endswith(":"))
    is_amount = float(bool(_RE_AMOUNT.match(text)))
    is_int = float(bool(_RE_INT.match(text)))
    is_date = float(bool(_RE_DATE.match(text)))
    is_upper = float(text.isupper() and has_alpha)
    is_title = float(text.istitle())
    frac_digits = digits / n if n else 0.0
    kw_money = float(stripped in _KW_MONEY)
    kw_id = float(stripped in _KW_ID)
    kw_date = float(stripped in _KW_DATE)
    kw_table = float(stripped in _KW_TABLE)
    kw_party = float(stripped in _KW_PARTY)
    kw_invoice = float(stripped == "invoice")
    kw_po = float(stripped in ("po", "purchase", "order"))
    return (float(n), has_digit, has_alpha, has_dollar, ends_colon,
            is_amount, is_int, is_date, is_upper, is_title, frac_digits,
            kw_money, kw_id, kw_date, kw_table, kw_party, kw_invoice, kw_po)


def extract_features(table: dict[str, list]) -> np.ndarray:
    """Build the dense feature matrix for a token table.

    ``table`` must contain the columns produced by ``documents_to_table``.
    Returns a ``float32`` array of shape ``(n_tokens, len(FEATURE_NAMES))``.
    """
    x = np.asarray(table["x"], dtype=np.float64)
    y = np.asarray(table["y"], dtype=np.float64)
    w = np.asarray(table["w"], dtype=np.float64)
    h = np.asarray(table["h"], dtype=np.float64)
    doc_id = np.asarray(table["doc_id"])
    token_idx = np.asarray(table["token_idx"], dtype=np.float64)
    texts = table["text"]
    n = len(texts)

    # --- positional (vectorized) ---
    cx = (x + w / 2) / PAGE_W
    cy = (y + h / 2) / PAGE_H
    x0 = x / PAGE_W
    y0 = y / PAGE_H
    w_n = w / PAGE_W
    h_n = h / PAGE_H
    area = w_n * h_n
    aspect = w / np.maximum(h, 1e-6)
    aspect = np.clip(aspect / 20.0, 0, 1)
    left_half = (cx < 0.5).astype(np.float64)
    right_third = (cx > 0.66).astype(np.float64)
    top_quarter = (cy < 0.25).astype(np.float64)
    bottom_quarter = (cy > 0.72).astype(np.float64)
    mid_band = ((cy >= 0.25) & (cy <= 0.72)).astype(np.float64)
    # reading-order rank within each doc, normalized 0..1
    rank = np.zeros(n)
    if n:
        # docs are contiguous; compute per-doc max token_idx via reduceat
        order = np.argsort(doc_id, kind="stable")
        # simple approach: normalize token_idx by per-doc max
        uniq, inv = np.unique(doc_id, return_inverse=True)
        maxidx = np.zeros(len(uniq))
        np.maximum.at(maxidx, inv, token_idx)
        denom = np.maximum(maxidx[inv], 1.0)
        rank = token_idx / denom

    pos = np.stack([cx, cy, x0, y0, w_n, h_n, area, aspect,
                    left_half, right_third, top_quarter, bottom_quarter,
                    mid_band, rank], axis=1)

    # --- textual (per-token) ---
    txt = np.empty((n, 18), dtype=np.float64)
    for i in range(n):
        txt[i] = _text_flags(texts[i])
    # txt column indices used below: ends_colon=4, kw_money=11, kw_id=12,
    # kw_date=13, kw_party=15, kw_invoice=16, kw_po=17

    # --- neighbour (previous reading-order token, same doc) ---
    prev_ends_colon = np.zeros(n)
    prev_kw_money = np.zeros(n)
    prev_kw_id = np.zeros(n)
    prev_kw_date = np.zeros(n)
    prev_kw_party = np.zeros(n)
    # txt columns: ends_colon=4, kw_money=11, kw_id=12, kw_date=13, kw_party=15
    same_doc = np.zeros(n, dtype=bool)
    same_doc[1:] = doc_id[1:] == doc_id[:-1]
    prev_ends_colon[1:] = txt[:-1, 4] * same_doc[1:]
    prev_kw_money[1:] = txt[:-1, 11] * same_doc[1:]
    prev_kw_id[1:] = txt[:-1, 12] * same_doc[1:]
    prev_kw_date[1:] = txt[:-1, 13] * same_doc[1:]
    prev_kw_party[1:] = txt[:-1, 15] * same_doc[1:]
    prev_is_key_like = np.clip(prev_ends_colon + prev_kw_money + prev_kw_id +
                               prev_kw_date + prev_kw_party, 0, 1)
    # 2-back context: does "Invoice"/"PO" appear within the two preceding tokens?
    ctx_invoice = np.zeros(n)
    ctx_po = np.zeros(n)
    same_doc2 = np.zeros(n, dtype=bool)
    same_doc2[2:] = doc_id[2:] == doc_id[:-2]
    ctx_invoice[1:] = txt[:-1, 16] * same_doc[1:]
    ctx_po[1:] = txt[:-1, 17] * same_doc[1:]
    ctx_invoice[2:] = np.clip(ctx_invoice[2:] + txt[:-2, 16] * same_doc2[2:], 0, 1)
    ctx_po[2:] = np.clip(ctx_po[2:] + txt[:-2, 17] * same_doc2[2:], 0, 1)
    nbr = np.stack([prev_ends_colon, prev_kw_money, prev_kw_id, prev_kw_date,
                    prev_kw_party, prev_is_key_like, ctx_invoice, ctx_po], axis=1)

    feats = np.concatenate([pos, txt, nbr], axis=1).astype(np.float32)
    assert feats.shape[1] == len(FEATURE_NAMES), (feats.shape[1], len(FEATURE_NAMES))
    return feats
