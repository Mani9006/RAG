"""Multimodal Document Intelligence — layout-aware document understanding.

A fully offline, deterministic pipeline that turns *documents* (invoices,
receipts, forms, purchase orders) into structured data:

* ``generator``  — synthetic document generator. Every document is a page of
  word-level **tokens** with ``(text, x, y, w, h)`` bounding boxes and gold
  labels. This is the standard way to prototype document-AI with controllable
  ground truth (real OCR / LayoutLM drops in behind the same token interface).
* ``features``   — positional + textual + neighbour feature extraction per token.
* ``extraction`` — token-classification model (field typing), geometric
  key/value pairing, and line-item TABLE extraction via row/column clustering.
* ``router``     — document-type classifier / router.
* ``embedding``  — offline TF-IDF + SVD embeddings of extracted content
  (semantic search / dedup).
* ``evaluation`` — per-field precision/recall/F1, table accuracy, doc-type
  accuracy, throughput.
* ``render``     — renders a token layout to a PNG that looks like a real doc.

Everything is seeded and needs no network, GPU, or paid API.
"""
from __future__ import annotations

__version__ = "1.0.0"

SEED = 42

# Page geometry (US Letter, in points).
PAGE_W = 612.0
PAGE_H = 792.0

# Document types the router distinguishes.
DOC_TYPES = ["INVOICE", "RECEIPT", "PURCHASE_ORDER", "FORM"]

# Token field taxonomy. ``O`` is background; ``KEY``/``HEADER`` are printed
# labels; the rest are value/field types. Line-item cells are ``LI_*``.
LABELS = [
    "O",
    "KEY",
    "HEADER",
    "INVOICE_NO",
    "PO_NO",
    "DATE",
    "VENDOR",
    "BILL_TO",
    "SUBTOTAL",
    "TAX",
    "TOTAL",
    "LI_DESC",
    "LI_QTY",
    "LI_PRICE",
    "LI_AMOUNT",
]
LABEL_TO_ID = {lab: i for i, lab in enumerate(LABELS)}
ID_TO_LABEL = {i: lab for i, lab in enumerate(LABELS)}

# Line-item column labels in canonical left-to-right order.
LI_COLUMNS = ["LI_DESC", "LI_QTY", "LI_PRICE", "LI_AMOUNT"]
