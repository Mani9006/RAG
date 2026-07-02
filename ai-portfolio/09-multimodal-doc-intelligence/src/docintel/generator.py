"""Synthetic document generator.

Each document is a **structured layout**: a list of word-level tokens, every
token carrying its text, an ``(x, y, w, h)`` bounding box (points, origin
top-left), a gold ``label`` from :data:`docintel.LABELS`, and a ``row_group``
(the visual line-item row index, or ``-1``). This mirrors what a real OCR /
vision model emits (Tesseract words, LayoutLM tokens) but with perfect,
controllable ground truth.

The generator is a *stream*: :func:`iter_documents` yields one document at a
time so we can produce millions of documents (billions of tokens) in bounded
memory. :func:`documents_to_table` flattens a batch to a columnar dict ready
for pyarrow/DuckDB.

Determinism: document ``i`` is a pure function of ``(seed, i)`` via a per-doc
``numpy`` PCG64 stream, so generation is reproducible *and* embarrassingly
parallel / shardable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from . import DOC_TYPES, PAGE_H, PAGE_W

# ---------------------------------------------------------------------------
# Synthetic vocab (no real company names / data).
# ---------------------------------------------------------------------------
_VENDOR_PREFIX = ["Acme", "Globex", "Initech", "Umbrella", "Soylent", "Vandelay",
                  "Stark", "Wayne", "Wonka", "Cyberdyne", "Hooli", "Pied Piper",
                  "Aperture", "Tyrell", "Nakatomi", "Gekko", "Bluth", "Prestige"]
_VENDOR_SUFFIX = ["LLC", "Inc", "Corp", "Co", "Group", "Industries", "Systems",
                  "Partners", "Holdings", "Labs", "Trading", "Supply"]
_STREETS = ["Market", "Main", "Oak", "Maple", "Pine", "Cedar", "Elm", "Broad",
            "Lake", "Hill", "River", "Sunset", "Union", "King", "Queen"]
_CITIES = ["Springfield", "Riverton", "Fairview", "Lakewood", "Georgetown",
           "Kingston", "Ashland", "Clinton", "Franklin", "Greenville"]
_STATES = ["CA", "NY", "TX", "WA", "IL", "MA", "CO", "OR", "GA", "FL"]
_ITEMS = ["Widget", "Gadget", "Bracket", "Fastener", "Cable", "Adapter",
          "Module", "Sensor", "Panel", "Bearing", "Valve", "Connector",
          "Battery", "Resistor", "Filter", "Gasket", "Sprocket", "Coupling",
          "Actuator", "Relay", "Bushing", "Flange", "Grommet", "Spindle"]
_ITEM_ADJ = ["Steel", "Brass", "Nylon", "Carbon", "Alloy", "Ceramic", "Copper",
             "Titanium", "Rubber", "Composite", "Aluminum", "Plastic"]
_FORM_FIELDS = [
    ("Full Name", "VENDOR"),
    ("Employee ID", "INVOICE_NO"),
    ("Department", "BILL_TO"),
    ("Date of Birth", "DATE"),
    ("Hire Date", "DATE"),
    ("Manager", "VENDOR"),
    ("Cost Center", "PO_NO"),
    ("Location", "BILL_TO"),
]

# Approximate glyph width as a fraction of font size (DejaVu-ish, monospaced-ish
# estimate good enough for a synthetic layout).
_CHAR_W = 0.52


@dataclass
class Token:
    text: str
    x: float
    y: float
    w: float
    h: float
    label: str
    row_group: int = -1


@dataclass
class Document:
    doc_id: int
    doc_type: str
    tokens: list[Token] = field(default_factory=list)
    # gold line-item grid: rows of {col_label: text}
    table: list[dict[str, str]] = field(default_factory=list)


def _text_w(text: str, fs: float) -> float:
    return max(4.0, len(text) * fs * _CHAR_W)


class _Cursor:
    """Places word tokens left-to-right on a line, tracking a bounding box."""

    def __init__(self, doc: Document) -> None:
        self.doc = doc

    def put_words(self, text: str, x: float, y: float, fs: float, label: str,
                  row_group: int = -1) -> float:
        """Emit each whitespace word as a token; return the x after the text."""
        cx = x
        for word in text.split():
            w = _text_w(word, fs)
            self.doc.tokens.append(Token(word, cx, y, w, fs * 1.15, label, row_group))
            cx += w + fs * _CHAR_W  # inter-word space
        return cx


def _money(rng: np.random.Generator, lo: float, hi: float) -> float:
    return round(float(rng.uniform(lo, hi)), 2)


def _date(rng: np.random.Generator) -> str:
    y = int(rng.integers(2018, 2026))
    m = int(rng.integers(1, 13))
    d = int(rng.integers(1, 28))
    return f"{m:02d}/{d:02d}/{y}"


def _vendor(rng: np.random.Generator) -> str:
    p = _VENDOR_PREFIX[rng.integers(len(_VENDOR_PREFIX))]
    s = _VENDOR_SUFFIX[rng.integers(len(_VENDOR_SUFFIX))]
    return f"{p} {s}"


def _address(rng: np.random.Generator) -> str:
    num = int(rng.integers(10, 9999))
    st = _STREETS[rng.integers(len(_STREETS))]
    city = _CITIES[rng.integers(len(_CITIES))]
    state = _STATES[rng.integers(len(_STATES))]
    zc = int(rng.integers(10000, 99999))
    return f"{num} {st} St, {city}, {state} {zc}"


def _line_items(rng: np.random.Generator, n: int) -> list[dict[str, str]]:
    rows = []
    for _ in range(n):
        adj = _ITEM_ADJ[rng.integers(len(_ITEM_ADJ))]
        item = _ITEMS[rng.integers(len(_ITEMS))]
        qty = int(rng.integers(1, 40))
        price = _money(rng, 1.5, 480.0)
        amount = round(qty * price, 2)
        rows.append({
            "LI_DESC": f"{adj} {item}",
            "LI_QTY": str(qty),
            "LI_PRICE": f"${price:,.2f}",
            "LI_AMOUNT": f"${amount:,.2f}",
        })
    return rows


# Column x-positions for the line-item table.
_COL_X = {"LI_DESC": 60.0, "LI_QTY": 300.0, "LI_PRICE": 380.0, "LI_AMOUNT": 480.0}
_HDR_TEXT = {"LI_DESC": "Description", "LI_QTY": "Qty",
             "LI_PRICE": "Unit Price", "LI_AMOUNT": "Amount"}


def _emit_table(cur: _Cursor, rng: np.random.Generator, rows: list[dict[str, str]],
                y0: float, fs: float) -> float:
    """Emit table header + rows; return y after the table."""
    for col, x in _COL_X.items():
        cur.put_words(_HDR_TEXT[col], x, y0, fs, "HEADER")
    y = y0 + fs * 1.9
    for r_idx, row in enumerate(rows):
        for col, x in _COL_X.items():
            cur.put_words(row[col], x, y, fs, col, row_group=r_idx)
        y += fs * 1.7
    return y


def _emit_totals(cur: _Cursor, rng: np.random.Generator, rows: list[dict[str, str]],
                 y0: float, fs: float, with_tax: bool = True) -> None:
    subtotal = round(sum(float(r["LI_AMOUNT"].replace("$", "").replace(",", ""))
                         for r in rows), 2)
    tax_rate = float(rng.uniform(0.04, 0.095))
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + (tax if with_tax else 0.0), 2)
    key_x, val_x = 380.0, 480.0
    y = y0
    cur.put_words("Subtotal", key_x, y, fs, "KEY")
    cur.put_words(f"${subtotal:,.2f}", val_x, y, fs, "SUBTOTAL")
    y += fs * 1.7
    if with_tax:
        cur.put_words("Tax", key_x, y, fs, "KEY")
        cur.put_words(f"${tax:,.2f}", val_x, y, fs, "TAX")
        y += fs * 1.7
    cur.put_words("Total", key_x, y, fs, "KEY")
    cur.put_words(f"${total:,.2f}", val_x, y, fs, "TOTAL")


def _gen_invoice(doc: Document, rng: np.random.Generator, po: bool = False) -> None:
    cur = _Cursor(doc)
    title = "PURCHASE ORDER" if po else "INVOICE"
    cur.put_words(title, 60, 55, 22, "HEADER")
    # vendor block top-left
    cur.put_words(_vendor(rng), 60, 100, 13, "VENDOR")
    cur.put_words(_address(rng), 60, 120, 9, "O")
    # metadata top-right (key/value)
    id_key = "PO No" if po else "Invoice No"
    id_label = "PO_NO" if po else "INVOICE_NO"
    id_val = f"{rng.integers(1000, 99999)}"
    cur.put_words(f"{id_key}:", 400, 100, 11, "KEY")
    cur.put_words(id_val, 490, 100, 11, id_label)
    cur.put_words("Date:", 400, 120, 11, "KEY")
    cur.put_words(_date(rng), 490, 120, 11, "DATE")
    # bill-to block
    cur.put_words("Bill To:", 60, 160, 11, "KEY")
    cur.put_words(_vendor(rng), 60, 178, 11, "BILL_TO")
    # table
    n = int(rng.integers(3, 11))
    rows = _line_items(rng, n)
    doc.table = rows
    y = _emit_table(cur, rng, rows, 220, 10)
    # totals
    _emit_totals(cur, rng, rows, y + 14, 11, with_tax=True)


def _gen_receipt(doc: Document, rng: np.random.Generator) -> None:
    cur = _Cursor(doc)
    cur.put_words(_vendor(rng), 60, 55, 15, "VENDOR")
    cur.put_words(_address(rng), 60, 78, 8, "O")
    cur.put_words("Date:", 60, 100, 10, "KEY")
    cur.put_words(_date(rng), 130, 100, 10, "DATE")
    n = int(rng.integers(2, 7))
    rows = _line_items(rng, n)
    doc.table = rows
    y = _emit_table(cur, rng, rows, 130, 9)
    _emit_totals(cur, rng, rows, y + 12, 10, with_tax=bool(rng.integers(0, 2)))


def _gen_form(doc: Document, rng: np.random.Generator) -> None:
    cur = _Cursor(doc)
    cur.put_words("EMPLOYEE RECORD FORM", 60, 55, 18, "HEADER")
    fields = list(_FORM_FIELDS)
    rng.shuffle(fields)
    fields = fields[: int(rng.integers(5, len(fields) + 1))]
    y = 110.0
    for key_text, val_label in fields:
        cur.put_words(f"{key_text}:", 60, y, 11, "KEY")
        if val_label == "DATE":
            val = _date(rng)
        elif val_label in ("INVOICE_NO", "PO_NO"):
            val = f"{rng.integers(1000, 999999)}"
        else:
            val = _vendor(rng) if rng.integers(0, 2) else \
                _CITIES[rng.integers(len(_CITIES))]
        cur.put_words(val, 220, y, 11, val_label)
        y += 30.0


_GEN = {
    "INVOICE": lambda d, r: _gen_invoice(d, r, po=False),
    "PURCHASE_ORDER": lambda d, r: _gen_invoice(d, r, po=True),
    "RECEIPT": _gen_receipt,
    "FORM": _gen_form,
}

# Class mix (sums to 1); invoices dominate as in real AP pipelines.
_TYPE_P = np.array([0.45, 0.20, 0.15, 0.20])


def generate_document(doc_id: int, seed: int = 0) -> Document:
    """Deterministically generate document ``doc_id`` (pure fn of seed+id)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, doc_id]))
    doc_type = DOC_TYPES[int(rng.choice(len(DOC_TYPES), p=_TYPE_P))]
    doc = Document(doc_id=doc_id, doc_type=doc_type)
    _GEN[doc_type](doc, rng)
    # Clamp everything to the page.
    for t in doc.tokens:
        t.x = float(np.clip(t.x, 0, PAGE_W - 2))
        t.y = float(np.clip(t.y, 0, PAGE_H - 2))
    return doc


def iter_documents(n: int, seed: int = 0, start: int = 0) -> Iterator[Document]:
    """Stream ``n`` documents (bounded memory — scales to billions)."""
    for i in range(start, start + n):
        yield generate_document(i, seed=seed)


def documents_to_table(docs: Iterator[Document]) -> dict[str, list]:
    """Flatten documents to a columnar token table (pyarrow/DuckDB-ready)."""
    cols: dict[str, list] = {k: [] for k in
                             ("doc_id", "doc_type", "token_idx", "text",
                              "x", "y", "w", "h", "label", "row_group")}
    for d in docs:
        for j, t in enumerate(d.tokens):
            cols["doc_id"].append(d.doc_id)
            cols["doc_type"].append(d.doc_type)
            cols["token_idx"].append(j)
            cols["text"].append(t.text)
            cols["x"].append(t.x)
            cols["y"].append(t.y)
            cols["w"].append(t.w)
            cols["h"].append(t.h)
            cols["label"].append(t.label)
            cols["row_group"].append(t.row_group)
    return cols
