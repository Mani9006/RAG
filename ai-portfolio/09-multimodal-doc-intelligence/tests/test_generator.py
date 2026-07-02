from __future__ import annotations

import numpy as np

from docintel import DOC_TYPES, LABELS, PAGE_H, PAGE_W
from docintel.generator import (documents_to_table, generate_document,
                                 iter_documents)


def test_determinism():
    a = generate_document(123, seed=7)
    b = generate_document(123, seed=7)
    assert a.doc_type == b.doc_type
    assert [t.text for t in a.tokens] == [t.text for t in b.tokens]
    assert [(t.x, t.y) for t in a.tokens] == [(t.x, t.y) for t in b.tokens]


def test_different_ids_differ():
    a = generate_document(1, seed=7)
    b = generate_document(2, seed=7)
    assert [t.text for t in a.tokens] != [t.text for t in b.tokens]


def test_tokens_on_page_and_labeled():
    for i in range(200):
        d = generate_document(i, seed=3)
        assert d.doc_type in DOC_TYPES
        assert len(d.tokens) > 0
        for t in d.tokens:
            assert 0 <= t.x <= PAGE_W
            assert 0 <= t.y <= PAGE_H
            assert t.w > 0 and t.h > 0
            assert t.label in LABELS


def test_invoice_has_required_fields():
    # find an invoice
    inv = next(d for d in iter_documents(50, seed=1) if d.doc_type == "INVOICE")
    labels = {t.label for t in inv.tokens}
    for required in ("INVOICE_NO", "DATE", "TOTAL", "LI_DESC", "LI_AMOUNT"):
        assert required in labels, required
    # line-item rows are grouped
    rgs = {t.row_group for t in inv.tokens if t.label.startswith("LI_")}
    assert rgs and min(rgs) == 0


def test_table_flattening_columns():
    cols = documents_to_table(iter_documents(20, seed=2))
    keys = {"doc_id", "doc_type", "token_idx", "text", "x", "y", "w", "h",
            "label", "row_group"}
    assert keys <= set(cols)
    n = len(cols["text"])
    assert all(len(cols[k]) == n for k in keys)


def test_type_mix_covers_all():
    types = {d.doc_type for d in iter_documents(400, seed=11)}
    assert types == set(DOC_TYPES)
