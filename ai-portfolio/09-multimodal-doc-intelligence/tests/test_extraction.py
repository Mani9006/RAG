from __future__ import annotations

import numpy as np

from docintel import LABELS
from docintel.evaluation import accuracy, macro_f1, prf_per_class
from docintel.extraction import (TokenClassifier, extract_table,
                                  group_entities, majority_baseline,
                                  pair_key_values)
from docintel.generator import documents_to_table, iter_documents


# --------------------------------------------------------------------------
# Token classifier beats the majority-class baseline.
# --------------------------------------------------------------------------
def test_classifier_beats_majority_baseline():
    train = documents_to_table(iter_documents(1600, seed=1))
    test = documents_to_table(iter_documents(500, seed=2))
    clf = TokenClassifier(seed=1).fit(train)
    pred = clf.predict_labels(test)
    acc = accuracy(test["label"], pred)
    base = majority_baseline(train["label"], test["label"])
    prf = prf_per_class(test["label"], pred, LABELS)
    assert base < 0.35  # sanity: no single class dominates
    assert acc > base + 0.4  # comfortably beats majority baseline
    assert macro_f1(prf) > 0.75


# --------------------------------------------------------------------------
# Key/value geometric pairing on a fixed layout.
# --------------------------------------------------------------------------
def _tok(text, x, y, w=40, h=12):
    return {"text": text, "x": float(x), "y": float(y), "w": float(w), "h": float(h)}


def test_kv_pairing_fixture():
    tokens = [
        _tok("Invoice", 400, 100, 42), _tok("No:", 446, 100, 22),
        _tok("12345", 492, 100, 40),
        _tok("Date:", 400, 122, 32),
        _tok("01/02/2020", 492, 122, 60),
        _tok("Total", 400, 150, 34),
        _tok("$99.50", 492, 150, 44),
    ]
    labels = ["KEY", "KEY", "INVOICE_NO", "KEY", "DATE", "KEY", "TOTAL"]
    ents = group_entities(tokens, labels)
    pairs = pair_key_values(ents)
    got = {p.key_text: (p.value_label, p.value_text) for p in pairs}
    assert got["Invoice No:"] == ("INVOICE_NO", "12345")
    assert got["Date:"] == ("DATE", "01/02/2020")
    assert got["Total"] == ("TOTAL", "$99.50")


def test_kv_pairing_below_fallback():
    # value sits directly below the key (no same-line value)
    tokens = [_tok("Bill", 60, 160, 24), _tok("To:", 86, 160, 20),
              _tok("Globex", 60, 178, 50)]
    labels = ["KEY", "KEY", "BILL_TO"]
    pairs = pair_key_values(group_entities(tokens, labels))
    assert len(pairs) == 1
    assert pairs[0].value_label == "BILL_TO"
    assert pairs[0].value_text == "Globex"


# --------------------------------------------------------------------------
# Table row/column clustering recovers a known grid.
# --------------------------------------------------------------------------
def test_table_clustering_recovers_grid():
    grid = [
        ["Steel Widget", "3", "$10.00", "$30.00"],
        ["Brass Gear", "5", "$4.00", "$20.00"],
        ["Nylon Bolt", "12", "$1.50", "$18.00"],
    ]
    col_x = {0: 60, 1: 300, 2: 380, 3: 480}
    col_lab = ["LI_DESC", "LI_QTY", "LI_PRICE", "LI_AMOUNT"]
    tokens, labels = [], []
    for r, row in enumerate(grid):
        y = 200 + r * 18
        for c, cell in enumerate(row):
            x = col_x[c]
            for word in cell.split():
                w = len(word) * 6
                tokens.append(_tok(word, x, y, w))
                labels.append(col_lab[c])
                x += w + 4
    out = extract_table(tokens, labels)
    assert out == grid


def test_table_ignores_non_line_items():
    tokens = [_tok("Total", 400, 150, 34), _tok("$5.00", 492, 150, 40)]
    labels = ["KEY", "TOTAL"]
    assert extract_table(tokens, labels) == []
