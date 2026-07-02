from __future__ import annotations

import numpy as np

from docintel.evaluation import (accuracy, confusion_matrix, macro_f1,
                                  micro_f1, prf_per_class, table_cell_accuracy,
                                  table_shape_match)


def test_prf_known_values():
    # 2 classes, hand-computable
    y_true = ["A", "A", "A", "B", "B"]
    y_pred = ["A", "A", "B", "B", "B"]
    prf = prf_per_class(y_true, y_pred, ["A", "B"])
    # A: tp=2 fp=0 fn=1 -> P=1.0 R=2/3 F1=0.8
    assert prf["A"].precision == 1.0
    assert abs(prf["A"].recall - 2 / 3) < 1e-9
    assert abs(prf["A"].f1 - 0.8) < 1e-9
    # B: tp=2 fp=1 fn=0 -> P=2/3 R=1.0 F1=0.8
    assert abs(prf["B"].precision - 2 / 3) < 1e-9
    assert prf["B"].recall == 1.0
    assert abs(prf["B"].f1 - 0.8) < 1e-9
    assert prf["A"].support == 3 and prf["B"].support == 2


def test_perfect_and_macro():
    y = ["A", "B", "C", "A"]
    prf = prf_per_class(y, y, ["A", "B", "C"])
    assert all(v.f1 == 1.0 for v in prf.values())
    assert macro_f1(prf, exclude=()) == 1.0


def test_micro_f1_excludes_background():
    y_true = ["O", "O", "TOTAL", "DATE"]
    y_pred = ["O", "O", "TOTAL", "O"]
    # keep = TOTAL, DATE ; tp=1 (TOTAL) ; pred_pos=1 ; total_true=2
    # prec=1, rec=0.5 -> F1 = 2*1*.5/1.5 = 0.6667
    assert abs(micro_f1(y_true, y_pred) - 2 / 3) < 1e-9


def test_accuracy_and_confusion():
    y_true = ["A", "B", "A", "B"]
    y_pred = ["A", "B", "B", "B"]
    assert accuracy(y_true, y_pred) == 0.75
    cm = confusion_matrix(y_true, y_pred, ["A", "B"])
    assert cm.tolist() == [[1, 1], [0, 2]]


def test_table_cell_accuracy():
    gold = [["a", "b"], ["c", "d"]]
    assert table_cell_accuracy(gold, gold) == 1.0
    assert table_cell_accuracy(gold, [["a", "x"], ["c", "d"]]) == 0.75
    assert table_cell_accuracy(gold, []) == 0.0
    assert table_shape_match(gold, [["1", "2"], ["3", "4"]]) is True
    assert table_shape_match(gold, [["1", "2"]]) is False
