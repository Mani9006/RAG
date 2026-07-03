from __future__ import annotations

import numpy as np

from distillkit.experiments import run_gate_demo
from distillkit.gate import evaluate_gate


def test_gate_blocks_bad_model_and_approves_good_one():
    demo = run_gate_demo(dim=128, hidden=32, n_source=1500, epochs=45)
    assert demo.good.passed, "a well-trained model should clear the gate"
    assert not demo.bad.passed, "a deliberately-bad model must be blocked"
    assert demo.bad_acc < demo.good_acc


def test_gate_absolute_floor():
    res = evaluate_gate(0.62, min_accuracy=0.80)
    assert not res.passed
    floor = [c for c in res.checks if c.name == "absolute_accuracy_floor"][0]
    assert not floor.passed and floor.value == 0.62


def test_gate_regression_check():
    # Above the absolute floor, but a regression vs the incumbent baseline.
    res = evaluate_gate(0.83, min_accuracy=0.80, baseline_accuracy=0.90, max_regression=0.02)
    assert not res.passed
    reg = [c for c in res.checks if c.name == "no_regression_vs_baseline"][0]
    assert not reg.passed


def test_gate_worst_class_recall():
    # Model ignores class 2 entirely -> worst-class recall = 0 -> blocked.
    y = np.array([0, 1, 2, 2, 0, 1])
    pred = np.array([0, 1, 0, 0, 0, 1])
    res = evaluate_gate(
        0.9, min_accuracy=0.5, pred=pred, y=y, n_classes=3, min_class_recall=0.5
    )
    assert not res.passed
    rec = [c for c in res.checks if c.name == "worst_class_recall_floor"][0]
    assert rec.value == 0.0


def test_gate_all_pass():
    res = evaluate_gate(0.95, min_accuracy=0.80, baseline_accuracy=0.90, max_regression=0.05)
    assert res.passed
    assert all(c.passed for c in res.checks)
    assert "APPROVED" in res.summary()
