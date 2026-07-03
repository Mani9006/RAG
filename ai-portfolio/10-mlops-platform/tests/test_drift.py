"""Drift detection: PSI rises on a shifted stream; performance monitor fires."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops.config import FEATURES
from mlops.data import make_data, stream_batches
from mlops.drift import DriftMonitor, PerformanceMonitor, kl_divergence, psi


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert psi(x, x, bins=10) < 1e-9


def test_psi_increases_with_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 5000)
    small = rng.normal(0.3, 1, 5000)
    large = rng.normal(1.5, 1.6, 5000)
    p_small = psi(ref, small)
    p_large = psi(ref, large)
    assert 0 < p_small < p_large
    assert p_large > 0.25  # major-shift regime


def test_kl_nonnegative_and_grows_with_shift():
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 5000)
    near = rng.normal(0.1, 1, 5000)
    far = rng.normal(2.0, 1, 5000)
    assert kl_divergence(ref, ref) >= -1e-9
    assert kl_divergence(ref, far) > kl_divergence(ref, near)


def test_psi_rises_monotonically_on_shifted_stream():
    """The core drift-monitoring guarantee: as the stream drifts, PSI climbs
    from stable (<0.1) into the alerting regime (>0.25)."""
    ref_X, _ = make_data(8000, seed=3, covariate_drift=0.0)
    mon = DriftMonitor(ref_X, FEATURES, bins=10, psi_threshold=0.25)

    early_max, late_max = [], []
    fired_any = False
    for b in stream_batches(20, 1200, seed=3, drift_start=5, drift_plateau=12,
                            max_covariate=1.5, max_concept=1.0):
        per_feat, alert = mon.check(b.X, b.t)
        m = max(per_feat.values())
        if b.t < 5:
            early_max.append(m)
        if b.t >= 12:
            late_max.append(m)
        fired_any = fired_any or (alert is not None)

    assert max(early_max) < 0.1          # stable before drift
    assert min(late_max) > 0.25          # clearly drifted after plateau
    assert max(late_max) > 5 * max(early_max)
    assert fired_any                     # an alert was raised


def test_performance_monitor_requires_sustained_drop():
    mon = PerformanceMonitor(baseline_accuracy=0.85, rel_drop_threshold=0.10, window=3)
    # A single bad window must NOT fire (transient).
    _, a0 = mon.update(0.84, 0)
    _, a1 = mon.update(0.60, 1)  # one bad window, window not yet full of drops
    assert a0 is None and a1 is None
    # Sustained degradation fires.
    _, a2 = mon.update(0.55, 2)
    _, a3 = mon.update(0.50, 3)
    assert a3 is not None
    assert a3.kind == "performance_drift"


def test_performance_monitor_stable_never_fires():
    mon = PerformanceMonitor(baseline_accuracy=0.85, rel_drop_threshold=0.10, window=3)
    alerts = [mon.update(0.84, t)[1] for t in range(10)]
    assert all(a is None for a in alerts)
