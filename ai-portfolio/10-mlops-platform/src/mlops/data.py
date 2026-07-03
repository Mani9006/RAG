"""Synthetic data generation with controllable covariate and concept drift.

The generator is parameterised so it can stream arbitrarily many rows in bounded
memory (see ``stream_batches`` and ``benchmarks/bench_scoring.py``). Two distinct
kinds of drift can be injected:

* **Covariate drift** shifts the marginal feature distributions. This is what a
  PSI monitor sees on the input stream.
* **Concept drift** changes the *relationship* between features and the label
  (the decision boundary rotates). This is what a performance monitor sees: a
  model trained before the shift degrades even on data it can still "explain".

Keeping the two knobs separate lets the tests assert that PSI reacts to covariate
drift and that accuracy reacts to concept drift.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .config import FEATURES

# Baseline logistic weights defining the "true" concept at t=0.
_BASE_WEIGHTS = np.array([1.3, -0.9, 0.7, -1.1, 1.6, -0.6])
# Post-drift concept: a rotation of the same weight magnitudes (permuted + sign
# flips). Signal strength is preserved so the new regime is just as learnable,
# but the decision boundary points in a different direction — a model fit to the
# old concept mislabels systematically, while one retrained on the new regime
# recovers full accuracy.
_TARGET_WEIGHTS = np.array([-1.1, 1.6, -0.6, 1.3, -0.9, 0.7])
_BASE_BIAS = -0.15
_BASE_MEANS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_BASE_STDS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


@dataclass(frozen=True)
class Batch:
    """A labelled batch of streamed scoring data."""

    X: np.ndarray  # shape (n, n_features)
    y: np.ndarray  # shape (n,)
    t: int         # window index in the stream
    drift: float   # covariate drift level applied to this batch


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def make_data(
    n: int,
    seed: int,
    covariate_drift: float = 0.0,
    concept_drift: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``n`` labelled rows at a given drift level.

    Parameters
    ----------
    n:
        Number of rows.
    seed:
        RNG seed (fully determines the output).
    covariate_drift:
        0 = reference distribution. Positive values shift feature means and
        inflate variance, raising PSI against the reference.
    concept_drift:
        0 = original concept. Positive values rotate the decision boundary so a
        model trained at concept_drift=0 loses accuracy.
    """
    rng = np.random.default_rng(seed)
    n_feat = len(FEATURES)

    # Covariate shift: move means along a fixed direction, grow spread.
    shift_dir = np.array([1.0, -0.7, 0.9, 0.8, 1.2, -0.5])
    means = _BASE_MEANS + covariate_drift * shift_dir
    stds = _BASE_STDS * (1.0 + 0.35 * covariate_drift)
    X = rng.normal(loc=means, scale=stds, size=(n, n_feat))

    # Concept shift: rotate the weight vector toward the target concept. At
    # concept_drift=1 the boundary is fully rotated but equally learnable.
    c = float(np.clip(concept_drift, 0.0, 1.0))
    weights = (1.0 - c) * _BASE_WEIGHTS + c * _TARGET_WEIGHTS
    logits = X @ weights + _BASE_BIAS
    probs = _sigmoid(logits)
    y = (rng.random(n) < probs).astype(np.int64)
    return X.astype(np.float64), y


def drift_level_at(
    t: int,
    drift_start: int,
    drift_plateau: int,
    max_level: float,
) -> float:
    """Drift fraction of ``max_level`` at window ``t``.

    Stationary until ``drift_start``, ramps linearly to ``max_level`` by
    ``drift_plateau``, then holds constant. The plateau is what lets a retrained
    model recover for good: once the world stops moving, a model fit to the new
    regime stays accurate.
    """
    if t < drift_start:
        return 0.0
    if t >= drift_plateau:
        return max_level
    frac = (t - drift_start + 1) / max(1, drift_plateau - drift_start)
    return max_level * frac


def stream_batches(
    n_windows: int,
    batch_size: int,
    seed: int,
    drift_start: int,
    drift_plateau: int,
    max_covariate: float,
    max_concept: float,
) -> Iterator[Batch]:
    """Yield ``n_windows`` batches with drift ramping to a plateau.

    Memory is bounded by ``batch_size`` regardless of ``n_windows`` — the stream
    can run indefinitely. Covariate and concept drift ramp together from
    ``drift_start`` to ``drift_plateau`` then hold, emulating a regime change that
    settles into a new steady state.
    """
    for t in range(n_windows):
        cov = drift_level_at(t, drift_start, drift_plateau, max_covariate)
        con = drift_level_at(t, drift_start, drift_plateau, max_concept)
        X, y = make_data(batch_size, seed=seed + 1000 + t, covariate_drift=cov, concept_drift=con)
        yield Batch(X=X, y=y, t=t, drift=cov)


def data_hash(X: np.ndarray, y: np.ndarray) -> str:
    """Content hash of a dataset, used as lineage fingerprint in the registry."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()[:16]
