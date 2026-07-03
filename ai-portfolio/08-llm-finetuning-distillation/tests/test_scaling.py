from __future__ import annotations

import numpy as np

from distillkit.experiments import run_scaling_study


def test_more_data_helps_and_is_monotonic():
    study = run_scaling_study(
        dim=128, hidden=32, sizes=(100, 400, 1600), n_eval=800, epochs=45
    )
    accs = np.array(study.accuracies)
    assert len(accs) == 3
    # More training data never hurts (allowing tiny noise), and clearly helps overall.
    assert np.all(np.diff(accs) >= -0.02)
    assert accs[-1] > accs[0]
