from __future__ import annotations

import numpy as np

from distillkit import data
from distillkit.experiments import run_lora_experiment
from distillkit.features import HashingVectorizer
from distillkit.lora import LoRAModel
from distillkit.model import MLPClassifier


def _small_experiment():
    return run_lora_experiment(
        dim=256, hidden=48, rank=4, n_source=1400, n_target=1200, epochs=45, batch_size=512
    )


def test_lora_uses_far_fewer_params_but_matches_full_finetune():
    r = _small_experiment()
    # (1) Far fewer trainable parameters than full fine-tuning.
    assert r.lora_trainable < r.full_trainable
    assert r.reduction_pct > 60.0, f"expected big reduction, got {r.reduction_pct:.1f}%"
    # (2) Yet reaches the target accuracy on the adapted domain...
    assert r.lora_target_acc >= 0.80, f"LoRA under target (acc={r.lora_target_acc:.3f})"
    # (3) ...essentially matching full fine-tuning.
    assert r.lora_target_acc >= r.full_target_acc - 0.05


def test_base_starts_near_chance_on_shifted_domain():
    # The whole point of adaptation: the frozen base can't handle the new domain.
    r = _small_experiment()
    assert r.base_target_zeroshot < 0.5
    assert r.base_source_acc > 0.80  # but it's competent on its own domain


def test_adapter_merges_exactly():
    r = _small_experiment()
    assert r.merge_ok
    assert r.merge_max_logit_diff < 1e-9


def test_lora_freezes_base_and_merge_reproduces_logits():
    # Build a tiny end-to-end case to check freezing + merge equivalence directly.
    vec = HashingVectorizer(128)
    ds = data.generate(600, seed=0, domain="source")
    tr, _, te = data.split(ds, seed=0)
    Xtr, ytr = vec.transform(tr.texts), tr.y
    Xte = vec.transform(te.texts)

    base = MLPClassifier(128, 32, data.N_CLASSES, seed=0).fit(Xtr, ytr, epochs=20, lr=1e-2)
    W1_before = base.params["W1"].copy()
    b1_before = base.params["b1"].copy()

    lora = LoRAModel(base, r=4, alpha=8.0, seed=0)
    # ΔW starts at zero (B initialised to zero) => identical to base at init.
    assert np.allclose(lora.w1_eff(), base.params["W1"])

    lora.fit(Xtr, ytr, epochs=20, lr=1e-2)
    # Frozen base weights are untouched by adapter training.
    assert np.array_equal(base.params["W1"], W1_before)
    assert np.array_equal(base.params["b1"], b1_before)
    # The adapter actually moved (B left the zero init).
    assert np.abs(lora.adapter["B"]).max() > 0

    merged = lora.merge()
    assert np.allclose(merged.logits(Xte), lora.logits(Xte), atol=1e-10)


def test_lora_trainable_count_formula():
    base = MLPClassifier(200, 32, 9, seed=0)
    lora = LoRAModel(base, r=4, alpha=8.0, train_head=True)
    # A(r x H) + B(D x r) + head W2(H x C) + b2(C)
    expected = 4 * 32 + 200 * 4 + 32 * 9 + 9
    assert lora.n_trainable() == expected
    assert lora.n_trainable() < base.n_params()
