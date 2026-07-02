"""End-to-end experiment drivers shared by scripts, tests, and screenshots.

Each function trains real models and returns a structured, serializable result.
Production-scale defaults are defined here; tests call the same functions with
small overrides so the test suite exercises the *identical* code path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from . import data
from .distill import DistillResult, TeacherEnsemble, distill
from .features import HashingVectorizer
from .gate import GateResult, evaluate_gate
from .lora import LoRAModel
from .model import MLPClassifier


def _featurize(ds: data.Dataset, vec: HashingVectorizer) -> tuple[np.ndarray, np.ndarray]:
    return vec.transform(ds.texts), ds.y


# ---------------------------------------------------------------------------
# 1. LoRA vs full fine-tuning (with a base pretrained model + domain shift)
# ---------------------------------------------------------------------------
@dataclass
class LoRAExperiment:
    dim: int
    hidden: int
    rank: int
    base_source_acc: float
    base_target_zeroshot: float
    full_target_acc: float
    lora_target_acc: float
    base_params: int
    full_trainable: int
    lora_trainable: int
    reduction_pct: float
    merge_max_logit_diff: float

    @property
    def merge_ok(self) -> bool:
        return self.merge_max_logit_diff < 1e-9

    def as_dict(self) -> dict:
        d = asdict(self)
        d["merge_ok"] = self.merge_ok
        return d


def run_lora_experiment(
    *,
    dim: int = 512,
    hidden: int = 64,
    rank: int = 4,
    alpha: float = 8.0,
    n_source: int = 3600,
    n_target: int = 2600,
    source_noise: float = 0.20,
    target_noise: float = 0.15,
    epochs: int = 70,
    lr: float = 1e-2,
    batch_size: int = 512,
    seed: int = 42,
) -> LoRAExperiment:
    vec = HashingVectorizer(dim)

    # Pretrain the base model on the SOURCE domain.
    src = data.generate(n_source, seed=seed, domain="source")
    s_tr, _s_va, s_te = data.split(src, seed=0)
    Xs_tr, ys_tr = _featurize(data.add_label_noise(s_tr, source_noise, seed=seed + 5), vec)
    Xs_te, ys_te = _featurize(s_te, vec)

    base = MLPClassifier(dim, hidden, data.N_CLASSES, seed=seed)
    base.fit(Xs_tr, ys_tr, epochs=epochs, lr=lr, batch_size=batch_size)
    base_source_acc = base.score(Xs_te, ys_te)

    # Adapt to the shifted TARGET domain (disjoint vocabulary).
    tgt = data.generate(n_target, seed=seed + 1, domain="target")
    t_tr, _t_va, t_te = data.split(tgt, seed=0)
    Xt_tr, yt_tr = _featurize(data.add_label_noise(t_tr, target_noise, seed=seed + 6), vec)
    Xt_te, yt_te = _featurize(t_te, vec)

    base_target_zeroshot = base.score(Xt_te, yt_te)

    # Full fine-tune: start from base weights, train everything.
    full = MLPClassifier(dim, hidden, data.N_CLASSES, seed=seed)
    full.params = {k: v.copy() for k, v in base.params.items()}
    full.fit(Xt_tr, yt_tr, epochs=epochs, lr=lr, batch_size=batch_size)

    # LoRA: freeze base, train adapter + head.
    lora = LoRAModel(base, r=rank, alpha=alpha, seed=0)
    lora.fit(Xt_tr, yt_tr, epochs=epochs, lr=lr, batch_size=batch_size)
    merged = lora.merge()
    merge_diff = float(np.max(np.abs(merged.logits(Xt_te) - lora.logits(Xt_te))))

    return LoRAExperiment(
        dim=dim,
        hidden=hidden,
        rank=rank,
        base_source_acc=base_source_acc,
        base_target_zeroshot=base_target_zeroshot,
        full_target_acc=full.score(Xt_te, yt_te),
        lora_target_acc=lora.score(Xt_te, yt_te),
        base_params=base.n_params(),
        full_trainable=full.n_params(),
        lora_trainable=lora.n_trainable(),
        reduction_pct=100.0 * (1.0 - lora.n_trainable() / full.n_params()),
        merge_max_logit_diff=merge_diff,
    )


# ---------------------------------------------------------------------------
# 2. Knowledge distillation
# ---------------------------------------------------------------------------
def run_distillation(
    *,
    dim: int = 256,
    teacher_hidden: int = 48,
    n_members: int = 3,
    student_hidden: int = 8,
    n_source: int = 3000,
    transfer_size: int = 1000,
    transfer_noise: float = 0.30,
    temperature: float = 4.0,
    alpha: float = 0.2,
    teacher_epochs: int = 55,
    student_epochs: int = 120,
    lr: float = 1e-2,
    seed: int = 7,
) -> DistillResult:
    vec = HashingVectorizer(dim)
    src = data.generate(n_source, seed=seed, domain="source")
    tr, _va, te = data.split(src, seed=0)
    Xtr, ytr = _featurize(tr, vec)  # teacher sees CLEAN labels
    Xte, yte = _featurize(te, vec)

    teacher = TeacherEnsemble(dim, data.N_CLASSES, hidden=teacher_hidden, n_members=n_members, seed=seed)
    teacher.fit(Xtr, ytr, epochs=teacher_epochs, lr=lr)

    # Transfer set: a noisy-labelled subset (mimics cheap/noisy downstream labels).
    rng = np.random.default_rng(seed + 3)
    idx = rng.permutation(len(ytr))[:transfer_size]
    Xsub = Xtr[idx]
    y_clean = ytr[idx]
    y_noisy = y_clean.copy()
    flip = rng.random(len(y_noisy)) < transfer_noise
    y_noisy[flip] = rng.integers(0, data.N_CLASSES, size=int(flip.sum()))

    return distill(
        teacher,
        Xsub,
        y_noisy,
        Xte,
        yte,
        student_hidden=student_hidden,
        temperature=temperature,
        alpha=alpha,
        epochs=student_epochs,
        lr=lr,
        seed=seed + 14,
    )


# ---------------------------------------------------------------------------
# 3. Synthetic-data scaling study
# ---------------------------------------------------------------------------
@dataclass
class ScalingStudy:
    sizes: list[int] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"sizes": self.sizes, "accuracies": self.accuracies}


def run_scaling_study(
    *,
    dim: int = 256,
    hidden: int = 32,
    sizes: tuple[int, ...] = (25, 50, 100, 250, 500, 1000, 2000, 4000),
    n_eval: int = 1500,
    epochs: int = 45,
    lr: float = 1e-2,
    batch_size: int = 256,
    train_noise: float = 0.15,
    seed: int = 3,
) -> ScalingStudy:
    # A deliberately *harder* regime (few signal words, many distractors, noisy
    # training labels) so that data volume actually matters and the curve is not
    # saturated from the first point.
    hard = dict(n_signal=2, n_noise=6, distractor_p=0.45)
    vec = HashingVectorizer(dim)
    pool = data.generate(max(sizes), seed=seed, domain="source", **hard)
    pool = data.add_label_noise(pool, train_noise, seed=seed + 1)
    Xpool, ypool = _featurize(pool, vec)
    ev = data.generate(n_eval, seed=seed + 99, domain="source", **hard)  # clean eval labels
    Xev, yev = _featurize(ev, vec)

    accs = []
    for m in sizes:
        clf = MLPClassifier(dim, hidden, data.N_CLASSES, seed=seed)
        clf.fit(Xpool[:m], ypool[:m], epochs=epochs, lr=lr, batch_size=batch_size)
        accs.append(clf.score(Xev, yev))
    return ScalingStudy(sizes=list(sizes), accuracies=accs)


# ---------------------------------------------------------------------------
# 4. Eval-gate demo (a good model passes, a deliberately-bad one is blocked)
# ---------------------------------------------------------------------------
@dataclass
class GateDemo:
    good: GateResult
    bad: GateResult
    good_acc: float
    bad_acc: float
    baseline_acc: float


def run_gate_demo(
    *,
    dim: int = 256,
    hidden: int = 48,
    n_source: int = 3000,
    min_accuracy: float = 0.80,
    max_regression: float = 0.02,
    epochs: int = 60,
    lr: float = 1e-2,
    seed: int = 11,
) -> GateDemo:
    vec = HashingVectorizer(dim)
    src = data.generate(n_source, seed=seed, domain="source")
    tr, va, _te = data.split(src, seed=0)
    Xtr, ytr = _featurize(tr, vec)
    Xva, yva = _featurize(va, vec)

    baseline_acc = min_accuracy + 0.05  # a plausible incumbent to regress against

    good = MLPClassifier(dim, hidden, data.N_CLASSES, seed=seed).fit(Xtr, ytr, epochs=epochs, lr=lr)
    good_acc = good.score(Xva, yva)
    good_pred = good.predict(Xva)
    good_gate = evaluate_gate(
        good_acc,
        min_accuracy=min_accuracy,
        baseline_accuracy=baseline_acc,
        max_regression=max_regression,
        pred=good_pred,
        y=yva,
        n_classes=data.N_CLASSES,
        min_class_recall=0.5,
    )

    # Deliberately-bad candidate: trained on CORRUPTED labels (a broken data
    # pipeline / label bug). It cannot generalise, so it must be blocked.
    rng = np.random.default_rng(seed + 7)
    y_corrupt = ytr[rng.permutation(len(ytr))]
    bad = MLPClassifier(dim, hidden, data.N_CLASSES, seed=seed).fit(Xtr, y_corrupt, epochs=epochs, lr=lr)
    bad_acc = bad.score(Xva, yva)
    bad_pred = bad.predict(Xva)
    bad_gate = evaluate_gate(
        bad_acc,
        min_accuracy=min_accuracy,
        baseline_accuracy=baseline_acc,
        max_regression=max_regression,
        pred=bad_pred,
        y=yva,
        n_classes=data.N_CLASSES,
        min_class_recall=0.5,
    )

    return GateDemo(
        good=good_gate,
        bad=bad_gate,
        good_acc=good_acc,
        bad_acc=bad_acc,
        baseline_acc=baseline_acc,
    )
