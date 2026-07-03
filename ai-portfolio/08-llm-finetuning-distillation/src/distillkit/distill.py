"""Knowledge distillation: a strong teacher ensemble -> a tiny student.

We build a **teacher** as a bagged ensemble of wider MLPs (higher accuracy, many
parameters) and distill its knowledge into a **student** with a fraction of the
parameters. The student is trained with the classic Hinton et al. (2015) loss:

    L = alpha · CE(hard_labels, student)
      + (1 - alpha) · T² · KL(softmax(teacher / T) || softmax(student / T))

The ``T²`` factor keeps soft-target gradients on the same scale as the hard-label
gradients. The soft targets carry the teacher's *dark knowledge* (its relative
confidence across wrong classes), which regularizes the small student and lets it
generalize better than an identically-sized student trained on hard labels alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import accuracy, softmax
from .model import MLPClassifier


@dataclass
class TeacherEnsemble:
    """Bagged ensemble of wide MLPs. Prediction = mean of member probabilities."""

    in_dim: int
    n_classes: int
    hidden: int = 128
    n_members: int = 5
    seed: int = 7
    l2: float = 1e-4
    members: list[MLPClassifier] = None  # type: ignore[assignment]

    def fit(self, X: np.ndarray, y: np.ndarray, *, epochs: int = 120, lr: float = 1e-2) -> "TeacherEnsemble":
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        self.members = []
        for m in range(self.n_members):
            # Bootstrap resample so members disagree (that disagreement is the
            # dark knowledge distillation transfers).
            boot = rng.integers(0, n, size=n)
            clf = MLPClassifier(self.in_dim, self.hidden, self.n_classes, seed=self.seed + 100 * m + 1, l2=self.l2)
            clf.fit(X[boot], y[boot], epochs=epochs, lr=lr)
            self.members.append(clf)
        return self

    def logits(self, X: np.ndarray) -> np.ndarray:
        """Mean logits across members (used to form temperature soft targets)."""
        return np.mean([mem.logits(X) for mem in self.members], axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.mean([mem.predict_proba(X) for mem in self.members], axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy(self.predict_proba(X), y)

    def n_params(self) -> int:
        return int(sum(mem.n_params() for mem in self.members))


def soft_targets(teacher: TeacherEnsemble, X: np.ndarray, temperature: float) -> np.ndarray:
    """Teacher soft labels ``softmax(mean_logits / T)`` for every row of ``X``."""
    return softmax(teacher.logits(X), temperature=temperature)


def _kd_output_grad(y_soft: np.ndarray, y_hard: np.ndarray, temperature: float, alpha: float):
    """Build an ``output_grad(logits, y_batch, idx) -> (loss, dlogits)`` closure.

    ``idx`` indexes into the precomputed ``y_soft`` matrix so each batch uses the
    correct per-example teacher distribution.
    """
    T = float(temperature)

    def output_grad(logits: np.ndarray, y_batch: np.ndarray, idx: np.ndarray):
        n = logits.shape[0]
        # Hard-label CE gradient.
        p_hard = softmax(logits)
        d_hard = p_hard.copy()
        d_hard[np.arange(n), y_batch] -= 1.0
        d_hard /= n
        # Soft-label (distillation) gradient at temperature T.
        p_soft = softmax(logits, temperature=T)
        q = y_soft[idx]
        # d/dz [ T^2 * KL(q || softmax(z/T)) ] = T * (softmax(z/T) - q)
        d_soft = (T * (p_soft - q)) / n
        d = alpha * d_hard + (1.0 - alpha) * d_soft
        return 0.0, d

    return output_grad


@dataclass
class DistillResult:
    student_kd: MLPClassifier
    student_nokd: MLPClassifier
    teacher_acc: float
    student_kd_acc: float
    student_nokd_acc: float
    teacher_params: int = 0
    student_params: int = 0
    kd_curve: list[float] = None  # type: ignore[assignment]
    nokd_curve: list[float] = None  # type: ignore[assignment]
    temperature: float = 0.0
    alpha: float = 0.0

    @property
    def kd_recovery(self) -> float:
        """Fraction of the teacher's accuracy the KD student recovers."""
        return self.student_kd_acc / self.teacher_acc if self.teacher_acc else 0.0

    @property
    def compression(self) -> float:
        """How many times smaller the student is than the teacher."""
        return self.teacher_params / self.student_params if self.student_params else 0.0


def distill(
    teacher: TeacherEnsemble,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    student_hidden: int = 8,
    temperature: float = 3.0,
    alpha: float = 0.3,
    epochs: int = 150,
    lr: float = 1e-2,
    seed: int = 21,
) -> DistillResult:
    """Train a KD student and an identical no-KD baseline; evaluate both."""
    in_dim = X_train.shape[1]
    n_classes = int(y_train.max()) + 1

    y_soft = soft_targets(teacher, X_train, temperature)
    kd_grad = _kd_output_grad(y_soft, y_train, temperature, alpha)

    student_kd = MLPClassifier(in_dim, student_hidden, n_classes, seed=seed)
    student_kd.fit(X_train, y_train, epochs=epochs, lr=lr, output_grad=kd_grad, eval_set=(X_eval, y_eval))

    # Same architecture, same seed, hard labels only — the controlled baseline.
    student_nokd = MLPClassifier(in_dim, student_hidden, n_classes, seed=seed)
    student_nokd.fit(X_train, y_train, epochs=epochs, lr=lr, eval_set=(X_eval, y_eval))

    return DistillResult(
        student_kd=student_kd,
        student_nokd=student_nokd,
        teacher_acc=teacher.score(X_eval, y_eval),
        student_kd_acc=student_kd.score(X_eval, y_eval),
        student_nokd_acc=student_nokd.score(X_eval, y_eval),
        teacher_params=teacher.n_params(),
        student_params=student_kd.n_params(),
        kd_curve=list(student_kd.history_),
        nokd_curve=list(student_nokd.history_),
        temperature=temperature,
        alpha=alpha,
    )
