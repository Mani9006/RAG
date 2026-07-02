from __future__ import annotations

from distillkit.experiments import run_distillation


def _small_distill():
    return run_distillation(
        dim=128,
        teacher_hidden=32,
        n_members=2,
        student_hidden=6,
        n_source=1600,
        transfer_size=600,
        transfer_noise=0.35,
        teacher_epochs=40,
        student_epochs=90,
    )


def test_kd_student_beats_no_kd_student_of_same_size():
    r = _small_distill()
    # Same architecture, same seed, same data — only the loss differs.
    assert r.student_kd.hidden == r.student_nokd.hidden
    assert r.student_kd.n_params() == r.student_nokd.n_params()
    assert r.student_kd_acc > r.student_nokd_acc + 0.03, (
        f"KD did not help: kd={r.student_kd_acc:.3f} nokd={r.student_nokd_acc:.3f}"
    )


def test_kd_student_recovers_most_of_teacher_at_fraction_of_params():
    r = _small_distill()
    assert r.kd_recovery > 0.90, f"KD recovery only {100 * r.kd_recovery:.1f}%"
    # The student is much smaller than the teacher ensemble.
    assert r.student_params * 4 < r.teacher_params
    assert r.compression > 4.0


def test_kd_curves_recorded():
    r = _small_distill()
    assert len(r.kd_curve) == 90 and len(r.nokd_curve) == 90
    # KD curve ends above the no-KD curve.
    assert r.kd_curve[-1] > r.nokd_curve[-1]
