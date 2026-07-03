"""Run every experiment at production scale, print headline numbers, and persist
structured results (JSON + benchmark CSV/MD) that the screenshot renderer reads.

    python scripts/run_pipeline.py            # full run
    python scripts/run_pipeline.py --fast     # quicker, smaller run

This is the single heavy driver: everything downstream (screenshots, README
tables) consumes ``benchmarks/results.json`` so the models are trained once.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import csv
import json
import os
import time

from distillkit import experiments as E

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH = os.path.join(ROOT, "benchmarks")


def _cfg(fast: bool) -> dict:
    if fast:
        return {
            "lora": dict(dim=256, hidden=48, n_source=1600, n_target=1400, epochs=45),
            "distill": dict(dim=128, teacher_hidden=32, n_members=2, student_hidden=6,
                            n_source=1600, transfer_size=600, teacher_epochs=40, student_epochs=90),
            "scaling": dict(dim=128, hidden=32, sizes=(100, 400, 1000, 2000), n_eval=1000, epochs=45),
            "gate": dict(dim=128, hidden=32, n_source=1500, epochs=45),
        }
    return {
        "lora": dict(dim=512, hidden=64, n_source=3600, n_target=2600, epochs=65),
        "distill": dict(dim=256, teacher_hidden=48, n_members=3, student_hidden=8,
                        n_source=3000, transfer_size=1000, transfer_noise=0.30,
                        teacher_epochs=50, student_epochs=120),
        "scaling": dict(dim=256, hidden=48, sizes=(100, 250, 500, 1000, 2000, 4000),
                        n_eval=1500, epochs=50),
        "gate": dict(dim=256, hidden=48, n_source=3000, epochs=55),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="smaller/faster configuration")
    args = ap.parse_args()
    cfg = _cfg(args.fast)
    os.makedirs(BENCH, exist_ok=True)

    t0 = time.time()
    print("[1/4] LoRA vs full fine-tuning ...")
    lora = E.run_lora_experiment(**cfg["lora"])
    print(f"      base(src)={lora.base_source_acc:.3f}  base(tgt,0-shot)={lora.base_target_zeroshot:.3f}")
    print(f"      full FT={lora.full_target_acc:.3f} ({lora.full_trainable:,} params)  "
          f"LoRA={lora.lora_target_acc:.3f} ({lora.lora_trainable:,} params)")
    print(f"      trainable-param reduction = {lora.reduction_pct:.1f}%  merge_ok={lora.merge_ok}")

    print("[2/4] Knowledge distillation ...")
    dist = E.run_distillation(**cfg["distill"])
    print(f"      teacher={dist.teacher_acc:.3f} ({dist.teacher_params:,})  "
          f"student+KD={dist.student_kd_acc:.3f}  student-noKD={dist.student_nokd_acc:.3f} "
          f"({dist.student_params:,})")
    print(f"      KD accuracy recovery = {100 * dist.kd_recovery:.1f}%  "
          f"compression = {dist.compression:.1f}x  KD gain = {dist.student_kd_acc - dist.student_nokd_acc:+.3f}")

    print("[3/4] Data-scaling study ...")
    scaling = E.run_scaling_study(**cfg["scaling"])
    print("      " + "  ".join(f"{n}:{a:.3f}" for n, a in zip(scaling.sizes, scaling.accuracies)))

    print("[4/4] Eval-gate demo ...")
    gate = E.run_gate_demo(**cfg["gate"])
    print(f"      good acc={gate.good_acc:.3f} -> passed={gate.good.passed}   "
          f"bad acc={gate.bad_acc:.3f} -> passed={gate.bad.passed}")

    results = {
        "config": {"fast": args.fast},
        "lora": lora.as_dict(),
        "distill": {
            "teacher_acc": dist.teacher_acc,
            "student_kd_acc": dist.student_kd_acc,
            "student_nokd_acc": dist.student_nokd_acc,
            "teacher_params": dist.teacher_params,
            "student_params": dist.student_params,
            "kd_recovery": dist.kd_recovery,
            "compression": dist.compression,
            "temperature": dist.temperature,
            "alpha": dist.alpha,
            "kd_curve": dist.kd_curve,
            "nokd_curve": dist.nokd_curve,
        },
        "scaling": scaling.as_dict(),
        "gate": {
            "good_acc": gate.good_acc,
            "bad_acc": gate.bad_acc,
            "baseline_acc": gate.baseline_acc,
            "good_checks": [c.__dict__ for c in gate.good.checks],
            "bad_checks": [c.__dict__ for c in gate.bad.checks],
            "good_passed": gate.good.passed,
            "bad_passed": gate.bad.passed,
        },
    }
    with open(os.path.join(BENCH, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Flat CSV summaries.
    with open(os.path.join(BENCH, "lora.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "target_acc", "trainable_params"])
        w.writerow(["full_finetune", f"{lora.full_target_acc:.4f}", lora.full_trainable])
        w.writerow(["lora", f"{lora.lora_target_acc:.4f}", lora.lora_trainable])
    with open(os.path.join(BENCH, "scaling.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["train_size", "eval_accuracy"])
        for n, a in zip(scaling.sizes, scaling.accuracies):
            w.writerow([n, f"{a:.4f}"])

    _write_summary_md(results)
    print(f"\nDone in {time.time() - t0:.1f}s. Results -> benchmarks/results.json")


def _write_summary_md(r: dict) -> None:
    lo, di, ga = r["lora"], r["distill"], r["gate"]
    lines = [
        "# Benchmark Results\n",
        "_Surrogate MLP over hashed instruction-text features. Numbers produced by "
        "`scripts/run_pipeline.py`._\n",
        "## LoRA vs full fine-tuning (domain adaptation)\n",
        "| Method | Target-domain accuracy | Trainable params |",
        "|---|---|---|",
        f"| Full fine-tune | {lo['full_target_acc']:.3f} | {lo['full_trainable']:,} |",
        f"| **LoRA (r={lo['rank']})** | **{lo['lora_target_acc']:.3f}** | **{lo['lora_trainable']:,}** |",
        f"\n**Trainable-parameter reduction: {lo['reduction_pct']:.1f}%** at matching accuracy. "
        f"Adapter merges exactly (max logit diff {lo['merge_max_logit_diff']:.1e}).\n",
        "## Knowledge distillation\n",
        "| Model | Accuracy | Params |",
        "|---|---|---|",
        f"| Teacher (ensemble) | {di['teacher_acc']:.3f} | {di['teacher_params']:,} |",
        f"| **Student + KD** | **{di['student_kd_acc']:.3f}** | **{di['student_params']:,}** |",
        f"| Student, no KD | {di['student_nokd_acc']:.3f} | {di['student_params']:,} |",
        f"\n**KD accuracy recovery: {100 * di['kd_recovery']:.1f}%** of the teacher at "
        f"**{di['compression']:.1f}x** compression (T={di['temperature']}, alpha={di['alpha']}). "
        f"KD lifts the same-size student by {di['student_kd_acc'] - di['student_nokd_acc']:+.3f}.\n",
        "## Eval / promotion gate\n",
        f"- Good model: val acc {ga['good_acc']:.3f} -> **{'APPROVED' if ga['good_passed'] else 'BLOCKED'}**",
        f"- Deliberately-bad model: val acc {ga['bad_acc']:.3f} -> **{'APPROVED' if ga['bad_passed'] else 'BLOCKED'}**\n",
    ]
    with open(os.path.join(BENCH, "results.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
