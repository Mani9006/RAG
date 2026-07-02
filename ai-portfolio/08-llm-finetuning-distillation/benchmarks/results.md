# Benchmark Results

_Surrogate MLP over hashed instruction-text features. Numbers produced by `scripts/run_pipeline.py`._

## LoRA vs full fine-tuning (domain adaptation)

| Method | Target-domain accuracy | Trainable params |
|---|---|---|
| Full fine-tune | 0.934 | 33,417 |
| **LoRA (r=4)** | **0.934** | **2,889** |

**Trainable-parameter reduction: 91.4%** at matching accuracy. Adapter merges exactly (max logit diff 0.0e+00).

## Knowledge distillation

| Model | Accuracy | Params |
|---|---|---|
| Teacher (ensemble) | 1.000 | 38,331 |
| **Student + KD** | **1.000** | **2,137** |
| Student, no KD | 0.844 | 2,137 |

**KD accuracy recovery: 100.0%** of the teacher at **17.9x** compression (T=4.0, alpha=0.2). KD lifts the same-size student by +0.156.

## Eval / promotion gate

- Good model: val acc 0.998 -> **APPROVED**
- Deliberately-bad model: val acc 0.138 -> **BLOCKED**
