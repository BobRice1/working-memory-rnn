# N-back final checkpoint-family failure audit

Date recorded: 2026-07-27
Frozen final-family pre-registration commit: `b339814`
Frozen final configuration commit: `a4191b9`
Perturbation outcomes inspected: **none**

## Decision

The first final checkpoint, seed `20260901`, passed. Seed `20260902` then
failed its Stage 2 training-validation gate after the frozen 12,000 Stage 2
updates. The stop-on-first-failure runner correctly stopped before seeds
`20260903-10`, and it did not run the untouched competence bank for the failed
checkpoint.

The final family therefore failed and is ineligible for perturbation
calibration or outcome analysis.

## Failed seed result

Seed `20260902` passed the unchanged 0-back acquisition stage at global step
200. At the final Stage 2 validation:

| Metric | 0-back | 2-back | Required |
| --- | ---: | ---: | ---: |
| Accuracy | 1.000 | 0.847 | 0.950 |
| Barrett discriminability | 1.000 | 0.688 | 0.900 |
| One-back-lure accuracy | NA | 0.835 | 0.900 |
| Mean cross-entropy | 0.0024 | 0.3460 | descriptive |

All three high-load checks failed.

## Training trajectory

The 2-back solution emerged very late. Held-out accuracy remained at or near
the chance/majority solutions through global step 10,800, then improved:

| Global step | 2-back accuracy | Discriminability | Lure accuracy |
| ---: | ---: | ---: | ---: |
| 11,000 | 0.633 | 0.144 | 0.676 |
| 11,200 | 0.736 | 0.242 | 0.995 |
| 11,800 | 0.710 | 0.522 | 0.595 |
| 12,000 | 0.796 | 0.643 | 0.727 |
| 12,200 | 0.847 | 0.688 | 0.835 |

This differs from a stable majority-class collapse: every primary high-load
metric was rising at the frozen endpoint. The 12,000-update ceiling truncated
a late-learning trajectory for this initialization.

## Interpretation

The successful three-seed development family did not establish adequate
training-budget robustness across the final seed family. This is a baseline
optimization failure, not evidence about a perturbation mechanism.

The smallest evidence-aligned rescue is to extend the Stage 2 maximum while
holding the task, model, loss, sampling, validation frequency, and competence
gates fixed. Any such rescue requires new development and final seed families.

## Recorded outputs

```text
outputs/nback_working_memory_final/seed_sweep/seed_20260901/
outputs/nback_working_memory_final/seed_sweep/seed_20260902/
outputs/nback_working_memory_final/metrics/nback_working_memory_final_seed_sweep_summary.json
```
