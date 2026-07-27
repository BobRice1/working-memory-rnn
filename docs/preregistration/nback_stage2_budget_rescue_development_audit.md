# N-back Stage 2 budget rescue development audit

Date recorded: 2026-07-27
Frozen rescue pre-registration commit: `4158040`
Frozen rescue configuration commit: `64759b8`
Perturbation outcomes inspected: **none**

## Decision

The Stage 2 budget rescue passed its three-seed development gate. Seeds
`20260827-29` each passed two consecutive curriculum-validation checks and
then passed every criterion on untouched 1,024-sequence-per-condition
competence banks.

## Untouched competence results

| Seed | Training steps | 0-back accuracy | 0-back discriminability | 2-back accuracy | 2-back discriminability | Lure accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `20260827` | 7,800 | 1.000 | 1.000 | 0.965 | 0.929 | 0.997 |
| `20260828` | 10,000 | 1.000 | 1.000 | 0.965 | 0.944 | 0.967 |
| `20260829` | 7,600 | 1.000 | 1.000 | 0.970 | 0.941 | 0.996 |

Values are rounded for display. Machine-readable values remain in the seed
sweep and per-checkpoint JSON outputs.

## Interpretation

The configuration transported across all three new initializations. Each seed
passed before the old 12,000-update ceiling, so this development family does
not by itself demonstrate that the additional budget was necessary. It does
show that extending the maximum did not impair the established baseline
solution.

The motivating final seed `20260902` remains excluded. The next test is an
entirely new ten-seed final family under the frozen 20,000-update maximum.

This is a baseline learnability result, not evidence for a perturbation
mechanism.

## Recorded outputs

```text
outputs/nback_working_memory_budget_rescue/
outputs/nback_working_memory_budget_rescue/seed_sweep/seed_20260827/
outputs/nback_working_memory_budget_rescue/seed_sweep/seed_20260828/
outputs/nback_working_memory_budget_rescue/seed_sweep/seed_20260829/
```
