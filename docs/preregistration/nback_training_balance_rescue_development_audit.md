# N-back training-balance rescue development audit

Date recorded: 2026-07-27
Frozen rescue pre-registration commit: `7efd0af`
Frozen rescue implementation commit: `ca74be7`
Seed-family runner commit: `ecbd01f`
Perturbation outcomes inspected: **none**

## Decision

The training-balance rescue **passed its three-seed development gate**.
Development seeds `20260824-26` each passed two consecutive curriculum
validation checks and then passed every criterion on their untouched
1,024-sequence-per-condition competence banks.

Final-family training is eligible only after its seed banks are frozen in a
separate pre-registration.

## Untouched competence results

| Seed | Training steps | 0-back accuracy | 0-back discriminability | 2-back accuracy | 2-back discriminability | Lure accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `20260824` | 9,400 | 1.000 | 1.000 | 0.965 | 0.938 | 0.988 |
| `20260825` | 11,600 | 1.000 | 1.000 | 0.963 | 0.938 | 0.973 |
| `20260826` | 7,400 | 1.000 | 1.000 | 0.959 | 0.924 | 0.971 |

Values are rounded for display. Machine-readable values remain in the recorded
JSON outputs.

## Interpretation

The rescue transported across three independent recurrent initializations.
It preserved the easy 0-back control while raising held-out 2-back accuracy,
discriminability, and lure rejection above the frozen gates.

This supports the baseline engineering diagnosis: the initial actual N-back
task was learnable, but equal rule allocation and unweighted class prevalence
were inefficient for the high-load match decision. It does not support any
psilocybin-related perturbation mechanism.

## Recorded outputs

```text
outputs/nback_working_memory_balance_rescue/
outputs/nback_working_memory_balance_rescue/seed_sweep/seed_20260825/
outputs/nback_working_memory_balance_rescue/seed_sweep/seed_20260826/
```
