# N-back competence-screened final-pool audit

## Audit status

Recorded on 2026-07-27 after the screened pool reached its registered target.

Frozen design commits:

- competence-screened pool pre-registration: `a31558c`;
- disjoint bank-seed addendum: `060a1c8`;
- screened-pool implementation: `530d300`.

Perturbation outcomes run or inspected before selection: **none**.

## Registered terminal result

The runner retained the first ten competent checkpoints from ten attempted
candidates:

```text
20260912, 20260913, 20260914, 20260915, 20260916,
20260917, 20260918, 20260919, 20260920, 20260921
```

All ten passed every unchanged untouched competence gate. The runner stopped
with:

```text
passed: true
stop_reason: target_reached
n_passed: 10
pool_pass_rate: 1.0
failed_seeds: []
unattempted_seeds: [20260922, 20260923, 20260924, 20260925, 20260926]
```

The five remaining candidates were not trained, as required by the
target-reached stop rule.

## Per-checkpoint untouched competence results

| Seed | Updates | 0-back accuracy | 0-back HR-FAR | 2-back accuracy | 2-back HR-FAR | 2-back lure accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260912 | 8,600 | 1.0000 | 1.0000 | 0.9721 | 0.9550 | 0.9863 |
| 20260913 | 9,000 | 1.0000 | 1.0000 | 0.9717 | 0.9443 | 0.9896 |
| 20260914 | 6,800 | 1.0000 | 1.0000 | 0.9598 | 0.9169 | 0.9856 |
| 20260915 | 7,800 | 1.0000 | 1.0000 | 0.9745 | 0.9556 | 0.9792 |
| 20260916 | 10,800 | 1.0000 | 1.0000 | 0.9859 | 0.9711 | 0.9965 |
| 20260917 | 9,400 | 1.0000 | 1.0000 | 0.9855 | 0.9655 | 0.9955 |
| 20260918 | 9,400 | 1.0000 | 1.0000 | 0.9575 | 0.9271 | 0.9641 |
| 20260919 | 8,800 | 0.9999 | 0.9998 | 0.9626 | 0.9348 | 0.9888 |
| 20260920 | 6,600 | 1.0000 | 1.0000 | 0.9777 | 0.9612 | 0.9941 |
| 20260921 | 8,200 | 1.0000 | 1.0000 | 0.9799 | 0.9650 | 0.9879 |

No failed competence check was recorded for any retained checkpoint. All
checkpoints were trained and evaluated on the NVIDIA GeForce RTX 3060 Laptop
GPU through the repository CUDA environment.

## Baseline-only calibration implication

Untouched 0-back accuracy is effectively at ceiling. Mean 0-back
cross-entropy nevertheless varies across checkpoints from approximately
`0.00145` to `0.00478` nats. Therefore neither proportional classification
error nor proportional change relative to baseline cross-entropy is a stable
matched-cost unit: the denominator is near zero and differs more than
threefold across checkpoints.

The next phase must be pre-registered before execution and must use a fresh,
unperturbed 0-back precision-reference bank to plan an additive
per-sequence-log-loss cost. The competence banks cannot be reused because
they participated in checkpoint selection.

## Decision

The ten retained checkpoints form the final N-back inferential family. They
are eligible for the registered **baseline-only calibration-precision**
phase.

They are not yet eligible for perturbation-strength calibration,
candidate-versus-P5 comparison, or a psilocybin-signature claim. Those steps
remain firewalled until the additive-cost definition, precision planning,
bank mapping, and failure rules are frozen and the baseline-only phase has
passed.

## Authoritative outputs

Central pool manifest:

```text
outputs/nback_working_memory_screened_final/metrics/
nback_working_memory_screened_final_screened_pool_summary.json
```

Per-seed checkpoints and untouched competence records:

```text
outputs/nback_working_memory_screened_final/seed_sweep/seed_<SEED>/
```
