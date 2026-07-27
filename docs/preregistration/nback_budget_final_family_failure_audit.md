# N-back budget-rescue final-family failure audit

Date recorded: 2026-07-27
Frozen final-family pre-registration commit: `f0082e1`
Frozen final configuration commit: `d986f1b`
Perturbation outcomes inspected: **none**

## Decision

Replacement final seed `20260911` passed curriculum training at global step
6,000 but failed one untouched competence criterion. The stop-on-first-failure
runner correctly stopped before seeds `20260912-20`.

The replacement final family therefore failed and remains ineligible for
calibration or perturbation outcomes.

## Untouched competence result

| Metric | Value | Required | Decision |
| --- | ---: | ---: | --- |
| 0-back accuracy | 1.000 | 0.950 | pass |
| 0-back discriminability | 1.000 | 0.900 | pass |
| 2-back accuracy | 0.952 | 0.950 | pass |
| 2-back discriminability | 0.888 | 0.900 | **fail** |
| 2-back lure accuracy | 0.996 | 0.900 | pass |

The 2-back discriminability comprised:

```text
hit rate          0.9186
false-alarm rate  0.0307
HR - FAR          0.8879
```

## Interpretation

This was a narrow held-out discriminability failure, not a majority-class or
lure-strategy collapse. Increasing the training ceiling did not address it:
the seed passed curriculum validation well before the old ceiling.

The repeated final-family failures also expose a design issue in the 10/10
acceptance rule. Wan et al. (2022), the closest exact RNN precedent, trained 12
independently initialized networks, discarded two below their competence
criterion, and analyzed the 10 competent networks. Competence screening before
representational analysis is therefore literature-grounded and remains blind
to perturbation outcomes.

A replacement design may preregister a larger independent seed pool and retain
the first ten checkpoints meeting the unchanged competence gates. It may not
relax those gates or inspect perturbations during selection.

## Recorded outputs

```text
outputs/nback_working_memory_budget_final/seed_sweep/seed_20260911/
outputs/nback_working_memory_budget_final/metrics/nback_working_memory_budget_final_seed_sweep_summary.json
```
