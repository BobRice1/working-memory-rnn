# Dense state-persistence neighbourhood result

Date: 29 July 2026

## Design

Frozen before outcome inspection in
`docs/preregistration/state_persistence_dense_variable_timing_1024.md`.

The evaluation used persistence strengths

```text
0.80, 0.85, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94,
0.95, 0.96, 0.97, 0.98, 0.99, 1.00
```

on the same 10 variable-timing circular checkpoints and 10 screened N-back
checkpoints, with 1,024 trials or sequences per cell. Circular distractor
timing remained the frozen randomized balanced bank over starts `0`--`15`.

## Result

The previously reported persistence `0.95` cell reproduced:
settling `+0.032`, slowing-with-preservation `6/10`, delay selectivity
`9/10`, distractor selectivity `8/10`, and N-back load selectivity
`+0.251` (`10/10`).

Nearby values were directionally similar. Persistence `0.96` improved the
slowing-with-preservation count to `7/10` while retaining delay `9/10`,
distractor `8/10`, and N-back load `10/10`. Persistence `0.80` produced
large settling (`+2.143`) but only `1/10` preservation passes and inverted
N-back load selectivity (`-0.601`, `0/10` positive).

## Interpretation

The descriptive persistence candidate is not a single-grid-point accident at
`0.95`, but the useful neighbourhood is mild. Deep persistence reduction
behaves as broad damage rather than a stronger signature match.

This remains descriptive neighbourhood mapping. Matched-cost Gaussian
specificity is still outstanding.

## Artifacts

- Config: `configs/state_persistence_dense_variable_timing_1024.yaml`
- Circular grid: `outputs/state_persistence_dense_variable_timing_1024/circular_variable_timing/metrics/circular_variable_timing_grid.csv`
- N-back signatures: `outputs/state_persistence_dense_variable_timing_1024/nback/pilot_signatures.csv`
- Figure: `docs/reports/figures/full_candidate_perturbation/comparison/persistence_response_dense_10seed_variable_timing.png`
- Comparison page: `docs/reports/figures/full_candidate_perturbation/comparison/README.md`
