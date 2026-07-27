# Full 1,024-sample candidate-only perturbation results

Date: 27 July 2026

## Result

The expanded evaluation reproduced the pilot's main finding. Across all five
original circular-task checkpoints and all ten competent N-back checkpoints,
**state persistence `0.95` remained the only operator-strength setting that
met the complete descriptive pattern**.

| Outcome | Mean contrast | Predicted direction |
|---|---:|---:|
| Delay-20 angular-error impairment | +1.38% | preservation passed 5/5 |
| Restricted settling change | +0.194 steps | 5/5 |
| Long-minus-short delay selectivity | +0.264 | 5/5 |
| Distractor-minus-clean selectivity | +0.159 | 5/5 |
| 2-back-minus-0-back selectivity | +0.250 | 10/10 |

The N-back result was especially consistent: every independently trained model
showed greater normalized discriminability impairment on 2-back than 0-back.
The individual load-selectivity contrasts ranged from approximately `0.110` to
`0.363`.

The settling result was consistently positive but small. Individual circular
checkpoint changes were approximately `0.010`, `0.012`, `0.033`, `0.354`, and
`0.561` model steps. It should therefore be described as a replicated
directional slowing analogue, not a large reaction-time effect.

## Scope

- Five circular checkpoints, seeds `20260714–20260718`.
- Ten N-back checkpoints, seeds `20260912–20260921`.
- 1,024 trials or twenty-item sequences per condition and strength.
- Seven candidate perturbation families plus circular distractor-window gain.
- No retraining, Gaussian-noise comparison, cost matching, hybrids, or
  confirmatory p-values.
- CUDA execution time: 1,075.9 seconds, or 17 minutes 56 seconds.

## Alternatives

No other setting reproduced the complete pattern.

- Sensory-input gain `1.20` showed distractor selectivity in 5/5 circular seeds
  and load selectivity in 9/10 N-back seeds, but mean settling became slightly
  faster and the delay contrast was not reliable.
- State persistence `0.90` produced larger delay and N-back effects, but
  slowing with preservation passed only 2/5 circular checkpoints.
- Time constant `0.90` produced delay and distractor selectivity but accelerated
  settling.
- Recurrent gain `0.95` showed positive load selectivity in 9/10 and delay
  selectivity in 4/5, but distractor selectivity in only 1/5.

This pattern supports a narrow persistence change around `0.95` rather than a
claim that any sufficiently damaging perturbation passes.

## Validity and interpretation

All 945 circular result rows passed the fixation gate, all ten N-back native
checkpoints were competent, and the state-persistence delay-20 clean cells used
for settling were latency-valid. Technical heterogeneous-gain replicates were
averaged within checkpoint.

The distractor caveat remains. Native distractor mean errors ranged from
approximately `25.8` to `50.4` degrees and only `26.7–35.7%` of unperturbed
trials settled. Increasing the trial count narrowed sampling uncertainty but
did not repair this weak out-of-distribution baseline. The Carter-style
distractor component therefore remains provisional.

The appropriate conclusion is:

> A small reduction in carried-state persistence is computationally sufficient
> to reproduce the direction of several selected psilocybin-associated
> working-memory contrasts across independently trained RNNs. This is a robust
> candidate-versus-baseline result, not evidence that state persistence is the
> biological action of psilocybin or that the full human signature has been
> conclusively recovered.

## Artifacts

- `outputs/full_candidate_perturbation_1024/circular_family_a/metrics/circular_family_a_grid.csv`
- `outputs/full_candidate_perturbation_1024/nback/pilot_cells.json`
- `outputs/full_candidate_perturbation_1024/nback/pilot_signatures.csv`
- `outputs/full_candidate_perturbation_1024/summary/cross_task_signature_summary.csv`
- `outputs/full_candidate_perturbation_1024/summary/leading_profile_seed_points.png`
