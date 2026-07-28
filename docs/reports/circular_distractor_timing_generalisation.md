# Circular distractor-timing generalisation

Date: 2026-07-28
Status: post-result robustness analysis

## Question

This analysis asked whether the five competent distractor-trained circular
checkpoints learned a general ability to filter irrelevant circular input, or
whether filtering depended strongly on the distractor appearing at its trained
midpoint location.

The result supports **timing-specific filtering with a pronounced late-delay
asymmetry**. It does not show that the networks use timestep location
exclusively: recurrent state and remaining recovery time also differ across
the delay.

## Frozen design

- Checkpoints: `20260731`, `20260732`, `20260733`, `20260735`, and `20260736`.
- Weights: frozen; no perturbation or retraining.
- Delay: 20 model steps.
- Distractor duration: 5 steps.
- Delay-relative distractor starts: `0`, `4`, `8`, `11`, and `15`.
- Trained position: midpoint, start `8`.
- Trials: 1,024 per checkpoint and condition, in eight batches of 128.
- Conditions: one clean condition and five distractor timings.
- Replication unit: independently trained checkpoint, not trial.

Targets were exactly identical across all six conditions, and distractor
angles were exactly identical across the five distractor timings. All
checkpoint and configuration hashes passed. The midpoint condition reproduced
the archived native midpoint errors to within `4.91e-7` degrees; the permitted
numerical tolerance was `1e-6` degrees.

The run used CPU because CUDA was unavailable to the active Python process.

## Response error

Values are mean angular response error in degrees.

| Checkpoint | Clean | Start 0 | Quarter 4 | Midpoint 8 | Three-quarter 11 | End 15 |
|---:|---:|---:|---:|---:|---:|---:|
| 20260731 | 3.959 | 4.604 | 4.457 | 4.437 | 4.471 | 4.936 |
| 20260732 | 2.829 | 3.994 | 3.805 | 3.760 | 3.799 | 4.279 |
| 20260733 | 3.382 | 4.360 | 4.251 | 4.248 | 4.322 | 5.298 |
| 20260735 | 2.636 | 3.822 | 3.556 | 3.461 | 3.466 | 4.077 |
| 20260736 | 4.087 | 4.986 | 4.880 | 4.828 | 4.887 | 5.609 |
| Mean | 3.379 | 4.353 | 4.190 | 4.147 | 4.189 | 4.840 |
| SD | 0.651 | 0.468 | 0.525 | 0.543 | 0.561 | 0.653 |
| 95% interval | [2.571, 4.187] | [3.773, 4.934] | [3.538, 4.841] | [3.473, 4.821] | [3.492, 4.886] | [4.029, 5.651] |

Every distractor position increased error relative to its paired clean
condition in all five checkpoints.

## Distractor costs and midpoint displacement

Raw distractor cost is distractor-condition error minus paired clean error.
Timing-minus-midpoint displacement is each cost minus the trained-midpoint
cost. Positive displacement means that moving the same distractor away from
the trained midpoint increased response error.

| Checkpoint | Start cost | Quarter cost | Midpoint cost | Three-quarter cost | End cost |
|---:|---:|---:|---:|---:|---:|
| 20260731 | 0.645 | 0.497 | 0.477 | 0.512 | 0.977 |
| 20260732 | 1.165 | 0.976 | 0.931 | 0.970 | 1.450 |
| 20260733 | 0.978 | 0.869 | 0.865 | 0.939 | 1.916 |
| 20260735 | 1.186 | 0.920 | 0.825 | 0.830 | 1.441 |
| 20260736 | 0.899 | 0.793 | 0.741 | 0.800 | 1.521 |
| Mean | 0.975 | 0.811 | 0.768 | 0.810 | 1.461 |
| SD | 0.221 | 0.188 | 0.176 | 0.181 | 0.334 |
| 95% interval | [0.700, 1.249] | [0.578, 1.044] | [0.549, 0.987] | [0.585, 1.035] | [1.047, 1.875] |

| Timing versus midpoint | Mean difference | SD | 95% interval | Direction |
|---|---:|---:|---:|---:|
| Start minus midpoint | +0.206 | 0.096 | [0.087, 0.326] | 5/5 positive |
| Quarter minus midpoint | +0.043 | 0.035 | [-0.00007, 0.086] | 5/5 positive |
| Three-quarter minus midpoint | +0.042 | 0.026 | [0.010, 0.075] | 5/5 positive |
| End minus midpoint | +0.693 | 0.229 | [0.409, 0.977] | 5/5 positive |

The midpoint was therefore the lowest-cost distractor position in every
checkpoint. Generalisation was nevertheless graded rather than absent:
quarter and three-quarter positions were only about `0.04` degrees worse than
midpoint on average, the start was `0.21` degrees worse, and the end was
`0.69` degrees worse. The late effect was more than three times the early
effect, indicating asymmetry across the delay.

## Supporting validity and state measures

- All 30 cells passed fixation and latency validity.
- Minimum fixation accuracy was `0.965`.
- Every cell had a settled fraction of `1.00`.
- Mean restricted settling times were `2.903` steps clean, `2.784` start,
  `2.694` quarter, `2.615` midpoint, `2.604` three-quarter, and `2.838` end.
- Mean delay-decoder errors were `1.036` degrees clean and ranged from `2.124`
  to `3.476` degrees across distractor timings.
- Distractor drift was computed under the existing definition. Recovery is
  undefined for the end condition because the distractor occupies the final
  five delay steps and there is no post-distractor delay window; its recovery
  values are therefore recorded as `NaN`, while peak drift and end attraction
  remain available.

The validity checks rule out fixation collapse or widespread failure to settle
as explanations for the timing pattern.

## Interpretation

These checkpoints did not learn fully timing-invariant distractor filtering.
They performed best when the distractor appeared at the midpoint used during
training, and all five checkpoints showed larger costs at every displaced
position. The clearest vulnerability occurred when the distractor arrived at
the end of the delay, immediately before the response period, with a smaller
but consistent cost at delay onset.

This is evidence of timing dependence, but it is not evidence for a literal
midpoint gate or proof that the networks memorised an absolute timestep.
Moving the distractor changes both the recurrent state present when it arrives
and the time available to recover before response. The strong end effect may
therefore reflect limited recovery time, state-dependent susceptibility, a
learned temporal expectation, or a combination of these mechanisms.

The practical conclusion is that midpoint-only evaluation overstates the
generality of learned filtering. Future training intended to support robust
distractor rejection should randomise distractor timing, and subsequent
dynamics analysis should separate immediate attraction from post-distractor
recovery.

## Claim boundary

This was a post-result descriptive robustness analysis. It is not a
preregistered confirmatory test, a human-task replication, or evidence for a
biological mechanism of psilocybin.

## Recorded outputs

- `outputs/circular_distractor_timing_generalisation/metrics/timing_metrics.csv`
- `outputs/circular_distractor_timing_generalisation/metrics/timing_comparisons.csv`
- `outputs/circular_distractor_timing_generalisation/metrics/timing_summary.json`
