# Candidate perturbation rerun on the variable-timing circular family

Date: 28 July 2026

## Design

The existing seven-operator candidate grid was rerun on the 10 competent
circular checkpoints trained with variable distractor timing. The completed
10-checkpoint N-back table was reused unchanged.

The circular conditions retained the earlier 1,024 trials per cell and clean
delays of 10, 20, 40, and 80 steps. The distractor condition used a 20-step
delay and a five-step distractor. Rather than placing every distractor at the
midpoint, each trial was assigned a frozen randomized onset from every valid
relative start `0`--`15`. The bank was exactly balanced, with 64 trials at
each onset. Target angles, distractor angles, and onset assignments were
paired across baseline and every perturbation.

This is a descriptive family rerun. It is not a new blind screen, a
matched-cost Gaussian comparison, or a biological model of psilocybin.

## Leading result

State persistence `0.95` remained the only operator-strength profile meeting
the complete descriptive majority pattern.

| Circular outcome | Mean +/- SD | Student-t 95% interval | Seed direction |
|---|---:|---:|---:|
| Delay-20 proportional error change | `-0.003 +/- 0.114` | `[-0.085, 0.079]` | 4/10 positive |
| Restricted settling change | `+0.032 +/- 0.062` steps | `[-0.013, 0.076]` | 7/10 positive |
| Long-minus-short delay selectivity | `+0.137 +/- 0.143` | `[0.035, 0.240]` | 9/10 positive |
| Randomized-timing distractor selectivity | `+0.035 +/- 0.063` | `[-0.010, 0.080]` | 8/10 positive |

The formal slowing-with-preservation rule held in 6/10 checkpoints rather
than 7/10 because one seed with slower settling exceeded the 20% clean-cost
ceiling. All 10 persistence `0.95` clean delay-20 cells passed latency
validity.

The unchanged N-back load-selectivity mean was `+0.250 +/- 0.077`, positive
in 10/10 checkpoints. It is included to complete the cross-task descriptive
pattern but is reused evidence, not a new N-back replication.

## Comparison with the historical circular rerun

| Outcome at persistence 0.95 | Historical 5-seed midpoint family | New 10-seed randomized-timing family |
|---|---:|---:|
| Clean proportional error change | `+0.049` | `-0.003` |
| Settling change | `+0.104` steps | `+0.032` steps |
| Delay selectivity | `+0.089` | `+0.137` |
| Distractor selectivity | `+0.015` | `+0.035` |
| Positive distractor direction | 4/5 | 8/10 |

The new result is directionally consistent with the earlier candidate result
and uses twice as many independent circular checkpoints. Its delay-selectivity
interval now excludes zero, whereas the settling and distractor intervals
still include zero.

The numerical changes are not a controlled between-family comparison. Both
the independently initialized checkpoint family and the distractor outcome
changed, from a fixed midpoint to randomized timing across the delay.

## Alternative profiles

No other shared operator-strength profile reproduced the full pattern.

- State persistence `0.90` produced positive delay selectivity in 10/10,
  randomized-timing distractor selectivity in 7/10, and N-back load
  selectivity in 10/10, but the slowing-with-preservation rule held in only
  5/10.
- Sensory-input gain `1.20` produced randomized-timing distractor selectivity
  in 10/10 and N-back load selectivity in 9/10, but settling accelerated on
  average and delay selectivity was not positive.
- Effective time-constant scale `1.10` slowed settling in 10/10 and retained
  the N-back direction in 10/10, but delay and distractor selectivity were
  negative in every circular checkpoint.
- Synaptic-drive gain `1.05` produced positive delay and distractor
  selectivity in 7/10 and 8/10 respectively, but accelerated settling in all
  10 checkpoints.

The circular-only distractor-input-gain manipulation remained an
implementation check. Gains `1.10`, `1.25`, and `1.50` increased randomized
distractor impairment by means of `3.5%`, `9.6%`, and `26.4%`; each direction
was reproduced in 10/10 checkpoints.

## Baseline and validity

Native clean delay-20 error averaged `3.680` degrees. Native randomized-timing
distractor error averaged `4.382` degrees and ranged from `3.327` to `5.758`
degrees across checkpoints. Every native distractor cell passed fixation and
settling validity; minimum fraction settled was `0.981` and minimum fixation
accuracy was `0.968`.

Fifteen of 1,890 circular grid rows were latency-invalid. All occurred under
heterogeneous-drive settings, primarily strength `0.30` at longer delays.
None entered the persistence `0.95` comparison.

Common-window distractor drift and recovery are undefined for the mixed-onset
condition. Computing those quantities requires aligning each trajectory to
its own distractor onset and should be handled in a dedicated hidden-state
analysis.

## Interpretation

The new result supports the following cautious conclusion:

> A small reduction in carried-state persistence remains computationally
> sufficient to reproduce the majority direction of selected working-memory
> contrasts across a 10-checkpoint circular family when distractor timing is
> distributed across the delay. The circular settling and distractor effects
> remain modest and uncertain, and the result does not identify a unique or
> biological mechanism of psilocybin.

The next decisive test remains matched-cost specificity against Gaussian
state disruption. A denser persistence analysis around `0.95` and
onset-aligned hidden-state analysis should be specified before inspecting
those outcomes.

## Artifacts

- Frozen configuration:
  `configs/full_candidate_perturbation_variable_timing_1024.yaml`
- Circular grid:
  `outputs/full_candidate_perturbation_variable_timing_1024/circular_variable_timing/metrics/circular_variable_timing_grid.csv`
- Metadata:
  `outputs/full_candidate_perturbation_variable_timing_1024/circular_variable_timing/metrics/circular_variable_timing_metadata.json`
- Cross-task summary:
  `outputs/full_candidate_perturbation_variable_timing_1024/summary/cross_task_signature_summary.csv`
- Seed-point figure:
  `outputs/full_candidate_perturbation_variable_timing_1024/summary/leading_profile_seed_points.png`
- Pre-outcome implementation commit: `8e8adc3`
