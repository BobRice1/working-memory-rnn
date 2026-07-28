# Full candidate perturbation results with trained distractor filtering

Date: 27 July 2026

## Result

The circular perturbation grid was repeated on five newly trained one-item
checkpoints that learned both clean recall and distractor filtering. The
existing ten-checkpoint N-back results were unchanged and reused for the
cross-task summary. Persistence `0.95` had been nominated by an earlier
three-seed exploratory pilot; the present run is therefore a circular-family
retest plus a partially overlapping N-back reuse, not a fresh blind screen.

State persistence `0.95` remained the only operator-strength setting that met
the complete descriptive majority pattern (mean positive and simple majority
seed agreement on slowing-with-preservation, distractor selectivity, and
N-back load selectivity; delay selectivity supporting only).

| Outcome | Mean ± SD | 95% t-interval | Predicted direction |
|---|---:|---:|---:|
| Delay-20 proportional error change | `+0.049 ± 0.092` | `[-0.065, 0.164]` | below `0.20` in 5/5 |
| Restricted settling change | `+0.104 ± 0.167` steps | `[-0.103, 0.312]` | 4/5 positive |
| Long-minus-short delay selectivity | `+0.089 ± 0.075` | `[-0.004, 0.181]` | 4/5 positive |
| Distractor-minus-clean selectivity | `+0.015 ± 0.044` | `[-0.040, 0.071]` | 4/5 positive |
| 2-back-minus-0-back selectivity | `+0.250 ± 0.077` | `[0.195, 0.305]` | 10/10 positive |

The trained-distractor result is more credible but less uniform than the
earlier evaluation-only result. Circular seed-level intervals include zero;
only the N-back load contrast excludes zero. One circular checkpoint showed a
negative distractor contrast and a second showed a slightly negative delay
contrast. Relative to the pilot, circular delay selectivity shrank from
`0.290` to `0.089` and distractor selectivity from `0.148` to `0.015`. State
persistence `0.95` therefore passes a descriptive majority rule, not an
all-seed or robust-interval criterion.

## Scope

- Five distractor-trained circular checkpoints retained from six trained seeds:
  `20260731`, `20260732`, `20260733`, `20260735`, and `20260736`. Seed
  `20260734` failed preconfigured competence limits and was excluded before the
  screen.
- Ten unchanged N-back checkpoints: `20260912`--`20260921`. Three of these
  (`20260912`--`20260914`) contributed to the exploratory nomination pilot.
- 1,024 trials or twenty-item sequences per condition and strength.
- Seven candidate perturbation families, including a circular-only
  distractor-window gain. Across shared families there were 23 non-neutral
  profiles (29 rows including neutrals).
- No matched-cost Gaussian-noise comparison, calibration, hybrids, or
  confirmatory p-values in this run. An exploratory three-seed N-back hint that
  persistence `0.95` exceeded noise `0.025` on load selectivity
  (`0.296` versus `0.052`) is secondary only.
- Circular CUDA rerun time: 590.7 seconds, or 9 minutes 51 seconds.

## Alternatives

No other shared operator-strength setting reproduced the complete pattern.

- Sensory-input gain `1.20` produced distractor selectivity in 5/5 circular
  checkpoints and N-back load selectivity in 9/10, but settling became slightly
  faster (`-0.010 ± 0.023` steps; 1/5 positive) and long-delay selectivity was
  positive in only 1/5.
- Effective time-constant scale `1.10` slowed settling with preservation in
  5/5 (`+0.605 ± 0.176` steps) and showed load selectivity in 10/10, but
  distractor selectivity was negative in all five circular checkpoints.
- Effective time-constant scale `0.90` showed long-delay selectivity in 5/5,
  distractor selectivity in 4/5, and load selectivity in 9/10, but accelerated
  settling in every circular checkpoint (`-0.681 ± 0.170` steps).
- Synaptic-drive gain `1.05` showed long-delay selectivity in 4/5,
  distractor selectivity in 3/5, and load selectivity in 10/10, but accelerated
  settling in every circular checkpoint (`-0.496 ± 0.240` steps).

All of these near-miss clean delay-20 settling cells were latency-valid. The
persistence-versus-conserved-time-constant contrast therefore supports the
asymmetric carried-state reading, while remaining uncontrolled for overall
clean-task cost.

The circular-only distractor-window manipulation behaved as an implementation
check: increasing its gain produced monotonic mean distractor impairment of
`3.7%`, `9.8%`, and `21.5%` at gains `1.10`, `1.25`, and `1.50`, respectively,
with the direction reproduced in 5/5 checkpoints. Because it acts only when a
distractor is present, it is not a complete cross-task mechanism candidate.

## Validity and interpretation

The earlier out-of-distribution concern is resolved for the new circular
family. Native delay-20 distractor error ranged from `3.46°` to `4.83°`, all
native distractor cells had a settled fraction of `1.00`, and every
state-persistence `0.95` clean cell was latency-valid. Twelve of 945 circular
rows were latency-invalid elsewhere in the grid (heterogeneous-drive settings);
none entered the near-miss settling comparisons above. The distractor contrast
is now interpretable as altered performance of a learned filtering behaviour.

The effect itself remains modest. Mean distractor selectivity for persistence
`0.95` was only `0.015`, with a between-checkpoint SD of `0.044`. It supports a
small replicated majority tendency rather than a large or universal
Carter-style effect.

The appropriate conclusion is:

> A small reduction in carried-state persistence remains computationally
> sufficient to reproduce the majority direction of selected
> psilocybin-associated working-memory contrasts when distractor filtering is a
> learned task property. The result is descriptive, not mechanistically unique,
> and does not identify the biological action of psilocybin. The next decisive
> test is matched-cost specificity against Gaussian disruption.

## Artifacts

- `outputs/full_candidate_perturbation_trained_distractor_1024/circular_trained_distractor/metrics/circular_trained_distractor_grid.csv`
- `outputs/full_candidate_perturbation_trained_distractor_1024/circular_trained_distractor/metrics/circular_trained_distractor_metadata.json`
- `outputs/full_candidate_perturbation_trained_distractor_1024/summary/cross_task_signature_summary.csv`
- `outputs/full_candidate_perturbation_1024/nback/pilot_signatures.csv`
- `docs/reports/figures/full_candidate_perturbation/`
- `docs/reports/full_candidate_perturbation_scientific_writeup.tex`
- `docs/reports/Perturbations.pdf`
- `docs/reports/Perturbations_QA.md`
