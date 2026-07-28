# Variable-timing circular checkpoint family

## Outcome

The frozen protocol produced the planned family of 10 competent,
independently initialized circular RNN checkpoints. Eleven candidates were
needed because seed `20260807` did not learn the task and failed the
predeclared competence gates.

Retained seeds:

`20260801`, `20260802`, `20260803`, `20260804`, `20260805`, `20260806`,
`20260808`, `20260809`, `20260810`, and `20260811`.

## Competence results

All retained checkpoints passed clean performance, every individual
distractor timing, and fixation accuracy. Values below are mean +/- SD across
the 10 retained seeds.

| Measure | Retained-family result | Frozen gate |
|---|---:|---:|
| Clean mean angular error | 2.972 +/- 0.527 degrees | <= 10 degrees |
| Mean error across distractor timings | 3.813 +/- 0.641 degrees | descriptive |
| Worst timing error per checkpoint | 3.899 +/- 0.638 degrees | <= 15 degrees at each timing |
| Worst observed timing error | 5.294 degrees | <= 15 degrees |
| Minimum fixation accuracy | 0.9766 +/- 0.0003 | >= 0.90 |
| Within-checkpoint timing range | 0.190 +/- 0.107 degrees | descriptive |

The failed seed had clean error of 90.751 degrees and a worst distractor error
of 91.104 degrees. It was excluded as a general task-learning failure; its
small difference between distractor timings is not evidence of robust
filtering.

## Timing profile

| Delay-relative distractor onset | Mean error | Mean distractor cost over clean |
|---:|---:|---:|
| 0.00 | 3.859 degrees | 0.887 degrees |
| 0.25 | 3.839 degrees | 0.867 degrees |
| 0.50 | 3.830 degrees | 0.858 degrees |
| 0.75 | 3.806 degrees | 0.834 degrees |
| 1.00 | 3.734 degrees | 0.762 degrees |

The mean within-checkpoint timing range was 0.190 degrees, compared
descriptively with 0.693 degrees in the earlier five-seed midpoint-trained
family. This is consistent with the intended reduction in timing dependence.
It is not a randomized between-family test: the two families have independent
initializations and separate held-out trial banks.

The slight decline in mean cost for later distractors should not be
overinterpreted. Timing changes both the recurrent state at distractor arrival
and the amount of post-distractor delay before response.

## Execution note

Training used the CUDA environment. An initial process completed the first
four checkpoints and was interrupted during the fifth candidate before any
checkpoint for that candidate was written. The resumed pool runner reused and
re-evaluated seeds `20260801`--`20260804`, restarted `20260805` from its
configured seed, and then completed the frozen retention rule. Thus every
retained checkpoint is a complete 4,000-step run; no partial seed was used.

## Interpretation and next use

This family is the circular side of the supervisor-requested balanced
10-circular/10-N-back comparison. It replaces the midpoint-trained
five-checkpoint family for future confirmatory comparisons, while the old
family remains the historical candidate-screen pool.

Passing these gates establishes task competence across the trained timing
distribution. It does not itself establish a psilocybin-like behavioural
signature, a biological mechanism, or a matched-cost advantage over Gaussian
disruption.

## Reproducibility

- Frozen protocol:
  `docs/reports/variable_distractor_timing_training_protocol.md`
- Configuration:
  `configs/fixation_circular_variable_distractor_working_memory.yaml`
- Pool table:
  `outputs/fixation_circular_variable_distractor_working_memory/metrics/fixation_circular_variable_distractor_working_memory_pool.csv`
- Machine-readable summary:
  `outputs/fixation_circular_variable_distractor_working_memory/metrics/fixation_circular_variable_distractor_working_memory_pool_summary.json`
- Pre-outcome implementation commit: `bc86089`
