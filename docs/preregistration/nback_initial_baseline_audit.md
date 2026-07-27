# Initial actual N-back baseline audit

Date recorded: 2026-07-27
Frozen pre-registration commit: `ec6ad30`
Frozen implementation commit: `7cfea6c`
Development seed: `20260821`
Device: NVIDIA GeForce RTX 3060 Laptop GPU
Perturbation outcomes inspected: **none**

## Decision

The initial actual N-back baseline **failed its joint development gate**.
Stage 1 passed, but Stage 2 reached its 12,000-step maximum without two
consecutive joint 0-back/2-back competence passes.

Under the frozen stop rule, development seeds `20260822-23`, the ten final
seeds, and all N-back perturbation analyses were not run.

## Stage 1

The shared 64-unit CTRNN learned 0-back rapidly:

- validation at global step 100: passed;
- validation at global step 200: passed;
- two consecutive passes advanced the model to Stage 2.

The failure is therefore not a general inability to learn the stream,
stimulus inputs, rule context, readout, or fixed-target decision.

## Final Stage 2 validation

Validation used 256 fixed sequences per condition from the preregistered
development-validation seed family.

| Metric | 0-back | 2-back | Joint gate |
| --- | ---: | ---: | --- |
| Accuracy | 1.000 | 0.891 | 0.950 |
| Hit rate | 1.000 | 0.778 | descriptive |
| False-alarm rate | 0.000 | 0.052 | descriptive |
| Barrett discriminability | 1.000 | 0.726 | 0.900 |
| Mean cross-entropy | 0.0005 | 0.1966 | descriptive |
| One-back-lure accuracy | NA | 0.991 | 0.900 |

The 2-back gate failures were:

```text
two_back_accuracy          FAIL
two_back_discriminability  FAIL
```

All 0-back, class-count, and lure checks passed.

## Diagnostic trajectory

The model initially sat at the majority-class solution: approximately `0.667`
2-back accuracy with cross-entropy near `0.637`. It began improving after
roughly global step 7,600 and reached a transient held-out accuracy of `0.909`
and discriminability of `0.791` at step 11,600, still below the frozen gates.

At the final check, low false-alarm rate, lower hit rate, and near-perfect lure
accuracy show a conservative match decision rather than a one-back heuristic:

```text
2-back hit rate                 0.778
2-back false-alarm rate         0.052
2-back one-back-lure accuracy   0.991
```

## Interpretation

The task reformulation fixed the old Family B learnability problem partially:
the CTRNN acquired the control rule, retained it during joint training, learned
to reject one-back lures, and improved genuine 2-back decisions. It did not
reach confirmatory baseline competence under the initial training allocation.

Two frozen design choices plausibly made the remaining gate unnecessarily
difficult:

1. equal 0-back/2-back batch sampling spent half of Stage 2 updates on a
   0-back rule whose held-out accuracy had already reached 1.0;
2. unweighted cross-entropy preserved the published 1:2 match/non-match
   prevalence but gave match errors half the aggregate weight of non-match
   errors, consistent with the observed conservative decision pattern.

This is a baseline training diagnosis, not evidence about any psilocybin-like
perturbation mechanism.

## Frozen stop

No failed checkpoint will enter perturbation analysis. A training-only rescue
requires a new pre-registration and new development-validation seed family.

## Recorded outputs

```text
outputs/nback_working_memory/checkpoints/nback_working_memory.pt
outputs/nback_working_memory/metrics/nback_working_memory_train_history.csv
outputs/nback_working_memory/metrics/nback_working_memory_train_metrics.json
```
