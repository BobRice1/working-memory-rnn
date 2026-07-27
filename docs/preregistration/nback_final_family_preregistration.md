# N-back final checkpoint-family pre-registration

## Registration status

Frozen on 2026-07-27 after all three training-balance rescue development seeds
passed and before training any final-family checkpoint.

Development evidence is recorded in
`docs/preregistration/nback_training_balance_rescue_development_audit.md`.
No perturbation outcome has been inspected.

## Frozen final family

Training seeds:

```text
20260901, 20260902, 20260903, 20260904, 20260905,
20260906, 20260907, 20260908, 20260909, 20260910
```

Per-seed curriculum-validation seed offset:

```text
500000
```

Per-seed untouched competence-bank offset:

```text
600000
```

Each competence bank contains 1,024 sequences per condition. These seed
families are disjoint from initial-development and rescue-development banks.

## Frozen training and competence

The exact successful rescue configuration is retained:

- shared 64-unit continuous-time tanh RNN;
- one 0-back and three 2-back batches per shuffled Stage 2 block;
- ordinary 0-back cross-entropy;
- 2-back class weights `[1.0, 2.0]`;
- all task timings, event counts, lures, optimizer settings, maximum steps,
  validation frequency, and metrics unchanged.

Every final checkpoint must pass:

- 0-back accuracy at least `0.95`;
- 2-back accuracy at least `0.95`;
- 0-back discriminability at least `0.90`;
- 2-back discriminability at least `0.90`;
- 2-back one-back-lure accuracy at least `0.90`;
- both classes and lures represented.

## Execution and stop rule

Use the stop-on-failure seed runner in numerical seed order. If any seed fails
training validation or untouched competence:

1. stop before the next seed;
2. record a baseline-only final-family failure audit;
3. do not inspect any N-back perturbation outcome.

All ten final checkpoints must pass before the N-back calibration-precision
addendum or perturbation runner may use them.

## Claim boundary

A passing final family establishes a stable computational baseline for testing
the Barrett 0-back/2-back dissociation. It does not establish a psilocybin
effect, biological equivalence, or a human neural mechanism.
