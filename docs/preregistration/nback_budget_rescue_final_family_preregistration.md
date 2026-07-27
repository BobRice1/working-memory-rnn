# N-back budget-rescue final-family pre-registration

## Registration status

Frozen on 2026-07-27 after the three-seed budget-rescue development family
passed and before training any replacement final checkpoint.

Development evidence is recorded in
`docs/preregistration/nback_stage2_budget_rescue_development_audit.md`. No
perturbation outcome has been inspected.

## Frozen final family

Training seeds:

```text
20260911, 20260912, 20260913, 20260914, 20260915,
20260916, 20260917, 20260918, 20260919, 20260920
```

Curriculum-validation offset:

```text
900000
```

Untouched competence-bank offset:

```text
1000000
```

Each competence bank contains 1,024 sequences per condition.

## Frozen procedure

The exact budget-rescue configuration is retained:

- actual shared 0-back/2-back task;
- 64-unit continuous-time tanh RNN;
- Stage 1 0-back acquisition;
- Stage 2 shuffled 1:3 rule allocation;
- 2-back class weights `[1.0, 2.0]`;
- Stage 2 maximum 20,000 updates;
- validation every 200 updates;
- early stop after two consecutive joint passes;
- unchanged metrics and competence thresholds.

## Execution and stop rule

Run seeds in numerical order with the stop-on-first-failure runner. If any
training or untouched competence gate fails:

1. stop before the next seed;
2. record a baseline-only audit;
3. do not run calibration or inspect perturbation outcomes.

All ten checkpoints must pass before the baseline-only calibration-precision
audit can begin.

## Claim boundary

A passing family establishes a sufficiently robust computational baseline for
the registered N-back load contrast. It does not constitute a psilocybin
result or biological mechanism claim.
