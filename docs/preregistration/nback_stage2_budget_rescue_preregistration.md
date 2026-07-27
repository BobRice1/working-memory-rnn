# N-back Stage 2 budget rescue pre-registration

## Registration status

Frozen on 2026-07-27 after the first final family failed and before
implementing or training this rescue.

The triggering baseline-only evidence is recorded in
`docs/preregistration/nback_final_family_failure_audit.md`. No perturbation
outcome has been inspected.

## Question

Was the successful training-balance procedure insufficiently robust because
the 12,000-update Stage 2 maximum truncated late-learning initializations?

Seed `20260902` preserved 0-back and showed rising 2-back accuracy,
discriminability, and lure accuracy at the final three validation checks. This
rescue tests training duration only.

## Sole frozen training change

```text
Stage 2 maximum:
12,000 updates -> 20,000 updates
```

Validation remains every 200 updates. Training still stops as soon as two
consecutive joint validation checks pass, so the extension is only used by
initializations that need it.

## Components held fixed

- actual 0-back/2-back task and exact sequence generator;
- six identities, event timing, match ratio, and lure requirement;
- shared 64-unit continuous-time tanh RNN;
- `dt = 20`, `tau = 100`;
- no recurrent or input noise;
- Stage 1 0-back acquisition procedure;
- one 0-back and three 2-back batches per Stage 2 block;
- ordinary 0-back cross-entropy;
- 2-back class weights `[1.0, 2.0]`;
- Adam learning rate `1e-3`;
- gradient-norm clipping at `1.0`;
- two consecutive validation passes;
- all competence metrics and thresholds;
- 1,024-sequence untouched competence banks;
- stop-on-first-failure execution.

No architecture, initialization, loss, task, or gate change is permitted.

## New development family

Training seeds:

```text
20260827, 20260828, 20260829
```

Validation seed offset:

```text
700000
```

Untouched competence seed offset:

```text
800000
```

Train in numerical order. Stop on the first failed training or competence
gate. All three must pass before a new final family is frozen.

## Prospective final family

If development passes, use entirely new training seeds:

```text
20260911-20260920
```

Their validation and competence offsets must be frozen in a separate final
pre-registration. Previously attempted final seeds will not be reused.

## Implementation gate

Before seed `20260827`:

- add an exact config-difference test proving the Stage 2 maximum and seed
  banks are the only scientific changes;
- run the full repository test suite;
- verify the stop-on-failure runner resolves the new output paths;
- do not load any checkpoint in perturbation code.

## Decision and claim boundary

A passing rescue would establish robust baseline learnability under a wider
optimization budget. It would not be evidence for a psilocybin-like mechanism.

A failed development seed stops the rescue. Further changes require another
versioned pre-registration. Perturbation calibration remains locked until a
complete ten-seed final family passes.
