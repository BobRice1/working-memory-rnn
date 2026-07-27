# N-back training-balance rescue pre-registration

## Registration status

Frozen on 2026-07-27 after the initial baseline audit and before implementing
or training this rescue.

The triggering evidence is restricted to baseline development seed `20260821`
and is recorded in
`docs/preregistration/nback_initial_baseline_audit.md`. No perturbation outcome
has been inspected.

## Question

Did the initial N-back baseline miss competence because Stage 2 allocated too
few effective updates to 2-back matches?

The initial model retained perfect 0-back performance and nearly perfect
one-back-lure rejection, but 2-back hit rate remained lower than specificity.
This rescue changes training allocation and loss balance without changing the
task, architecture, validation metrics, or competence thresholds.

## Frozen changes

Exactly two linked training changes are permitted.

### R1. Rule allocation

Stage 2 shuffled blocks change from:

```text
[0-back, 2-back]
```

to:

```text
[0-back, 2-back, 2-back, 2-back]
```

This retains continual 0-back rehearsal while allocating 75% of Stage 2
updates to the unresolved working-memory rule.

### R2. Within-rule class weighting

Stage 1 and Stage 2 0-back batches retain ordinary cross-entropy.

Stage 2 2-back batches use class weights:

```text
non-match = 1.0
match     = 2.0
```

The generator remains at six matches and twelve non-matches per scored
sequence. Weighting therefore equalizes the aggregate match and non-match
contributions without changing the task prevalence observed by the network.

## Components held fixed

- six one-hot identities plus two rule-context channels;
- 20 events, three stimulus steps, and six blank steps;
- six matches among 18 scored events;
- at least three one-back lures per 2-back sequence;
- 64-unit continuous-time tanh RNN;
- `dt = 20`, `tau = 100`;
- no recurrent or input noise;
- Adam at `1e-3`;
- gradient-norm clipping at `1.0`;
- Stage 1 procedure and thresholds;
- Stage 2 maximum of 12,000 updates;
- validation every 200 Stage 2 updates;
- two consecutive passes;
- 0-back and 2-back accuracy at least `0.95`;
- both discriminabilities at least `0.90`;
- 2-back lure accuracy at least `0.90`;
- item-level metric definitions and settling definitions.

No hidden-size increase, LSTM/GRU substitution, timing change, target-ratio
change, or gate relaxation is permitted.

## Independent development family

The rescue development training seeds are:

```text
20260824, 20260825, 20260826
```

Validation uses the new offset `300000`; competence evaluation uses the new
offset `400000`. These banks are disjoint from the initial rescue-diagnostic
bank.

Train seed `20260824` first. If it fails either training stage or fresh
competence evaluation, stop without training the other development seeds.
All three development seeds must pass before final-family training.

The untouched final training seeds remain:

```text
20260901-20260910
```

Their validation and competence offsets will be frozen after a successful
three-seed development audit and before final training.

## Implementation gate

Before seed `20260824`:

- test that every Stage 2 block contains one 0-back and three 2-back entries;
- test deterministic shuffled blocks;
- test exact weighted cross-entropy against a hand calculation;
- test that 0-back remains unweighted;
- test that the old configuration still produces balanced two-rule blocks;
- run the full repository test suite;
- verify a CUDA weighted-loss smoke step.

## Interpretation and stop rule

A successful rescue would show that the actual N-back CTRNN is trainable under
an allocation aligned to the unresolved high-load decision. It would not be a
psilocybin result.

A failed first development seed stops this rescue. Further task, architecture,
optimizer, or gate changes require another versioned pre-registration. Failed
checkpoints remain ineligible for perturbation analysis.
