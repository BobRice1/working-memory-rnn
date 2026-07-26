# N-back baseline and perturbation-integration pre-registration

## Registration status

Frozen on 2026-07-26 before implementing the N-back task, training loop,
evaluation metrics, or any N-back perturbation analysis.

This specification replaces the failed synthetic two-slot Family B baseline.
No perturbation outcome from Family B has been inspected. The existing stable
delayed-response Family A checkpoints and their registered perturbation
definitions remain unchanged.

The evidence and architecture rationale are recorded in
`docs/nback_rnn_literature_and_architecture.md`.

## Scientific purpose

Train one shared continuous-time RNN on actual 0-back and 2-back conditions,
then use the human load dissociation as a model constraint:

```text
candidate excess 2-back impairment relative to 0-back
>
matched Gaussian excess 2-back impairment relative to 0-back
```

This N-back model replaces only the failed Family B load component. The stable
delayed-response Family A remains the source of delay-length, distractor, and
post-cue settling analyses. No single baseline must carry every behavioural
signature.

## Frozen task

- task types: `0-back` and `2-back`;
- one shared model with two constant mutually exclusive context channels;
- six one-hot stimulus identities;
- identity `0` is the fixed 0-back target;
- 20 stimulus events per sequence;
- three stimulus steps and six blank steps per event;
- events 0 and 1 unscored for both conditions;
- exactly six match events among the 18 scored events;
- at least three valid one-back lures in every 2-back sequence;
- targets: class `0` non-match, class `1` match;
- loss: cross-entropy on every step of each scored event;
- no response target during the two warm-up events;
- homogeneous 0-back or 2-back batches, sampled in shuffled balanced
  two-batch blocks.

Task batches expose event identities, match labels, one-back-lure flags, scored
event indices, condition, and event slices so all metrics are auditable.

## Frozen architecture

- existing `WorkingMemoryRNN` and continuous-time tanh `CTRNN`;
- input size 8;
- hidden size 64;
- output size 2;
- `dt = 20.0`;
- `tau = 100.0`;
- baseline recurrent noise `0.0`;
- baseline training input noise `0.0`.

LSTM, GRU, attention, trainable task embeddings, and architecture-specific
gates are out of scope for this first baseline.

## Frozen training procedure

Optimizer: Adam, initial learning rate `1e-3`, gradient-norm clipping at `1.0`.

Training is competence-gated rather than advanced only by a fixed clock:

### Stage 1: 0-back acquisition

- train homogeneous 0-back batches;
- evaluate every 100 optimizer steps on a fixed, independently generated
  development-validation set;
- advance only when two consecutive evaluations satisfy:
  - accuracy at least `0.98`;
  - Barrett discriminability (`hit rate - false-alarm rate`) at least `0.95`;
- minimum 200 and maximum 2,000 optimizer steps.

### Stage 2: shared 0-back/2-back acquisition

- train shuffled balanced blocks containing one homogeneous batch of each
  condition;
- evaluate every 200 optimizer steps;
- stop successfully only when two consecutive evaluations satisfy all of:
  - 0-back accuracy at least `0.95`;
  - 2-back accuracy at least `0.95`;
  - 0-back discriminability at least `0.90`;
  - 2-back discriminability at least `0.90`;
  - 2-back one-back-lure accuracy at least `0.90`;
- maximum 12,000 additional optimizer steps.

The checkpoint with the lowest mean validation cross-entropy among
gate-passing evaluations is retained. Validation is used for training control,
not confirmatory inference.

Training batch size is 128 sequences. Each validation condition contains 256
fixed sequences generated from seed families disjoint from training and final
evaluation.

## Development and final seed families

Development seeds are:

```text
20260821, 20260822, 20260823
```

Development validation seeds are derived by adding `100000`; development
evaluation seeds by adding `200000`.

Phase 2 passes only if all three development checkpoints satisfy the Stage 2
gate on fresh development-evaluation sequences. If any development seed fails,
stop before final-family training and record a baseline-only audit. Changes
after a failure require a new versioned pre-registration.

Untouched final checkpoint seeds are:

```text
20260901-20260910
```

Final validation seeds are derived by adding `100000`; final competence seeds
by adding `200000`. All ten final checkpoints must pass before any N-back
perturbation outcome is inspected. Checkpoint is the inferential unit.

## Frozen baseline metrics

Per condition, report:

- accuracy;
- mean cross-entropy;
- hit rate;
- false-alarm rate;
- Barrett discriminability: `hit rate - false-alarm rate`;
- Barrett response bias:
  `false-alarm rate / (1 - discriminability)`;
- signal-detection `d_prime`, using the log-linear half-count correction;
- match count and non-match count.

For 2-back, also report one-back-lure accuracy and lure false-alarm rate.

The Barrett response-bias measure is descriptive because it is unstable when
discriminability approaches one.

## Frozen settling analogue

At each scored event, correct-class probability is measured from stimulus
onset. A trial-event settles at the first step for which:

1. correct-class probability is at least `0.80`; and
2. the correct-minus-incorrect probability margin is at least `0.60`; and
3. both conditions persist for three consecutive steps.

Report median settling among settled event-trials, fraction settled, and
restricted-mean settling with unsettled event-trials assigned the nine-step
event cap. Settling is reported separately for correct behavioural decisions
and all event-trials. It is an RNN decision-settling analogue, not human
reaction time.

If fraction settled is below `0.80`, latency is marked invalid and the cell is
reported as a failure-rate effect.

## Frozen Phase 1 implementation gate

Before training:

- deterministic generator tests pass;
- every sequence has the frozen match count;
- 2-back labels equal the two-event comparison exactly;
- every 2-back sequence passes the lure requirement;
- contexts and warm-up masks are correct;
- metric tests cover perfect, all-non-match, false-alarm, miss, and
  non-settling cases;
- a CPU smoke run completes;
- a CUDA smoke run confirms that model parameters, inputs, targets, and loss
  reside on CUDA when CUDA is available;
- the complete pre-existing test suite passes.

## Frozen N-back perturbation question

N-back perturbation analysis begins only after all final baselines pass.
Operators retain the definitions in
`docs/preregistration/psilocybin_signature_preregistration.md`.

For each checkpoint and candidate operator, select a strength that produces a
30% proportional increase in 0-back error relative to that checkpoint's
baseline, where:

```text
error = 1 - accuracy
```

P5 Gaussian state noise is calibrated to the same 0-back proportional cost.
Because ceiling-level 0-back error can make proportional ratios unstable, the
Phase 0 baseline-only precision audit must estimate the attainable 0-back error
and may replace this denominator with log loss before strength calibration.
That choice must be frozen in an addendum before any perturbation outcome is
computed.

The primary N-back load contrast is:

```text
Delta_candidate =
  (2-back discriminability change - 0-back discriminability change)

Delta_P5 =
  (2-back discriminability change - 0-back discriminability change)

C2_NBACK = Delta_candidate - Delta_P5
```

Signs are oriented so that positive `C2_NBACK` means greater selective 2-back
impairment by the candidate than by matched Gaussian disruption. The same
contrast for restricted-mean settling is secondary and valid only when both
cells pass the settling-fraction guard.

The inferential family, checkpoint-level analysis, operator correction, and
generic-noise requirement remain those of the original Phase 0 registration.
This document changes the load task and its observable, not the biological
claim.

## Phase order and stop rules

1. Freeze this document and its evidence note in Git.
2. Implement and verify the N-back generator, metrics, trainer, evaluator, and
   configuration without training a development seed.
3. Train development seed `20260821`. Stop immediately on failure.
4. If it passes, train `20260822-23`. Stop if either fails.
5. If all pass, freeze a development audit and train the ten untouched final
   seeds.
6. If all final baselines pass, run the baseline-only calibration-precision
   addendum.
7. Only then integrate and run N-back perturbations.

At every failed competence gate, perturbation outcomes remain uninspected.
