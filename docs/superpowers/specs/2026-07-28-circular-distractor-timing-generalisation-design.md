# Circular Distractor-Timing Generalisation Design

Date: 2026-07-28

## Question

Did the five competent distractor-trained circular RNN checkpoints learn a
general ability to reject irrelevant circular input, or is their performance
strongly dependent on the distractor appearing at its trained midpoint
location?

This evaluation can detect timing dependence. It cannot prove that the network
uses timestep location exclusively, because similar behaviour could arise from
other recurrent-state differences across the delay.

## Frozen checkpoint set

Use the five retained competent checkpoints:

- `20260731`
- `20260732`
- `20260733`
- `20260735`
- `20260736`

Weights remain frozen. No retraining or perturbation is applied.

## Evaluation design

- Delay length: 20 model steps.
- Distractor duration: 5 model steps.
- Trials: 1,024 per checkpoint and condition.
- Batch size: 128.
- Pre-cue duration: 25 model steps.
- Cue duration: 20 model steps.
- Response duration: 25 model steps.
- Distractor positions:
  - `start`: delay-relative start 0;
  - `quarter`: delay-relative start 4;
  - `midpoint`: delay-relative start 8;
  - `three_quarter`: delay-relative start 11;
  - `end`: delay-relative start 15.
- Include one clean condition with no distractor.

These starts are obtained from the existing five-step placement rule at onset
fractions `0.00`, `0.25`, `0.50`, `0.75`, and `1.00`.

## Pairing

Within each checkpoint:

- use the same frozen evaluation seed for clean and all five timing conditions;
- verify that target-angle arrays are exactly equal across conditions;
- verify that distractor-angle arrays are exactly equal across the five
  distractor positions;
- change only distractor timing.

The clean condition has no distractor angle, but its target-angle bank remains
paired with the distractor conditions.

## Outcomes

### Primary outcome

Mean absolute angular response error in degrees for each checkpoint and timing
condition.

For each distractor timing:

```text
raw distractor cost =
    distractor-condition angular error - clean angular error
```

The primary timing-generalisation comparisons are:

```text
early displacement =
    start distractor cost - midpoint distractor cost

quarter displacement =
    quarter distractor cost - midpoint distractor cost

three-quarter displacement =
    three-quarter distractor cost - midpoint distractor cost

late displacement =
    end distractor cost - midpoint distractor cost
```

Positive displacement means that moving the distractor away from the trained
midpoint increased error.

### Supporting outcomes

- median angular error;
- fixation accuracy and fixation validity;
- restricted mean settling time;
- fraction of trials settled;
- latency validity;
- delay-decoder error, if available from the existing frozen decoder;
- distractor-induced drift and recovery metrics, if already returned by the
  existing evaluator without changing their definitions.

Settling outcomes will only be interpreted when their existing validity gates
pass.

## Summaries

For every timing condition and timing-minus-midpoint comparison, report:

- all five checkpoint values;
- checkpoint mean;
- checkpoint SD;
- student-t 95% interval across checkpoints;
- number of checkpoints with the same effect direction.

Checkpoint is the inferential unit. Trials are not treated as independent
replicates for across-model inference.

## Interpretation rules

- **Evidence of timing-specific filtering:** non-midpoint distractor costs are
  consistently larger than midpoint cost, especially at both delay extremes.
- **Evidence of timing generalisation:** distractor costs remain similar across
  positions without a consistent midpoint advantage.
- **Asymmetric generalisation:** performance differs mainly for early or late
  distractors, suggesting a delay-state-dependent vulnerability rather than a
  simple midpoint gate.
- **Invalid comparison:** fixation or latency validity fails broadly, or paired
  target/distractor banks cannot be verified.

No arbitrary numerical pass threshold will be introduced after inspecting the
results. Conclusions will be descriptive and seed-level uncertainty will be
reported.

## Outputs

Write to a new directory without modifying the completed candidate-screen
outputs:

```text
outputs/circular_distractor_timing_generalisation/
```

Required artifacts:

- full checkpoint-by-condition metric CSV;
- paired-bank verification metadata;
- checkpoint-level timing-comparison CSV;
- JSON summary;
- concise Markdown report under `docs/reports/`;
- changelog and vault project-state updates.

## Verification

- Unit tests for timing-position resolution and paired-bank equality.
- Exact expected row counts.
- All five checkpoint hashes verified before evaluation.
- Neutral midpoint condition checked against the existing native midpoint
  evaluation within deterministic numerical tolerance.
- Focused test suite and `git diff --check`.

## Scientific boundary

This is a post-result robustness analysis prompted by supervisor discussion.
It tests whether the trained circular family generalises distractor filtering
across time. It is not a preregistered confirmatory test, a human-task
replication, or evidence for a biological psilocybin mechanism.
