# N-back additive-cost perturbation pre-registration

## Registration status

Frozen on 2026-07-27 after:

- ten competent N-back checkpoints were retained and committed;
- the baseline-only additive-cost precision phase passed;
- `n_cost_check` was fixed at 1,024 sequences;
- and before constructing any non-neutral operator, evaluating a strength
  grid, running P5, or inspecting a 2-back perturbation outcome.

Authoritative prior documents:

- `nback_competence_screened_final_pool_audit.md`;
- `nback_additive_cost_precision_preregistration.md`;
- `nback_additive_cost_precision_audit.md`;
- `psilocybin_signature_preregistration.md`.

## Scientific question and scope

The N-back branch tests the Barrett-style load component:

> At the same small additive 0-back confidence cost, does a candidate
> perturbation impair 2-back discriminability more selectively than generic
> Gaussian state noise?

This branch supplies only **C2**, the load-selective component of the
cross-task signature. Family A supplies C1 settling and C3 distractor
selectivity. N-back alone cannot establish a complete psilocybin-signature
match.

The biological claim remains computational sufficiency only. None of the
operators is treated as pharmacologically equivalent to psilocybin.

## Frozen checkpoint family

Checkpoint is the inferential unit. The exact ten checkpoints, in ordinal
order, are:

```text
0: 20260912
1: 20260913
2: 20260914
3: 20260915
4: 20260916
5: 20260917
6: 20260918
7: 20260919
8: 20260920
9: 20260921
```

The runner must load checkpoint paths from the retained manifest, verify its
hash and every checkpoint hash, and never discover checkpoints by globbing.

## Frozen operator-profile manifest

Stable profile IDs, names, grids, and analysis classes are:

| ID | Operator | Variant | Branch | Ordered grid from neutral outward | Class |
| ---: | --- | --- | --- | --- | --- |
| 0 | P1 synaptic-drive gain | `bias_outside` | below | `[1.0, 0.975, 0.95, 0.90]` | descriptive |
| 1 | P1 synaptic-drive gain | `bias_outside` | above | `[1.0, 1.025, 1.05, 1.10, 1.15, 1.20]` | confirmatory |
| 2 | P1 synaptic-drive gain | `bias_inside` | below | `[1.0, 0.975, 0.95, 0.90]` | descriptive |
| 3 | P1 synaptic-drive gain | `bias_inside` | above | `[1.0, 1.025, 1.05, 1.10, 1.15, 1.20]` | descriptive |
| 4 | P2 heterogeneous drive gain | `bias_outside` | above | `[0.0, 0.025, 0.05, 0.075, 0.10, 0.15]` | confirmatory |
| 5 | P2 heterogeneous drive gain | `bias_inside` | above | `[0.0, 0.025, 0.05, 0.075, 0.10, 0.15]` | descriptive |
| 6 | P3a sensory-input gain | `six_sensory_channels` | below | `[1.0, 0.975, 0.95, 0.90]` | descriptive |
| 7 | P3a sensory-input gain | `six_sensory_channels` | above | `[1.0, 1.025, 1.05, 1.10, 1.15, 1.20]` | confirmatory |
| 8 | P4 recurrent gain | `weights_only` | below | `[1.0, 0.975, 0.95, 0.90]` | descriptive |
| 9 | P4 recurrent gain | `weights_only` | above | `[1.0, 1.025, 1.05, 1.10, 1.15, 1.20]` | confirmatory |
| 10 | P6 state persistence | `carried_state_only` | below | `[1.0, 0.975, 0.95, 0.90]` | confirmatory |
| 11 | P6 state persistence | `carried_state_only` | above | `[1.0, 1.025, 1.05, 1.10, 1.15, 1.20]` | descriptive |
| 12 | P7 effective time constant | `conserved_integrator` | below | `[1.0, 0.95, 0.90, 0.80]` | confirmatory |
| 13 | P7 effective time constant | `conserved_integrator` | above | `[1.0, 1.05, 1.10, 1.25]` | descriptive |
| 14 | P5 Gaussian state noise | `generic_control` | above | `[0.0, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10]` | comparator |

P3b is absent rather than an NA profile because this N-back task has no
distractor window. The six confirmatory profiles remain IDs
`1, 4, 7, 9, 10, 12`.

## Frozen phase sizes and task banks

| Phase | Bank base | Conditions | Batches | Batch size | Sequences |
| --- | ---: | --- | ---: | ---: | ---: |
| strength calibration | 132000000 | 0-back | 4 | 128 | 512 per checkpoint and setting |
| held-out cost check | 133000000 | 0-back | 8 | 128 | 1,024 per checkpoint and cell |
| confirmatory outcome | 134000000 | 0-back and 2-back | 8 each | 128 | 1,024 per checkpoint and condition |

Task seeds retain the frozen formula:

```text
task_seed =
  bank_base
  + 10000 * checkpoint_ordinal
  + 1000 * condition_code
  + batch_index
```

Condition code is zero for 0-back and one for 2-back. Native baseline,
candidate, and P5 receive identical task sequences within a phase and cell.
No task bank is reused between phases.

The confirmatory bank contains exactly 18,432 scored decisions per condition
and checkpoint: 6,144 matches and 12,288 nonmatches.

## Frozen P2 and P5 replicate streams

Replicates are nested within checkpoint and never increase inferential `n`.

### P2

Preserve the already frozen literal gain-vector seeds:

```text
3101, 3102, 3103
```

Each seed defines one static per-unit gain assignment for a checkpoint and
log-standard-deviation. The same seed is reused across phase, condition,
batch, strength evaluation, and bias variant. This supplies common random
numbers and permits direct comparison of bias placement.

The reserved base `137000000` is not used by the core P2 construction. It
remains reserved for a separately pre-registered future assignment/permutation
analysis and may not replace the literal seeds above.

### P5

Persist display labels `4101, 4102, 4103`, mapped to replicate ordinals
`0, 1, 2`. The actual collision-free generator seed is:

```text
p5_seed =
  138000000
  + 1000000 * checkpoint_ordinal
  + 100000 * phase_code
  + 10000 * condition_code
  + 1000 * replicate_ordinal
  + batch_index
```

Phase codes are calibration `0`, cost check `1`, and confirmatory `2`.
Strength is deliberately absent, giving common Gaussian draws across grid and
bisection strengths.

For P2 and P5 additive costs, average the three perturbed CE values
sequence-wise before subtracting the corresponding native baseline CE. For
behavioural outcomes, calculate each replicate's pooled-count metric first,
then average the three scalar replicate metrics. Do not concatenate replicate
trials or average logits.

Any missing or nonfinite replicate invalidates the cell; do not average the
remaining two.

## Required code corrections before execution

The implementation must:

1. replace the proportional N-back `SequenceLogLossCost` helper with
   `additive_cost = mean(perturbed CE) - mean(baseline CE)`;
2. never call the proportional calibration APIs for this branch;
3. add an explicit branch monotonicity validator before bisection;
4. make N-back metrics require exact equality between supplied
   targets/masks and batch metadata;
5. reject nonfinite logits, output sizes other than two, invalid settling
   thresholds, and sequences with no scored timestep;
6. report specificity, balanced accuracy, ordinary-nonmatch performance, and
   explicit `failure_rate = 1 - fraction_settled`;
7. assert at runner level that inputs 6 and 7 are mutually exclusive rule
   context channels and that P3a always targets only inputs 0 through 5.

## Neutral-equivalence firewall

Before any non-neutral calibration strength:

- load every checkpoint in evaluation mode;
- require embedded `recurrent_noise_std == 0`;
- run native and neutral forward passes on the calibration 0-back bank for
  every applicable profile;
- include all three P2 seeds at `log_std=0` and all three P5 replicates at
  `sigma=0`;
- require exact `torch.equal` equality for logits and hidden states;
- require exactly zero additive cost;
- persist maximum absolute differences and pass/fail.

At the start of the confirmatory phase, repeat exact neutral equality for both
rule contexts on the confirmatory banks before evaluating non-neutral
outcomes. These comparisons may establish implementation equality only; their
behavioural values may not be used to alter strengths or gates.

Any neutral failure stops that profile before a non-neutral outcome.

## Additive calibration algorithm

The target and tolerance are:

```text
target = 0.050 nats
numerical tolerance = 0.0025 nats
maximum bisection iterations = 12
```

For each checkpoint and profile:

1. evaluate native baseline once on the fixed 512 calibration sequences;
2. evaluate every registered branch-grid point on the same sequences;
3. form one paired sequence-difference vector, averaging P2/P5 replicates
   sequence-wise first;
4. calculate its mean additive cost;
5. cache the units and result for every strength;
6. order strengths from neutral outward;
7. require costs to be non-decreasing with branch distance, allowing an
   adjacent downward inversion of at most `0.0025` nats;
8. if multiple grid points lie within tolerance, select the closest target,
   breaking exact ties toward neutral;
9. otherwise use the first adjacent monotone bracket around `0.050`;
10. bisect in raw strength coordinates for at most 12 iterations;
11. require every midpoint cost to lie between its current endpoint costs,
    allowing the same `0.0025` tolerance;
12. stop only at absolute target error at most `0.0025`.

A reversal beyond tolerance gives `NA: nonmonotone_calibration`. No bracket
gives `NA: unreachable_matched_strength`. Failure after 12 valid midpoint
evaluations gives `NA: calibration_numerical_failure`.

Do not extrapolate, extend a grid, change a branch, increase calibration
sequences, or choose a different point after held-out checking. P5 is
calibrated once per checkpoint and reused for every candidate comparison.

## Held-out additive cost validation

Use exactly 1,024 paired 0-back sequences per checkpoint and selected cell.
For each sequence:

```text
paired_delta =
  replicate-averaged perturbed CE - native baseline CE
```

Bootstrap the paired sequence differences 10,000 times. Cost-check bootstrap
seeds are:

```text
136000000 + 1000 * checkpoint_ordinal + profile_id
```

The cell passes only if:

```text
0.040 <= mean(paired_delta) <= 0.060
95% percentile-CI half-width <= 0.005
```

Every candidate also requires:

```text
abs(candidate point cost - checkpoint P5 point cost) <= 0.005
```

P5 must independently pass its point-cost and precision gates. A failure does
not trigger recalibration, grid extension, or more sequences.

A profile is C2-testable only if its candidate and P5 cells pass for all ten
checkpoints.

## Confirmatory baseline transport

Before non-neutral confirmatory outcomes, native baseline on the new outcome
bank must pass:

| Condition | Metric | Required |
| --- | --- | ---: |
| 0-back | accuracy | at least 0.95 |
| 0-back | HR - FAR | at least 0.90 |
| 2-back | accuracy | at least 0.95 |
| 2-back | HR - FAR | at least 0.90 |
| 2-back | one-back-lure accuracy | at least 0.90 |

Both classes and the lure subset must have nonzero raw counts. If any
checkpoint fails, N-back C2 is `not_testable_validity` for every profile and
candidate outcomes are not run. Baseline failure is not repaired or replaced.

## Primary N-back C2 estimand

For checkpoint `s`, condition `c` in `{0-back, 2-back}`, native baseline `0`,
candidate profile `o`, and matched P5 `g`, calculate discriminability from
pooled item counts:

```text
D = hit_rate - false_alarm_rate
```

Do not average per-sequence discriminabilities. Native baseline denominators
must be finite and strictly positive.

Define:

```text
I_s(o,c) = [D_s(0,c) - D_s(o,c)] / D_s(0,c)
I_s(g,c) = [D_s(0,c) - D_s(g,c)] / D_s(0,c)

S_s(o) = I_s(o,2-back) - I_s(o,0-back)
S_s(g) = I_s(g,2-back) - I_s(g,0-back)

X2_NBACK,s(o) = S_s(o) - S_s(g)
```

Positive `X2_NBACK` means greater selective 2-back impairment by the candidate
than by equally costly Gaussian state noise.

Each profile yields exactly ten checkpoint values. Use a one-sided
checkpoint-level Student t-test:

```text
H0: mean(X2_NBACK) <= 0
H1: mean(X2_NBACK) > 0
```

Report all checkpoint points, mean, SD, paired `d_z`, one-sided p-value, and
two-sided 95% t interval. At the strictest future Holm threshold
`0.05 / 6`, the registered 80% power minimum detectable effect for N-back C2
is paired `d_z = 1.2235`.

The C2 p-value is a diagnostic component, not an independent confirmatory
discovery.

## Cross-task intersection-union integration

For each of the six confirmatory operator profiles, the eventual complete
profile is:

```text
C1: Family A excess settling slowing versus matched P5, n=5
C2: N-back excess load selectivity versus matched P5, n=10
C3: Family A excess distractor selectivity versus matched P5, n=5
```

Family A and N-back strengths are calibrated independently under their own
cost units. The hypothesis concerns the same operator profile and direction,
not an identical numerical strength across architectures.

Define:

```text
p_IUT = max(p_C1, p_C2, p_C3)
```

No multiplicity correction is applied within this conjunction. Holm correction
is applied only across the six frozen profile-level `p_IUT` values.

A complete match requires:

- all three component means positive;
- predicted sign in at least 4/5 Family A checkpoints for C1;
- predicted sign in at least 8/10 N-back checkpoints for C2;
- predicted sign in at least 4/5 Family A checkpoints for C3;
- Holm-adjusted `p_IUT < 0.05`;
- every required validity gate across both task families.

If a profile is invalid, keep the Holm family size at six and insert
`p_IUT=1` for multiplicity calculation only. Report its p-values as NA and
its outcome as `not_testable_validity`. Do not shrink the family.

If all validity gates pass but the complete conjunction fails, use
`tested_no_match_at_registered_power`, not language implying proof that the
mechanism is absent. The five-checkpoint Family A components are the power
bottleneck.

## Descriptive N-back outcomes

Persist, without promotion to C2:

- accuracy, hit rate, false-alarm rate, specificity, balanced accuracy, raw
  counts, corrected d-prime, and Barrett response-bias analogue;
- one-back-lure and ordinary-nonmatch accuracy/false alarms;
- native, candidate, and P5 cross-entropy;
- raw discriminability changes and unnormalized difference-in-differences;
- 2-back additive CE cost and 2-back-minus-0-back CE interaction;
- fraction settled, explicit response-failure rate, and restricted-mean
  settling across all decisions;
- correct-decision-only settling as selection-biased descriptive output only.

N-back settling is a model decision-dynamics analogue, not human reaction
time. Latency contrasts use `settling_all` only and are valid only when native,
candidate, and P5 each have `fraction_settled >= 0.80` in both conditions.
Otherwise latency is NA and failure rates replace latency interpretation.

Within-checkpoint descriptive intervals resample complete sequences, never
items or timepoints. They do not change inferential `n`.

## Descriptive dose-ordering phase

Run only after the primary confirmatory outputs are frozen. For each valid
profile, evaluate:

1. its matched strength;
2. the next registered grid point farther from neutral;
3. the following registered grid point farther from neutral.

Use the unchanged confirmatory task bank. If two farther grid points do not
exist, dose ordering is `NA: insufficient_registered_grid`.

For the N-back C2 component:

- `preserved`: predicted candidate-versus-baseline load-selectivity sign at
  all three points and non-decreasing magnitude;
- `degraded`: predicted sign at the matched point and at least one farther
  point, but the pattern is not non-decreasing;
- `scrambled`: the predicted sign is absent at both farther points or
  reverses.

Higher points are descriptive and are not P5 cost-matched constraint tests.
Overlay P5's full registered strength-cost curve. Do not refit a dose-response
model or promote dose ordering to the primary claim.

## Validity and NA rules

A profile is `not_testable_validity` if any required checkpoint has:

- neutral-equivalence failure;
- unreachable, nonmonotone, extrapolated, or numerically failed calibration;
- held-out point cost outside `[0.040, 0.060]`;
- held-out half-width above `0.005`;
- candidate-P5 point-cost gap above `0.005`;
- missing or nonfinite P2/P5 replicate;
- missing or nonfinite logits, raw counts, or metric;
- nonpositive native discriminability denominator;
- failed confirmatory baseline transport.

Do not drop, impute, replace, or rerun a checkpoint with more sequences.

Low or negative perturbed discriminability, all-match/all-nonmatch model
predictions with defined task class counts, negative C2, and increased
response failure are valid adverse outcomes rather than NA.

Settling or response-bias invalidity affects only those secondary metrics, not
primary C2.

## Persistence, phase barriers, and safe resume

Use:

```text
outputs/nback_additive_perturbation/
  manifest/
  neutral/
  calibration/
  cost_check/
  confirmatory/
  dose/
  metrics/
  state/
```

Every cell is written atomically through a temporary file and replace. The
run manifest must hash:

- this pre-registration;
- the screened-pool manifest and each checkpoint;
- the additive precision summary and arrays;
- the executable config;
- profile and seed manifests;
- the implementation commit/design hash.

Expose explicit phases:

```text
neutral-calibration
calibration
cost-check
neutral-confirmatory
confirmatory
dose
finalize
```

Each later phase refuses to start until the prior phase is complete and all
recorded hashes validate. Confirmatory 2-back data cannot be generated during
calibration or cost checking.

On resume, require an exact design-hash match, verify completed artifact
hashes, reuse valid cells, and run only missing cells. A corrupt cell or
changed design/checkpoint stops the run. Never silently overwrite a valid
completed cell.

Run checkpoint-by-checkpoint, discard tensors immediately, record synchronized
phase wall time, and do not use mixed precision.

## Required implementation tests

Before execution, tests must cover:

- exact profile IDs, grids, classes, and P3b exclusion;
- additive rather than proportional cost;
- target/mask equality and finite-logit checks;
- monotonicity pass, tolerated inversion, hard inversion, bisection, midpoint
  violation, unreachable target, and no extrapolation;
- P2 literal seed invariance and P5 seed uniqueness/invariance across
  strength;
- per-sequence replicate averaging without increasing `n`;
- exact neutral equality for all operator types;
- task-bank non-overlap and identical sequence hashes within cells;
- deterministic paired bootstrap and all three held-out gates;
- inability of held-out results to modify selected strength;
- baseline transport and phase barriers;
- pooled-count discriminability and positive C2 sign convention;
- settling failure-rate and latency-NA propagation;
- P3a context-channel protection;
- atomic resume, design-hash mismatch, and corrupt-artifact refusal;
- static rejection of proportional calibration imports/calls.

The full repository test suite must pass before the first non-neutral
operator is constructed.

## Claim boundary

A positive N-back C2 component means an operator class reproduces the
registered load-selective behavioral pattern better than matched generic
noise in this RNN task. It does not establish a complete human psilocybin
signature, causal biology, receptor pharmacology, or mechanistic uniqueness.
