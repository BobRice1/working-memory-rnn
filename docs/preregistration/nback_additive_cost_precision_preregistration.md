# N-back additive-cost precision pre-registration

## Registration status

Frozen on 2026-07-27 after the ten-checkpoint screened family was committed
and before collecting a new precision-reference sequence, calibrating any
strength, or running any perturbation.

Final retained checkpoints, in inferential order:

```text
20260912, 20260913, 20260914, 20260915, 20260916,
20260917, 20260918, 20260919, 20260920, 20260921
```

The family audit is
`docs/preregistration/nback_competence_screened_final_pool_audit.md`.

## Reason for replacing proportional cost

The original N-back registration allowed the Phase 0 baseline audit to
replace proportional classification error with log loss before any
perturbation was computed. That replacement is required:

- untouched 0-back accuracy is at or extremely near 1.0;
- mean untouched 0-back cross-entropy ranges from approximately `0.00145` to
  `0.00478` nats across checkpoints;
- dividing by classification error is undefined or unstable;
- dividing by baseline cross-entropy would make the same absolute confidence
  change more than three times as severe across this family.

The N-back matched-cost unit is therefore changed from proportional error to
an **additive mean sequence log-loss increase**:

```text
delta_CE =
  mean(perturbed sequence CE) - mean(native baseline sequence CE)
```

Each sequence CE is the mean natural-log loss over that sequence's registered
scored timepoints. The complete 20-item sequence is one observation; items
and timepoints are not independent units.

## Frozen future matched-cost targets

These values are fixed before the precision-reference run:

```text
target additive cost:                  0.050 nats
held-out acceptable point-cost band:  [0.040, 0.060] nats
maximum 95% bootstrap half-width:      0.005 nats
candidate-versus-P5 absolute gap:      0.005 nats
calibration numerical tolerance:       0.0025 nats
paired-bootstrap draws:                10000
```

An additive `0.050`-nat increase multiplies the geometric mean probability
assigned to the correct class by `exp(-0.050)`, approximately `0.951`. It is
a small confidence cost, not a human dose estimate.

## Baseline-only precision-reference run

For each retained checkpoint:

1. generate exactly 8,192 new 0-back sequences in 64 homogeneous batches of
   128;
2. evaluate only the native, unperturbed checkpoint;
3. collect one `sequence_log_loss_units` value per sequence;
4. require all 8,192 values to be finite and non-negative;
5. record the mean, sample standard deviation (`ddof=1`), median, interquartile
   range, 90th, 95th, and 99th percentiles, and maximum.

Do not evaluate 2-back in this phase. Do not construct an operator, calibrate
a strength, generate P2/P5 stochastic streams, or inspect a candidate/P5
outcome.

## Family-wide variance planning

Use 10,000 deterministic bootstrap draws. Within every draw:

1. resample 8,192 sequences with replacement independently within each of
   the ten checkpoints;
2. calculate the sample SD for each checkpoint;
3. retain the maximum of the ten checkpoint SDs.

Define `sigma_upper` as the 95th percentile of those 10,000 maximum-SD
values. This protects planning against selecting the most favorable
checkpoint.

Baseline-only data cannot identify future paired-difference variance because
the perturbed variance and baseline-perturbed covariance do not yet exist.
The planning calculation therefore freezes these conservative assumptions:

```text
perturbed sequence-CE SD <= kappa * baseline sequence-CE SD
kappa = 2
baseline-perturbed covariance >= 0
```

Ignoring non-negative covariance is conservative. With desired absolute
half-width `h = 0.005`, calculate:

```text
n_required =
  (1.96 * sigma_upper * sqrt(1 + kappa^2) / h)^2

n_cost_check =
  max(1024, round_up_to_next_multiple_of_128(n_required))
```

The maximum permitted `n_cost_check` is 8,192 sequences per checkpoint and
cell. If the formula exceeds 8,192, stop before strength calibration and
pre-register a revised compute budget or cost design. Do not widen the band
after observing any operator.

The eventual paired bootstrap on actual baseline-perturbed sequence
differences remains the binding precision gate. If its half-width exceeds
`0.005`, that cell is `NA`; its sample size may not be increased after seeing
the result.

## Frozen data-bank addressing

The retained checkpoint ordinal is its zero-based position in the frozen
family above. For task sequences use:

```text
task_seed =
  bank_base
  + 10000 * checkpoint_ordinal
  + 1000 * condition_code
  + batch_index
```

Condition codes are `0` for 0-back and `1` for 2-back. Batch indices begin at
zero. The non-overlapping task-bank bases are:

| Bank | Base |
| --- | ---: |
| baseline precision reference | 131000000 |
| strength calibration | 132000000 |
| held-out cost check | 133000000 |
| confirmatory 0-/2-back outcomes | 134000000 |

The precision-reference phase uses only base `131000000`, condition code
zero, and batch indices `0-63`.

Future native baseline, candidate, and P5 evaluations within a calibration
or cost-check cell must reuse identical task sequences. Different phases must
not reuse task banks.

Bootstrap bases are also frozen:

| Purpose | Base |
| --- | ---: |
| family-wide precision SD bootstrap | 135000000 |
| held-out paired cost bootstrap | 136000000 |

The precision SD bootstrap uses seed `135000000`. Exact future cell offsets
from the held-out base must be materialized and committed before strength
calibration.

Stochastic-operator streams must not reuse task or bootstrap seeds. Future
P2 and P5 stream bases are reserved as `137000000` and `138000000`,
respectively. Their exact checkpoint, variant, replicate, branch, and batch
mapping must be frozen before calibration. Existing P2 and P5 three-replicate
within-checkpoint averaging remains unchanged; replicates are not inferential
units.

## Operator and branch decisions not reopened

The operator definitions, variants, neutral points, and grids already frozen
in `docs/preregistration/psilocybin_signature_preregistration.md` remain in
force for the N-back branch, except that P3b distractor gain is inapplicable.
The applicable profiles are:

- P1 synaptic-drive gain, both bias placements, below- and above-neutral;
- P2 heterogeneous synaptic-drive gain;
- P3a six-channel sensory-input gain, below- and above-neutral;
- P4 recurrent gain, below- and above-neutral;
- P5 Gaussian state noise as the generic comparator;
- P6 state persistence, below- and above-neutral;
- P7 effective time constant, below- and above-neutral.

Calibration remains branch-restricted, monotonicity-checked, and
no-extrapolation. A branch that cannot bracket the additive target is `NA`;
the grid may not be extended after inspecting outcomes.

## Precision-phase acceptance and next decision

The precision phase passes only if:

- every checkpoint yields 8,192 valid sequence units;
- `sigma_upper` and `n_required` are finite;
- the frozen formula yields `n_cost_check <= 8192`;
- the complete seed map and descriptive summaries are persisted.

After a pass, record and commit the derived `n_cost_check`. Then freeze the
exact additive calibration, held-out checking, stochastic-stream mapping,
and N-back outcome-runner specification before computing any perturbation.

If the phase fails, stop. Do not run strength grids, P5, 2-back outcomes, or
the candidate-versus-noise contrast.

## Claim boundary

This phase estimates computational precision for matching small 0-back costs.
It does not test a behavioural dissociation, establish a psilocybin effect,
or validate any perturbation as a biological drug mechanism.
