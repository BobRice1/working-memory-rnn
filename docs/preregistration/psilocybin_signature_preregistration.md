# Psilocybin-Signature Perturbation Experiment: Phase 0 Pre-registration

## Registration status

Frozen on 2026-07-26 before implementing the Phase 1 outcome metrics module and
before running any perturbation outcome grid.

This document freezes the confirmatory and descriptive analysis decisions from
the current vault specification:

`wiki/experiments/psilocybin-signature-perturbation-experiment-plan.md`

The Git commit that first adds this file is the pre-registration timestamp.
That commit hash must be stored in the later run metadata. No experimental
outcomes are reported here. The sole numerical pilot result included below is
the explicitly permitted, baseline-only Phase 0 precision pilot used to fix the
cost-check sample size.

## Scientific question and claim boundary

Can a post-training perturbation of a task-trained working-memory RNN reproduce
the selective pattern of acute psilocybin-related working-memory effects
reported in humans more convincingly than generic Gaussian disruption, and
what recurrent dynamics explain that match?

The human literature supplies qualitative, ordinal constraints rather than a
quantitative target vector. Model angular degrees and settling steps are not
commensurable with human milliseconds or signal-detection measures. Nothing is
fitted to human effect sizes, and no distance from a human target vector will
be reported.

A successful operator supports the claim that it produces a **qualitative
behavioural-signature match beyond matched generic Gaussian degradation within
these RNN task analogues**. It does not establish biological equivalence to
psilocybin, reproduce a human reaction time directly, or identify
psilocybin's biological mechanism.

The primary confirmatory analysis is restricted to Family B, which supplies
all three constraints within one trained task family and ten independently
trained checkpoints. Family A supplies descriptive continuity and
mechanism-diagnostic analyses only.

## Model families

- **Family A:** the five existing frozen fixation-gated circular working-memory
  checkpoints with seeds `20260714` through `20260718`.
- **Family B:** ten independently trained checkpoints using a factorial
  two-slot task with `load1_clean`, `load1_distractor`, `load2_clean`, and
  `load2_distractor` conditions. The first five seeds match Family A. Family B
  uses matched two-slot timing for load 1 and load 2, records actual per-item
  retention, and supports a balanced probe channel (`-1` first item, `+1`
  second item, `0` before response).

Checkpoint, not trial or batch, is the inferential unit. Trials and nested
noise/gain-vector replicates are averaged within checkpoint before inference.

## Frozen perturbation families

- **P1 `synaptic_drive_gain`:** scalar gain `g` on synaptic drive, with
  `bias_outside` and `bias_inside` variants.
- **P2 `heterogeneous_drive_gain`:** a fixed, positive, per-unit lognormal gain
  vector on synaptic drive, rescaled to exact population mean 1.0 and constant
  across time and trials, with `bias_outside` and `bias_inside` variants.
- **P3a `sensory_input_gain`:** gain only on the tuned sensory-channel
  contribution; fixation, probe contributions, and biases remain unscaled.
- **P3b `distractor_input_gain`:** gain only on the tuned distractor
  contribution during the distractor slice; it is identity on clean trials.
- **P4 `recurrent_gain`:** gain only on `W_h h`, with recurrent bias re-added
  outside the gain.
- **P5 `gaussian_state_noise`:** seeded additive Gaussian noise on the
  pre-activation update. This is the generic degradation comparator.
- **P6 `state_persistence`:** gain on the carried-state term only. This breaks
  the conserved leaky-integrator form and is not a time-constant manipulation.
- **P7 `time_constant`:** scale effective `tau` and recompute both the
  carried-state and drive coefficients consistently.

## Locked decisions D1-D10

### D1. Gain placement and naming

P1 and P2 apply gain to synaptic drive inside the nonlinearity. P6 alone scales
the full carried-state contribution. Because this CTRNN applies `tanh` after
mixing drive and carried state, P1/P2 are named **synaptic-drive gain** and
described as the closest available response-gain analogue. They must not be
called response gain.

### D2. Bias and threshold treatment

P1 and P2 have two variants:

- `bias_outside = g * (W_x x + W_h h) + b`
- `bias_inside = g * (W_x x + W_h h + b)`

Both are run. `bias_outside`, which leaves the threshold-like bias unscaled, is
the primary literature-motivated variant.

### D3. Strength grids and dose analogy

The frozen grids are:

- P1, P3a, P4, and P6:
  `[0.90, 0.95, 0.975, 1.00, 1.025, 1.05, 1.10, 1.15, 1.20]`
- P2 log standard deviation:
  `[0.0, 0.025, 0.05, 0.075, 0.10, 0.15]`
- P5 Gaussian sigma:
  `[0.0, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10]`
- P7 `tau_scale`:
  `[0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25]`

P2 gain-vector seeds are `(3101, 3102, 3103)`. Their estimates are averaged
within checkpoint. P5 generator replicates are `(4101, 4102, 4103)`, averaged
within checkpoint for calibration, cost checks, and final outcomes. The full
generator seed is derived from the D8 batch seed, checkpoint seed, and replicate
ID. Replicates are not inferential units.

Strength is only an ordinal dose analogue, not an estimate of a human dose.
Bidirectional sweeps are retained where meaningful. The pre-specified
dose-ordering rule appears below.

### D4. Dynamical estimator and analysis window

For bounded `tanh` states, estimate per-unit fixed-bin histogram differential
entropy on `[-1, 1]` with 64 bins and average over units. This differs from
Herzog's estimator and must not be compared numerically with it.

The primary entropy, activation-slope, and saturation window is the final
`min(10, delay_steps)` delay steps. Response-window versions may be secondary
but may not replace this matched late-delay window. Entropy is a secondary
outcome.

### D5. Settling-time analogue

Settling is an analogue of reaction time because response onset is externally
imposed. A trial first settles at the response-phase step at which both
conditions hold for at least three consecutive steps:

1. decoded angular error is below 15 degrees; and
2. output population-vector length is at least 50% of the checkpoint,
   condition, and delay-specific median baseline response-phase vector length
   estimated on independent D8 metric-reference batches.

The amplitude condition is mandatory. Report:

- `median_settling_steps`, conditional on settled trials;
- `fraction_settled`; and
- `restricted_mean_settling_steps`, assigning unsettled trials the full
  response-window length.

Restricted-mean settling is primary only when the D9 response-failure gate
passes. Conditional median settling is secondary.

### D6. Perturbation window

Operators apply at every timestep of the whole trial. P3b is the sole
exception and applies only during the pre-specified distractor slice.

### D7. Proportional matched-cost calibration

For each checkpoint, target the delay-20 proportional clean-task cost

```text
(perturbed_mean_error - baseline_mean_error) / baseline_mean_error = 0.30
```

using Family A `clean` trials or Family B `load1_clean` trials. Both
proportional and absolute angular-error changes are reported.

The held-out validity band is:

```text
0.20 <= proportional_clean_cost <= 0.40
```

It is a calibration-comparability gate, not evidence that accuracy was
independently preserved and not a human equivalence margin.

#### Frozen baseline-only precision pilot

The permitted Phase 0 pilot used 1,024 clean delay-20 trials per frozen Family A
checkpoint with pilot seeds `202607050`-`202607065`.

| Checkpoint seed | Mean trial response error (degrees) | Trial-level SD (degrees) |
| --- | ---: | ---: |
| `20260714` | 7.013 | 6.144 |
| `20260715` | 3.869 | 1.632 |
| `20260716` | 2.853 | 1.382 |
| `20260717` | 3.951 | 2.015 |
| `20260718` | 3.936 | 1.971 |

The conservative independent-means precision rule is:

```text
n_required = 2 * (1.96 * SD / (0.10 * baseline_mean))^2
```

The worst Family A checkpoint requires 590 trials, rounded to 640 trials
(10 complete 64-trial batches). The frozen choice is nevertheless:

```text
n_cost_check_A = 1024
```

For Family B, before inspecting any perturbation outcome, estimate baseline-only
mean and SD on D8 metric-reference trials and set:

```text
n_cost_check_B = max(1024, round_up_to_64(max_seed(n_required)))
```

This blinded precision adaptation may increase sample size but may not change
the target or validity band. The expected conservative 95% half-width and
selected count are recorded in run metadata.

Baseline, each calibrated candidate, and calibrated P5 are evaluated on the
same dedicated D8 cost-check trials. Those trials may not move or reselect a
strength. Use 10,000 deterministic paired-trial bootstrap resamples. A cost
check is valid only if the point estimate lies in `[0.20, 0.40]` and its
bootstrap interval half-width is at most `0.10`.

Candidate and P5 must also satisfy, per checkpoint:

```text
abs(achieved_proportional_cost(candidate)
    - achieved_proportional_cost(P5)) <= 0.05
```

Record the signed difference as `p5_cost_gap`. A larger gap is
`p5_cost_mismatch` and makes the checkpoint not confirmatorily testable. Report
the gap distribution and both unadjusted and `p5_cost_gap`-adjusted C2/C3
estimates. Material disagreement is reported as unresolved.

For P1, P3a, P4, P6, and P7, calibrate below-neutral and above-neutral branches
separately by bisection within a monotone bracket found from the D3 grid. Do not
extrapolate if the target is unreachable. P2 and P5 have only
`above_neutral`. P3b is inactive on clean trials; match it to P3a's
distractor-attraction effect and classify it `matched_distractor` and
`descriptive_only`.

### D8. Data separation

Use disjoint, frozen seed families:

| Purpose | Seed base |
| --- | ---: |
| Family A precision pilot; baseline only, never scored | `202607050` |
| Decoder and metric references | `202607100` |
| Matched-cost calibration | `202607200` |
| Held-out high-precision cost check | `202607250` |
| Final outcome evaluation | `202607300` |

No trial used to fit the hidden decoder, set the settling amplitude reference,
estimate the cost-check count, or select perturbation strength may enter final
outcome scoring. Cost-check trials validate calibration transport but do not
estimate C1-C3. Store the exact seed-offset formula and hashes of all generated
angle arrays in run metadata.

### D9. Validity gates

1. **Fixation:** a perturbed seed-condition-delay cell must have fixation
   accuracy at least `0.90`. Otherwise settling-derived quantities and C1 are
   `NA`, reason `fixation_failure`. An operator-level settling score requires at
   least 80% valid checkpoint cells.
2. **Response failure:** if either baseline or perturbed
   `fraction_settled < 0.50`, latency is not interpretable. Mark latency `NA`,
   reason `low_fraction_settled`, and report failure rate and its paired change.
   Do not use cap-pinned restricted means to claim latency similarity. An
   operator-level latency score requires at least 80% valid checkpoint cells.

### D10. Outcome taxonomy

Every candidate profile receives exactly one label:

- `confirmatory_match`: all four confirmatory criteria below are met.
- `tested_null`: all required validity gates pass, the profile is genuinely
  testable, and it fails one or more substantive match criteria. This is
  evidence against the mechanism only at the pre-registered power.
- `not_testable_validity`: a required gate fails. Record one or more of
  `cost_band_failure`, `cost_precision_failure`, `p5_cost_mismatch`,
  `fixation_failure`, `low_fraction_settled`,
  `unreachable_matched_strength`, or `p5_reference_invalid`, plus affected
  checkpoint count. This is not evidence for or against the mechanism.
- `descriptive_only`: the profile was never in the confirmatory family.

Do not drop invalid checkpoints to rescue a profile. For
`not_testable_validity`, report the valid-checkpoint estimate as exploratory
without null language. Report counts for all four labels.

## Empty descriptive scoring schema

The following Section 9a table is frozen with every scoring cell empty.
P5 is the comparator and therefore is not a candidate row. Values may later be
filled only as `yes`, `partial`, `no`, or `NA` under the rule below.

| Operator profile | Settling slowing | Response failure | Retention-dependent | Distractor-selective | Load-dependent | Dose-ordered | Dynamics differ from P5 | Assignment-sensitive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 `bias_outside`, below neutral |  |  |  |  |  |  |  |  |
| P1 `bias_outside`, above neutral |  |  |  |  |  |  |  |  |
| P1 `bias_inside`, below neutral |  |  |  |  |  |  |  |  |
| P1 `bias_inside`, above neutral |  |  |  |  |  |  |  |  |
| P2 `bias_outside`, above neutral |  |  |  |  |  |  |  |  |
| P2 `bias_inside`, above neutral |  |  |  |  |  |  |  |  |
| P3a, below neutral |  |  |  |  |  |  |  |  |
| P3a, above neutral |  |  |  |  |  |  |  |  |
| P3b, matched distractor |  |  |  |  |  |  |  |  |
| P4, below neutral |  |  |  |  |  |  |  |  |
| P4, above neutral |  |  |  |  |  |  |  |  |
| P6, below neutral |  |  |  |  |  |  |  |  |
| P6, above neutral |  |  |  |  |  |  |  |  |
| P7, below neutral |  |  |  |  |  |  |  |  |
| P7, above neutral |  |  |  |  |  |  |  |  |

### Exact descriptive decision rule

At the proportional matched-cost strength unless otherwise stated:

- `yes`: predicted sign in at least 80% of valid checkpoints and the two-sided
  95% Student-t interval excludes zero in that direction, using actual degrees
  of freedom (`t_0.975,4 = 2.776` for five valid Family A checkpoints and
  `t_0.975,9 = 2.262` for ten valid Family B checkpoints);
- `partial`: predicted sign in at least 80% of valid checkpoints, but the 95%
  interval includes zero;
- `no`: otherwise; and
- `NA`: structurally inapplicable, matched cost unreachable, fewer than 80% of
  checkpoints pass applicable D7/D9 gates, or latency is cap-dominated.

This table is descriptive. Its intervals are not multiplicity-controlled and
must not be described as confirmatory.

### Frozen feature definitions

| Feature | Operational test |
| --- | --- |
| Settling slowing at matched clean cost | Report restricted-mean change versus baseline and excess slowing versus calibrated P5; only the P5 contrast enters C1 |
| Response failure | `delta_failure_rate > 0`; replace latency interpretation when `fraction_settled < 0.50` |
| Retention-dependent | Family A: regress delta angular error on `log2(delay_steps / 10)`. Family B: use actual probed retention, estimate slopes within serial position over trained delays, and report retention-by-position interaction |
| Distractor-selective | Absolute-degree DiD is descriptive; the primary proportional, P5-referenced contrast is C3 |
| Load-dependent | Absolute-degree DiD is descriptive; the primary proportional, P5-referenced contrast is C2 |
| Dose-ordered | Apply the frozen dose-ordering rule below |
| Dynamics differ from Gaussian | Two-sided late-delay entropy change versus P5 at matched clean cost; distractor attraction/recovery and response geometry are secondary |
| Assignment-sensitive | Result-contingent Phase 8, P2 only: consistent non-zero checkpoint-level mean of within-checkpoint outcome versus gain-strength-alignment slopes; `NA` otherwise |

## Noise-referenced ordinal constraints

For candidate operator `o`, calibrated P5 Gaussian noise `g`, baseline `0`, and
condition `c`, define:

```text
r(o,c) = (error(o,c) - error(0,c)) / error(0,c)
```

Every candidate and P5 uses its own checkpoint-specific D7-calibrated strength,
must pass the high-precision cost check and pairwise P5 cost-gap gate, and is
evaluated on identical D8 final trials.

| Constraint | Frozen contrast |
| --- | --- |
| C1: excess settling slowing versus P5 | `X1(o) = [RMST(o,load1_clean) - RMST(0,load1_clean)] - [RMST(g,load1_clean) - RMST(0,load1_clean)] > 0`, with D7 and D9 valid |
| C2: excess proportional load selectivity versus P5 | `L(o) = r(o,load2_clean) - r(o,load1_clean)` and `X2(o) = L(o) - L(g) > 0` |
| C3: excess proportional distractor selectivity versus P5 | `D(o) = r(o,load1_distractor) - r(o,load1_clean)` and `X3(o) = D(o) - D(g) > 0` |

Absolute-degree load and distractor DiDs remain descriptive and cannot satisfy
C2/C3. C1 does not independently test preserved accuracy: it tests greater
settling slowing than Gaussian noise at a controlled proportional accuracy cost.

## Primary confirmatory family

Exactly six Family B profiles are confirmatory:

| Profile | Direction | Frozen prior justification |
| --- | --- | --- |
| P1 `bias_outside` | `above_neutral` | Psilocybin is a 5-HT2A agonist; Herzog models agonism as increased gain, and `bias_outside` leaves the threshold-like bias unscaled |
| P2 `bias_outside` | `above_neutral` | Primary literature-motivated heterogeneous, strictly positive, static gain increase |
| P3a | `above_neutral` | Increased sensory drive is the pre-specified sensitivity route to distractor vulnerability |
| P4 | `above_neutral` | Increased recurrent amplification is the pre-specified REBUS-adjacent interpretation; Herzog's system effect arose through recurrent amplification |
| P6 | `below_neutral` | Reduced state retention is the memory-degrading direction of the persistence hypothesis |
| P7 | `below_neutral` | Shortened effective `tau` is the conserved-integrator counterpart of P6 |

The following nine profiles are run but permanently `descriptive_only`:

- P1 `bias_outside`, `below_neutral`;
- P1 `bias_inside`, `above_neutral`;
- P1 `bias_inside`, `below_neutral`;
- P2 `bias_inside`, `above_neutral`;
- P3a, `below_neutral`;
- P4, `below_neutral`;
- P6, `above_neutral`;
- P7, `above_neutral`; and
- P3b, `matched_distractor`.

They cannot be promoted after inspecting results. P5 is the comparator, not a
candidate profile.

## Confirmatory intersection-union and multiplicity procedure

For each of the six profiles, perform three one-sided paired-checkpoint
Student-t tests:

```text
H0: mean(Xk) <= 0
H1: mean(Xk) > 0
```

Define the operator-level intersection-union p-value:

```text
p_IUT(o) = max(p_C1(o), p_C2(o), p_C3(o))
```

This tests the union null that at least one required component is absent.
No multiplicity correction is applied within the conjunction. Apply Holm
correction only across the six primary operator-level `p_IUT` values.
Individual component p-values are diagnostics, not separate confirmatory
discoveries.

A profile is a `confirmatory_match` only if all four conditions hold:

1. all three mean excess contrasts are positive;
2. at least 80% of checkpoints have the predicted sign for each contrast;
3. Holm-adjusted operator-level `p_IUT < 0.05`; and
4. all ten checkpoints pass candidate cost check, P5 cost check, pairwise
   `p5_cost_gap` tolerance, and every constraint-specific D9 gate.

If criterion 4 fails, the outcome is `not_testable_validity`, not a null. If
criterion 4 passes but criteria 1-3 do not, the outcome is `tested_null`.

Report component p-values, `p_IUT`, Holm-adjusted `p_IUT`, all ten checkpoint
points, paired standardized effects `d_z`, and two-sided 95% Student-t
intervals.

## Pre-registered power boundary

With `n = 10`, six primary profiles, one-sided family alpha 0.05, and strictest
Holm threshold `0.05 / 6 = 0.008333`, the critical value is `t = 2.9333` at
9 degrees of freedom.

The frozen calculation is:

```python
from scipy import optimize, stats
import numpy as np


def mde(n, alpha, power):
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha, df)
    return optimize.brentq(
        lambda d: stats.nct.sf(t_crit, df, d * np.sqrt(n)) - power,
        0.01,
        10.0,
    )


print(mde(10, 0.05 / 6, 0.80))
print(mde(10, 0.05 / 6, 0.80 ** (1 / 3)))
print(mde(10, 0.05 / 14, 0.80))
print(mde(10, 0.05 / 14, 0.80 ** (1 / 3)))
```

Frozen reference values:

| Family | Power target | Minimum detectable paired `d_z` |
| --- | ---: | ---: |
| 6 profiles | 80% per component | 1.2235 |
| 6 profiles | 92.83% per component, independence illustration for 80% three-component profile power | 1.4655 |
| Earlier 14-profile reference | 80% per component | 1.4051 |
| Earlier 14-profile reference | 92.83% per component, independence illustration | 1.6633 |

Complete-profile power depends on unknown cross-component correlations. A
`tested_null` at this design excludes only large, consistent excess-over-P5
effects. Moderate effects with wide intervals are reported as inconclusive at
the pre-registered power, not as evidence of no selective mechanism.

## Dose-ordering rule

For each operator, inspect the three underlying signature directions at three
increasing above-neutral strengths: the matched-cost point and the next two
higher grid points. Report proportional clean cost alongside them.

At the matched point, use the formal excess-over-P5 contrasts. At higher points,
report the three underlying candidate-versus-baseline effects and overlay P5's
full strength-cost curve descriptively. Higher points are not constraint tests
because they are not D7/P5 cost-matched.

Record exactly:

```text
dose_ordering: preserved | degraded | scrambled
```

This is an ordinal robustness check, not a dose-response fit or inferential
test. A pattern present at only one strength is a weaker candidate.

## Operator distinguishability rule

Construct each checkpoint-level core profile from five pre-specified paired
contrasts:

1. response-failure rate;
2. retention slope;
3. distractor difference-in-differences;
4. load difference-in-differences; and
5. late-delay entropy change versus baseline.

Standardize each feature once using its pooled across-checkpoint standard
deviation over all operators in the same family.

Also report a six-feature latency-augmented profile adding restricted-mean
settling only for operator pairs where both operators have D9-valid latency in
at least 80% of checkpoints. Otherwise that distance is `NA`; do not impute the
response-window cap.

For operator `i`, let `V_i` be the root-mean-square Euclidean distance of its
checkpoint profiles from its centroid. For operators `i,j`, report:

```text
R_ij = distance(centroid_i, centroid_j)
       / sqrt((V_i^2 + V_j^2) / 2)
```

Classify `R_ij < 1` descriptively as not distinguishable by that profile. Report
the full core matrix and validity-limited latency-augmented matrix. Do not
attach an inferential p-value to this heuristic.

## Standing mechanism prediction

P6 and P7 should be more dependent on actual per-item retention and relatively
flatter across distractor and load, whereas P1/P3 sensitivity changes should be
more distractor- and load-selective and flatter across retention. In Family B,
estimate retention effects within serial position and report the
retention-by-position interaction. Nominal post-cue delay must not replace
per-item retention.

This is a mechanism-diagnostic prediction, not a scored human signature.

## Frozen result schemas

### `metrics/profile_match.csv`

```text
family, operator, variant, branch, profile_class, n_checkpoints,
mean_x1, dz_x1, sign_fraction_x1, p_c1,
mean_x2, dz_x2, sign_fraction_x2, p_c2,
mean_x3, dz_x3, sign_fraction_x3, p_c3,
mean_x2_gap_adjusted, mean_x3_gap_adjusted,
p_iut, p_iut_holm, all_cost_checks_valid, all_metric_gates_valid,
max_abs_p5_cost_gap, all_p5_gaps_valid,
strictest_holm_alpha, component_mde_dz_80,
outcome_label, invalid_reason
```

`profile_class` is `primary` or `descriptive_only`. Only primary rows receive a
Holm-adjusted p-value. The valid outcome labels are exactly those in D10.

### `metrics/cost_match_check.csv`

```text
family, operator, variant, branch, seed, strength, n_trials,
n_noise_replicates,
baseline_mean_angular_error_degrees, mean_angular_error_degrees,
delta_angular_error_degrees, proportional_clean_cost,
paired_difference_sd_degrees, paired_se_degrees,
bootstrap_ci_lower_proportional, bootstrap_ci_upper_proportional,
bootstrap_ci_half_width, cost_precision_valid,
band_lower, band_upper, cost_match_valid,
p5_proportional_clean_cost, p5_cost_gap, p5_cost_gap_valid
```

## Frozen interpretation rules

- C1 is excess settling slowing relative to matched P5 at controlled modest
  accuracy cost. It is not evidence that human-like accuracy preservation was
  independently recovered.
- C2 and C3 use proportional condition effects relative to each condition's own
  baseline and then subtract matched P5. Absolute-degree DiDs are descriptive.
- Delay/retention dependence is a mechanism diagnostic, not a scored human
  target.
- Barrett's signal-detection response bias is not represented. Signed angular
  error and output-vector length are exploratory response-geometry measures.
- The two-item retro-cue manipulation is a graded load analogue, not N-back.
- Settling is an externally cued latency analogue, not human reaction time.
- P1/P2 are synaptic-drive gain analogues, not exact F-I response gain.
- P6 is asymmetric state persistence; P7 is the conserved time-constant
  manipulation.
- A validity failure is never reported as evidence of no mechanism.
- Multiple matching operators imply mechanistic non-identifiability under this
  task battery.
- A signature match demonstrates computational sufficiency only.

## Freeze declaration

The scoring cells above are empty. No perturbation grid, Family B outcome, or
confirmatory C1-C3 result was inspected to create this document. Any change to
the locked decisions, primary family, contrasts, gates, or decision rules after
this commit requires an explicitly labelled amendment made before inspecting
the affected outcomes; otherwise the affected analysis is exploratory.
