# N-back additive perturbation experiment: final audit

Date: 2026-07-27

## Bottom line

The N-back model and measurement design worked, but the registered
candidate-versus-Gaussian perturbation comparison was not testable.

All ten retained CTRNN checkpoints remained competent on independent 0-back
and 2-back sequences. Both exact-neutral firewalls passed. The blocking result
was matched-cost validity: Gaussian comparator P5 could reach the registered
`0.050 +/- 0.0025`-nat calibration target on only nine of ten checkpoints.
The frozen all-ten rule therefore made C2 `not_testable_validity` for every
operator profile. No non-neutral confirmatory or dose outcome was run.

This is not evidence that the candidate mechanisms do or do not reproduce the
human load-selective signature. It is evidence that the present strength grids
do not support the registered equal-cost comparison across the full checkpoint
family.

## Why this is still the right human-signature task

Barrett et al. reported a selective 2-back versus 0-back impairment under
psilocybin. The shared-context N-back CTRNN represents that contrast directly:
the same recurrent weights solve both conditions, 0-back controls for
vigilance and response demands, and 2-back adds working-memory load.

The architecture is literature-grounded:

- Wan, Menendez, and Postle (2022) provide direct six-identity, 20-event,
  match/non-match 2-back RNN precedent
  ([DOI](https://doi.org/10.1371/journal.pcbi.1009062)).
- Lei, Ito, and Bashivan (2024) show that a vanilla RNN can acquire N-back
  competence, supporting retention of the project's interpretable CTRNN
  dynamics ([DOI](https://doi.org/10.52202/079017-3191)).
- Yang and Yu (2025) support interrogating independently trained N-back RNN
  dynamics without equating them with the human mechanism
  ([PubMed](https://pubmed.ncbi.nlm.nih.gov/41326245/)).

The ten-checkpoint screened pool therefore remains an appropriate model family
for attempting the Barrett-style component. The failed comparator gate does
not invalidate the trained N-back task.

## Frozen question

The registered N-back question was:

> At the same small additive 0-back confidence cost, does a candidate
> perturbation impair 2-back discriminability more selectively than generic
> Gaussian state noise?

N-back supplies only C2, the load-selective component:

```text
C1 = Family A excess settling slowing versus matched P5
C2 = N-back excess load selectivity versus matched P5
C3 = Family A excess distractor selectivity versus matched P5
```

A complete psilocybin-signature match requires all three components. N-back
alone was never registered as sufficient.

## Execution audit

The scientific runner was frozen in commit `38f45d8` after 100 focused tests
and 300 repository-wide tests passed. The first non-neutral outcome was
evaluated only after that commit.

| Phase | Outcome | Active seconds |
| --- | --- | ---: |
| neutral-calibration | 150/150 exact-neutral cells passed | 201.8215 |
| calibration | 75/150 cells reached the registered target | 1157.2079 |
| cost-check | 53/150 cells valid; no profile passed all ten | 377.7230 |
| neutral-confirmatory | 300/300 exact-neutral cells passed | 1328.3175 |
| confirmatory | 20/20 native cells passed; 300/300 perturbation cells NA | 961.4707 |
| dose | 150/150 cells NA; no dose arrays generated | 468.7234 |
| finalize | Summary and CSV artifacts completed | 3.6560 |
| **Total** | One uninterrupted CUDA attempt per phase | **4498.9199** |

All phases used the NVIDIA GeForce RTX 3060 Laptop GPU. Every state-recorded
artifact rehashed successfully in the closing audit.

## Calibration result

The following profiles calibrated on every checkpoint:

| Profile | Operator and direction | Calibration |
| ---: | --- | ---: |
| 1 | synaptic-drive gain, `bias_outside`, above | 10/10 |
| 3 | synaptic-drive gain, `bias_inside`, above | 10/10 |
| 9 | recurrent gain, above | 10/10 |
| 11 | carried-state persistence, above | 10/10 |
| 12 | conserved-integrator time constant, below | 10/10 |
| 13 | conserved-integrator time constant, above | 10/10 |

Of the six confirmatory candidates, profiles 1, 9, and 12 passed this first
all-ten barrier. Profiles 4, 7, and 10 calibrated on 1, 0, and 5 checkpoints,
respectively.

Gaussian P5 calibrated on 9/10 checkpoints. For checkpoint seed `20260913`,
its additive cost increased monotonically across the registered grid but
reached only `0.0433148` nats at the maximum strength `0.1`. The runner
correctly refused extrapolation.

There were no nonfinite numerical failures.

## Held-out cost result

On the independent 1,024-sequence cost bank:

- P5 transported on all nine checkpoints for which it had a calibrated
  strength, with point costs from `0.044388` to `0.051666` nats.
- P5 remained invalid for seed `20260913`, so it failed the registered all-ten
  comparator barrier.
- Confirmatory valid-checkpoint counts after every held-out gate were:
  profile 1, 7/10; profile 4, 1/10; profile 7, 0/10; profile 9, 2/10;
  profile 10, 4/10; and profile 12, 8/10.
- Additional failures included candidate-P5 point-cost gaps above `0.005` and
  paired-bootstrap half-widths above `0.005`.

No recalibration, grid extension, checkpoint replacement, or extra sampling
was performed.

## Confirmatory-bank competence

All native baselines passed the independent confirmatory transport gates:

| Metric | Range across checkpoints |
| --- | ---: |
| 0-back accuracy | 0.999946 to 1.000000 |
| 2-back accuracy | 0.956868 to 0.987142 |
| 2-back discriminability | 0.918620 to 0.973063 |

Thus the final NA is not caused by failed N-back training. It is specifically a
matched-cost perturbation-validity result.

## Final registered outcome

The final checkpoint CSV is intentionally header-only because no profile has
ten valid candidate-versus-P5 contrasts. All 14 non-P5 profile rows are:

```text
valid = false
invalid_reason = not_testable_validity
n_checkpoints = 0
```

Consequently:

- no C2 mean, effect size, t statistic, p-value, or sign count exists;
- no profile can enter the C1-C2-C3 intersection-union test;
- dose ordering is NA rather than preserved, degraded, or scrambled;
- and the run neither recovers nor rejects a psilocybin-like behavioral
  signature.

## Does the plan hold true to the dissertation aim?

Yes as an attempted test, but not as a successful recovery.

It holds true because it:

1. trains the RNN on the actual human task contrast rather than a loose
   two-slot proxy;
2. uses a within-network 2-back versus 0-back dissociation;
3. compares candidates with generic Gaussian disruption at matched small
   clean-task cost;
4. uses independently trained checkpoints as the inferential units;
5. separates calibration, held-out validity, and confirmatory task banks;
6. treats N-back as only the load component of the broader signature; and
7. prevents biological claims about psilocybin from being inferred from an
   abstract RNN operator.

The present run cannot answer the comparative mechanism question because the
common-cost support was inadequate. Preserving that NA is more faithful to the
aim than weakening the gate after seeing the result.

## Recommended next experiment

Keep this run frozen as the first registered attempt. Do not extend P5's grid
inside it.

For a separate second-generation experiment:

1. Run a calibration-feasibility study on new development-only 0-back banks,
   with no 2-back outcomes, to estimate the intersection of reachable additive
   costs across all ten checkpoints for P5 and the retained candidate
   profiles.
2. Decide prospectively whether to widen P5's strength grid, lower the common
   target, or drop candidate profiles whose registered directions cannot
   reach a shared small cost.
3. Freeze new task and stochastic seed banks, rerun the baseline-only precision
   calculation for the chosen target, and create a new preregistration rather
   than amending this result.
4. Repeat the exact-neutral and all-ten held-out gates.
5. Only if C2 becomes valid should it be integrated with Family A C1 settling
   and C3 distractor selectivity for a complete signature test.

This sequencing protects the dissertation from a post-hoc “rescue” while
turning the present failure into a concrete design result: equal-cost
mechanism comparison requires demonstrating common cost support before the
confirmatory banks are opened.

## Authoritative artifacts

- Preregistration:
  `docs/preregistration/nback_additive_perturbation_preregistration.md`
- Run state:
  `outputs/nback_additive_perturbation/run_state.json`
- Final JSON:
  `outputs/nback_additive_perturbation/metrics/nback_c2_summary.json`
- Profile table:
  `outputs/nback_additive_perturbation/metrics/nback_c2_profile.csv`
- Checkpoint table:
  `outputs/nback_additive_perturbation/metrics/nback_c2_checkpoint.csv`
