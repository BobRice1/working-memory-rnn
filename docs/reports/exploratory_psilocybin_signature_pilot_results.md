# Exploratory psilocybin-signature pilot: first-pass results

Date: 27 July 2026

## Headline

The small two-task pilot produced one formal profile worth taking forward:
**carried-state persistence scaled to 0.95**. It was the only
operator-strength setting that met all three primary behavioural-pattern rules
in the same direction across both tasks.

This is preliminary evidence that a small reduction in effective state
persistence is a computationally sufficient candidate for reproducing this
selected set of behavioural contrasts. The distractor component is provisional
because the unperturbed distractor condition was already weak. This is not
evidence that psilocybin
literally changes this RNN parameter, and it does not uniquely identify a
biological mechanism.

## What was run

- Three independently trained original circular-task checkpoints:
  `20260714–20260716`.
- Three independently trained N-back checkpoints: `20260912–20260914`.
- All seven single perturbation families, plus the circular-task
  distractor-window gain, on the fixed grids committed before outcome
  evaluation.
- 256 circular trials per cell at clean delays 10, 20, 40, and 80, plus a
  delay-20 distractor condition.
- 256 twenty-item sequences per 0-back and 2-back cell.
- CUDA execution on an NVIDIA GeForce RTX 3060 Laptop GPU.

The circular sweep took 135.6 seconds and the N-back sweep 75.6 seconds.
No models were retrained.

## Leading result

For state persistence `0.95`, all three circular checkpoints showed:

- increased post-cue settling time with final error preserved under the
  pre-specified 20% ceiling;
- greater impairment at delay 80 than delay 10;
- greater impairment on distractor than matched clean trials.

All three N-back checkpoints also showed greater normalized discriminability
impairment in 2-back than 0-back.

| Pilot outcome | Mean contrast | Seeds in predicted direction |
|---|---:|---:|
| Clean delay-20 angular-error impairment | +1.35% | preservation passed 3/3 |
| Post-cue restricted settling change | +0.19 steps | 3/3 |
| Long-minus-short delay selectivity | +0.290 | 3/3 |
| Distractor-minus-clean selectivity | +0.148 | 3/3 |
| 2-back-minus-0-back selectivity | +0.296 | 3/3 |

The small mean settling change needs emphasis. It was positive in every seed,
but two checkpoint effects were close to zero; this is a directional pilot
signal, not a stable reaction-time effect estimate.

## Interpretation against the human findings

- **Barrett-style load effect:** recovered descriptively. The perturbation
  impaired 2-back discriminability more than 0-back in all three N-back
  models.
- **Vollenweider/Yousefi-style slowing with relative accuracy
  preservation:** directionally recovered in the circular task. Post-cue
  settling is only an RNN analogue of reaction time because response onset is
  externally imposed.
- **Wittmann-style interval dependence:** recovered descriptively. The
  perturbation cost was larger at the long than short delay in all three
  circular models.
- **Carter-style distractor vulnerability:** recovered descriptively. The
  perturbation cost was larger with an irrelevant delay-period cue in all
  three circular models. This condition is out of distribution because the
  original models were not trained with distractors. More importantly, native
  distractor mean errors were already approximately 31.5–53.7 degrees and only
  about 26% of native distractor trials settled, so floor/headroom distortion
  is plausible.

Thus, the first-pass answer is a **qualified yes**: the plan is capable of
finding an RNN perturbation that resembles several selected human psilocybin
working-memory signatures across the two task families. The present evidence
supports continued testing of reduced state persistence, while the
Carter-style component needs a better-behaved distractor baseline. It does not
yet establish human-signature recovery.

## Other useful findings

- Direct distractor-window gain behaved as expected: proportional distractor
  error increased monotonically from approximately 0 at gain 1.0 to +14.8%,
  +39.7%, and +77.0% at gains 1.1, 1.25, and 1.5, respectively, with 3/3
  checkpoints positive at every non-neutral setting. This is a manipulation
  check, not a complete cross-task account.
- Several operators reproduced isolated components but not the full pattern.
  For example, recurrent gain `0.95` showed slowing/preservation in 2/3,
  delay selectivity in 3/3, and N-back load selectivity in 3/3, but distractor
  selectivity in only 1/3.
- Gaussian noise itself produced positive N-back load selectivity, confirming
  that a difficult condition can be more vulnerable to generic disruption.
  At nearly matched 0-back cost, state persistence `0.95` had mean load
  selectivity `0.296` versus `0.052` for Gaussian noise `0.025`. The circular
  clean-cost grid did not provide an informative non-neutral Gaussian match
  within the frozen 0.05 tolerance, so cross-task superiority to noise is not
  claimed.

## Validity audit

- Every neutral operator setting reproduced its native baseline exactly.
- All 792 circular rows passed the fixation gate, and all N-back native
  checkpoints passed competence.
- The delay-20 clean cells used for the slowing result were latency-valid.
- There were 169 latency-invalid circular rows elsewhere in the grid; their
  settling values must not be interpreted as latency.
- Technical P2 and P5 replicates were averaged within checkpoint before sign
  counting and were never treated as extra trained seeds.
- The frozen pilot omitted a distractor-baseline competence gate. Consequently,
  distractor selectivity can nominate a follow-up but cannot currently carry a
  strong recovery claim.

## What can be reported tomorrow

> In an exploratory GPU pilot using three independently trained RNNs per task,
> a 5% reduction in carried-state persistence was the only tested setting to
> meet the complete pre-specified qualitative pattern: relatively
> preserved final circular-task accuracy with slower post-cue settling,
> stronger long-delay and distractor costs, and selective 2-back impairment.
> Every contrast had the predicted mean direction in all three seeds. Because
> this was a small descriptive screen and the circular distractor condition
> was out of distribution and weak even at baseline, the result nominates a
> candidate mechanism for a larger independent evaluation rather than
> demonstrating a recovered psilocybin signature or mechanism.

## Next step

Freeze state persistence around the local region `0.925–0.975`, retain the
strongest alternative profiles and Gaussian noise, and evaluate all available
independent checkpoints with larger task banks. The next run should report
effect uncertainty, explicitly test dose ordering, and improve the circular
Gaussian cost match. The distractor contrast should first be redesigned or
calibrated so the unperturbed models retain usable performance; that may
require a weaker distractor pulse, but it does not require retraining unless a
trained distractor task is preferred.

## Artifacts

- `outputs/exploratory_psilocybin_signature_pilot/circular_family_a/metrics/circular_family_a_grid.csv`
- `outputs/exploratory_psilocybin_signature_pilot/nback/pilot_cells.json`
- `outputs/exploratory_psilocybin_signature_pilot/nback/pilot_signatures.csv`
- `outputs/exploratory_psilocybin_signature_pilot/summary/cross_task_signature_summary.csv`
- `outputs/exploratory_psilocybin_signature_pilot/summary/leading_profile_seed_points.png`
