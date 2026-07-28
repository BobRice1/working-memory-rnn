# Variable-timing circular training protocol

Status: frozen before the full candidate run on 2026-07-28.

## Aim

Train a new family of 10 independently initialized circular working-memory
RNNs that cannot rely on one fixed delay location for distractor filtering.
This family is separate from the five checkpoints trained with a midpoint
distractor.

## Training distribution

- Clean and distractor batches occur in a shuffled 1:1 block.
- Distractor batches use delay-relative onset fractions `0.00`, `0.25`,
  `0.50`, `0.75`, and `1.00`.
- Each onset fraction occurs once per shuffled block of five distractor
  batches.
- The onset fraction is applied to each sampled delay length, preserving the
  existing delay choices of 10, 20, 40, and 80 time steps.
- Pre-cue and cue durations retain their existing variable choices.
- All other architecture, noise, loss, and optimizer settings match the
  midpoint-trained circular family.

## Candidate and retention rule

- Ordered candidate seeds: `20260801` through `20260815`.
- Target: 10 competent checkpoints.
- Retain the first 10 candidates that pass every frozen competence gate.
- A later candidate replaces an earlier candidate only when the earlier
  candidate fails a gate.
- Do not select on distractor timing range or on downstream perturbation
  results.

## Held-out evaluation

Each candidate is evaluated on one clean condition and all five distractor
onset fractions. Every condition uses 1,024 trials in batches of 128. Matching
batch seeds pair target-angle banks across all conditions and pair
distractor-angle banks across timing conditions.

A candidate passes only when:

- clean mean angular error is at most 10 degrees;
- mean angular error at every individual distractor timing is at most 15
  degrees; and
- fixation accuracy is at least 0.90 in the clean condition and at every
  distractor timing.

Timing range is an outcome to report, not a retention criterion. These are
preregistered model-validity and competence thresholds informed by task
structure, not empirical psilocybin effect sizes.

## Planned outputs

- configuration:
  `configs/fixation_circular_variable_distractor_working_memory.yaml`
- checkpoints and histories:
  `outputs/fixation_circular_variable_distractor_working_memory/seed_sweep/`
- pool table and summary:
  `outputs/fixation_circular_variable_distractor_working_memory/metrics/`
