# Family B Phase 4 competence audit

Date: 2026-07-26  
Pre-registration commit: `5f888be`  
Pre-outcome implementation commit: `5fa38fc`  
Device: NVIDIA GeForce RTX 3060 Laptop GPU (CUDA)

## Scope and stopping rule

This is a baseline-only competence audit. No perturbation calibration, strength
grid, profile score, assignment sensitivity, hybrid test, or perturbation
outcome was generated or inspected.

The frozen specification requires every Family B seed to pass:

- pooled `load1_clean` mean angular error below 10 degrees;
- first- and second-position `load2_clean` error below 45 degrees;
- first- and second-position error below 45 degrees in both distractor
  conditions;
- fixation accuracy at least 0.94 in every condition.

It also states that a baseline unable to perform a tested condition must not be
used for perturbation inference.

## Baseline-only attempts

All attempts used the frozen 64-unit tanh CTRNN, five-step random distractor,
8-step serial cues, 2-step gap, trained delay choices 10/20/40/80, homogeneous
batches, and seed `20260713`. Thresholds were never weakened.

| Attempt | Schedule | Distractor prevalence | Final-stage learning rate | Held-out result |
|---|---|---:|---:|---|
| A1 | 6,000 steps, full mixture from initialization | 0.50 | 0.001 | Fixation 0.977; every memory cell near chance (86.44-90.36 degrees) |
| A2 | 12,000 steps, full mixture from initialization | 0.50 | 0.001 | Fixation 0.977; every memory cell near chance (87.43-93.21 degrees) |
| A3 | 2,000 load1 clean; 4,000 clean load1/load2; 6,000 full mixture | 0.50 | 0.001 | Fixation 0.966; every memory cell near chance (87.44-93.27 degrees) |
| A4 | Same curriculum, deterministic 13/7 clean/distractor counts at each load | 0.35 | 0.001 | Fixation 0.977; every memory cell near chance (87.96-92.90 degrees) |
| A5 | Same 0.35 curriculum, tenfold lower learning rate in final stage | 0.35 | 0.0001 | Fixation 0.966; partial learning, but seven of eight competence checks failed |

A1 implements the frozen primary schedule. A2 implements “raise steps.” A3-A5
exercise the frozen “reduce mixture difficulty before weakening criteria”
contingency. A5 was fixed before it ran after A4 showed that distractor
prevalence alone did not prevent collapse.

## Final A5 held-out gate values

Evaluation used 20 fresh homogeneous batches per condition at delay 20.

| Condition | Position | Mean angular error (degrees) | Fixation accuracy | Gate |
|---|---|---:|---:|---|
| load1_clean | pooled | 39.65 | 0.966 | fail (<10) |
| load1_clean | first | 43.41 | 0.966 | descriptive |
| load1_clean | second | 35.89 | 0.966 | descriptive |
| load1_distractor | first | 68.13 | 0.966 | fail (<45) |
| load1_distractor | second | 66.42 | 0.966 | fail (<45) |
| load2_clean | first | 90.21 | 0.966 | fail (<45) |
| load2_clean | second | 39.24 | 0.966 | pass (<45) |
| load2_distractor | first | 89.20 | 0.966 | fail (<45) |
| load2_distractor | second | 68.61 | 0.966 | fail (<45) |

The fixation gate passed. Only the second-position `load2_clean` memory gate
passed.

## Component diagnostics

The component tasks were tested only to locate the baseline-training failure:

| Diagnostic | Steps | Held-out result |
|---|---:|---:|
| load1 clean only | 2,000 | 9.61 degrees; fixation 0.966 |
| load2 clean only | 3,000 | first 54.26 degrees; second 21.70 degrees; fixation 0.970 |
| load1 distractor only | 2,000 | 25.69 degrees; fixation 0.967 |

These results rule out a universal target/mask failure and show that clean
one-item maintenance and distractor resistance are individually learnable. The
joint schedules instead converge toward the low-amplitude, near-chance
population-output solution; the lower final-stage learning rate slows but does
not prevent loss of clean and first-position retrieval.

## Phase decision

Phase 4 is **failed / upstream-blocking**. The required ten accepted Family B
checkpoints do not exist. Therefore:

- Phase 5 calibration is not run;
- Phase 6 smoke/full perturbation grids are not run;
- Phase 7 confirmatory and descriptive signature scoring is not run;
- Phase 8 assignment sensitivity is not triggered;
- Phase 9 hybrid testing is not triggered;
- Phase 10 perturbation report is replaced by this competence audit;
- Phase 11 records the stop decision and exact gate values.

Running Family A alone would not rescue the frozen confirmatory question:
Family A lacks the trained load and distractor outcome space, and its
five-checkpoint table is explicitly diagnostic/descriptive.

## Interpretation and next design decision

This audit does not test a psilocybin-like perturbation mechanism. It shows that
the current 64-unit RNN and training objective cannot jointly instantiate the
measurement apparatus needed to test the selective human signatures. Generic
impairment, selective impairment, and excess-over-noise contrasts are therefore
not identifiable in the planned Family B model.

A follow-up should be separately pre-registered. Plausible choices include an
explicit distractor/context channel, a larger or structured recurrent state, a
training loss that avoids the sparse-population zero-output attractor, or a
different retro-cue architecture. Acceptance gates should remain unchanged so
the redesign improves the model rather than redefining competence.
