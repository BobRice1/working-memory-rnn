# Family B circular-distribution-loss rescue audit

Date: 2026-07-26  
Pre-registration commit: `5cc18af`  
Implementation commit: `23cb77b`  
Device: NVIDIA GeForce RTX 3060 Laptop GPU (CUDA)  
Perturbation outcomes inspected: **none**

## Decision

R1 **failed its development gate**. Seed `20260801` failed seven memory
competence checks while passing fixation. Under the frozen rule, training was
stopped before completing seeds `20260802` and `20260803`. No final-family seed
was trained.

## Held-out development result

Evaluation used 20 fresh homogeneous batches per condition at delay 20 and
softmax-normalized population decoding.

| Condition | Position | Mean error (degrees) | Cross-entropy | Resultant length | Fixation accuracy | Gate |
|---|---|---:|---:|---:|---:|---|
| load1_clean | pooled | 42.14 | 2.678 | 0.593 | 0.968 | fail (<10) |
| load1_clean | first | 42.91 | 2.712 | 0.577 | 0.968 | descriptive |
| load1_clean | second | 41.37 | 2.644 | 0.608 | 0.968 | descriptive |
| load1_distractor | first | 46.49 | 2.812 | 0.510 | 0.967 | fail (<45) |
| load1_distractor | second | 47.18 | 2.819 | 0.567 | 0.967 | fail (<45) |
| load2_clean | first | 54.29 | 3.141 | 0.412 | 0.967 | fail (<45) |
| load2_clean | second | 49.14 | 2.952 | 0.507 | 0.967 | fail (<45) |
| load2_distractor | first | 70.28 | 3.365 | 0.371 | 0.967 | fail (<45) |
| load2_distractor | second | 65.36 | 3.312 | 0.486 | 0.967 | fail (<45) |

The probabilities were not uniform: resultant lengths were 0.37-0.61. The
loss redesign therefore removed the original near-zero/raw-output loophole but
did not produce correct joint retrieval.

## Interpretation

R1 rules out the sparse raw-population MSE objective as the sole cause. The
remaining failure is consistent with task-role and selection interference:
target cues and distractors share the same tuned sensory channels, and the RNN
must infer whether to encode or ignore a pulse from timing and internal state.
The first-position load cost also remains substantial.

This is a baseline-training result, not evidence about a psilocybin-related
perturbation.

## Frozen consequence

- Development seeds `20260802` and `20260803` are not completed.
- Final seeds `20260811`-`20260820` remain untouched.
- Perturbation calibration and outcome phases remain sealed.
- A stimulus-role channel or capacity change requires a new pre-registration.
