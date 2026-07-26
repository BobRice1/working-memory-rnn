# Family B 128-unit capacity rescue audit

Date recorded: 2026-07-26  
Frozen R3 pre-registration commit: `3c082ac`  
Frozen R3 implementation commit: `fb85c49`  
Perturbation outcomes inspected: **none**

## Decision

R3 **failed its development gate**. Development seed `20260807` passed the
fixation guard but failed six of seven memory checks. Under the frozen stop
rule, seeds `20260808-09` were not trained and the untouched final family was
not started.

The planned baseline rescue ladder is now exhausted.

## Held-out competence result

Evaluation used 20 fresh homogeneous batches per condition at delay 20.

| Condition | Position | Mean angular error (degrees) | Fixation accuracy |
|---|---:|---:|---:|
| load1 clean | pooled | 46.556 | 0.966 |
| load1 distractor | first | 46.212 | 0.965 |
| load1 distractor | second | 44.894 | 0.965 |
| load2 clean | first | 73.115 | 0.965 |
| load2 clean | second | 46.833 | 0.965 |
| load2 distractor | first | 75.030 | 0.965 |
| load2 distractor | second | 48.988 | 0.965 |

Probability-output resultant lengths were `0.39-0.53`. The larger network
therefore also produced structured but behaviourally incorrect response
distributions.

## Acceptance checks

```text
load1_clean_under_10                  FAIL
load2_clean_first_under_45            FAIL
load2_clean_second_under_45           FAIL
load1_distractor_first_under_45       FAIL
load1_distractor_second_under_45      PASS
load2_distractor_first_under_45       FAIL
load2_distractor_second_under_45      FAIL
all_condition_fixation_at_least_0_94  PASS
```

## Interpretation

Doubling recurrent capacity from 64 to 128 units did not rescue Family B under
the frozen role-labelled task, normalized circular-distribution objective, and
A5 curriculum. Insufficient hidden size is therefore not a sufficient
explanation for the competence failure.

Across the rescue ladder:

1. normalized circular-distribution loss removed the raw zero-output loophole
   but did not restore retrieval;
2. explicit target-versus-distractor role labelling did not restore retrieval;
3. doubled recurrent capacity did not restore retrieval.

The remaining problem is more likely the training formulation or task
decomposition than output normalization, stimulus-role ambiguity, or 64-unit
capacity alone. This conclusion concerns baseline learnability, not a
psilocybin-related perturbation mechanism.

No perturbation outcome may be inspected from this failed Family B family. Any
further attempt requires a new scientific design decision and pre-registration.

## Recorded outputs

```text
outputs/multicondition_working_memory_distribution_role_h128/
outputs/multicondition_working_memory_distribution_role_h128/seed_sweep/seed_20260807/
```
