# Family B stimulus-role rescue audit

Date recorded: 2026-07-26  
Frozen R2 pre-registration commit: `cb8cf9a`  
Frozen R2 implementation commit: `cd031aa`  
Perturbation outcomes inspected: **none**

## Decision

R2 **failed its development gate**. Development seed `20260804` passed the
fixation guard but failed all seven memory checks. Under the frozen stop rule,
seeds `20260805-06` were not trained and the untouched final family was not
started.

## Held-out competence result

Evaluation used 20 fresh homogeneous batches per condition at delay 20.

| Condition | Position | Mean angular error (degrees) | Fixation accuracy |
|---|---:|---:|---:|
| load1 clean | pooled | 42.464 | 0.969 |
| load1 distractor | first | 46.575 | 0.967 |
| load1 distractor | second | 45.008 | 0.967 |
| load2 clean | first | 62.793 | 0.968 |
| load2 clean | second | 65.813 | 0.968 |
| load2 distractor | first | 66.224 | 0.967 |
| load2 distractor | second | 67.641 | 0.967 |

The probability-output resultant lengths were `0.42-0.48`. The model therefore
produced structured distributions, but the decoded locations were not
competent.

## Acceptance checks

```text
load1_clean_under_10                  FAIL
load2_clean_first_under_45            FAIL
load2_clean_second_under_45           FAIL
load1_distractor_first_under_45       FAIL
load1_distractor_second_under_45      FAIL
load2_distractor_first_under_45       FAIL
load2_distractor_second_under_45      FAIL
all_condition_fixation_at_least_0_94  PASS
```

## Interpretation

Adding an explicit memoranda-versus-distractor role label did not rescue the
64-unit network. This rejects stimulus-role ambiguity as a sufficient
explanation for the Family B competence failure under the frozen objective and
curriculum. It does not test or reject any psilocybin-related perturbation
mechanism.

The next permitted step is a separately pre-registered capacity rescue. No
perturbation outcome may be inspected unless a baseline family passes every
competence gate.

## Recorded outputs

```text
outputs/multicondition_working_memory_distribution_role/
outputs/multicondition_working_memory_distribution_role/seed_sweep/seed_20260804/
```
