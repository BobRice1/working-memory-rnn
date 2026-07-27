# N-back competence-screened final-pool pre-registration

## Registration status

Frozen on 2026-07-27 after replacement-final seed `20260911` failed one
untouched competence criterion and before training any seed in this pool.
No perturbation outcome has been run or inspected.

The preceding baseline-only result is recorded in
`docs/preregistration/nback_budget_final_family_failure_audit.md`.

## Rationale for the design change

Requiring ten of ten prespecified initializations to pass treats ordinary
between-initialization training variability as a failure of the task or
architecture. It is also stricter than the closest exact published precedent:
Wan et al. (2022) trained 12 independently initialized N-back RNNs, excluded
two that did not reach their performance criterion, and analyzed the ten
competent networks (https://doi.org/10.1371/journal.pcbi.1009062).

This registration therefore changes only checkpoint-family assembly. Model
competence is screened using baseline task performance before any
perturbation is applied. The task, architecture, curriculum, competence
thresholds, and perturbation-blind selection principle remain unchanged.

## Frozen candidate pool and order

Attempt these 15 independent training seeds in numerical order:

```text
20260912, 20260913, 20260914, 20260915, 20260916,
20260917, 20260918, 20260919, 20260920, 20260921,
20260922, 20260923, 20260924, 20260925, 20260926
```

Seeds `20260912-20` were named in the preceding family but were never trained
because its stop-on-first-failure rule stopped at seed `20260911`. They
therefore remain untouched. Seeds `20260921-26` are new.

Frozen curriculum-validation offset:

```text
1100000
```

Frozen untouched competence-bank offset:

```text
1200000
```

Each competence bank contains 1,024 sequences per condition.

## Frozen task, model, and training procedure

Retain the exact budget-rescue configuration:

- shared six-identity 0-back/2-back task with 20 items per sequence;
- identical timing and scoring windows across rules;
- fixed-target 0-back and lag-two-match 2-back decisions;
- 64-unit continuous-time tanh RNN;
- Stage 1 0-back acquisition;
- Stage 2 shuffled 1:3 allocation of 0-back and 2-back homogeneous batches;
- 2-back class weights `[1.0, 2.0]`;
- Stage 2 maximum 20,000 updates;
- validation every 200 updates;
- checkpoint at the first two-consecutive-pass joint validation gate;
- no recurrent training noise, with all other frozen optimizer settings
  unchanged.

No seed may receive a larger budget, altered learning rate, additional
training, or checkpoint selection based on its competence-bank result.

## Unchanged competence gates

A candidate checkpoint is retained only if its single untouched evaluation
passes every gate:

| Condition | Metric | Required |
| --- | --- | ---: |
| 0-back | item accuracy | at least 0.95 |
| 0-back | Barrett discriminability, HR - FAR | at least 0.90 |
| 2-back | item accuracy | at least 0.95 |
| 2-back | Barrett discriminability, HR - FAR | at least 0.90 |
| 2-back | one-back-lure accuracy | at least 0.90 |

All required metrics must also be finite and have the registered raw counts.
A checkpoint that fails training or any untouched gate is recorded as a
failed candidate and is never perturbed.

## Selection and stopping algorithm

1. Attempt candidate seeds once, in the frozen numerical order.
2. Run the registered curriculum and, if training passes, the single
   untouched competence evaluation.
3. Retain a checkpoint if and only if all unchanged gates pass.
4. Continue after failed candidates, preserving their results.
5. Stop successfully as soon as ten competent checkpoints have been retained.
6. Stop unsuccessfully if fewer than ten can still be obtained from the
   unattempted pool. With a 15-seed pool, the sixth scientific failure makes
   success impossible.

An infrastructure interruption that produces no scientific result may be
restarted with the same seed and frozen configuration; the interruption and
restart must be logged. A completed training or competence result may not be
rerun.

The final analysis family is the first ten competent checkpoints in frozen
seed order. Pool pass rate, all attempted seeds, and every failure reason will
be reported.

## Outcome firewall

Until ten checkpoints have been selected:

- do not calibrate perturbation strengths;
- do not run any perturbation;
- do not inspect candidate or generic-noise signature outcomes;
- do not change competence gates in response to pool results.

Checkpoint selection uses baseline competence only. This prevents selection
for the desired psilocybin-related load interaction.

## Decision after the pool

If ten checkpoints are retained, freeze their identities in a baseline audit
and proceed to the pre-registered baseline-only calibration-precision phase.
If the pool cannot yield ten, stop and report baseline robustness failure;
the current N-back branch remains ineligible for mechanistic comparison.

## Claim boundary

A successful screened family establishes ten independently initialized,
competent computational baselines for the 0-back/2-back comparison. Screening
does not demonstrate a psilocybin signature, validate a biological mechanism,
or make the RNN pharmacologically equivalent to humans receiving
psilocybin.
