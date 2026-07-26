# Family B 128-unit capacity rescue pre-registration

Date frozen: 2026-07-26  
Parent R2 audit commit: `ac08bbe`  
Perturbation outcomes inspected: **none**

## Purpose

The normalized circular-distribution objective and explicit stimulus-role
channel were not sufficient for the 64-unit network to learn joint Family B
retrieval. R3 tests the remaining planned explanation: insufficient recurrent
capacity for the two-item, probe-gated, distractor-resistant task family.

This is a baseline-only capacity rescue. It does not test a
psilocybin-related mechanism.

## Frozen R3 change

R3 retains the complete R2 task, role-channel semantics, circular-distribution
loss, probability decoding, A5 curriculum, deterministic 0.35 distractor
prevalence, timing, optimizer settings, noise levels, and competence gates.

It changes exactly one model hyperparameter:

```yaml
model:
  hidden_size: 128
```

The R2 value is 64. No 256-unit follow-up, modular architecture, auxiliary
decoder, loss change, curriculum extension, or task simplification is permitted
under R3.

## Required implementation checks

Before development training:

1. the R3 configuration differs from R2 only in hidden size and output paths;
2. task input/output dimensions remain 35/33;
3. the recurrent hidden dimension is 128;
4. a short CPU training smoke test completes with the R2 task and loss;
5. the full repository test suite passes.

## Development and final seeds

R3 development seeds:

```text
20260807, 20260808, 20260809
```

If and only if all three pass every gate, train the untouched final family:

```text
20260811, 20260812, 20260813, 20260814, 20260815,
20260816, 20260817, 20260818, 20260819, 20260820
```

No R1 or R2 development seed enters R3 inference.

## Competence gates

Every evaluated seed must satisfy the unchanged gates:

- pooled `load1_clean` mean angular error below 10 degrees;
- both `load2_clean` positions below 45 degrees;
- both positions in both distractor conditions below 45 degrees;
- fixation accuracy at least 0.94 in every condition.

Evaluation uses 20 fresh homogeneous batches per condition at delay 20.

## Decision rule

- **R3 development pass:** all three development seeds pass every gate. Freeze
  the implementation commit, then train the ten untouched final seeds.
- **R3 development fail:** any seed fails any gate. Stop R3 immediately. Do
  not train final seeds or inspect perturbation outcomes.
- **Final-family pass:** all ten final seeds pass every gate. Only then may the
  perturbation plan resume.
- **Final-family fail:** stop and report non-transport across seeds.

If R3 development fails, the planned baseline rescue ladder is exhausted. Any
further architecture, curriculum, objective, or task change requires a new
scientific design decision rather than another automatic rescue.

## Output locations

```text
configs/multicondition_working_memory_distribution_role_h128.yaml
outputs/multicondition_working_memory_distribution_role_h128/
```

The larger hidden state does not alter later perturbation definitions. The
stimulus-role channel remains exempt from sensory-gain perturbations unless a
new pre-registration explicitly places it in scope.
