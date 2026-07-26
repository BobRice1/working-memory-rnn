# Family B stimulus-role rescue pre-registration

Date frozen: 2026-07-26  
Parent R1 audit commit: `de45f30`  
Perturbation outcomes inspected: **none**

## Purpose

The circular-distribution objective removed the raw zero-output loophole but
R1 seed `20260801` still failed joint retrieval. R2 tests whether the remaining
bottleneck is ambiguity about stimulus role: target items and delay distractors
currently enter through identical tuned sensory channels.

This is a baseline-only task-interface rescue. It does not test a
psilocybin-related mechanism.

## Frozen R2 change

R2 retains the complete R1 architecture, circular-distribution loss,
probability decoding, A5 curriculum, deterministic 0.35 distractor prevalence,
and all competence gates.

It adds exactly one scalar stimulus-role input channel:

- `+1` while a present item is being cued for encoding;
- `-1` during the five-step distractor pulse;
- `0` during pre-cue fixation, item gaps, ordinary delay, and response.

For load1 trials, the role channel is `+1` only in the occupied slot. For load2
trials it is `+1` in both item slots. The probe remains a separate channel and
continues to indicate which serial position to report during response.

The role signal labels task relevance; it does not provide the item angle,
probe answer, delay length, or future condition. Humans likewise receive task
instructions that distinguish memoranda from irrelevant distractors.

No hidden-size increase, modularity, auxiliary decoder, task-timing change,
loss-coefficient change, or perturbation-specific modification is permitted.

## Required implementation tests

Before development training:

1. legacy task batches remain byte-identical when the role channel is disabled;
2. input size increases by exactly one when enabled;
3. the role channel is `+1` only for present item slots;
4. it is `-1` only during distractor presentation;
5. it is zero during all other phases;
6. the probe channel remains distinct and keeps its balanced `-1/+1` values;
7. model input dimensions follow the role-enabled task configuration;
8. a short CPU training smoke test completes with the R1 loss;
9. the full repository test suite passes.

## Development and final seeds

R2 development seeds:

```text
20260804, 20260805, 20260806
```

If and only if all three pass every gate, train the untouched final family:

```text
20260811, 20260812, 20260813, 20260814, 20260815,
20260816, 20260817, 20260818, 20260819, 20260820
```

R1 development seed `20260801` never enters R2 inference. R1 seeds
`20260802-03` remain incomplete.

## Competence gates

Every evaluated seed must satisfy the unchanged gates:

- pooled `load1_clean` mean angular error below 10 degrees;
- both `load2_clean` positions below 45 degrees;
- both positions in both distractor conditions below 45 degrees;
- fixation accuracy at least 0.94 in every condition.

Evaluation uses 20 fresh homogeneous batches per condition at delay 20.

## Decision rule

- **R2 development pass:** all three development seeds pass every gate. Freeze
  the implementation commit, then train the ten untouched final seeds.
- **R2 development fail:** any seed fails any gate. Stop R2 immediately. Do
  not train final seeds or inspect perturbation outcomes. A hidden-size change
  requires a separate pre-registration.
- **Final-family pass:** all ten final seeds pass every gate. Only then may the
  perturbation plan be revised for role-channel handling and resumed.
- **Final-family fail:** stop and report non-transport across seeds.

## Output locations

```text
configs/multicondition_working_memory_distribution_role.yaml
outputs/multicondition_working_memory_distribution_role/
```

The role channel is exempt from later sensory-gain perturbations unless a new
pre-registration explicitly places it in scope.
