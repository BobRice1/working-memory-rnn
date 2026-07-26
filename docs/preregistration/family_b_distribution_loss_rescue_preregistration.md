# Family B circular-distribution-loss rescue pre-registration

Date frozen: 2026-07-26  
Parent competence-audit commit: `fa00964`  
Perturbation outcomes inspected: **none**

## Purpose

The original Family B training objective converged toward a low-amplitude,
near-chance circular output while retaining fixation. Component diagnostics
showed that one-item clean and one-item distractor tasks were individually
learnable. This baseline-only rescue tests whether replacing sparse raw
population MSE with a normalized circular-distribution objective is sufficient
to train the existing joint task.

This is not a perturbation experiment and cannot provide evidence for or
against a psilocybin-related mechanism.

## Frozen R1 design

R1 changes only the tuned response objective and the corresponding population
normalization used for behavioural decoding.

Unchanged:

- 64-unit tanh CTRNN;
- fixation and probe channels;
- two sequential 8-step item slots separated by 2 steps;
- five-step random-angle distractor;
- trained delays 10/20/40/80;
- balanced probe positions;
- homogeneous trial-type batches;
- the A5 curriculum and deterministic 0.35 distractor prevalence;
- the A5 final-stage learning rate of `0.0001`;
- all Family B acceptance thresholds.

Changed:

- response population activity is interpreted as logits;
- the von Mises target is normalized to a probability distribution;
- response loss is cross-entropy against that continuous target distribution;
- fixation MSE is normalized and combined as a separate loss component;
- behavioural angle is decoded from softmax-normalized population activity.

No distractor-role channel, hidden-size increase, modularity, auxiliary decoder,
or perturbation-specific change is permitted in R1.

## Exact loss

For tuned response channels \(k=1,\ldots,32\), target activity \(t_k\), and
model logits \(y_k\):

\[
q_k = \frac{t_k}{\sum_j t_j},
\qquad
L_{\mathrm{circ}} =
-\frac{\sum_{tb}m^{\mathrm{resp}}_{tb}
\sum_kq_{tbk}\log\operatorname{softmax}(y_{tb})_k}
{\sum_{tb}m^{\mathrm{resp}}_{tb}}.
\]

Fixation is trained separately:

\[
L_{\mathrm{fix}} =
\frac{\sum_{tb}m^{\mathrm{fix}}_{tb}
(\hat f_{tb}-f_{tb})^2}
{\sum_{tb}m^{\mathrm{fix}}_{tb}}.
\]

The total objective is:

\[
L = L_{\mathrm{circ}} + 2L_{\mathrm{fix}}.
\]

`m_resp` includes response steps after the frozen five-step response transition.
`m_fix` includes pre-response steps after the frozen five-step initial
exclusion and response steps after the transition. Circular output remains
silent and unscored before response.

The circular target must have positive mass at every scored response sample.
Non-finite logits, targets, or loss values are errors.

## Decoding and baseline metrics

For R1 checkpoints, output probabilities are:

\[
p_k = \operatorname{softmax}(y)_k.
\]

Decoded angle uses the existing circular vector average over \(p_k\). Report:

- mean and median angular error;
- response cross-entropy;
- mean circular resultant length
  \(R=\left|\sum_kp_ke^{i\theta_k}\right|\);
- fixation MSE and accuracy;
- the existing per-condition and per-position acceptance values.

Legacy checkpoints retain raw-activity decoding. The normalization mode must be
read from the run configuration and must not silently change historical
analyses.

## Required implementation tests

Before model training:

1. zero/uniform logits have a finite, non-zero response gradient;
2. aligned logits have lower loss than uniform logits;
3. rotating logits and targets together leaves loss unchanged;
4. changing fixation predictions does not change circular loss;
5. changing circular logits does not change fixation loss;
6. softmax decoding recovers known preferred and between-unit angles;
7. legacy raw-population decoding is unchanged;
8. a short Family B CPU training smoke test records total, circular, and
   fixation losses plus the configured decoding mode;
9. the full repository test suite passes.

## Development and final seed separation

Development seeds are fixed as:

```text
20260801, 20260802, 20260803
```

They may be used only to decide whether R1 is viable.

If and only if all three development seeds pass every competence gate, train
the final untouched seed family:

```text
20260811, 20260812, 20260813, 20260814, 20260815,
20260816, 20260817, 20260818, 20260819, 20260820
```

The final seeds are the checkpoint-level inferential units for any later
perturbation experiment. Development seeds never enter perturbation inference.

## Unchanged competence gates

Every evaluated seed must satisfy:

- pooled `load1_clean` mean angular error below 10 degrees;
- first- and second-position `load2_clean` error below 45 degrees;
- first- and second-position `load1_distractor` error below 45 degrees;
- first- and second-position `load2_distractor` error below 45 degrees;
- fixation accuracy at least 0.94 in every condition.

Evaluation uses 20 fresh homogeneous batches per condition at delay 20.

## Decision rule

- **R1 development pass:** all three development seeds pass every gate. Freeze
  the implementation commit, then train the ten untouched final seeds.
- **R1 development fail:** any development seed fails any gate. Stop. Do not
  train final seeds and do not inspect perturbation outcomes. A role-channel or
  capacity change requires a separate pre-registration.
- **Final-family pass:** all ten final seeds pass every gate. Only then may the
  perturbation plan be updated for probability decoding and resumed at
  calibration.
- **Final-family fail:** stop and report baseline non-transport across seeds.

Thresholds, seed membership, condition definitions, and the loss coefficient
may not be changed after development training begins.

## Output locations

Implementation config:

```text
configs/multicondition_working_memory_distribution_loss.yaml
```

Development and final artifacts:

```text
outputs/multicondition_working_memory_distribution_loss/
```

The development acceptance table, implementation commit, config hash, CUDA
device, and stop/proceed decision must be recorded in the changelog and a
baseline-rescue audit.
