# Working Memory RNN

This repository contains the modelling code for the dissertation's baseline
working-memory RNN. The project asks whether a simple recurrent neural network
can maintain a cue across a delay, how that memory is represented in hidden
state dynamics, and how stable those dynamics are before any
psilocybin-informed perturbations are introduced.

## Project Aim

The immediate modelling aim is to build a defensible baseline working-memory
model before testing any altered-dynamics hypotheses. The baseline task is a
categorical delayed-response task:

```text
brief cue -> blank delay -> report remembered cue
```

The cue is currently an abstract class identity, not an image or visual object.
During the delay, the cue channel is removed. Successful performance therefore
depends on recurrent hidden activity carrying information forward until the
response period.

## Current Baseline

The current canonical baseline is a continuous-time RNN with a bounded `tanh`
hidden-state nonlinearity. It is trained with a fixed `20`-step delay and a
response-period-only classification loss.

The bounded `tanh` model produces settled delay-period dynamics and strong
long-delay generalization. The categorical delay baseline should be treated as
`tanh` only.

The next model iteration is a continuous population-code task. Instead of
remembering one of four abstract classes, the model remembers a sampled
circular location. The cue is encoded as a bump of activity across tuned units,
each with a preferred angle, and the readout is trained to reproduce that bump
after the delay. This makes memory precision, angular error, drift, and
population-code stability available as analysis measures.

## Model Progression

- `configs/baseline_delay.yaml`: current `tanh` baseline.
- `configs/tuned_delay.yaml`: continuous circular-location delayed-response
  model using Gaussian/von-Mises population tuning across input and output
  units. This is the next baseline iteration after the categorical task.
- `configs/tuned_delay_fixation_gate_stable.yaml`: stronger Yang-style baseline
  trained with randomized cue timing and `10`, `20`, `40`, or `80` step delays,
  a 25-step response period, transition-aware weighted loss, input/recurrent
  training noise, and gradient clipping.

These three models are the active dissertation progression: categorical
baseline, circular tuned task, and stabilized Yang-style fixation-gated task.
Superseded runs are preserved under `outputs/archive/`, with retained YAMLs
under `configs/archive/`. See `docs/model-run-archive.md`.

For the continuous tuned task, the fixed-delay `configs/tuned_delay.yaml` model
learns the trained `20`-step memory but drifts strongly when pushed to longer
or autonomous delays. The `configs/tuned_delay_stable.yaml` variant is much more
stable: mean angular error stays below `2` degrees through a `120`-step delay,
and the autonomous hidden-state probe shows only about `1.5` degrees mean drift
over `100` probe steps. This is strong evidence for a stable attractor-like
continuous memory state.

Follow-up fixed-point/Jacobian analysis on `tuned_delay_stable` found sampled
blank-delay fixed points that preserve the remembered angle closely
(`1.36` degrees mean fixed-point decoding error). The local Jacobian has a
near-neutral leading direction (`1.000` mean spectral radius) and contracting
secondary directions (`0.984` mean second-largest absolute eigenvalue), with the
leading eigenvector strongly aligned to the circular memory manifold
(`0.945` mean tangent alignment). This is now direct evidence for a sampled
ring-attractor-like memory structure, although it is still sampled rather than a
global proof of every possible fixed point.

A broader fixed-point landscape analysis now mirrors the reference RNN notebook
workflow by plotting fixed-point search endpoints in the same PCA space as
task-evoked trajectories. Starting from actual late-delay states, locally
perturbed late-delay states, and random bounded hidden states, most searches
land on the same ring-shaped fixed-point set. Across `320` starts, `85.6%`
fell below a `0.001` residual threshold; random starts converged most reliably
(`99.2%`), and known-angle starts preserved remembered angle with about
`3.48` degrees mean error.

## Analyses

The project currently includes four analysis layers:

- Task performance at the trained delay.
- Delay-length sweeps with frozen weights.
- PCA visualization of hidden-state trajectories.
- Hidden-state stability analysis using state norm, step-to-step speed, and a
  delay settling ratio.
- Autonomous hidden-state probing for tuned models, measuring drift when the
  recurrent state is run forward without the original cue.
- Fixed-point and Jacobian analysis for tuned models, measuring approximate
  blank-delay fixed points and local recurrent stability around them.
- Fixed-point landscape visualization for tuned models, projecting task
  trajectories and fixed-point search endpoints into the same PCA state space.
- Additional tuned dynamics figures for dissertation inspection: ring manifold,
  decoded angle over time, Jacobian spectrum, and deterministic perturbation
  recovery. Their rationale and literature anchors are recorded in
  `docs/analysis-figure-rationale.md`.

The delay sweep asks how far the trained memory can be pushed without
retraining. The stability analysis asks whether the hidden state settles during
the delay or continues moving in a ramping/phasic pattern. The trajectory-based
analyses are descriptive; the tuned fixed-point/Jacobian analysis is more direct
but still sampled rather than a global proof of the entire attractor landscape.

## Current Interpretation

The baseline model has learned to remember a briefly presented cue and report it
later. The current `tanh` version is the categorical baseline because its
bounded hidden activity supports settled delay-period dynamics and strong
delay-length generalization.

This is still a baseline working-memory model, not a psilocybin model. Any
future psilocybin-informed condition should be introduced as a separate
modelling hypothesis, with the literature-to-parameter mapping documented
before implementation.

## Repository Structure

- `src/wm_rnn/`: model, task generation, training, evaluation, and analysis code.
- `configs/`: active progression configurations; superseded configurations are
  retained under `configs/archive/`.
- `docs/model-architecture.md`: architecture and plain-English task explanation.
- `docs/changelog.md`: commit history, experiment run log, and interpretation history.
- `outputs/`: active generated artifacts; superseded runs are under
  `outputs/archive/`. This folder is not tracked by Git.
