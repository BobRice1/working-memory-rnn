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

`tanh` replaced the original `relu` default after analysis showed that the
`relu` model solved the trained task but did so with hidden states that kept
growing through the delay. That ramping solution generalized poorly to longer
delays. The bounded `tanh` model produced more settled delay-period dynamics
and much stronger long-delay generalization, although the exact degree of
long-delay accuracy remains seed-dependent.

The archived original `relu` baseline remains available for comparison in
`configs/baseline_delay_relu.yaml` and `outputs/baseline_delay_relu/`.

## Model Variants

- `configs/baseline_delay.yaml`: current `tanh` baseline.
- `configs/baseline_delay_relu.yaml`: archived original `relu` baseline.
- `configs/baseline_delay_stable.yaml`: `relu` variant trained with randomized
delay lengths and loss over both delay and response periods.

The stable `relu` variant improved delay generalization but did not settle as
fully as the `tanh` baseline. This suggests that bounding the hidden-state
nonlinearity was more important than changing the training objective for this
first baseline.

## Analyses

The project currently includes four analysis layers:

- Task performance at the trained delay.
- Delay-length sweeps with frozen weights.
- PCA visualization of hidden-state trajectories.
- Hidden-state stability analysis using state norm, step-to-step speed, and a
delay settling ratio.

The delay sweep asks how far the trained memory can be pushed without
retraining. The stability analysis asks whether the hidden state settles during
the delay or continues moving in a ramping/phasic pattern. These analyses are
descriptive; they do not yet prove fixed-point or attractor structure.

## Current Interpretation

The baseline model has learned to remember a briefly presented cue and report it
later. The original `relu` version did this with a ramping hidden-state
trajectory that was accurate at the trained response time but fragile when the
delay was extended. The current `tanh` version is a stronger baseline because
its bounded hidden activity supports more settled delay-period dynamics and
better delay-length generalization.

This is still a baseline working-memory model, not a psilocybin model. Any
future psilocybin-informed condition should be introduced as a separate
modelling hypothesis, with the literature-to-parameter mapping documented
before implementation.

## Repository Structure

- `src/wm_rnn/`: model, task generation, training, evaluation, and analysis code.
- `configs/`: baseline and comparison experiment configurations.
- `docs/model-architecture.md`: architecture and plain-English task explanation.
- `docs/changelog.md`: commit history, experiment run log, and interpretation history.
- `outputs/`: generated checkpoints, metrics, arrays, and figures. This folder is  
not tracked by Git.

