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

The canonical perturbation baseline is the Yang-style fixation-gated circular
working-memory RNN. It uses a bounded `tanh` recurrent state, a circular
population code, a fixation input/output pair, randomized cue and delay timing,
training noise, weighted loss, and gradient clipping. The circular readout is
silent until response; remembered angle during maintenance is measured with a
cross-temporal hidden-state decoder.

## Model Progression

- `configs/categorical_working_memory.yaml`: four-class delayed-response RNN;
  the simplest categorical working-memory stage.
- `configs/circular_working_memory.yaml`: continuous circular-location RNN using
  a Gaussian/von-Mises population code.
- `configs/yang_fixation_circular_working_memory.yaml`: canonical Yang-style
  circular working-memory RNN with explicit fixation control and randomized
  timing. This is the baseline for perturbation experiments.

These names describe the task and representation rather than the order in which
the models happened to be developed. The three stages are categorical memory,
circular memory, and Yang-style fixation-gated circular memory.
Superseded runs are preserved under `outputs/archive/`, with retained YAMLs
under `configs/archive/`. See `docs/model-run-archive.md`.

Five independently trained Yang-style baselines have mean response error
`4.36 ± 1.55` degrees and mean fixation accuracy `0.968`. Cross-temporal ridge
decoding recovers remembered angle from the delay-period hidden state with mean
error `0.55 ± 0.46` degrees across seeds. Frozen-model delay sweeps remain
functional through an untrained `160`-step delay, with seed-level mean response
errors between `5.14` and `11.73` degrees.

## Analyses

The project currently includes four analysis layers:

- Task performance at the trained delay.
- Delay-length sweeps with frozen weights.
- PCA visualization of hidden-state trajectories.
- Cross-temporal hidden-state decoding of circular memory content.
- Independent multi-seed training with task, decoder, and delay-sweep summaries.
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
