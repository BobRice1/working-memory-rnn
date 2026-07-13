# Model Architecture

## Scope

This document describes the current baseline architecture in `working-memory-rnn`.
The model is a deliberately narrow continuous-time recurrent neural network
for a categorical delayed-response working-memory task. It is the baseline
condition only: it does not yet include psilocybin-informed perturbations,
E/I constraints, modular structure, distractors, N-back updating, fixed-point
landscape mapping, or full global attractor analysis.

## Latest Build State

The canonical categorical model is the `tanh` baseline defined by
`configs/categorical_working_memory.yaml`. The recurrent hidden-state nonlinearity is
bounded with `tanh`, and this is the only categorical delay baseline that should
be used in current comparisons.

The important build conclusion so far is that the current `tanh` baseline gives
a stable delayed-memory representation and generalizes well to longer delays.

## Plain-English Walkthrough

The model is learning a very simple version of a human working-memory task.

Imagine a person sees a cue on a screen for a short time. The cue could be
described as "cue 0", "cue 1", "cue 2", or "cue 3". In the current model these
are not pictures, colors, or real visual objects; they are abstract class
signals. A future interface could show them as colors or symbols, but the model
itself receives them as input channels.

One trial works like this:

```text
see the cue -> hold it in mind during a blank delay -> report the cue later
```

During the cue period, one class channel is switched on. During the delay, that
class signal is switched off, so the model no longer receives the answer from
the input. To respond correctly, it has to preserve enough information in its
recurrent hidden state. The hidden state is the model's working-memory-like
activity: it is what carries the cue forward after the cue disappears.

At the response period, a readout layer looks at the hidden state and predicts
which cue was shown. Training only scores this final response period. The model
is therefore not trained to label every moment of the trial; it is trained to
remember the cue and report it when asked.

In one sentence, the baseline asks whether a simple recurrent network can learn:

```text
I saw this cue earlier, it is gone now, but I can still report it later.
```

## Casual Progress Summary For Supervisor

So far, we have built a simple baseline RNN for a delayed-response
working-memory task. The task is deliberately minimal: the model briefly sees
one of four abstract cues, the cue disappears, and after a delay the model has
to report which cue it saw. We generate this task ourselves with NumPy rather
than using a task library, and we train the network in PyTorch with Adam and a
masked cross-entropy loss that only scores the response period.

The first version used a continuous-time `relu` RNN. It trained successfully:
at the trained `20`-step delay it reached perfect performance, so in the basic
behavioral sense it had learned to remember the cue. But when we pushed the
same frozen model to longer delays, accuracy degraded quite quickly. That told
us the model had learned a solution that worked at the trained response time
but was not especially stable when the memory period was extended.

We then looked at the hidden states rather than just the accuracy. The PCA plots
showed that the hidden-state trajectories separated by cue class, which is good:
the network was representing the remembered cue internally. However, for the
`relu` model those trajectories kept shooting outward rather than settling into
a stable region. A separate stability analysis confirmed this more directly:
hidden-state magnitude and step-to-step speed increased through the delay, so
the model looked more like a ramping or phasic solution than a settled
attractor-like memory state.

We tried a training-objective variant next. That version still used `relu`, but
we trained it with randomized delay lengths and scored the loss across the
delay period as well as the response period. This helped: the model generalized
to longer delays better than the original `relu` model. But the hidden state
still did not really settle; it was a better ramp rather than a clean stable
memory state.

The biggest improvement came from changing the hidden-state nonlinearity from
`relu` to `tanh`. This bounds the hidden activity, so the recurrent state cannot
grow without limit. With the same simple training setup as the original model,
the `tanh` RNN still learned the task and showed much better long-delay
generalization. The stability analysis also changed in the expected direction:
the hidden state rose after the cue and then slowed/flattened during the delay
instead of accelerating. Across multiple seeds, `tanh` is not perfectly
identical every time, but it consistently outperforms the archived `relu`
baseline at long delays.

The current interpretation is that the original model was a useful first pass
because it proved the pipeline and task worked, but it did not give the most
defensible working-memory dynamics. The current `tanh` baseline is stronger for
the dissertation because it behaves more like a stable maintained
representation. The next scientific step is not to call it a confirmed
attractor yet, but to test that more directly with fixed-point or Jacobian-style
analysis before adding psilocybin-informed perturbations.

After that, the next modelling phase would be to choose a small number of
carefully framed perturbations. The first candidate I would suggest is
hidden-state noise during the delay period. In practical terms, we would keep
the trained `tanh` model frozen, run the same cue-delay-response task, and add a
controlled amount of random noise to the recurrent hidden state only during the
delay:

```text
cue period: run normally
delay period: hidden_t = normal recurrent update + noise
response period: read out the remembered cue
```

This would let us ask whether a memory that is stable under normal conditions
becomes less accurate, less settled, or more variable when the internal state is
perturbed. We could then compare normal and noisy runs using the same metrics we
already have: response accuracy, delay-length sweeps, PCA trajectories, and the
delay settling ratio.

The psilocybin link would need to be stated cautiously. This would not be a
literal psilocybin simulation, and it would not model receptors directly.
Instead, it would be a simple computational hypothesis inspired by the
psychedelic literature: if psilocybin is associated with increased neural
variability, altered signal diversity, or less constrained network dynamics,
then adding controlled hidden-state variability is one way to test how a
working-memory RNN behaves when its maintained state is made less stable. The
interpretation would be:

```text
not "this is psilocybin",
but "this tests one candidate dynamical consequence that psilocybin-related
findings might motivate."
```

Before doing that perturbation, I would still do two baseline checks. First,
run fixed-point or Jacobian-style analysis to test whether the `tanh` model is
really settling into cue-specific stable states. Second, run the stability
analysis across the separate `tanh` seeds, because the multi-seed sweep showed
that long-delay generalization is strong but not identical across seeds. If
settling ratio predicts long-delay accuracy across seeds, that gives us a much
cleaner baseline mechanism to perturb.

## Task Interface

The task is generated by `src/wm_rnn/task.py` as a simple cue-delay-response
trial.

Each trial has three temporal phases:

1. Cue period: a one-hot cue identifies the class to remember.
2. Delay period: the class cue is removed, so the model must maintain the
   remembered class internally.
3. Response period: the model reports the remembered class.

Input tensors use shape:

```text
(time, batch, input_channels)
```

The input channels are:

```text
n_classes + 1
```

The first `n_classes` channels encode the categorical cue. The final channel is
a constant fixation/context input that is active at every time step.

Target tensors use shape:

```text
(time, batch)
```

The target class is present at every time step, but the loss mask scores only
the response period. This lets the network receive the cue early, maintain it
through the delay, and be trained only on the final report.

The default categorical configuration in `configs/categorical_working_memory.yaml` uses:

```text
n_classes: 4
cue_steps: 5
delay_steps: 20
response_steps: 5
batch_size: 64
```

This gives a sequence length of 30 time steps and an input size of 5 channels.

## Continuous Tuned Task Interface

The tuned model iteration keeps the same cue-delay-response structure but
changes the represented variable. Instead of a categorical cue, each trial
samples a continuous circular location in `[0, 2*pi)`.

The input population contains evenly spaced tuned units. Each unit has a
preferred angle, and cue activity is computed with a circular Gaussian /
von-Mises-like curve:

```text
activity_i = exp(kappa * (cos(theta - preferred_i) - 1))
```

The population bump is shown during the cue period and removed during the
delay. The target output is the same remembered population bump, scored during
the response period with masked mean-squared error. Evaluation decodes the
predicted angle with circular vector averaging and reports angular error in
degrees.

This is still a baseline working-memory model. It does not introduce
psilocybin-informed perturbations; it gives a richer representational format
for later stability, drift, and perturbation analyses.

The active model progression contains three explicitly named configurations:

- `configs/categorical_working_memory.yaml`: four-class delayed-response task.
- `configs/circular_working_memory.yaml`: fixed `20`-step circular delay task.
  This model learns accurate reporting at the trained delay but shows substantial
  drift when evaluated at much longer delays or in an autonomous hidden-state
  probe.
- `configs/yang_fixation_circular_working_memory.yaml`: canonical Yang-style
  circular working-memory baseline. Training randomizes pre-cue fixation (`15`, `25`, or `35`
  steps), cue duration (`10`, `20`, or `30` steps), and delay (`10`, `20`, `40`,
  or `80` steps). The response lasts `25` steps; its first `5` steps are an
  unscored transition and the remaining response receives weight `5`. The
  fixation output receives weight `2`. Training also uses input noise `0.01`,
  recurrent noise `0.05`, and gradient clipping at `1.0`.

The loss weights are applied linearly in a normalized weighted MSE. This is a
documented adaptation of Yang's implementation, where masks multiply the error
before squaring and therefore have squared effective weights.

## Hidden-State Memory Decoder

Because the Yang-style circular output is intentionally silent during fixation
and delay, output angle is not a valid maintenance-period measure. The
cross-temporal decoder fits ridge regressions from hidden state to the sine and
cosine of the remembered angle. Each decoder is trained at one time step and
tested at every time step on held-out trials, producing a train-time by
test-time angular-error matrix. This supports both same-time delay decoding and
tests of representational stability across time.

Across five independently trained Yang-style models, mean delay-period decoding
error was `0.55 ± 0.46` degrees (range `0.31–1.37`). This establishes that the
silent-output architecture retains readily decodable circular memory content in
its recurrent hidden state.

The stable tuned run now has direct sampled fixed-point/Jacobian evidence. A
follow-up analysis starts from late-delay hidden states, optimizes nearby states
under blank-delay input until one-step recurrent speed is very low, and then
computes the local recurrent Jacobian at each sampled point. For
`tuned_delay_stable`, these fixed points preserve the remembered angle closely
and show the expected ring-attractor-like local structure: a leading eigenvalue
near `1`, a strongly aligned tangent direction around the circular manifold, and
smaller secondary eigenvalues. This supports a sampled ring-attractor-like
interpretation, while still falling short of a global proof over every possible
state in hidden space.

The fixed-point landscape analysis adds the visual comparison used in the
reference RNN dynamical-systems notebook: PCA is fitted on task-evoked hidden
trajectories, and fixed-point search endpoints from late-delay, perturbed
late-delay, and random bounded hidden starts are projected into that same space.
For `tuned_delay_stable`, these endpoints form the same ring-shaped structure as
the task trajectories, giving broader evidence that the ring is not only visible
from the exact trained trajectory states.

## Recurrent Core

The recurrent core is implemented in `src/wm_rnn/model.py` as `CTRNN`. It is a
continuous-time, rate-style recurrent layer with a bounded `tanh` hidden-state
nonlinearity. For the categorical baseline, `tanh` is the intended activation.

The current `baseline_delay` checkpoint has a delay settling ratio of about
`0.062`, meaning late-delay hidden-state speed is much lower than early-delay
speed. In the recorded delay sweep it remains at `1.0` accuracy through the
tested `80`-step delay.

The model configuration is:

```text
input_size: n_classes + 1 for categorical tasks, or n_tuned_units + 1 for tuned tasks
hidden_size: configurable, default 64
output_size: n_classes for categorical tasks, or n_tuned_units for tuned tasks
dt: simulation time step, default 20.0
tau: recurrent time constant, default 100.0
activation: hidden-state nonlinearity, default "tanh" (also supports "relu")
```

The leak factor is:

```text
alpha = dt / tau
```

With the default config:

```text
alpha = 20.0 / 100.0 = 0.2
```

At each time step, the recurrent layer computes:

```text
pre_activation_t = input2h(input_t) + h2h(hidden_t_minus_1)
hidden_t = activation((1 - alpha) * hidden_t_minus_1 + alpha * pre_activation_t)
```

where `activation` is `tanh` by default, bounding hidden activity to
`(-1, 1)`, or `relu` if configured, which keeps activity non-negative but
allows unbounded growth.

The two learned affine projections are:

```text
input2h: input_size -> hidden_size
h2h: hidden_size -> hidden_size
```

The hidden state is initialized to zeros for each sequence:

```text
hidden_0: (batch, hidden_size)
```

The recurrent layer returns all hidden states:

```text
hidden_states: (time, batch, hidden_size)
```

## Readout

The full `WorkingMemoryRNN` wraps the recurrent core with a linear categorical
readout:

```text
readout: hidden_size -> output_size
```

The forward pass returns:

```text
logits: (time, batch, output_size)
hidden_states: (time, batch, hidden_size)
```

Outputs are produced at every time step. For categorical tasks they are class
logits scored with cross-entropy and response accuracy. For tuned tasks they
are population activity predictions scored with masked mean-squared error and
decoded into angular-error metrics.

## Training Objective

Training is implemented in `src/wm_rnn/train.py` with shared utilities in
`src/wm_rnn/training_utils.py`.

Each training step generates a fresh delayed-response batch. The configured
task seed is offset by the training step, giving deterministic but distinct
batches across training.

Categorical runs use masked cross-entropy:

```text
loss_per_time_batch = cross_entropy(logits, targets, reduction="none")
loss = sum(loss_per_time_batch * loss_mask) / sum(loss_mask)
```

Tuned runs use masked mean-squared error over the predicted population code:

```text
population_error = mean((predicted_population - target_population)^2 over tuned units)
loss = sum(population_error * loss_mask) / sum(loss_mask)
```

Because `loss_mask` is 1 only during the response period by default, the model
is trained to classify the remembered cue or reproduce the remembered
population bump only when the response phase begins.

The optimizer is Adam, using the configured learning rate. The default training
configuration is:

```text
steps: 1000
learning_rate: 0.001
device: auto
```

`device: auto` selects CUDA when available and otherwise falls back to CPU.

## Evaluation

Evaluation is implemented in `src/wm_rnn/evaluate.py`.

The evaluator loads a saved checkpoint and generates fresh held-out delayed
response batches with deterministic seed offsets. Categorical evaluation
reports response-period accuracy and writes a class-by-class confusion matrix.
Tuned evaluation reports mean and median angular error plus population MSE; it
does not write a confusion matrix because the output is continuous.

The default evaluation configuration uses:

```text
batches: 20
```

## Hidden-State And Stability Analysis

Hidden-state PCA analysis is implemented in `src/wm_rnn/analysis.py`.

The analysis pipeline:

1. Loads a trained checkpoint.
2. Generates a fresh analysis batch.
3. Runs the model and stores hidden states.
4. Flattens hidden states across time and trials.
5. Fits PCA to the flattened hidden activity.
6. Reshapes projected activity back to trajectory form.
7. Saves arrays and a two-dimensional trajectory plot colored by cue class.

The default analysis configuration uses:

```text
n_components: 2
n_trials: 64
```

This analysis is descriptive. It visualizes whether cue classes occupy
separable trajectories in hidden-state space, but it is not yet a fixed-point,
attractor, or manifold stability analysis.

Hidden-state stability analysis is implemented in
`src/wm_rnn/stability_analysis.py`. It measures whether the hidden state slows
down during the delay or keeps changing. It computes:

```text
hidden-state norm: size of the hidden-state vector
step-to-step speed: size of the hidden-state change from one time step to the next
delay settling ratio: late-delay speed / early-delay speed
```

A delay settling ratio well below `1` means the model is slowing down during
the delay, which is consistent with a more settled representation. A ratio near
or above `1` means the state is still changing at a similar or increasing rate.

This analysis showed a major difference between model versions:

- The archived `relu` baseline had a high settling ratio and kept accelerating
  through the delay.
- The randomized-delay / whole-delay-loss `relu` variant improved behavior but
  still did not fully settle.
- The current `tanh` baseline showed much lower late-delay speed and a much
  more settled hidden-state trajectory.

For tuned checkpoints, `src/wm_rnn.attractor_probe.py` and
`src/wm_rnn.fixed_point_analysis.py` add stronger continuous-memory diagnostics.
The attractor probe measures whether late-delay hidden states keep the same
decoded angle when run forward under blank-delay input. The fixed-point analysis
then searches for nearby approximate fixed points and computes Jacobian
eigenvalues around them. `src/wm_rnn.fixed_point_landscape.py` complements this
by searching from task-reached, perturbed, and random bounded starting states and
plotting the resulting fixed points in the same PCA space as task trajectories.
In the stable tuned run, these analyses found low-residual angle-preserving
fixed points with one near-neutral tangent direction, contracting secondary
directions, and a ring-shaped fixed-point landscape.

`src/wm_rnn.dynamics_figures.py` generates four additional mechanism figures
for dissertation inspection: a clean fixed-point ring plot, decoded angle over
time, Jacobian eigenspectrum, and deterministic perturbation recovery. The
reason for each plot and its reference anchor are recorded in
`docs/analysis-figure-rationale.md`.

## Data Flow Summary

The current pipeline is:

```text
DelayTaskConfig or TunedDelayTaskConfig
    -> generate_delay_batch or generate_tuned_delay_batch
    -> inputs, targets, loss_mask
    -> WorkingMemoryRNN
    -> outputs, hidden_states
    -> masked response-period cross-entropy or masked population MSE
    -> checkpoint and training metrics
    -> evaluation accuracy/confusion matrix or angular-error/population-MSE metrics
    -> delay-length sweep for categorical accuracy or tuned angular error
    -> PCA trajectory analysis with cue-class or angle labels
    -> hidden-state stability analysis
    -> autonomous hidden-state drift probe for tuned checkpoints
    -> fixed-point and Jacobian analysis for tuned checkpoints
    -> fixed-point landscape visualization for tuned checkpoints
    -> dynamics figure generation for tuned checkpoints
```

## Current Interpretation

Architecturally, the model tests whether a simple leaky recurrent population can
store a categorical cue across a delay and recover it at response time. The
recurrent state is the working-memory substrate: during the delay, there is no
class input, so successful performance must depend on maintained hidden-state
activity shaped by recurrent dynamics.

The categorical progression stage is `categorical_working_memory`, which uses a bounded
`tanh` recurrent nonlinearity. It stores the cue in hidden-state activity during
the delay, settles strongly late in the delay, and remains accurate across the
tested delay sweep. Older relu-based delay variants are historical experiments
and should not be treated as current baseline models.

The `dt` and `tau` parameters still make the memory timescale explicit. They
are not yet treated as psilocybin parameters, but they provide a natural place
to later test modelling hypotheses about altered recurrent stability or state
updating, provided those hypotheses are justified separately from the
literature.

## Explicit Non-Features

The present architecture does not implement:

- Dale's principle or explicit excitatory/inhibitory populations.
- Biophysical spiking dynamics.
- NMDA, serotonin, or receptor-level mechanisms.
- Psilocybin-specific parameter changes.
- Distractor input channels or distractor-response scoring.
- N-back or sequence updating.
- Multi-area or modular recurrent structure.
- Exhaustive fixed-point landscape mapping across hidden space.
- A formal global proof of attractor structure.

Those additions should be introduced as separate modelling decisions, with
their assumptions documented before implementation.
