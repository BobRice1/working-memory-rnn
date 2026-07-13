# Archived Intermediate Circular Attractor Model Summary

## Purpose

The historical `tuned_delay_stable` model is an archived intermediate circular
working-memory model. It is not the current perturbation baseline; the canonical
model is `yang_fixation_circular_working_memory`.
It extends the earlier categorical delayed-response task into a continuous
circular memory task. Instead of remembering one of four discrete cue classes,
the model remembers an angle on a circle.

This model is still a baseline model. It does not include psilocybin-informed
perturbations, noise manipulations, receptor mechanisms, excitatory/inhibitory
constraints, or biological circuit detail. Its purpose is to provide a stable
continuous working-memory system that can later be perturbed and analyzed.

## Task Structure

a continuous circular delayed-response working-memory task using population-coded angle cues

Each trial has three phases:

```text
cue period -> delay period -> response period
```

During the cue period, the model receives a circular population-code input. A
target angle is sampled from `[0, 2*pi)`, and this angle is encoded as a bump of
activity over tuned input units.

During the delay period, the tuned cue is removed. The model still receives the
context channel, but it no longer receives the remembered angle from the input.
To solve the task, it must maintain the angle in its recurrent hidden state.

During the response period, the model reconstructs the remembered population
bump. The output is decoded back into an angle, and performance is measured as
circular angular error in degrees.

## Population Code

The tuned task uses `32` units arranged evenly around the circle. Each unit has a
preferred angle. Units close to the sampled target angle are strongly active;
units far from it are weakly active.

The activity of each tuned unit is computed with a circular Gaussian /
von-Mises-like tuning curve:

```text
activity_i = exp(kappa * (cos(theta - preferred_i) - 1))
```

where:

- `theta` is the sampled target angle.
- `preferred_i` is the preferred angle of unit `i`.
- `kappa` controls the sharpness of the population bump.

The stable tuned config uses:

```text
n_tuned_units: 32
tuning_kappa: 8.0
cue_steps: 5
delay_steps_min: 20
delay_steps_max: 80
response_steps: 5
batch_size: 64
```

## Model Architecture

The model is a continuous-time recurrent neural network with a linear readout:

```text
tuned input population + context channel
        -> continuous-time recurrent layer
        -> linear readout
        -> reconstructed tuned output population
```

The recurrent layer has `64` hidden units. It uses a leaky continuous-time update
with:

```text
dt: 20.0
tau: 100.0
activation: tanh
```

The `dt / tau` ratio controls how strongly the previous hidden state carries
forward into the next time step. The `tanh` nonlinearity bounds the hidden state,
which helps prevent unbounded ramping dynamics.

At each time step, the recurrent state combines:

- the previous hidden state,
- the current input,
- learned recurrent weights,
- learned input weights,
- the leak term controlled by `dt / tau`.

The readout layer maps the `64`-dimensional hidden state back to the `32` tuned
output units.

## Training Objective

The model is trained with population mean-squared error. The target output is the
same tuned population bump that encoded the sampled target angle.

The stable tuned model differs from the first fixed-delay tuned model in two
important ways:

- The delay length is randomized during training between `20` and `80` time
  steps.
- The loss is applied during the delay and response periods.

This encourages the network to maintain a stable remembered angle throughout the
delay, rather than only being correct at one fixed response time.

## Decoding And Evaluation

The model output is decoded into an angle using circular vector averaging. The
decoded angle is then compared with the target angle using wrapped circular
error, reported in degrees.

The main evaluation measures are:

- mean angular error,
- median angular error,
- 95th-percentile angular error,
- maximum angular error,
- population mean-squared error.

Delay sweeps test whether the frozen trained model can preserve the remembered
angle over delays longer than the reference delay.

## Hidden-State Interpretation

The hidden state is the model's working-memory representation. During the cue,
the population-code input pushes the hidden state toward a representation of the
sampled angle. During the delay, the cue is absent, so the recurrent dynamics
must keep the remembered angle available internally.

Analyses of the stable tuned model suggest that remembered angles are organized
around a ring-like hidden-state structure. Different angles occupy different
positions around the ring. Movement around the ring corresponds to changing the
remembered angle.

This gives a ring-attractor-like interpretation:

- the ring stores the continuous memory variable,
- positions around the ring correspond to remembered angles,
- off-ring perturbations can return toward the ring,
- local dynamics contain a near-neutral direction around the ring and more
  stable directions away from it.

This is sampled evidence, not a complete proof over the full `64`-dimensional
hidden-state space.

## Fixed-Point And Jacobian Analysis

The fixed-point and Jacobian analyses test whether the ring-like hidden-state
structure behaves like a stable memory system, rather than only looking like a
ring in a PCA plot.

A fixed point is a hidden state that does not meaningfully change when the model
is run forward with no cue input. In working-memory terms, this is important
because a stable memory should be able to persist during the blank delay period.
If the model has a remembered angle stored in its hidden state, and the cue is
removed, a fixed-point-like state would keep representing that angle instead of
drifting away.

The analysis works in four broad steps:

1. Take hidden states from late in the delay period, when the model should
   already be maintaining the remembered angle.
2. Run an optimization that searches nearby hidden states for points where the
   next recurrent update is almost the same as the current state.
3. Decode each candidate fixed point back into an angle, to check whether it
   still represents the original remembered angle.
4. Compute the local Jacobian around each fixed point, which describes how small
   perturbations would grow, shrink, or move around that point.

The Jacobian is useful because it tells us about local stability. For a ring
attractor, the expected pattern is not that every direction is strongly stable.
Instead, one direction should be close to neutral: movement along the ring changes
the remembered angle without immediately collapsing back to one single point.
Directions away from the ring should be more stable, so perturbations off the
ring return toward the memory manifold.

The recorded `tuned_delay_stable` fixed-point/Jacobian results support this
interpretation at a sampled level:

- Mean fixed-point residual: `0.000792`.
- Median fixed-point residual: `0.000728`.
- Fraction below the `0.001` residual threshold: `0.766`.
- Mean fixed-point decoding error: `1.356` degrees.
- 95th-percentile fixed-point decoding error: `3.244` degrees.
- Mean spectral radius: `1.000036`.
- Maximum spectral radius: `1.005071`.
- Mean second-largest absolute eigenvalue: `0.984019`.
- Mean tangent alignment between the sampled ring direction and leading
  eigenvector: `0.945`.

At a high level, this means the optimized states are very close to true
blank-delay fixed points, and they still decode to almost the same remembered
angle. The leading Jacobian mode is near neutral and closely aligned with the
ring direction, which is what we would expect if the model has learned a
continuous circular memory manifold. Most other modes are smaller, suggesting
greater stability away from the ring.

The fixed-point landscape analysis adds a broader visual check. It searches from
actual late-delay states, perturbed late-delay states, and random bounded hidden
states, then projects the endpoints into the same PCA space as the task
trajectories. For `tuned_delay_stable`, those endpoints form the same ring-shaped
structure as the task trajectories. This suggests the ring is not only visible
from the exact trial trajectories, although the result is still sampled rather
than exhaustive.

## Historical Role In The Project

`tuned_delay_stable` was the strongest intermediate continuous baseline. It is more
useful than the categorical baseline for attractor-style working-memory analysis
because it provides:

- a continuous circular memory variable,
- angular precision metrics,
- drift measurements,
- fixed-point and Jacobian diagnostics,
- ring-manifold visualization,
- deterministic perturbation-recovery analysis,
- hidden-state movie visualization.

The next modelling phase should add perturbations only after keeping this
baseline condition clearly separate from any psilocybin-inspired assumptions.
