# Current Model Architecture

## Scope

The current experiment uses two task-trained recurrent-network families with a
shared continuous-time recurrent core. The models are behavioural and
computational abstractions. Perturbing them tests whether a dynamical mechanism
is sufficient to reproduce selected task signatures; it does not simulate
psilocybin pharmacology.

## Shared Recurrent Core

Both families use `WorkingMemoryRNN` and the continuous-time recurrent layer in
`src/wm_rnn/model.py`. For hidden state \(h_t\), external input \(x_t\), time
step \(dt\), and time constant \(\tau\), the unperturbed update is:

```text
alpha = dt / tau
drive = W_in x_t + W_rec h_(t-1) + b
h_t = (1 - alpha) h_(t-1) + alpha tanh(drive)
```

The canonical models use:

- 64 recurrent units;
- `tanh` activation;
- `dt = 20`;
- `tau = 100`;
- a linear task-specific readout.

## Fixation-Gated Circular Family

Configuration: `configs/fixation_circular_working_memory.yaml`

The circular task presents a population-coded angle, removes the angle input
during a variable delay, and requires the network to report the remembered
angle after fixation is released.

```text
fixation -> circular cue -> silent delay -> circular response
```

The model has:

- 32 circularly tuned stimulus channels;
- one fixation input;
- 32 circularly tuned output channels;
- one fixation output.

The circular readout is intentionally silent during fixation and delay.
Maintenance therefore must be assessed using hidden-state decoding rather than
output-angle decoding. The independently trained five-checkpoint family is the
canonical circular evaluation baseline.

## N-Back Family

Configuration: `configs/nback_working_memory_screened_final.yaml`

The same recurrent architecture receives streams drawn from six stimulus
identities and a rule context. Its readout classifies match versus non-match:

- 0-back: whether the current item matches a fixed target;
- 2-back: whether the current item matches the item two positions earlier.

The competence-screened ten-checkpoint family supplies the low-load versus
working-memory-load comparison.

## Perturbation Interface

The completed candidate screen applies frozen-weight interventions to:

- input and recurrent synaptic drive;
- heterogeneous unit-wise drive gain;
- sensory or distractor input strength;
- recurrent-weight contribution;
- carried-state persistence;
- effective recurrent time constant.

The operators are implemented in
`src/wm_rnn/perturbation_operators.py` and
`src/wm_rnn/nback_perturbation.py`. Operator names describe the implemented
mathematics and should not be treated as receptor-level mechanisms.

## Current Measurement Boundary

The circular family supports:

- response angular error;
- response settling;
- fixation validity;
- hidden-state memory decoding;
- delay-length dependence;
- exploratory response to an evaluation-only distractor.

The N-back family supports:

- discriminability and accuracy;
- match/non-match decision settling;
- 0-back versus 2-back selectivity.

Because the existing circular checkpoints were not trained to ignore delay
distractors, their distractor condition is an out-of-distribution robustness
probe. A separately trained distractor-capable single-item circular family is
the planned design for a stronger distractor-filtering claim.

## Historical Architectures

The categorical baseline, early circular baseline, two-slot multicondition
task, and their rescue variants are retained for scientific provenance. Their
configuration status is listed in `configs/README.md`; the original detailed
categorical architecture document is archived as
`docs/archive/categorical-model-architecture.md`.
