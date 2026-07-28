# Psilocybin-Related Working-Memory Signatures in Recurrent Neural Networks

This dissertation project asks whether controlled changes to a recurrent
neural network can reproduce selected behavioural patterns reported in humans
performing working-memory tasks under acute psilocybin. The aim is to test the
computational sufficiency of candidate mechanisms, not to model psilocybin's
pharmacology or claim that an artificial network is biologically equivalent to
the human brain.

## Model

The project uses a leaky continuous-time recurrent neural network with `tanh`
units. The network receives a sequence of task inputs, maintains relevant
information in its recurrent hidden state, and produces a response when
prompted.

Separate networks are trained for each task and evaluated across independently
trained checkpoints. Perturbations are applied only after training, with the
learned weights held fixed. This makes it possible to ask how changing a
particular aspect of the network's dynamics alters otherwise competent
working-memory performance.

## Tasks

### Circular delayed-response working memory

The network fixates, observes a cue at an angle on a circle, retains that angle
across a blank delay, and reports it after a response cue. This task supports
measurement of:

- angular recall accuracy;
- response-settling time as a computational analogue of reaction time;
- sensitivity to longer retention delays; and
- sensitivity to an irrelevant cue presented during the delay.

A dedicated circular checkpoint family is trained on a balanced mixture of
clean and distractor-present trials. This makes distractor filtering a learned
part of the task while preserving the original single-item architecture and
timeline. The earlier clean-trained checkpoints and their evaluation-only
distractor results remain historical exploratory analyses.

### N-back working memory

The network performs both 0-back and 2-back judgements using the same trained
weights. The 0-back condition provides a low-memory comparison, whereas the
2-back condition requires the network to maintain and update recent
information. The contrast tests whether a perturbation disproportionately
affects performance under higher working-memory demand.

## Perturbations

The candidate perturbations alter distinct components of the trained network:

- **Synaptic-drive gain** scales the combined input and recurrent drive entering
  the recurrent nonlinearity.
- **Heterogeneous synaptic-drive gain** applies stable unit-to-unit variation
  in that gain.
- **Sensory-input gain** changes the influence of task stimuli on the recurrent
  network.
- **Distractor-input gain** changes only the influence of the irrelevant
  delay-period cue in the circular task.
- **Recurrent gain** changes the contribution of recurrent connectivity to the
  network's next state.
- **State persistence** changes how strongly the previous hidden state is
  carried forward.
- **Effective time constant** changes the rate at which the recurrent state is
  updated.

These are competing computational interventions rather than direct
representations of receptor-level drug action. Gaussian state noise is retained
as a generic disruption control where a non-specific comparison is required,
but it is not treated as a candidate psilocybin mechanism.
