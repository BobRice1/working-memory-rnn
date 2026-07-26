# N-back RNN literature and architecture decision

Date: 2026-07-26

## Purpose

This note records the evidence used to replace the failed synthetic two-slot
Family B baseline with an actual N-back task. It separates published evidence
from project-specific modelling decisions.

## Human behavioural target

Barrett et al. (2018) administered letter 0-back, 1-back, and 2-back conditions.
Their reported outcomes included median response time on correct trials,
discriminability defined as hit rate minus false-alarm rate, and response bias
defined as false-alarm rate divided by one minus discriminability. Both 0-back
and 2-back require vigilance, whereas only 2-back requires working memory.
Psilocybin impaired the 2-back condition relative to 0-back and did not affect
the reported 0-back outcomes.

The model target is therefore a within-network dissociation:

1. preserved 0-back performance;
2. impaired 2-back discriminability and/or increased 2-back settling time;
3. a larger 2-back-minus-0-back change than matched Gaussian disruption.

Settling time remains a computational analogue because model time steps are not
human milliseconds.

## RNN precedents

### Wan, Menendez, and Postle (2022)

Wan et al. trained PyTorch LSTMs on a 2-back task with six stimulus identities.
The categorical version used six one-hot inputs, seven LSTM units, and one
binary match/non-match output. Each stimulus presentation was followed by two
blank delay steps. Sequences contained 20 stimuli; the first two decisions were
unscored. Match trials occurred at approximately a 1:2 ratio to non-match
trials. The networks were trained with Adam at `1e-3` and evaluated on
independently sampled sequences. A 99.5% test-accuracy competence criterion was
applied across independently initialized networks.

This directly supports the six-identity stream, binary decision, 20-event
sequence, two-event warm-up, approximate 1:2 match ratio, independent test
sequences, and checkpoint-level competence gating used here.

Reference:

- Quan Wan, Jorge A. Menendez, and Bradley R. Postle (2022), “Priority-based
  transformations of stimulus representation in visual working memory,”
  *PLOS Computational Biology* 18(6): e1009062.
  https://doi.org/10.1371/journal.pcbi.1009062

### Lei, Ito, and Bashivan (2024)

Lei et al. trained vanilla RNN, GRU, and LSTM architectures on nine N-back
tasks spanning N values 1, 2, and 3. Successful gateless vanilla RNNs show that
an LSTM is not required in principle for N-back competence.

This supports retaining the project's existing continuous-time tanh RNN. That
choice is scientifically important because the registered perturbation
operators are defined on its input drive, recurrent drive, carried state, and
time constant. Replacing it with an LSTM would introduce several gates and cell
states and would make those operator definitions non-equivalent.

Reference:

- Yiming Lei, Takuya Ito, and Pouya Bashivan (2024), “Geometry of naturalistic
  object representations in recurrent neural network models of working
  memory,” *NeurIPS 2024*. https://doi.org/10.52202/079017-3191

### Yang and Yu (2025)

Yang and Yu trained RNNs on N-back tasks and compared their representational
dynamics with human data. Their results support treating independently trained
N-back RNNs as computational models whose dynamics can be interrogated, while
not assuming that a successful network implements the human mechanism.

Reference:

- Yuxuan Yang and Qing Yu (2025), “Orthogonal-Rotational Dynamics Supports
  Efficient Encoding and Updating for Streaming Information in Working
  Memory,” *Journal of Neuroscience*. PMID: 41326245.
  https://pubmed.ncbi.nlm.nih.gov/41326245/

## Architecture decision

The N-back baseline will reuse `WorkingMemoryRNN`:

- continuous-time tanh recurrent layer;
- 64 hidden units for the first development specification;
- `dt = 20`, `tau = 100`, and therefore `alpha = 0.2`;
- two-class linear readout: non-match and match;
- no recurrent or input noise during baseline development.

The task will supply eight inputs:

- six one-hot stimulus-identity channels;
- one constant 0-back context channel;
- one constant 2-back context channel.

The task context is block-wise and mutually exclusive. The same weights solve
both conditions.

## Task decision

Each batch is homogeneous in task condition so that generation, validation,
and later perturbation contrasts are unambiguous. Training alternates complete
0-back and 2-back batches in balanced shuffled blocks.

Each sequence contains 20 stimulus events. An event contains three stimulus
steps followed by six blank steps. Events 0 and 1 are warm-up events and are
unscored in both task conditions, yielding 18 matched scored positions.

For 0-back, stimulus identity 0 is the fixed target. For 2-back, a match occurs
when the current identity equals the identity two events earlier. Every
sequence contains exactly six matches among its 18 scored positions. The
2-back generator also includes at least three one-back lures per sequence when
feasible, so a current-versus-previous-item heuristic cannot pass the
competence gate.

The two output classes are trained with cross-entropy throughout each scored
event. This provides a decision trajectory from stimulus onset through the
following blank interval and enables an output-settling measure.

The three-plus-six timing is a project-specific scaling of Wan et al.'s
one-plus-two event structure for an `alpha = 0.2` continuous-time RNN. It is not
claimed to map onto human stimulus durations.

## Why the failed Family B task is not retained

The two-slot retrocue task required serial encoding, long silent maintenance,
probe interpretation, circular recall, distractor rejection, and fixation
gating in a single fixed-clock trial. The distribution-loss, stimulus-role,
and 128-unit rescues all failed baseline memory competence despite adequate
fixation. Those failures do not test any perturbation mechanism.

The N-back task removes the externally cued serial-recall bottleneck. Each new
stimulus supplies a match/non-match teaching signal, and the exact human
0-back/2-back contrast is represented directly. It is therefore both more
learnable and more faithful to the Barrett behavioural signature.

## Claim boundary

Successful N-back training would establish only that this CTRNN can solve the
task and can be used to test selective perturbation effects. A perturbation
that reproduces the ordinal human pattern would demonstrate computational
sufficiency within this model and task battery. It would not be a biological
model of psilocybin or proof of its human neural mechanism.
