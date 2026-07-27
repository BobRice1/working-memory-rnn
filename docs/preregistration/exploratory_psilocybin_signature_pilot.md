# Exploratory two-task psilocybin-signature pilot

Frozen before outcome evaluation: 27 July 2026.

## Purpose and claim boundary

This small first pass asks which already implemented RNN perturbation directions
produce behavioural patterns resembling selected human psilocybin findings.
It is an exploratory screen for a larger preregistered evaluation, not a test of
pharmacological equivalence, mechanistic uniqueness, or statistical recovery.
The independently trained checkpoint is the unit of replication, but three
checkpoints per task are insufficient for confirmatory inference.

The primary comparison is each perturbation against that checkpoint's native
unperturbed baseline on identical task samples. Gaussian state noise is a
secondary generic-disruption control. Failure to exceed noise does not erase a
candidate-versus-baseline resemblance.

## Frozen task banks

- Original fixation-gated circular delayed-response task: seeds 20260714,
  20260715, and 20260716. Clean trials use delays 10, 20, 40, and 80. A
  delay-20 distractor condition adds one irrelevant tuned pulse during the
  delay. There are 256 trials per cell.
- N-back task: seeds 20260912, 20260913, and 20260914. Conditions are 0-back
  and 2-back, with 256 twenty-item sequences per cell.
- Existing checkpoints are reused. No model is retrained.
- Native and perturbed evaluations reuse identical generated trials within a
  checkpoint/task cell.

## Frozen perturbation grids

Only single-operator interventions are screened; no hybrids are included.

| ID | Operator | Strengths |
|---|---|---|
| P1 | synaptic-drive gain, bias outside | 0.90, 0.95, 1.00, 1.05, 1.10 |
| P2 | fixed heterogeneous drive gain, log-SD | 0.00, 0.10, 0.20, 0.30 |
| P3a | sensory-input gain | 0.80, 0.90, 1.00, 1.10, 1.20 |
| P3b | distractor-window input gain, circular task only | 1.00, 1.10, 1.25, 1.50 |
| P4 | recurrent gain | 0.90, 0.95, 1.00, 1.05, 1.10 |
| P5 | Gaussian state noise | 0.00, 0.025, 0.050, 0.075, 0.100 |
| P6 | carried-state persistence | 0.90, 0.95, 1.00, 1.05, 1.10 |
| P7 | conserved effective time constant | 0.80, 0.90, 1.00, 1.10, 1.25 |

P2 and P5 use three deterministic within-checkpoint replicates, averaged before
checkpoint-level summaries. These replicates are not independent observations.
Neutral settings are checked against the native forward pass but the baseline
is evaluated only once per task bank.

## Human-signature analogues and scores

Circular-task outcomes:

- slowing with relative preservation: restricted mean post-cue settling time
  increases at delay 20 while proportional mean-angular-error impairment is at
  most 0.20;
- delay selectivity: baseline-normalized error impairment at delay 80 exceeds
  that at delay 10; the four-delay slope is also reported;
- distractor selectivity: baseline-normalized impairment is greater for
  delay-20 distractor trials than matched clean trials.

Settling is an RNN response-dynamics analogue, not literal human reaction time.
It is scored only when fixation accuracy is at least 0.90 and at least 0.80 of
trials settle under the frozen criterion.

N-back outcome:

`I(c) = (D_native(c) - D_perturbed(c)) / D_native(c)`, where `D` is
discriminability and `c` is 0-back or 2-back. Load selectivity is
`I(2-back) - I(0-back)`; positive values resemble selective high-load
impairment.

For every signature, report its mean direction and the number of checkpoints
with the predicted sign (0/3 to 3/3). A pattern is called
`pilot-consistent` only when the mean has the predicted sign and at least two
of three checkpoints agree. `strongly-consistent` is reserved for three of
three. No p-values or multiplicity-adjusted match labels are used.

Dose ordering is supporting evidence: within each side of neutral, the target
effect must retain its predicted sign and not diminish at the farther setting.

## Secondary Gaussian comparison

For each candidate strength, identify the P5 strength with the nearest
native-normalized clean-task cost. Report the cost gap and excess signature
contrast. If the cost gap exceeds 0.05, label it `unmatched_descriptive`.
These comparisons are secondary and cannot gate the primary human-signature
screen.

## Interpretation

The pilot may select operator directions and strengths for a new independently
banked preregistered run. It cannot establish that an RNN perturbation is
psilocybin, that it is biologically correct, or that a mechanism is uniquely
identified. Human findings motivate the qualitative contrast directions; they
are not model training targets or seed-selection criteria.
