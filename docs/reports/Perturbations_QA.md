# Perturbation Screen: Supervisor Questions and Answers

This note anticipates questions arising from the supervisor-facing candidate
perturbation report. The answers preserve the report's distinction between a
descriptive computational result, a mechanism candidate, and a biological
claim about psilocybin.

## Core interpretation

### What is the central result?

Persistence `0.95` was the only tested operator-strength profile that met the
descriptive cross-task majority rule against each checkpoint's native
baseline. The N-back load effect was consistent across checkpoints, whereas
the circular settling, delay, and distractor effects were small and uncertain.

### Was persistence `0.95` selected before this analysis?

Yes. It was nominated in an earlier three-checkpoint exploratory pilot. The
new distractor-trained circular checkpoints provide a genuine circular-family
retest, but three of the ten N-back checkpoints also contributed to the
nomination pilot. The present result is therefore not a completely independent
blind confirmation.

### Does this show that psilocybin reduces neural persistence?

No. State persistence is an abstract intervention in the implemented RNN. The
result shows qualitative computational sufficiency against the native model
under the descriptive scoring rule; it does not identify a pharmacological or
biological action of psilocybin.

### Why call these psilocybin-related signatures?

The contrast directions were motivated by acute human findings concerning
response slowing relative to accuracy, greater 2-back than 0-back impairment,
and vulnerability in a task involving competing information. The model tests
qualitative analogues of those orderings. It does not reproduce the human
effect sizes, measurement units, or exact experimental tasks.

### Are the results statistically convincing?

The N-back load-selectivity interval excluded zero and the effect was positive
in all ten checkpoints. The circular settling, delay, and distractor intervals
included zero. Those circular results should therefore be described as
descriptive majority tendencies rather than robust confirmatory effects.

### Is the complete pattern mostly driven by N-back?

Yes, in evidential strength. The N-back component is consistent, while the
circular components supply the required mean directions and majority agreement
but remain weak. The result is best treated as a candidate nomination with a
strong N-back component and provisional circular support.

### Why use a majority rule?

The rule prevents a mean effect produced by one unusual checkpoint from
qualifying. It remains an operational descriptive rule rather than a
confirmatory statistical criterion. Although the component definitions came
from the earlier pilot, the exact numeric majority threshold was not restated
in the 1,024-trial follow-up preregistration.

### What can be concluded now?

A small asymmetric reduction of the carried-state coefficient can generate the
selected qualitative cross-task ordering in competent trained networks. It
cannot yet be concluded that persistence is superior to generic disruption,
that it is the unique computational explanation, or that it represents the
biological effect of psilocybin.

## Experimental design

### Why combine two different network families?

No single competent task family supplied every contrast. The circular networks
measure response settling, delay dependence, and learned distractor filtering;
the N-back networks measure updating-load selectivity. Effects are normalised
within task and checkpoint, and raw scores are not combined across task
families. The resulting profile is therefore a cross-model conjunction rather
than one unified-task phenotype.

### How independent is the N-back replication?

All ten N-back checkpoints showed the predicted load-selectivity direction.
Three were used in the nomination pilot and the seven additional checkpoints
were also positive. This strengthens checkpoint consistency, but the complete
ten-checkpoint result should not be described as an independent confirmation.

### Could competence screening bias the result?

The analysis is conditional on competent networks. Five of six trained circular
seeds and ten competence-screened N-back seeds were retained before
perturbation evaluation. Screening is necessary because perturbing failed
baselines is difficult to interpret, but the conclusions apply to trained,
competent networks rather than arbitrary initialisations.

### Why were trials not treated as independent observations?

Trials estimate behaviour within a frozen network. The inferentially relevant
replication unit is the independently trained checkpoint because trials from
the same checkpoint share weights and training history. Treating 1,024 trials
as 1,024 independent model replications would be pseudoreplication.

### Why report Student-\(t\) intervals if the analysis is descriptive?

The intervals summarise variation across independently trained checkpoints.
However, the leading profile was selected from 23 non-neutral profiles and no
confirmatory multiplicity correction was applied. An interval excluding zero
therefore should not be presented as a pre-specified corrected hypothesis test.

### Why is delay selectivity supporting rather than primary evidence?

The direct human evidence for delay dependence is weaker and comes from tasks
that are not equivalent to the circular RNN task. Delay selectivity is useful
for distinguishing persistence-like from sensitivity-like operators, but it
was not required to pass the primary descriptive pattern.

### Why use a 20% accuracy-preservation ceiling?

It is an operational ceiling intended to reject wholesale clean-task
impairment while allowing small costs. It is not a human-derived threshold.
A useful sensitivity analysis would repeat the selection under tighter
ceilings, such as 10% and 15%, to show whether the nomination depends on this
choice.

### Could another operator work at a different strength?

Yes. The screen evaluated a fixed discrete grid, and operators were not matched
for overall clean-task cost. A near-miss can therefore reflect strength
selection as well as mechanism. This motivates denser persistence sampling and
matched-cost comparisons among shortlisted operators.

## Meaning of the persistence manipulation

### What exactly does a 5% persistence reduction do?

With \(\alpha=0.2\), the native carried-state coefficient is
\(1-\alpha=0.8\). Persistence `0.95` changes that coefficient to

\[
0.95\times0.8=0.76,
\]

while leaving the incoming-drive coefficient at \(0.2\). The coefficient sum
therefore changes from \(1.0\) to \(0.96\). It is an asymmetric contraction of
the carried state, not a 5% change in the network time constant.

### How is persistence different from changing the effective time constant?

The conserved time-constant operator changes the carried-state and
incoming-drive coefficients together so that their sum remains one.
Persistence changes only the carried-state coefficient. Their different
behavioural profiles suggest that this asymmetry matters, although unequal
overall perturbation cost remains an alternative explanation.

### How is persistence different from recurrent gain?

Recurrent gain scales the learned recurrent weight contribution
\(W_{\mathrm{rec}}h_{t-1}\). Persistence instead scales the direct carried-state
term \((1-\alpha)h_{t-1}\) while leaving the learned recurrent drive unchanged.
They intervene at different places in the update equation.

### Why should persistence have anything to do with psychedelics?

There is no direct evidence mapping psilocybin to this RNN parameter.
Persistence was included as one competing manipulation of working-memory
stability. Its current status comes from the behavioural screen, not from a
receptor-level derivation.

### Does the behavioural result demonstrate altered attractor dynamics?

No. The persistence equation suggests a model-level explanation, but the
current analyses do not directly measure attractor geometry, slow-manifold
drift, local stability, or recovery dynamics. Those quantities require
separate hidden-state analyses.

## Human-to-model correspondence

### Is model settling time equivalent to human reaction time?

No. Response onset is externally imposed in the circular task. Settling time
measures how many model steps the output requires to approach the target after
that onset. It is a response-dynamics analogue, and the mean persistence effect
of `0.032` steps is very small.

### Is the distractor condition equivalent to Carter et al.'s task?

No. Carter et al. used multiple-object tracking, whereas the RNN receives a
single irrelevant circular cue during a memory delay. The model tests a related
filtering principle rather than reproducing the human task.

### Can model effect sizes be compared directly with human effect sizes?

No. Angular degrees, model settling steps, and normalised discriminability
changes are not commensurable with human milliseconds, standardised effects,
or task accuracy. Human evidence supplies qualitative orderings; statistical
comparisons remain internal to the model.

## Specificity and next experiments

### Why is Gaussian noise absent if specificity is the central question?

This run screened candidate operators against native baseline. Gaussian noise
must now be calibrated to a comparable clean-task cost. Without that
matched-cost comparison, the current result cannot establish that persistence
produces a more selective pattern than generic degradation.

### What would falsify or substantially weaken the persistence interpretation?

The interpretation would be weakened if persistence failed to outperform
matched-cost Gaussian disruption, disappeared when values around `0.95` were
sampled more densely, failed on independently banked checkpoints, or produced
no persistence-specific change in representational stability.

### What should the next experiment be?

Compare persistence `0.95` with Gaussian state noise matched on held-out
clean-task cost in both task families. Evaluate the pre-specified contrasts on
independently banked samples, then quantify delay-period drift, local Jacobian
spectra, and recovery trajectories following distractor input.

### What is the defensible one-sentence conclusion?

> Reduced carried-state persistence was the only abstract perturbation to meet
> the descriptive cross-task rule against native baseline, with strong N-back
> consistency but weak circular effects; matched-cost Gaussian specificity is
> the decisive next test.
