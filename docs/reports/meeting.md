# Meeting notes: candidate RNN perturbations

## Short opening

- I wanted to give you an update on the working-memory modelling.
- I have added two complementary task families:
  - a fixation-gated circular delayed-response task with trained distractor filtering;
  - a context-cued 0-back/2-back task.
- I used them to ask whether an abstract change to trained recurrent dynamics could reproduce several qualitative working-memory effects associated with acute psilocybin.
- The aim was to find selective behavioural dissociations, rather than a perturbation that simply makes every task worse.
- Important claim boundary:
  - these are computational interventions;
  - none is a literal pharmacological model of psilocybin;
  - behavioural resemblance establishes candidate computational sufficiency within these RNNs, not biological equivalence.

## The two task families

### Circular distractor task

- The network receives an angle as a population code over 32 tuned input channels.
- Trial sequence:
  - fixation;
  - circular cue;
  - silent delay;
  - fixation release and circular response.
- The circular output is deliberately silent during the delay:
  - behavioural error is measured after the go cue;
  - delay-period memory can be examined from the recurrent hidden state.
- Half of the training batches contain an irrelevant five-step circular cue in the middle of the delay.
- The required answer remains the original angle.
- This means distractor filtering is trained competence, rather than an unfamiliar perturbation introduced only at evaluation.
- Behavioural motivation:
  - Carter et al. (2005) reported impaired multiple-object tracking with comparatively spared spatial working-memory span;
  - they discussed weakened filtering of irrelevant information;
  - this model tests a related filtering principle, not a reproduction of their human task.

### N-back task

- Each sequence contains 20 items drawn from six stimulus identities.
- Two context channels indicate whether the current rule is 0-back or 2-back.
- 0-back:
  - report whether the current item is a fixed target identity;
  - acts as the lower-memory-load detection condition.
- 2-back:
  - report whether the current item matches the item two positions earlier;
  - requires recent information to be maintained and continually updated.
- One-back lures prevent the network from passing through a simple previous-item or familiarity strategy.
- Behavioural motivation:
  - Barrett et al. (2018) found clearer psilocybin effects on 2-back than on 0-back outcomes;
  - the model therefore tests whether a perturbation causes greater impairment when active updating is required.
- Training:
  - networks first learned 0-back and had to pass a competence gate;
  - training then introduced 2-back;
  - checkpoints were retained only if they passed 0-back, 2-back, and lure-performance gates.

## Baseline competence

### Circular family

- Six candidate networks were trained.
- Five passed the preconfigured competence gates and were retained.
- Across the five retained checkpoints:
  - mean clean angular error: approximately \(2.70^\circ\);
  - mean distractor-trial error: approximately \(3.54^\circ\);
  - mean distractor cost: approximately \(0.85^\circ\);
  - fixation accuracy: approximately \(0.98\) in both conditions.
- Interpretation:
  - clean recall was accurate;
  - the trained distractor produced a small but measurable cost;
  - the distractor comparison therefore did not depend on presenting an entirely unfamiliar task condition.

### N-back family

- Ten independently trained competent checkpoints were retained.
- Across those checkpoints:
  - 0-back accuracy and discriminability were effectively \(1.00\);
  - mean 2-back accuracy was approximately \(0.973\);
  - mean 2-back discriminability was approximately \(0.950\);
  - mean 2-back lure accuracy was approximately \(0.987\).
- Interpretation:
  - both task rules were competently solved before perturbation;
  - any later 0-back/2-back difference was evaluated within the same trained checkpoint.

## Are these the same RNN?

- They use the same general recurrent architecture but are not the same trained network.
- Both families use:
  - 64 recurrent units;
  - a `tanh` activation;
  - continuous-time parameters \(dt=20\) and \(\tau=100\);
  - native leak factor \(\alpha=dt/\tau=0.2\).
- Native update:

  \[
  h_t=\tanh\left[(1-\alpha)h_{t-1}+\alpha z_t\right],
  \]

  where \(z_t\) contains the input drive, recurrent drive, and learned biases.

- Circular interface:
  - 32 stimulus channels plus fixation;
  - circular population readout plus fixation output.
- N-back interface:
  - six stimulus channels plus two rule-context channels;
  - two classification outputs: non-match and match.
- Each family has its own learned weights and was trained only on its own task.
- The cross-task result is therefore a conjunction across two independently trained but architecturally matched model families.
- The circular family supplies:
  - response settling;
  - delay-length selectivity;
  - distractor selectivity.
- The N-back family supplies:
  - updating-load selectivity.

## How the perturbation screen was run

- All learned weights were frozen after training.
- Each perturbation changed only the forward dynamics; there was no retraining.
- Each non-neutral operator was compared with the same checkpoint's native forward pass.
- Except for distractor-window gain, operators were active at every timestep.
- Learned biases remained outside multiplicative gains unless explicitly stated.
- Neutral gain, persistence, and timescale values were \(1.0\).
- Not every operator has equal evidential provenance:
  - some are translations motivated by psychedelic or behavioural literature;
  - some are working-memory circuit hypotheses;
  - state persistence was recovered from an implementation audit;
  - time constant and Gaussian noise are mechanistic or generic-disruption controls.

## Perturbations and their exact implementation

### P1 — Synaptic-drive gain

- Task application:
  - circular task: evaluated on clean and distractor conditions;
  - N-back task: evaluated on both 0-back and 2-back.
- Implementation:
  - multiply both input-weight and recurrent-weight contributions by one scalar \(g\);
  - leave input and recurrent biases unscaled;
  - apply the gain before the `tanh` nonlinearity.

  \[
  h_t=\tanh\left[(1-\alpha)h_{t-1}
  +\alpha\left(g(W_{\mathrm{in}}x_t+W_{\mathrm{rec}}h_{t-1})
  b_{\mathrm{in}}+b_{\mathrm{rec}}\right)\right].
  \]

- Evaluated strengths: \(0.90, 0.95, 1.00, 1.05, 1.10\).
- Motivation:
  - Herzog et al. (2023) provide the closest computational precedent through receptor-dependent response-gain modulation.
- Boundary:
  - this is synaptic-drive gain in this particular CTRNN;
  - it is not an exact neuronal \(F\)-\(I\) slope manipulation or literal receptor model.

### P2 — Heterogeneous drive gain

- Task application:
  - circular task: evaluated on clean and distractor conditions;
  - N-back task: evaluated on both 0-back and 2-back.
- Implementation:
  - use the same weight-derived drive as P1;
  - give each of the 64 hidden units its own fixed positive gain;
  - draw gains from a log-normal distribution;
  - normalise each vector to an exact population mean of one;
  - leave learned biases outside the gain.

  \[
  h_t=\tanh\left[(1-\alpha)h_{t-1}
  +\alpha\left(\mathbf{g}\odot
  (W_{\mathrm{in}}x_t+W_{\mathrm{rec}}h_{t-1})
  b_{\mathrm{in}}+b_{\mathrm{rec}}\right)\right].
  \]

- Evaluated log-standard deviations: \(0.00, 0.10, 0.20, 0.30\).
- Three fixed gain vectors were averaged within each checkpoint:
  - these are technical replicates;
  - they are not additional independently trained networks.
- Motivation:
  - Herzog et al.'s receptor-density dependence motivates testing spatially nonuniform rather than purely global gain.
- Boundary:
  - the unit-wise gains are abstract;
  - they are not mapped to measured receptor densities.

### P3a — Sensory-input gain

- Task application:
  - circular task: evaluated on clean and distractor conditions;
  - N-back task: evaluated on both 0-back and 2-back.
- Implementation:
  - split input into stimulus and control channels;
  - multiply only the stimulus-channel weight contribution by \(g\);
  - leave fixation or N-back rule-context channels unchanged;
  - leave recurrent input unchanged.

  \[
  z_t^{\mathrm{in}}=
  gW_{\mathrm{in}}^{\mathrm{stim}}x_t^{\mathrm{stim}}
  W_{\mathrm{in}}^{\mathrm{ctrl}}x_t^{\mathrm{ctrl}}
  b_{\mathrm{in}}.
  \]

- Circular family:
  - scales the 32 tuned angle channels;
  - does not scale fixation.
- N-back family:
  - scales the six stimulus-identity channels;
  - does not scale the two rule-context channels.
- Evaluated strengths: \(0.80, 0.90, 1.00, 1.10, 1.20\).
- Motivation:
  - task-level translation of altered afferent or bottom-up influence under REBUS.
- Boundary:
  - REBUS does not specify this exact RNN operator.

### P3b — Distractor-window input gain

- Task application:
  - circular task: evaluated only when the irrelevant cue was present;
  - clean circular trials: operator inactive;
  - N-back task: not applied because there is no registered distractor window.
- Implementation:
  - use the same sensory-input scaling as P3a;
  - activate it only during the five-step irrelevant circular-cue window;
  - use native gain at every other timestep.
- Evaluated strengths: \(1.00, 1.10, 1.25, 1.50\).
- Scope:
  - circular distractor trials only;
  - inactive on clean circular trials;
  - no N-back equivalent because that task has no registered distractor window.
- Motivation:
  - direct manipulation check for sensitivity to irrelevant input, informed by Carter et al. (2005).
- Boundary:
  - it is not a complete cross-task candidate mechanism.

### P4 — Recurrent gain

- Task application:
  - circular task: evaluated on clean and distractor conditions;
  - N-back task: evaluated on both 0-back and 2-back.
- Implementation:
  - multiply only \(W_{\mathrm{rec}}h_{t-1}\) by \(g\);
  - leave input drive unchanged;
  - leave recurrent bias unscaled.

  \[
  h_t=\tanh\left[(1-\alpha)h_{t-1}
  +\alpha\left(W_{\mathrm{in}}x_t+b_{\mathrm{in}}
  gW_{\mathrm{rec}}h_{t-1}+b_{\mathrm{rec}}\right)\right].
  \]

- Evaluated strengths: \(0.90, 0.95, 1.00, 1.05, 1.10\).
- Motivation:
  - recurrent excitation is a general candidate substrate for working-memory maintenance.
- Boundary:
  - this is primarily a working-memory circuit contrast;
  - it is not directly derived from evidence about psilocybin.

### P5 — Gaussian state noise

- Task application:
  - current candidate screen: not applied to either task family;
  - planned matched-cost experiment: intended for both the circular and N-back
    families.
- Implementation:
  - add seeded, independent Gaussian noise to the pre-activation update before `tanh`;
  - noise is independently sampled across time, trials, and hidden units.

  \[
  h_t=\tanh\left[(1-\alpha)h_{t-1}+\alpha z_t
  +\sigma\varepsilon_t\right],
  \qquad \varepsilon_t\sim\mathcal{N}(0,I).
  \]

- Status:
  - implemented but not evaluated in this candidate-only screen;
  - planned as the generic-disruption comparator;
  - its magnitude will be calibrated to match the clean-task cost of persistence \(0.95\).
- Boundary:
  - it is not a model of psilocybin.

### P6 — State persistence

- Task application:
  - circular task: evaluated on clean and distractor conditions;
  - N-back task: evaluated on both 0-back and 2-back.
- Implementation:
  - multiply only the carried-state coefficient \((1-\alpha)\) by persistence scale \(p\);
  - leave the incoming-drive coefficient \(\alpha\) unchanged;
  - this deliberately breaks the coefficient sum of the native leaky integrator.

  \[
  h_t=\tanh\left[p(1-\alpha)h_{t-1}+\alpha z_t\right].
  \]

- With the native \(\alpha=0.2\):
  - native carried-state coefficient: \(0.80\);
  - at \(p=0.95\): \(0.95\times0.80=0.76\);
  - drive coefficient remains \(0.20\).
- Evaluated strengths: \(0.90, 0.95, 1.00, 1.05, 1.10\).
- Provenance:
  - this operator was not derived from evidence that psilocybin reduces neural persistence;
  - an implementation audit showed that an earlier whole-update manipulation labelled “response gain” was mainly altering carried-state retention;
  - the recovered operator was therefore retained under the accurate name **state persistence** as an operational competing hypothesis.
- Interpretation:
  - general working-memory literature makes persistence computationally interpretable;
  - the reason it is now interesting for psilocybin-related behaviour is that modifying it reproduced selected behavioural contrasts in the screen;
  - that is a model-generated candidate explanation, not evidence that psilocybin biologically acts on persistence.

### P7 — Conserved effective time constant

- Task application:
  - circular task: evaluated on clean and distractor conditions;
  - N-back task: evaluated on both 0-back and 2-back.
- Implementation:
  - rescale the effective leak factor as

    \[
    \alpha'=\min(\alpha/s,1);
    \]

  - update both carried-state and incoming-drive coefficients together:

    \[
    h_t=\tanh\left[(1-\alpha')h_{t-1}+\alpha'z_t\right].
    \]

- Evaluated strengths: \(0.80, 0.90, 1.00, 1.10, 1.25\).
- Interpretation:
  - \(s>1\) produces a slower effective timescale;
  - \(s<1\) produces a faster effective timescale;
  - unlike P6, the two coefficients still sum to one.
- Purpose:
  - mechanistic control for P6;
  - tests whether persistence results are explained by an ordinary conserved timescale change.
- Boundary:
  - it is not presented as a psychedelic parameter.

## Main perturbation result

- State persistence \(0.95\) was the only operator-strength profile to meet the complete descriptive cross-task majority rule against native baseline.
- Circular family:
  - clean delay-20 error increased by approximately \(4.9\%\) on average;
  - preservation ceiling was passed in all five checkpoints;
  - restricted settling increased by \(0.104\) model steps on average;
  - settling was slower in four of five checkpoints;
  - long-minus-short delay selectivity was positive in four of five;
  - distractor-minus-clean selectivity was positive in four of five.
- N-back family:
  - mean 2-back-minus-0-back impairment contrast: \(0.250\);
  - positive in all ten independently trained checkpoints.
- Strength of evidence:
  - N-back load selectivity was consistent across all ten checkpoints;
  - circular effects had the predicted mean directions and majority agreement;
  - circular effects were small and their checkpoint-level intervals included zero.
- Confirmation boundary:
  - persistence \(0.95\) was nominated in an earlier three-checkpoint pilot;
  - the circular family is a new distractor-trained retest;
  - three of the ten N-back checkpoints overlap with the nomination pilot;
  - this is therefore a descriptive candidate result, not a fully blind confirmation.

## What the result does and does not mean

- Supported model-level statement:
  - reducing carried-state persistence reproduced selected psilocybin-associated behavioural contrasts in these RNN task families;
  - altered persistence is therefore a computationally sufficient candidate within the model.
- Not supported:
  - psilocybin reduces neural state persistence;
  - 5-HT2A activation maps onto \(p=0.95\);
  - persistence is a unique explanation;
  - persistence is more selective than generic disruption.
- Current value of the result:
  - it nominates one concrete intervention for matched-control and hidden-dynamics analysis.

## Main limitations

- N-back evidence is substantially clearer than the circular evidence.
- Circular settling, delay, and distractor effects are small.
- Model settling time is not literal human reaction time.
- The circular task tests a filtering principle but does not reproduce Carter et al.'s multiple-object tracking task.
- The majority rule is descriptive, not a corrected confirmatory statistical family.
- Perturbation strengths were fixed discrete values rather than matched for overall impairment.
- Three N-back checkpoints overlap with the earlier candidate-nomination pilot.
- The matched-cost Gaussian comparison has not yet been run.
- The current behavioural screen does not establish:
  - altered attractor geometry;
  - manifold drift;
  - changed local stability;
  - altered distractor-recovery dynamics.

## Next steps

- First: compare persistence \(0.95\) with matched-cost Gaussian state noise.
  - calibrate Gaussian magnitude on separate held-out clean-task samples;
  - match overall clean-task impairment;
  - test whether persistence produces greater settling, load, delay, or distractor selectivity.
- Second: analyse hidden dynamics on identical banked trials.
  - delay-period representational drift;
  - hidden-state norm and speed;
  - local Jacobian eigenvalues around delay states;
  - distractor-induced displacement and recovery trajectories.
- Third: evaluate denser persistence values around \(0.95\).
  - determine whether the pattern is locally stable;
  - avoid interpreting an isolated grid point as a robust parameter region.
- Fourth: if the candidate survives specificity testing, formulate a new empirical hypothesis rather than retrospectively treating persistence as literature-derived.

## Questions for discussion

- Does combining the circular and N-back task families seem defensible if each task-specific contrast remains explicit?
- Is matched clean-task cost against Gaussian state noise the right next specificity test?
- Given the consistent N-back result but weak circular effects, should persistence be prioritised for deeper analysis or treated as insufficiently supported?
- Is the distinction clear between:
  - perturbations motivated before the screen by literature;
  - state persistence as an operational hypothesis recovered through implementation audit;
  - behavioural resemblance as the reason persistence is now relevant?

## Under-one-minute fallback

- I added a distractor-trained circular task and a context-cued 0-back/2-back task to test settling, delay, filtering, and updating-load contrasts.
- I screened frozen-weight interventions targeting synaptic drive, heterogeneous unit gain, sensory input, distractor input, recurrent drive, carried-state persistence, and effective timescale.
- A small reduction in carried-state persistence was the only profile to meet the full descriptive cross-task rule against baseline.
- That operator was not derived from a biological claim about psilocybin:
  - it was recovered from an implementation audit;
  - modifying it subsequently reproduced selected psilocybin-associated behavioural effects.
- The N-back effect was consistent, but the circular effects were small.
- I am treating persistence as a computational candidate, not a confirmed or biological mechanism.
- The next step is a matched-cost Gaussian comparison followed by hidden-state stability and recovery analyses.

## References mentioned

- Barrett et al. (2018), *Psychopharmacology*: `10.1007/s00213-018-4981-x`.
- Carhart-Harris and Friston (2019), *Pharmacological Reviews*: `10.1124/pr.118.017160`.
- Carter et al. (2005), *Journal of Cognitive Neuroscience*: `10.1162/089892905774597191`.
- Ghazizadeh and Ching (2021), *PLoS Computational Biology*: `10.1371/journal.pcbi.1009366`.
- Herzog et al. (2023), *Scientific Reports*: `10.1038/s41598-023-32649-7`.
- Wang et al. (2013), *Neuron*: `10.1016/j.neuron.2012.12.032`.
