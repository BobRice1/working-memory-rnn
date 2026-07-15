# Structured Noise Perturbation Protocol

## Status and scope

This protocol evaluates frozen `yang_fixation_circular_working_memory` models. It is a controlled modelling experiment, not a literal simulation of psilocybin, receptor pharmacology, dose, or human working-memory behaviour.

Decision recorded 2026-07-14: the completed temporal AR(1) and recurrent-
topology-covariance conditions are retained as a closed exploratory robustness
study but are benched from future hypothesis-testing experiments. Independent
Gaussian noise remains the generic stochastic control for comparisons with
input/distractor-gain and recurrent-stability or gain perturbations.

The four conditions are an unperturbed reference, independent Gaussian control, stationary AR(1) noise (`rho=0.9`), and AR(1) noise transformed by normalized recurrent topology covariance `0.25 I + 0.75 W_rec W_rec^T`. Realized RMS is normalized over active times, trials, and units. Main perturbations are delay-only; cue-only and response-only topology timing are secondary context controls.

## Design

- Frozen seeds: `20260714`–`20260718`.
- Delays: `20`, `80`, `160`.
- RMS: `0`, `0.01`, `0.025`, `0.05`, `0.10`.
- Five stochastic replicates and 20 paired 64-trial batches.
- Clean sine/cosine ridge decoder: 512 independent trials per seed and delay; fitting and inference occur on CUDA.
- Statistical unit: model seed. Trials and stochastic repeats are nested measurements, not independent inferential replicates.
- Checkpoint SHA-256 hashes are recorded before the run and verified unchanged afterward.

## Outcomes

Primary outcomes are response angular error, clean-baseline decoder error during delay, decoded drift, and fixation accuracy. Secondary outcomes include trajectory speed, within-angle dispersion, between-angle separation, covariance participation ratio, a normalized binary-transition diversity proxy, and post-perturbation recovery distance. The diversity value currently implemented is a lightweight smoke-compatible proxy; dissertation claims about Lempel–Ziv complexity require replacement with a validated LZ implementation before the full inferential run.

## Commands

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m wm_rnn.noise_structure_experiment
```

Reduced acceptance smoke run:

```powershell
.\.venv\Scripts\python.exe -m wm_rnn.noise_structure_experiment --seeds 20260714 --delays 20 --strengths 0 0.05 --replicates 1 --batches 1 --batch-size 16 --decoder-trials 64 --phases cue delay response --output outputs\yang_fixation_circular_working_memory\perturbation_experiments\noise_structure\smoke
```

Research figure generation is a separate analysis step. For the current
dissertation results layout:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m wm_rnn.noise_structure_dissertation_figures
```

The denser earlier analysis suite can still be regenerated with
`python -m wm_rnn.noise_structure_figures --reuse-hidden` after restoring or
pointing at archived `figure_data/`. That suite is preserved under
`outputs/archive/noise_structure_initial_figure_suite_2026-07-14/`.

## Interpretation boundary

Herzog et al. (2023) directly support topology-sensitive entropy changes in a whole-brain model; they do not prescribe this RNN covariance. Stoliker et al. (2026) directly motivate context sensitivity and cohesive trajectories, but not additive hidden-state noise. Bredenberg et al. (2026) motivate a structured-versus-noise control logic in neural-network models, not this working-memory perturbation. Schartner et al. (2017) motivate temporal signal-diversity measurement. Carhart-Harris et al. (2014) motivate entropy/reduced-constraint hypotheses. The AR coefficient, topology mixture, phase masks, strengths, and decoder design are project assumptions.

Structured reorganization would be supported by increased diversity with preserved decoding/response accuracy, cohesive geometry, or context-specific effects. Monotonic dispersion with degraded decoding and performance would instead indicate ordinary representation-destroying noise.
