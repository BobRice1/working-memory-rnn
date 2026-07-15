# Structured Noise Experiment: Full Results

## Scope and provenance

The full frozen-model experiment used five independently trained Yang-style
checkpoints, delays `20/80/160`, RMS strengths `0/0.01/0.025/0.05/0.10`, five
stochastic replicates, 20 paired 64-trial batches, and independent 512-trial
clean ridge-decoder fits per seed and delay. PyTorch `2.10.0+cu126` ran on the
NVIDIA GeForce RTX 3060 Laptop GPU with CUDA `12.6`. All five checkpoint hashes
were unchanged after execution.

The main dataset contains 30,000 condition rows. A separate 30,000-row timing
dataset evaluates cue-, delay-, and response-period perturbations at delay 80.
Values below are mean ± SD across the five seed-level means; trials, batches,
and stochastic replicates were not treated as independent inferential units.

## Main results

At RMS `0.05`, response error increased with perturbation structure and delay:

| Delay | Unperturbed | Independent | AR(1) | Topology-correlated AR(1) |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 4.34 ± 1.54° | 4.91 ± 1.45° | 8.14 ± 1.28° | 14.11 ± 1.48° |
| 80 | 6.49 ± 1.93° | 7.47 ± 1.68° | 16.33 ± 2.71° | 32.43 ± 4.75° |
| 160 | 9.03 ± 2.50° | 10.44 ± 2.24° | 23.16 ± 3.86° | 45.70 ± 5.82° |

The clean-baseline decoder showed the same ordering. At RMS `0.05`, delay
decoder error at delay 80 was `1.29 ± 0.82°` unperturbed, `3.71 ± 1.53°`
independent, `10.59 ± 2.89°` AR(1), and `20.15 ± 3.39°` topology-correlated.
At delay 160 the corresponding values were `2.32 ± 1.05°`, `5.93 ± 1.79°`,
`16.84 ± 4.13°`, and `31.12 ± 4.89°`.

At RMS `0.10`, topology-correlated perturbations produced response errors of
`29.53 ± 3.63°`, `64.61 ± 4.58°`, and `80.97 ± 2.39°` at delays 20, 80, and
160. This was substantially worse than independent noise (`6.42 ± 1.35°`,
`10.48 ± 1.92°`, `14.68 ± 2.57°`) at the same realized energy.

Decoded drift also followed independent < AR(1) < topology-correlated. At
delay 80 and RMS `0.05`, drift was `2.60 ± 1.84°` unperturbed,
`5.20 ± 2.11°` independent, `15.68 ± 3.89°` AR(1), and `31.69 ± 5.25°`
topology-correlated.

Fixation accuracy was comparatively preserved at RMS `0.05`. Clear fixation
loss emerged mainly at RMS `0.10` under structured perturbations; for example,
at delay 160 topology-correlated fixation accuracy was `0.953 ± 0.011` versus
`0.987 ± 0.000` unperturbed.

## Timing sensitivity

Topology-correlated effects were strongly phase-dependent at delay 80. At RMS
`0.05`, response error was `7.43 ± 1.65°` for cue-only, `32.12 ± 4.49°` for
delay-only, and `11.60 ± 1.64°` for response-only perturbation. Decoder error
was `3.74 ± 0.35°`, `19.96 ± 3.34°`, and `1.29 ± 0.82°`, respectively.
Response-only noise altered the output while leaving the earlier delay state
unchanged; delay-only noise disrupted both maintained content and response.

At RMS `0.10`, response errors were `9.89 ± 1.40°` cue-only,
`65.24 ± 4.41°` delay-only, and `22.22 ± 3.36°` response-only. The timing
result operationalizes context dependence within the task, but it is not direct
evidence about human psilocybin context effects.

## Manipulation validation

- Zero-strength mismatches across all primary outcomes: `0`.
- Maximum realized-RMS error: `<1.7e-8`.
- Maximum perturbation magnitude outside the active phase: `0`.
- Mean lag-1 correlation: independent `-0.00003`, AR(1) `0.90000`,
  topology-correlated AR(1) `0.90004`.
- Non-finite primary values: `0`.

## Interpretation

The first-order result supports representation-destroying structured noise, not
beneficial structured reorganization: greater temporal and topology alignment
increased decoder error, drift, and behavioural error at matched RMS. The
topology-correlated condition was not protective and did not identify a
diversity-with-preserved-memory regime in the primary outcomes.

This does not imply that psilocybin is representation-destroying noise. The
perturbations are project-defined interventions inspired by questions from
Herzog et al. (topology), Stoliker et al. (context and cohesive trajectories),
Bredenberg et al. (structured variability versus noise controls), Schartner et
al. (signal diversity), and Carhart-Harris et al. (entropy/reduced constraint).
None of those papers specifies this hidden-state noise model.

## Remaining analysis limits

The current `lzc_normalized` column is a binary-transition proxy and must not be
reported as validated Lempel–Ziv complexity. The current aggregate
cross-temporal and PCA-oriented figures remain provisional artifact-coverage
panels rather than the final time-by-time matrices and clean-PCA centroid
ellipse analyses in the protocol. The primary response, frozen-decoder, drift,
fixation, timing, RMS, autocorrelation, and phase-mask results above are based
on the completed full datasets.

## Outputs

- `outputs/yang_fixation_circular_working_memory/perturbation_experiments/noise_structure/full/summary_metrics.csv`
- `outputs/yang_fixation_circular_working_memory/perturbation_experiments/noise_structure/full/noise_diagnostics.csv`
- `outputs/yang_fixation_circular_working_memory/perturbation_experiments/noise_structure/epoch_timing_full/summary_metrics.csv`
- `outputs/yang_fixation_circular_working_memory/perturbation_experiments/noise_structure/epoch_timing_full/noise_diagnostics.csv`
