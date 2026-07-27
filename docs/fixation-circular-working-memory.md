# Fixation-Gated Circular Working-Memory RNN

## Role

`fixation_circular_working_memory` is the canonical circular baseline for
perturbation experiments. The task uses explicit fixation control and a
circular remembered variable.

## Architecture and Training

- 32 circularly tuned input and output units.
- One fixation input and one fixation output.
- 64-unit leaky continuous-time `tanh` RNN.
- Randomized pre-cue, cue, and 10/20/40/80-step delay timing.
- A 25-step response with an unscored 5-step transition.
- Weighted circular/fixation MSE, input and recurrent training noise, and
  gradient clipping.

The active configuration is
`configs/fixation_circular_working_memory.yaml` and generated artifacts are
stored under `outputs/fixation_circular_working_memory/`.

## Hidden-State Decoder

The circular output is intentionally silent before response. Memory content
during fixation and delay is therefore measured using cross-temporal ridge
decoding from hidden state to the sine and cosine of the target angle. Decoders
are fitted independently at every training time and evaluated at every test
time on held-out trials.

For the original canonical checkpoint, mean delay-period diagonal decoding
error is `0.427` degrees and the maximum delay-time error is `0.474` degrees.

## Independent-Seed Baseline

Five full 4,000-step models were trained with seeds `20260714` through
`20260718`. Across seeds:

- response angular error: `4.36 ± 1.55` degrees (range `2.95–7.03`);
- response population MSE: `0.00957 ± 0.00194`;
- fixation accuracy: `0.9677 ± 0.0016`;
- delay-period hidden-state decoding error: `0.55 ± 0.46` degrees (range
  `0.31–1.37`).

At the untrained 160-step delay, seed-level mean response errors ranged from
`5.14` to `11.73` degrees. These runs demonstrate that performance and hidden
memory decoding are reproducible across independently initialized networks,
while also quantifying meaningful between-seed variation.

## Reproduction Status

The trained checkpoint family and its baseline analysis artifacts are retained
under `outputs/fixation_circular_working_memory/`. The one-off PCA, delay-sweep,
cross-temporal, and generic seed-sweep modules were retired from the active
package during repository cleanup; their exact implementations remain
recoverable from Git commit `d64b2cf`. Current evaluation is performed by the
candidate perturbation runners documented in `docs/repository-map.md`.
