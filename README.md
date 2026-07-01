# Working Memory RNN

Continuous-time RNN (`tanh` hidden-state nonlinearity by default) trained on a categorical delayed-response working-memory task.

## Repository Layout

- `src/wm_rnn/`: package code for task generation, model, training, evaluation, and analysis.
- `configs/baseline_delay.yaml`: default baseline configuration.
- `docs/build-log.md`: build-process notes and decisions.
- `outputs/`: generated checkpoints, metrics, arrays, and figures.

For CUDA training on a GPU, install the CUDA requirements first:

```powershell
python -m pip install -r requirements-cuda.txt
python -m pip install -e . --no-deps --no-build-isolation
```

For CPU-only setup, use:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps --no-build-isolation
```

The code defaults to CUDA when available and falls back to CPU otherwise.

## Train

Run the default baseline training job:

```powershell
python -m wm_rnn.train --config configs/baseline_delay.yaml
```

Force CPU or CUDA explicitly:

```powershell
python -m wm_rnn.train --config configs/baseline_delay.yaml --device cpu
python -m wm_rnn.train --config configs/baseline_delay.yaml --device cuda
```

Training writes a checkpoint and training metrics under `outputs/baseline_delay/`.

## Evaluate

After training, evaluate the saved checkpoint:

```powershell
python -m wm_rnn.evaluate --config configs/baseline_delay.yaml --checkpoint outputs/baseline_delay/checkpoints/baseline_delay.pt
```

## PCA Analysis

Run hidden-state PCA trajectory analysis:

```powershell
python -m wm_rnn.analysis --config configs/baseline_delay.yaml --checkpoint outputs/baseline_delay/checkpoints/baseline_delay.pt
```

Analysis writes hidden-state arrays and a PCA trajectory figure under `outputs/baseline_delay/`.

## Hidden-State Stability Analysis

Check whether a trained checkpoint's hidden state settles (attractor-like)
or keeps changing/growing (ramping or phasic) during the delay period:

```powershell
python -m wm_rnn.stability_analysis --config configs/baseline_delay.yaml --checkpoint outputs/baseline_delay/checkpoints/baseline_delay.pt
```

This writes a two-panel figure (hidden-state norm and step-to-step speed
across trial time) and a JSON summary with phase-averaged values and a
"delay settling ratio" under `outputs/baseline_delay/`. A ratio well below
`1` indicates settling; a ratio near or above `1` indicates the hidden state
is still changing at a similar or increasing rate through the delay.

## Model Variants

- `configs/baseline_delay.yaml`: **the current baseline.** Fixed `20`-step
  training delay, loss scored only on the response period, `model.activation:
  tanh`. Outputs live under `outputs/baseline_delay/`. This became the
  baseline after hidden-state stability analysis showed it settles during
  the delay period, using the plain, original training setup. A four-seed
  sweep found this generalization is strong and typical but seed-dependent:
  some seeds hold perfect accuracy to at least four times the trained delay
  length, others settle to a lower-but-still-well-above-chance accuracy
  beyond roughly twice the trained length; every seed clearly outperforms
  every archived `relu` seed at long delays (see `docs/changelog.md`).
- `configs/baseline_delay_relu.yaml`: the **archived original baseline**,
  identical in every other respect but with `model.activation: relu`. Kept
  for comparison and reproducibility; its previously recorded outputs are
  archived under `outputs/baseline_delay_relu/` (including the original
  multi-seed sweep). This is the `relu` network whose unbounded hidden-state
  growth and delay-generalization failure motivated the switch to `tanh`.
- `configs/baseline_delay_stable.yaml`: an `relu`-based variant that trains
  with a randomized delay length (`15`-`45` steps) and scores the loss
  across the delay period plus the response period, to test whether that
  produces more settled hidden-state dynamics without changing the
  activation function. Outputs live under `outputs/baseline_delay_stable/`.
  It helped, but did not settle as fully as switching to `tanh` did; see
  `docs/changelog.md` for the three-way comparison. Whether to re-test this
  training-objective change on top of the new `tanh` baseline is an open
  question, also logged in the changelog.

The hidden-state activation is a configurable field (`model.activation:
relu` or `tanh`), not a hardcoded choice, so any of the above variants can
be reproduced or extended from its config file. See `docs/changelog.md` for
the full recorded reasoning and before/after results across all variants.

## Reproducible Baseline Run

From a fresh environment:

```powershell
python -m wm_rnn.train --config configs/baseline_delay.yaml
python -m wm_rnn.evaluate --config configs/baseline_delay.yaml --checkpoint outputs/baseline_delay/checkpoints/baseline_delay.pt
python -m wm_rnn.analysis --config configs/baseline_delay.yaml --checkpoint outputs/baseline_delay/checkpoints/baseline_delay.pt
```

The default task seed is fixed in `configs/baseline_delay.yaml`. Training batches use deterministic seed offsets by step, so repeated runs with the same PyTorch version and device should be broadly reproducible.
