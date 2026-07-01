# Working Memory RNN

Continuous-time ReLU RNN trained on a categorical delayed-response working-memory task.

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

- `configs/baseline_delay.yaml`: the original baseline. Fixed `20`-step
  training delay, loss scored only on the response period. Outputs live
  under `outputs/baseline_delay/`.
- `configs/baseline_delay_stable.yaml`: same architecture, but trains with a
  randomized delay length (`15`-`45` steps) and scores the loss across the
  delay period plus the response period, to test whether that produces more
  settled hidden-state dynamics. Outputs live under
  `outputs/baseline_delay_stable/`, in the same `checkpoints/metrics/figures/arrays`
  layout, kept fully separate from the original baseline's outputs.
- `configs/baseline_delay_tanh.yaml`: identical training setup to the
  original baseline (fixed `20`-step delay, response-period-only loss,
  `1000` steps), but with `model.activation: tanh` instead of the default
  `relu`, to test whether bounding the hidden-state nonlinearity alone
  produces settled dynamics. Outputs live under `outputs/baseline_delay_tanh/`.
  The hidden-state activation is a configurable field
  (`model.activation: relu` or `tanh`) rather than a hardcoded choice, so
  either setting can be reproduced from its config file.

Run any of the commands above against `configs/baseline_delay_stable.yaml`
and its checkpoint to reproduce or extend that comparison. See
`docs/changelog.md` for the recorded reasoning and before/after results.

## Reproducible Baseline Run

From a fresh environment:

```powershell
python -m wm_rnn.train --config configs/baseline_delay.yaml
python -m wm_rnn.evaluate --config configs/baseline_delay.yaml --checkpoint outputs/baseline_delay/checkpoints/baseline_delay.pt
python -m wm_rnn.analysis --config configs/baseline_delay.yaml --checkpoint outputs/baseline_delay/checkpoints/baseline_delay.pt
```

The default task seed is fixed in `configs/baseline_delay.yaml`. Training batches use deterministic seed offsets by step, so repeated runs with the same PyTorch version and device should be broadly reproducible.
