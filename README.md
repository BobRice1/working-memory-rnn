# Working Memory RNN

Continuous-time ReLU RNN trained on a categorical delayed-response working-memory task.

## Repository Layout

- `src/wm_rnn/`: package code for task generation, model, training, evaluation, and analysis.
- `configs/baseline_delay.yaml`: default baseline configuration.
- `docs/build-log.md`: build-process notes and decisions.
- `outputs/`: generated checkpoints, metrics, arrays, and figures.

For CUDA training on the RTX 3060 notebook GPU, install the CUDA requirements first:

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

## Reproducible Baseline Run

From a fresh environment:

```powershell
python -m wm_rnn.train --config configs/baseline_delay.yaml
python -m wm_rnn.evaluate --config configs/baseline_delay.yaml --checkpoint outputs/baseline_delay/checkpoints/baseline_delay.pt
python -m wm_rnn.analysis --config configs/baseline_delay.yaml --checkpoint outputs/baseline_delay/checkpoints/baseline_delay.pt
```

The default task seed is fixed in `configs/baseline_delay.yaml`. Training batches use deterministic seed offsets by step, so repeated runs with the same PyTorch version and device should be broadly reproducible.
