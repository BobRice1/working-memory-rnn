# Working Memory RNN

This repository contains the first build-phase baseline model for the dissertation project: a continuous-time PyTorch RNN trained on a categorical delayed-response working-memory task.

The baseline is intentionally narrow. It trains a simple leaky/rate RNN, evaluates task performance, and plots hidden-state trajectories with PCA. It does not yet implement psilocybin-informed perturbations, E/I constraints, modular structure, N-back updating, or fixed-point analysis.

## Repository Layout

- `src/wm_rnn/`: package code for task generation, model, training, evaluation, and analysis.
- `configs/baseline_delay.yaml`: default baseline configuration.
- `docs/build-log.md`: build-process notes and decisions.
- `outputs/`: generated checkpoints, metrics, arrays, and figures. This directory is ignored by git.

## Environment Setup

Create and activate a virtual environment from this repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

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

Check the current install with:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
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

Evaluation writes accuracy metrics and a confusion matrix under `outputs/baseline_delay/metrics/`.

## Delay-Length Sweep

Evaluate a trained checkpoint across longer or shorter delay periods without retraining:

```powershell
python -m wm_rnn.delay_sweep --config configs/baseline_delay.yaml --checkpoint outputs/baseline_delay/checkpoints/baseline_delay.pt --delays 5 10 20 40 80 120
```

The sweep keeps the trained weights frozen, changes only `task.delay_steps` at evaluation time, and writes JSON/CSV metrics under `outputs/baseline_delay/metrics/`.
It also writes a plot under `outputs/baseline_delay/figures/`.

## Multiple Training Seeds

Train and evaluate independent baseline models across several seeds:

```powershell
python -m wm_rnn.seed_sweep --config configs/baseline_delay.yaml --seeds 20260629 20260630 20260631 --delays 20 25 30 35 40 50 60 70 80
```

Each seed gets its own output directory under `outputs/baseline_delay/seed_sweep/`.
When `--delays` is provided, the command also runs a frozen-weight delay sweep for each trained seed and saves each seed's sweep plot.

Plot all seed delay sweeps on one graph:

```powershell
python -m wm_rnn.plot_seed_sweeps --summary outputs/baseline_delay/metrics/baseline_delay_seed_sweep_summary.json
```

The combined plot is written to `outputs/baseline_delay/figures/baseline_delay_seed_sweep_delay_curves.png`.

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

## Dissertation Boundary

This repository implements the baseline working-memory model only. Any later psilocybin-informed condition should be documented as a modelling hypothesis and grounded separately in the dissertation wiki.
