# Build Log

## Purpose

Create the first self-contained baseline working-memory RNN repository for the dissertation build phase.

## Repository Initialization

- Project folder: `working-memory-rnn/`.
- Git scope: this folder only, not the whole Obsidian vault.
- Generated artifacts are written to `outputs/` and ignored by git.

## Environment Decisions

- Language: Python.
- Model framework: PyTorch.
- Test runner: pytest.
- Analysis stack: NumPy, scikit-learn PCA, matplotlib.
- Config format: YAML.
- Environment: project-local `.venv`.
- Device handling: `auto` selects CUDA when `torch.cuda.is_available()` is true; otherwise CPU is used. The CLI also accepts `--device cpu` and `--device cuda`.
- CUDA setup: `requirements-cuda.txt` installs `torch==2.10.0+cu126` for the RTX 3060 path.

## Task Design

The first task is a categorical delayed-response working-memory task:

- Cue period: one-hot class cue is visible.
- Delay period: class input is removed.
- Response period: the model reports the remembered class.
- A constant fixation/context input channel is present at every step.
- Loss is scored only during the response period.

This matches the vault's recommendation to start with a simple delayed-response task before adding distractors or N-back updating.

## Model Design

The model is a continuous-time ReLU RNN with a linear readout:

- recurrent update uses `alpha = dt / tau`;
- hidden state is nonnegative after ReLU;
- output logits are produced at every time step;
- cross-entropy is masked to response steps only.

This follows the professor-provided notebook's core pattern while keeping the code modular and testable.

## Commands

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Train:

```powershell
.\.venv\Scripts\python.exe -m wm_rnn.train --config configs/baseline_delay.yaml
```

Evaluate:

```powershell
.\.venv\Scripts\python.exe -m wm_rnn.evaluate --config configs/baseline_delay.yaml --checkpoint outputs/baseline_delay/checkpoints/baseline_delay.pt
```

PCA analysis:

```powershell
.\.venv\Scripts\python.exe -m wm_rnn.analysis --config configs/baseline_delay.yaml --checkpoint outputs/baseline_delay/checkpoints/baseline_delay.pt
```

## Known Limitations

- No distractor condition yet.
- No N-back or sequence-updating task yet.
- No psilocybin-inspired perturbation yet.
- No E/I constraint or Dale's principle yet.
- No fixed-point or Jacobian analysis yet.

## Verification Record

Verified on 2026-06-29 through the project-local `.venv`:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Observed:

```text
2.10.0+cu126
True
NVIDIA GeForce RTX 3060 Laptop GPU
```

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Observed:

```text
9 passed
```

Default training selected CUDA and reached response-period accuracy of 1.000 by the end of the run. Evaluation on generated held-out batches reported accuracy of 1.000. PCA analysis wrote `outputs/baseline_delay/figures/baseline_delay_pca_trajectories.png`.

## Next Build Steps

1. Confirm the baseline trains above chance across several seeds.
2. Add a delayed-response-with-distractors task.
3. Add delay-duration sweeps.
4. Add fixed-point and Jacobian analysis after the baseline dynamics are stable.
5. Define psilocybin-informed perturbations only after documenting the literature-to-parameter mapping.
