# Project Changelog

This changelog tracks two related histories:

1. Git commits: code and documentation changes committed to the repository.
2. Run log: experiment actions performed with the baseline model, including
   training, delay sweeps, multi-seed runs, and plotting.

## Git Commit History

<details>
<summary>2026-06-29 - ed92033 - feat: add baseline working-memory rnn</summary>

Initial repository build for the baseline delayed-response working-memory RNN.

File changes:

- `.gitignore`: Added Python, environment, checkpoint, output, and temporary-file ignores.
- `README.md`: Added setup, training, evaluation, PCA analysis, and reproducible-run instructions.
- `configs/baseline_delay.yaml`: Added the default baseline task, model, training, evaluation, analysis, and output-path configuration.
- `docs/build-log.md`: Added the original build record and implementation decisions. This file was later removed from Git tracking.
- `pyproject.toml`: Added package metadata, dependencies, setuptools configuration, and initial test configuration.
- `requirements-cuda.txt`: Added the CUDA-oriented dependency path.
- `requirements.txt`: Added the CPU/default dependency path.
- `src/wm_rnn/__init__.py`: Created the package namespace.
- `src/wm_rnn/analysis.py`: Added hidden-state PCA trajectory analysis for trained checkpoints.
- `src/wm_rnn/config.py`: Added default config and YAML config loading.
- `src/wm_rnn/device.py`: Added automatic CPU/CUDA device selection.
- `src/wm_rnn/evaluate.py`: Added checkpoint evaluation and confusion-matrix output.
- `src/wm_rnn/io.py`: Added output-directory and metrics-writing helpers.
- `src/wm_rnn/model.py`: Added the continuous-time ReLU RNN and linear readout.
- `src/wm_rnn/task.py`: Added the categorical cue-delay-response task generator.
- `src/wm_rnn/train.py`: Added the baseline training entry point.
- `src/wm_rnn/training_utils.py`: Added shared tensor conversion, model creation, masked loss, accuracy, and confusion-matrix utilities.
- `tests/*`: Added the original pytest coverage for config/device, model, task, and training pipeline behavior. These were later removed from Git.

</details>

<details>
<summary>2026-06-29 - df8dd04 - docs: add pydoc docstrings</summary>

Documentation-only cleanup across the Python package.

File changes:

- `src/wm_rnn/analysis.py`: Added docstrings for PCA analysis structures and functions.
- `src/wm_rnn/config.py`: Added docstrings for configuration loading and merging.
- `src/wm_rnn/device.py`: Added docstrings for device-selection helpers.
- `src/wm_rnn/evaluate.py`: Added docstrings for evaluation result structures and functions.
- `src/wm_rnn/io.py`: Added docstrings for output and metrics file helpers.
- `src/wm_rnn/model.py`: Added docstrings for RNN config, recurrent layer, and readout model.
- `src/wm_rnn/task.py`: Added docstrings for task config, generated batches, and task generation.
- `src/wm_rnn/train.py`: Added docstrings for training result structures and CLI flow.
- `src/wm_rnn/training_utils.py`: Added docstrings for shared training and evaluation utilities.

</details>

<details>
<summary>2026-07-01 - 71f9fd8 - Removed pytests from git, Added seed sweep for baseline model with plots to visualise memory degradation</summary>

Added the main delay-generalization and multi-seed analysis tooling, while removing the tracked pytest suite.

File changes:

- `README.md`: Added usage instructions for delay sweeps, multi-seed training, and combined seed-sweep plotting.
- `src/wm_rnn/delay_sweep.py`: Added frozen-checkpoint evaluation across multiple delay lengths, writing JSON, CSV, and a delay-accuracy plot.
- `src/wm_rnn/plot_seed_sweeps.py`: Added combined plotting for multiple seed-specific delay sweeps.
- `src/wm_rnn/seed_sweep.py`: Added multi-seed training, evaluation, optional delay sweeps, and seed-level summary outputs.
- `tests/test_config_and_device.py`: Removed from Git.
- `tests/test_model.py`: Removed from Git.
- `tests/test_task.py`: Removed from Git.
- `tests/test_training_pipeline.py`: Removed from Git.

</details>

<details>
<summary>2026-07-01 - dc54e4c - Revise README for</summary>

README simplification pass.

File changes:

- `README.md`: Removed a large amount of earlier README material and kept the documentation more focused on the project workflow.

</details>

<details>
<summary>2026-07-01 - bf7ccc6 - Update README for CUDA training instructions</summary>

Small CUDA setup documentation update.

File changes:

- `README.md`: Revised the CUDA training instructions.

</details>

<details>
<summary>2026-07-01 - 32bdec9 - removed build log</summary>

Removed the build log from Git history going forward.

File changes:

- `docs/build-log.md`: Deleted from Git tracking.

</details>

<details>
<summary>2026-07-01 - ce2dffa - Merge branch 'main' of https://github.com/BobRice1/working-memory-rnn</summary>

Merge commit.

File changes:

- No file-level changes were listed by `git log --name-status` for this merge commit.

</details>

<details>
<summary>2026-07-01 - f673574 - updated gitignore</summary>

Ignore-rule update.

File changes:

- `.gitignore`: Updated ignore patterns.

</details>

## Current Working Changes

<details>
<summary>Uncommitted documentation and configuration changes</summary>

These changes exist in the working tree at the time this changelog was written.

File changes:

- `docs/model-architecture.md`: Added a model architecture document, including a plain-English walkthrough of the cue-delay-response task and the recurrent model.
- `docs/changelog.md`: Added this changelog.
- `pyproject.toml`: Removed remaining pytest configuration from the package metadata.
- `docs/build-log.md`: Kept locally but removed from Git tracking and ignored.

</details>

<details>
<summary>Uncommitted: added hidden-state stability analysis</summary>

Reasoning:

- Discussion of whether the baseline model should be expected to show
  attractor-like (settled, tonic) hidden-state dynamics raised a testable
  question: does the trained checkpoint actually settle into a steady
  hidden-state during the delay period, or does it keep changing?
- The existing PCA trajectory analysis (`src/wm_rnn/analysis.py`) is
  descriptive of trajectory shape in 2D but does not directly measure
  whether the hidden state is settling (attractor-like) or continuing to
  move/grow (ramping or phasic) over time.
- The existing delay-length sweep already showed accuracy collapsing once
  the delay length exceeds the trained value (see the 2026-07-01 delay-sweep
  run log entry below), which is consistent with a time-locked or ramping
  solution rather than a stable attractor, but this was only inferred
  indirectly from behavior, not measured directly from the hidden state.
- This analysis was added as a fast, low-cost first diagnostic before
  committing to heavier dynamical-systems analysis (for example fixed-point
  or Jacobian analysis), and before deciding whether to change training to
  explicitly encourage more attractor-like dynamics.

File changes:

- `src/wm_rnn/stability_analysis.py`: Added hidden-state stability analysis.
  For a trained checkpoint, this loads a fresh analysis batch, runs the
  model, and computes two per-time-step quantities across trials: hidden-state
  norm (magnitude) and step-to-step speed (the size of the change in hidden
  state from one time step to the next). It reports phase-averaged norm and
  speed for the cue, delay, and response periods, a "delay settling ratio"
  (late-delay speed divided by early-delay speed), a two-panel figure of
  norm and speed across trial time with cue/delay/response periods shaded,
  and a JSON summary with an interpretation note explaining how to read the
  settling ratio.

</details>

## Chronological Run Log

<details>
<summary>2026-06-29 - Baseline model trained once</summary>

Action:

- Trained the default baseline model with `configs/baseline_delay.yaml`.

Configuration:

- Task: categorical delayed-response working-memory task.
- Trained delay length: `20` time steps.
- Number of cue classes: `4`.
- Hidden size: `64`.
- Training steps: `1000`.
- Device: CUDA on `NVIDIA GeForce RTX 3060 Laptop GPU`.

Recorded result:

- Final training accuracy: `1.0`.
- Final training loss: `1.415610206834117e-08`.
- Checkpoint: `outputs/baseline_delay/checkpoints/baseline_delay.pt`.
- Training history: `outputs/baseline_delay/metrics/baseline_delay_train_history.csv`.
- Training metrics: `outputs/baseline_delay/metrics/baseline_delay_train_metrics.json`.

</details>

<details>
<summary>2026-06-29 - Baseline checkpoint evaluated</summary>

Action:

- Evaluated the trained baseline checkpoint on held-out generated delay-task batches.

Recorded result:

- Evaluation accuracy: `1.0`.
- Evaluation batches: `20`.
- Metrics: `outputs/baseline_delay/metrics/baseline_delay_eval_metrics.json`.
- Confusion matrix: `outputs/baseline_delay/metrics/baseline_delay_confusion_matrix.csv`.

</details>

<details>
<summary>2026-06-29 - Hidden-state PCA analysis generated</summary>

Action:

- Ran PCA analysis on hidden-state trajectories from the trained checkpoint.

Recorded outputs:

- PCA summary: `outputs/baseline_delay/metrics/baseline_delay_pca_summary.json`.
- Hidden-state arrays: `outputs/baseline_delay/arrays/baseline_delay_hidden_states.npz`.
- PCA figure: `outputs/baseline_delay/figures/baseline_delay_pca_trajectories.png`.

</details>

<details>
<summary>2026-07-01 - Single-checkpoint delay sweep run</summary>

Action:

- Loaded the trained baseline checkpoint.
- Kept model weights frozen.
- Evaluated response accuracy across longer delay lengths without retraining.

Latest recorded sweep:

| Delay steps | Accuracy |
| ---: | ---: |
| 20 | 1.000 |
| 25 | 1.000 |
| 30 | 0.907 |
| 35 | 0.749 |
| 40 | 0.745 |
| 50 | 0.519 |
| 60 | 0.485 |
| 70 | 0.483 |
| 80 | 0.503 |

Interpretation:

- The model remains reliable around the trained delay length.
- Accuracy begins to degrade around `30` delay steps.
- The major transition occurs around `30-50` delay steps.

Recorded outputs:

- Metrics: `outputs/baseline_delay/metrics/baseline_delay_delay_sweep_metrics.json`.
- CSV: `outputs/baseline_delay/metrics/baseline_delay_delay_sweep.csv`.
- Figure: `outputs/baseline_delay/figures/baseline_delay_delay_sweep.png`.

</details>

<details>
<summary>2026-07-01 - Multiple seeds trained and swept</summary>

Action:

- Trained independent baseline models for seeds `101`, `102`, `103`, and `104`.
- Evaluated each trained model.
- Ran a frozen-weight delay sweep for each seed.

Shared delay sweep:

```text
20 25 30 35 40 50 60 70 80
```

Per-seed training/evaluation result:

| Seed | Final train accuracy | Evaluation accuracy |
| ---: | ---: | ---: |
| 101 | 1.000 | 1.000 |
| 102 | 1.000 | 1.000 |
| 103 | 1.000 | 1.000 |
| 104 | 1.000 | 1.000 |

Per-seed outputs:

- Seed `101`: `outputs/baseline_delay/seed_sweep/seed_101/`.
- Seed `102`: `outputs/baseline_delay/seed_sweep/seed_102/`.
- Seed `103`: `outputs/baseline_delay/seed_sweep/seed_103/`.
- Seed `104`: `outputs/baseline_delay/seed_sweep/seed_104/`.

Summary outputs:

- Summary JSON: `outputs/baseline_delay/metrics/baseline_delay_seed_sweep_summary.json`.
- Summary CSV: `outputs/baseline_delay/metrics/baseline_delay_seed_sweep.csv`.

</details>

<details>
<summary>2026-07-01 - Combined multi-seed delay plot generated</summary>

Action:

- Read the multi-seed summary file.
- Loaded each seed's delay-sweep CSV.
- Plotted all seed curves on one graph with an aggregate curve.

Included seeds:

```text
101 102 103 104
```

Included delays:

```text
20 25 30 35 40 50 60 70 80
```

Recorded output:

- Combined figure: `outputs/baseline_delay/figures/baseline_delay_seed_sweep_delay_curves.png`.

Interpretation:

- This plot is the current best overview of whether delay-related memory degradation is consistent across independently trained models.

</details>

<details>
<summary>2026-07-01 - Hidden-state stability analysis run on the baseline checkpoint</summary>

Action:

- Loaded the trained baseline checkpoint (`outputs/baseline_delay/checkpoints/baseline_delay.pt`), kept weights frozen.
- Generated a fresh 64-trial analysis batch at the trained task settings (`cue_steps=5`, `delay_steps=20`, `response_steps=5`).
- Measured hidden-state norm (magnitude) and step-to-step speed (the size of the change in hidden state between consecutive time steps) at every time step, averaged across trials.

Recorded result:

| Phase | Mean hidden-state norm | Mean step-to-step speed |
| --- | ---: | ---: |
| Cue (steps 0-5) | 1.86 | 0.73 |
| Delay (steps 5-25) | 39.54 | 7.69 |
| Response (steps 25-30) | 232.06 | 41.36 |

- Early-delay speed (first third of the delay period): `1.58`.
- Late-delay speed (last third of the delay period): `17.09`.
- Delay settling ratio (late-delay speed / early-delay speed): `10.83`.

Interpretation:

- The hidden-state norm and step-to-step speed both grow smoothly and
  substantially throughout the delay period and continue accelerating into
  the response period, rather than flattening out.
- A delay settling ratio well above `1` (here, about `10.8x`) means the
  hidden state is moving faster, not slower, by the end of the delay. This
  is the opposite of a settling, attractor-like signature, where step-to-step
  speed would be expected to shrink toward zero as the network approaches a
  stable state.
- This is consistent with the baseline model having learned a ramping or
  phasic solution rather than a tonic fixed-point attractor: the network
  only needs the hidden state to be decodable at the one fixed, trained
  response time, so nothing in the training objective rewards the state for
  settling down earlier.
- This also gives a direct mechanistic explanation for the delay-length
  sweep result recorded above: a growing, never-settled trajectory is
  expected to become progressively less predictable and harder to read out
  correctly the further past the trained delay length it is pushed, which
  matches the observed accuracy collapse beyond `30` delay steps.
- This finding does not mean the baseline model is broken; per Ghazizadeh and
  Ching (2021, "Slow Manifolds in Recurrent Networks Encode Working Memory
  Efficiently and Robustly"; see the dissertation wiki literature notes),
  trained working-memory RNNs can legitimately use fixed-point, limit-cycle,
  or slow-manifold/ramping mechanisms, and behavioral accuracy alone does
  not reveal which one is present. It does mean this baseline currently
  looks like the ramping case rather than a settled attractor case, which is
  a meaningful data point before layering psilocybin-informed perturbations
  (for example reduced recurrent stability) on top of it.

Recorded outputs:

- Figure: `outputs/baseline_delay/figures/baseline_delay_stability.png`.
- Summary: `outputs/baseline_delay/metrics/baseline_delay_stability_summary.json`.
- Arrays: `outputs/baseline_delay/arrays/baseline_delay_stability.npz`.

Next action under consideration:

- Decide whether to retrain with the delay period included in the training
  loss and/or with randomized delay length per batch, specifically to test
  whether either change produces a lower (more settled) delay settling
  ratio, before doing any fixed-point or Jacobian-level analysis.

</details>

