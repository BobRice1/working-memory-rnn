# Project Changelog

This changelog tracks two related histories:

1. Git commits: code and documentation changes committed to the repository.
2. Run log: experiment actions performed with the baseline model, including
   training, delay sweeps, multi-seed runs, and plotting.

## Note On Archived Baseline Artifacts (2026-07-01)

As of 2026-07-01, the baseline hidden-state nonlinearity changed from `relu`
to `tanh` (see the "Promoted `tanh` to the canonical baseline activation"
entry below for the full reasoning and results). Every run-log entry below
this note that references `outputs/baseline_delay/...` paths and was
recorded **before** this change describes the original `relu`-based
network. Those artifacts still exist and were not deleted; they were moved
to `outputs/baseline_delay_relu/` (same internal file names, unchanged),
and the exact config used to produce them is preserved as
`configs/baseline_delay_relu.yaml`. `outputs/baseline_delay/` and
`configs/baseline_delay.yaml` now refer to the current `tanh`-based
baseline. Run-log entries recorded after this note that reference
`outputs/baseline_delay/...` describe the new `tanh`-based artifacts.

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

<details>
<summary>Uncommitted: added a whole-delay-loss and randomized-delay training variant, kept as a separate versioned run</summary>

Reasoning:

- The stability analysis above showed the original baseline checkpoint does
  not settle during the delay period; step-to-step speed grows throughout
  the delay instead of shrinking. Two changes were identified as candidate,
  complementary fixes to test before any architecture change:
  1. Score the training loss across the whole delay period, not only the
     response window, so the network is directly rewarded for staying
     decodable at every delay time step, not only at one fixed offset.
  2. Randomize the delay length used in each training batch, so the network
     cannot rely on a solution that only needs to be correct at one exact,
     fixed trained delay.
- These two changes were judged complementary rather than redundant: (1)
  creates pressure to stay correct at every moment, and (2) extends that
  pressure across a range of durations instead of one fixed duration.
- Both changes are implemented as opt-in configuration so the original
  `configs/baseline_delay.yaml` config and its already-recorded results are
  completely unaffected.
- Outputs for this variant are written to a separate top-level output
  directory (`outputs/baseline_delay_stable/`) rather than being mixed into
  `outputs/baseline_delay/`, so figures, metrics, checkpoints, and arrays
  are kept in their own folder per model version and can be compared
  side by side without overwriting or interleaving files. This follows the
  same per-run folder pattern already used by `seed_sweep.py`.

File changes:

- `src/wm_rnn/training_utils.py`: Added `with_delay_steps`, a small helper
  that returns a copy of a task config with a different delay length,
  mirroring the existing `with_batch_size` helper.
- `src/wm_rnn/train.py`: Added two opt-in training behaviors, both disabled
  by default:
  - If `task.delay_steps_min` and `task.delay_steps_max` are both set, each
    training step now samples a random delay length from that inclusive
    range (using a seeded, reproducible random generator) instead of always
    using the fixed `task.delay_steps`.
  - If `training.score_delay_period` is `true`, the training loss is
    computed over the delay period plus the response period, instead of the
    response period only. Response-period accuracy is still always logged
    using the original response-only mask, so training curves stay
    comparable across configs regardless of this setting. The sampled delay
    length for each step is now also recorded in the training history CSV.
  - Final training metrics JSON now also records whether whole-delay
    scoring and delay randomization were used, and the configured delay
    range if randomization was on.
- `configs/baseline_delay_stable.yaml`: Added a new, separate experiment
  config that keeps the model architecture identical to the original
  baseline (`hidden_size=64`, `dt=20.0`, `tau=100.0`) and only changes the
  training objective: `delay_steps_min=15`, `delay_steps_max=45`,
  `score_delay_period=true`, and `steps=2000` (increased from `1000` because
  the task is harder). `paths.output_dir` is set to
  `outputs/baseline_delay_stable` and `paths.run_name` to
  `baseline_delay_stable`, so every output lands in its own folder tree
  separate from the original baseline run. `task.delay_steps` is kept at
  `20` so evaluation, delay-sweep, PCA, and stability analyses still probe
  the same reference delay length as the original baseline for comparison.

</details>

<details>
<summary>Uncommitted: added a configurable tanh activation and an isolated tanh test variant</summary>

Reasoning:

- The whole-delay-loss / randomized-delay variant improved how far the
  delay length could be pushed before accuracy collapsed, but the hidden
  state was still accelerating rather than settling (delay settling ratio
  about `4.8x`, still above `1`). That pointed at a second, separate cause:
  the recurrent layer's `relu` nonlinearity has no upper bound, so nothing
  in the architecture itself prevents the hidden state from growing forever,
  regardless of how the loss is scored.
- Switching the hidden-state nonlinearity to `tanh` was proposed as a
  structural fix rather than a training-incentive fix: `tanh` bounds hidden
  activity to `(-1, 1)`, so unbounded growth becomes mathematically
  impossible rather than merely discouraged.
- This was deliberately tested in isolation: same fixed `20`-step delay,
  same response-period-only loss, same `1000` training steps as the
  original baseline, changing only the activation function. This isolates
  whether bounding alone (independent of the training-objective changes
  already tested) produces settled hidden-state dynamics.
- `relu` gives only non-negative activity, loosely resembling a firing rate;
  `tanh` allows negative activity and is the standard choice in most
  trained-RNN dynamical-systems literature (for example the kind of
  fixed-point-finding analysis flagged as a future next step), so this is
  recorded as a real modelling choice, not only an implementation detail.

File changes:

- `src/wm_rnn/model.py`: Added an `activation` field to `RNNConfig`
  (`"relu"` by default, or `"tanh"`), with validation against unsupported
  values. `CTRNN` now resolves the configured activation once at
  construction time and applies it in `recurrence()` instead of a hardcoded
  `torch.relu` call. Docstrings updated to describe both activation options.
- `src/wm_rnn/training_utils.py`: `model_config_from_dict` now reads
  `model.activation` from the config dictionary (defaulting to `"relu"` if
  absent) and passes it through to `RNNConfig`.
- `src/wm_rnn/config.py`: Added `activation: "relu"` to the default model
  configuration so the setting is discoverable and documented even when a
  YAML config omits it.
- `configs/baseline_delay_tanh.yaml`: Added a new experiment config that is
  otherwise identical to `configs/baseline_delay.yaml` (same task timing,
  same `1000`-step response-only training) except `model.activation: tanh`.
  `paths.output_dir` is `outputs/baseline_delay_tanh` and `paths.run_name`
  is `baseline_delay_tanh`, giving this variant its own separate output
  folder tree, consistent with the other variants.

</details>

<details>
<summary>2026-07-01 - Trained and analyzed the whole-delay-loss / randomized-delay variant, compared against the original baseline</summary>

Action:

- Trained a new model with `configs/baseline_delay_stable.yaml`: same
  architecture as the original baseline, but with the training delay length
  sampled per step from `[15, 45]` and the training loss scored across the
  delay period plus the response period, for `2000` steps.
- Evaluated the trained checkpoint, ran the same frozen-weight delay-length
  sweep used for the original baseline (`20, 25, 30, 35, 40, 50, 60, 70,
  80`), ran the PCA trajectory analysis, and ran the hidden-state stability
  analysis, all against this new checkpoint.
- All outputs were written to `outputs/baseline_delay_stable/` (separate
  `checkpoints/`, `metrics/`, `figures/`, and `arrays/` subfolders), so they
  do not overwrite or mix with `outputs/baseline_delay/`.

Recorded result - training and evaluation:

- Final training loss: `0.0038`. Final training (response-period) accuracy:
  `1.000`, reached consistently across randomly sampled delay lengths from
  step `100` onward.
- Held-out evaluation accuracy at the reference `20`-step delay: `1.000`.

Recorded result - delay-length sweep, compared with the original baseline:

| Delay steps | Original baseline accuracy | Stable-variant accuracy |
| ---: | ---: | ---: |
| 20 | 1.000 | 1.000 |
| 25 | 1.000 | 1.000 |
| 30 | 0.907 | 1.000 |
| 35 | 0.749 | 1.000 |
| 40 | 0.745 | 1.000 |
| 50 | 0.519 | 1.000 |
| 60 | 0.485 | 0.712 |
| 70 | 0.483 | 0.510 |
| 80 | 0.503 | 0.511 |

Recorded result - hidden-state stability, compared with the original baseline:

| Metric | Original baseline | Stable variant |
| --- | ---: | ---: |
| Early-delay speed | 1.58 | 1.20 |
| Late-delay speed | 17.09 | 5.72 |
| Delay settling ratio (late / early) | 10.83 | 4.76 |

Interpretation:

- The delay-length sweep shows a large, direct improvement: the original
  baseline started degrading past `25-30` steps, while the stable variant
  holds perfect accuracy all the way to `50` steps, which lines up closely
  with the trained random-delay upper bound of `45`. Accuracy still falls
  off beyond that, roughly toward the same chance-level plateau seen in the
  original baseline by `70-80` steps.
- The hidden-state stability analysis shows real but partial improvement:
  the delay settling ratio dropped from about `10.8x` to about `4.8x`, and
  both early- and late-delay speeds are lower. The hidden state is still
  accelerating through the delay rather than flattening toward zero, so
  this is not yet a settled, tonic attractor state; it is a less extreme,
  longer-range version of the same ramping signature seen in the original
  baseline (confirmed visually in
  `outputs/baseline_delay_stable/figures/baseline_delay_stable_pca_trajectories.png`,
  where hidden-state trajectories still diverge outward by cue class rather
  than curving toward fixed points).
- Taken together, this suggests the two training changes successfully
  widened the range of delay lengths the network can handle reliably
  (matching the literal range it was trained on, plus some margin), but did
  not by themselves convert the underlying mechanism into a settled
  attractor. This is consistent with the ramping/phasic account of the
  original baseline rather than a full mechanism change: the network
  appears to have learned a ramp that stays correct over a wider trained
  window, rather than learning to stop ramping.
- This is a meaningful, literature-consistent result rather than a null
  result: Ghazizadeh and Ching (2021) describe slow-manifold/phasic
  solutions as a legitimate, sometimes more efficient alternative to tonic
  fixed-point attractors, and this result is consistent with the network
  favoring that kind of solution even when the training objective directly
  rewards delay-period stability.

Recorded outputs:

- Training metrics: `outputs/baseline_delay_stable/metrics/baseline_delay_stable_train_metrics.json`.
- Training history: `outputs/baseline_delay_stable/metrics/baseline_delay_stable_train_history.csv`.
- Evaluation metrics: `outputs/baseline_delay_stable/metrics/baseline_delay_stable_eval_metrics.json`.
- Delay-sweep metrics/CSV/figure: `outputs/baseline_delay_stable/metrics/baseline_delay_stable_delay_sweep_metrics.json`, `outputs/baseline_delay_stable/metrics/baseline_delay_stable_delay_sweep.csv`, `outputs/baseline_delay_stable/figures/baseline_delay_stable_delay_sweep.png`.
- PCA figure/arrays: `outputs/baseline_delay_stable/figures/baseline_delay_stable_pca_trajectories.png`, `outputs/baseline_delay_stable/arrays/baseline_delay_stable_hidden_states.npz`.
- Stability figure/summary/arrays: `outputs/baseline_delay_stable/figures/baseline_delay_stable_stability.png`, `outputs/baseline_delay_stable/metrics/baseline_delay_stable_stability_summary.json`, `outputs/baseline_delay_stable/arrays/baseline_delay_stable_stability.npz`.

Next action under consideration:

- If a genuinely settled, tonic attractor state is desired rather than a
  wider-range ramp, consider a change that directly bounds or penalizes
  hidden-state growth (for example a bounded nonlinearity such as `tanh`,
  or an explicit hidden-state norm penalty) rather than further widening the
  trained delay range.
- Fixed-point or Jacobian-level analysis would give a more direct answer
  than trajectory speed alone, and is a reasonable next step now that two
  checkpoints with meaningfully different delay-generalization behavior
  exist to compare.

</details>

<details>
<summary>2026-07-01 - Trained and analyzed the isolated tanh-activation variant, compared against both prior runs</summary>

Action:

- Trained a new model with `configs/baseline_delay_tanh.yaml`: identical
  training setup to the original baseline (fixed `20`-step delay,
  response-period-only loss, `1000` steps), with only `model.activation`
  changed from `relu` to `tanh`.
- Ran the same evaluation, delay-length sweep (`20, 25, 30, 35, 40, 50, 60,
  70, 80`), PCA trajectory analysis, and hidden-state stability analysis
  used for the previous two runs, against this checkpoint.
- All outputs were written to `outputs/baseline_delay_tanh/`, kept separate
  from both `outputs/baseline_delay/` and `outputs/baseline_delay_stable/`.

Recorded result - training and evaluation:

- Final training loss: `0.0019`. Final training accuracy: `1.000`, reached
  by step `150` and stable for the remainder of training.
- Held-out evaluation accuracy at the reference `20`-step delay: `1.000`.

Recorded result - delay-length sweep, compared with both prior runs:

| Delay steps | Original baseline (relu) | Whole-delay-loss variant (relu) | Tanh variant |
| ---: | ---: | ---: | ---: |
| 20 | 1.000 | 1.000 | 1.000 |
| 25 | 1.000 | 1.000 | 1.000 |
| 30 | 0.907 | 1.000 | 1.000 |
| 35 | 0.749 | 1.000 | 1.000 |
| 40 | 0.745 | 1.000 | 1.000 |
| 50 | 0.519 | 1.000 | 1.000 |
| 60 | 0.485 | 0.712 | 1.000 |
| 70 | 0.483 | 0.510 | 1.000 |
| 80 | 0.503 | 0.511 | 1.000 |

Recorded result - hidden-state stability, compared with both prior runs:

| Metric | Original baseline (relu) | Whole-delay-loss variant (relu) | Tanh variant |
| --- | ---: | ---: | ---: |
| Early-delay speed | 1.58 | 1.20 | 0.56 |
| Late-delay speed | 17.09 | 5.72 | 0.03 |
| Delay settling ratio (late / early) | 10.83 | 4.76 | **0.06** |

Interpretation:

- The tanh variant reaches perfect response accuracy at every tested delay
  length, including `80` steps, four times the trained `20`-step delay, and
  it does this using the plain original training setup (fixed delay,
  response-only loss, no extra training steps). This outperforms the
  whole-delay-loss/randomized-delay variant, which needed a harder training
  procedure and still degraded past `50` steps.
- The hidden-state stability numbers now show genuine settling rather than
  a smaller ramp: the delay settling ratio dropped from `10.83` (original)
  and `4.76` (whole-delay-loss variant) to `0.06` for the tanh variant. A
  ratio well below `1` means late-delay speed is much smaller than
  early-delay speed, that is, the hidden state is slowing down and
  flattening out as the delay continues, not accelerating.
- This is visible directly in the recorded figures: hidden-state norm
  (`outputs/baseline_delay_tanh/figures/baseline_delay_tanh_stability.png`)
  rises during the cue period and then visibly plateaus partway through the
  delay instead of continuing to climb; step-to-step speed rises briefly
  after the cue, then decays toward near zero by the end of the delay. The
  PCA trajectory figure
  (`outputs/baseline_delay_tanh/figures/baseline_delay_tanh_pca_trajectories.png`)
  shows trajectories that curve and slow into a settled region per cue
  class, rather than the straight, ever-diverging rays seen in both `relu`
  runs.
- Taken together, this points to the unbounded `relu` nonlinearity, not the
  training objective, as the primary cause of the original baseline's poor
  delay generalization. The whole-delay-loss and randomized-delay changes
  did help the `relu` network cope with a wider range of delays, but they
  could not make it settle, because nothing in that architecture prevents
  continued growth. Bounding the nonlinearity removed the underlying
  capacity for unbounded growth directly, and a settled representation
  emerged from the same simple training setup as the original baseline,
  with no need for the more complex training changes.
- This is the first run in this project to show a hidden-state signature
  consistent with a genuinely settled, tonic, attractor-like mechanism
  rather than a ramping or phasic one, in the terms used by Ghazizadeh and
  Ching (2021). It should still be confirmed with a direct fixed-point or
  Jacobian-level analysis rather than trajectory speed alone before treating
  it as a confirmed attractor.

Recorded outputs:

- Training metrics: `outputs/baseline_delay_tanh/metrics/baseline_delay_tanh_train_metrics.json`.
- Training history: `outputs/baseline_delay_tanh/metrics/baseline_delay_tanh_train_history.csv`.
- Evaluation metrics: `outputs/baseline_delay_tanh/metrics/baseline_delay_tanh_eval_metrics.json`.
- Delay-sweep metrics/CSV/figure: `outputs/baseline_delay_tanh/metrics/baseline_delay_tanh_delay_sweep_metrics.json`, `outputs/baseline_delay_tanh/metrics/baseline_delay_tanh_delay_sweep.csv`, `outputs/baseline_delay_tanh/figures/baseline_delay_tanh_delay_sweep.png`.
- PCA figure/arrays: `outputs/baseline_delay_tanh/figures/baseline_delay_tanh_pca_trajectories.png`, `outputs/baseline_delay_tanh/arrays/baseline_delay_tanh_hidden_states.npz`.
- Stability figure/summary/arrays: `outputs/baseline_delay_tanh/figures/baseline_delay_tanh_stability.png`, `outputs/baseline_delay_tanh/metrics/baseline_delay_tanh_stability_summary.json`, `outputs/baseline_delay_tanh/arrays/baseline_delay_tanh_stability.npz`.

Next action under consideration:

- Decide whether `tanh` should become the new default baseline activation
  (updating `configs/baseline_delay.yaml` and `docs/model-architecture.md`)
  given this result, or whether `relu` should be kept as the documented
  original baseline with `tanh` recorded as a compared variant.
- Run a fixed-point or Jacobian-level analysis on the tanh checkpoint to
  directly confirm attractor structure rather than inferring it from
  trajectory speed and norm alone.
- Consider whether combining `tanh` with the whole-delay-loss/randomized-delay
  training changes gives any further benefit, now that each change has been
  tested in isolation.

</details>

<details>
<summary>Uncommitted: promoted tanh to the canonical baseline activation</summary>

Reasoning:

- The isolated tanh test above showed perfect response accuracy at every
  tested delay length (up to `80` steps, four times the trained length) and
  a delay settling ratio of `0.06`, using the exact same simple training
  setup as the original `relu` baseline. This was judged strong enough
  evidence to make `tanh` the default baseline activation rather than a
  separately tracked variant, since it is a strictly better-generalizing,
  more settled network produced by the same architecture family and the
  same training procedure, not a different modelling approach.
- The original `relu`-based baseline was not discarded. Its config and
  recorded outputs were archived under new, clearly named files/folders so
  the full history remains reproducible and inspectable, consistent with
  how `docs/changelog.md` otherwise preserves a complete run history.
- The standalone `configs/baseline_delay_tanh.yaml` config and its
  `outputs/baseline_delay_tanh/` outputs were removed, because after this
  change they would be an exact duplicate of the new canonical
  `configs/baseline_delay.yaml` / `outputs/baseline_delay/`.

File changes:

- `configs/baseline_delay_relu.yaml`: Added as an exact archival copy of the
  pre-change `configs/baseline_delay.yaml` (fixed `20`-step delay,
  response-only loss, `1000` steps, `model.activation: relu`), with
  `paths.output_dir` set to `outputs/baseline_delay_relu` and
  `paths.run_name` set to `baseline_delay_relu` so it can be re-run
  independently without colliding with the new baseline.
- `configs/baseline_delay.yaml`: Added `model.activation: tanh`. No other
  fields changed; task timing, training steps, and output paths
  (`outputs/baseline_delay/`, run name `baseline_delay`) are unchanged, so
  this file now defines the new canonical baseline in place.
- `configs/baseline_delay_tanh.yaml`: Removed (superseded by the change
  above; would otherwise duplicate `configs/baseline_delay.yaml`).
- `docs/model-architecture.md`: Updated the "Recurrent Core" section to
  describe the hidden-state nonlinearity as configurable with `tanh` as the
  baseline default, documented the `activation` config field, generalized
  the per-step update formula from a hardcoded `relu` call to
  `activation(...)`, and added a short explanation of why `tanh` replaced
  `relu` as the default, pointing to this changelog for full detail.
- `README.md`: Updated the top-line description, and rewrote "Model
  Variants" to describe `configs/baseline_delay.yaml` as the current
  (`tanh`) baseline, `configs/baseline_delay_relu.yaml` as the archived
  original baseline, and `configs/baseline_delay_stable.yaml` as a
  still-open comparison; removed the now-deleted `baseline_delay_tanh`
  entry.

Filesystem changes (outputs, not tracked by Git since `outputs/` is
git-ignored):

- Renamed `outputs/baseline_delay/` to `outputs/baseline_delay_relu/`,
  archiving the original `relu` baseline's checkpoint, metrics, figures,
  arrays, and multi-seed sweep results without modification. File names
  inside this archived folder still carry the original `baseline_delay_*`
  prefix rather than `baseline_delay_relu_*`, since they were generated
  before this archiving step; the containing folder name is authoritative
  for identifying these as the archived `relu` run.
- Deleted `outputs/baseline_delay_tanh/` (superseded by the retrained
  canonical baseline below).
- Retrained a fresh checkpoint into `outputs/baseline_delay/` using the
  updated `configs/baseline_delay.yaml`, and re-ran evaluation, the
  delay-length sweep, PCA analysis, and stability analysis against it (see
  run-log entry below). Because this config is otherwise identical to the
  earlier isolated tanh test, results are the same; this run exists so the
  checkpoint's embedded config and all output file names correctly reflect
  the `baseline_delay` identity going forward.

</details>

<details>
<summary>2026-07-01 - Retrained the canonical baseline_delay checkpoint under tanh and confirmed results</summary>

Action:

- Trained `outputs/baseline_delay/checkpoints/baseline_delay.pt` using the
  updated `configs/baseline_delay.yaml` (`model.activation: tanh`, fixed
  `20`-step delay, response-only loss, `1000` steps).
- Re-ran evaluation, the same delay-length sweep (`20, 25, 30, 35, 40, 50,
  60, 70, 80`), PCA trajectory analysis, and hidden-state stability
  analysis against this checkpoint, all writing into `outputs/baseline_delay/`.

Recorded result:

- Final training loss: `0.0019`. Final training accuracy: `1.000`.
- Held-out evaluation accuracy at the `20`-step reference delay: `1.000`.
- Delay-length sweep: `1.000` accuracy at every tested delay length from
  `20` through `80` steps.
- Hidden-state stability: early-delay speed `0.56`, late-delay speed
  `0.03`, delay settling ratio `0.06`.

Interpretation:

- These numbers are identical (within floating-point rounding) to the
  isolated tanh test recorded earlier, as expected: the only configuration
  difference is the output path and run name, not the model, data, or
  training procedure. This run exists to make `outputs/baseline_delay/`
  and its embedded checkpoint config correctly represent the current
  canonical baseline, rather than to test a new hypothesis.
- `outputs/baseline_delay/` now documents a settled, delay-length-robust
  baseline. `outputs/baseline_delay_relu/` remains available as the
  archived original baseline for any comparison that specifically needs
  the earlier `relu`-based, ramping-dynamics network.

Recorded outputs:

- Training metrics: `outputs/baseline_delay/metrics/baseline_delay_train_metrics.json`.
- Training history: `outputs/baseline_delay/metrics/baseline_delay_train_history.csv`.
- Evaluation metrics: `outputs/baseline_delay/metrics/baseline_delay_eval_metrics.json`.
- Delay-sweep metrics/CSV/figure: `outputs/baseline_delay/metrics/baseline_delay_delay_sweep_metrics.json`, `outputs/baseline_delay/metrics/baseline_delay_delay_sweep.csv`, `outputs/baseline_delay/figures/baseline_delay_delay_sweep.png`.
- PCA figure/arrays: `outputs/baseline_delay/figures/baseline_delay_pca_trajectories.png`, `outputs/baseline_delay/arrays/baseline_delay_hidden_states.npz`.
- Stability figure/summary/arrays: `outputs/baseline_delay/figures/baseline_delay_stability.png`, `outputs/baseline_delay/metrics/baseline_delay_stability_summary.json`, `outputs/baseline_delay/arrays/baseline_delay_stability.npz`.

Next action under consideration:

- Run a fixed-point or Jacobian-level analysis on this checkpoint to
  directly confirm attractor structure, since this is now the checkpoint
  that will anchor future baseline comparisons and psilocybin-informed
  perturbations.
- Decide whether to retrain `configs/baseline_delay_stable.yaml`-style
  whole-delay-loss/randomized-delay training on top of `tanh` rather than
  `relu`, now that `tanh` is the baseline activation.
- Consider whether the existing multi-seed sweep tooling
  (`src/wm_rnn/seed_sweep.py`) should be re-run under the new `tanh`
  baseline to confirm the settled, delay-robust behavior holds across
  seeds, since the original multi-seed results in
  `outputs/baseline_delay_relu/seed_sweep/` were only ever measured for the
  `relu` network.

</details>

<details>
<summary>2026-07-01 - Re-ran the multi-seed delay sweep under the tanh baseline</summary>

Action:

- Trained independent `tanh`-based baseline models for seeds `101`, `102`,
  `103`, and `104` using `configs/baseline_delay.yaml`, evaluated each, and
  ran the same frozen-weight delay sweep (`20, 25, 30, 35, 40, 50, 60, 70,
  80`) used for every other run in this project.
- Plotted all four seed curves together with a mean and `+/- 1 SD` band,
  same as the original `relu` multi-seed plot.
- All outputs were written under `outputs/baseline_delay/seed_sweep/`,
  replacing the prior `tanh`-baseline seed-sweep summary location (the
  original `relu` four-seed results remain untouched and available at
  `outputs/baseline_delay_relu/seed_sweep/`).

Recorded result - training and evaluation:

- All four seeds reached `1.000` final training accuracy and `1.000`
  held-out evaluation accuracy at the `20`-step reference delay.

Recorded result - delay-length sweep, `tanh` seeds compared with the archived `relu` seeds:

| Delay steps | relu seed 101 | relu seed 102 | relu seed 103 | relu seed 104 | relu mean | tanh seed 101 | tanh seed 102 | tanh seed 103 | tanh seed 104 | tanh mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 30 | 0.950 | 1.000 | 1.000 | 1.000 | 0.988 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 40 | 0.747 | 0.723 | 0.517 | 0.910 | 0.724 | 0.781 | 1.000 | 0.899 | 1.000 | 0.920 |
| 60 | 0.730 | 0.418 | 0.509 | 0.484 | 0.535 | 0.747 | 1.000 | 0.757 | 1.000 | 0.876 |
| 80 | 0.754 | 0.245 | 0.455 | 0.504 | 0.489 | 0.745 | 1.000 | 0.711 | 1.000 | 0.864 |

Interpretation:

- This is a more complete and more honest picture than the single-seed
  result recorded earlier. Under `relu`, all four seeds degrade toward or
  below the neighborhood of chance (`0.25`) by `80` steps, with high
  seed-to-seed variability (`0.245` to `0.754`) and no seed staying
  reliable. Under `tanh`, two of four seeds (`102`, `104`) hold perfect
  `1.000` accuracy through `80` steps, matching the single-seed result
  reported earlier, while the other two (`101`, `103`) settle to roughly
  `0.71`-`0.78` beyond `40` steps rather than continuing to degrade toward
  chance.
- The earlier claim that switching to `tanh` produces generalization "at
  least four times longer than trained" was accurate for the one seed used
  in the initial checkpoint analysis, but should not be read as a
  guarantee for every randomly initialized network. `docs/model-architecture.md`
  has been updated to state this more precisely: strong, well-above-chance
  generalization is typical under `tanh`, and perfect generalization occurs
  for some but not all seeds.
- Even the weaker `tanh` seeds are still a clear improvement over every
  `relu` seed at long delays: the worst `tanh` seed at `80` steps (`0.711`)
  is comparable to or better than the best `relu` seed at `80` steps
  (`0.754`), and the `tanh` mean (`0.864`) is far above the `relu` mean
  (`0.489`) with a much tighter spread. The core conclusion, that bounding
  the nonlinearity substantially improves delay-length generalization, is
  unchanged; what changes is that it should be described as a strong,
  typical improvement rather than a uniform guarantee.
- This also means the `tanh` network's settling behavior may itself vary
  by seed (some seeds may find a fully settled fixed-point-like solution,
  others a partially settled or slower-drifting one). This is a natural
  candidate follow-up: running the hidden-state stability analysis on the
  `101`/`103` (partial) and `102`/`104` (perfect) checkpoints and comparing
  their settling ratios would show whether degree of settling predicts
  degree of long-delay accuracy, rather than assuming it from accuracy
  alone.

Recorded outputs:

- Summary JSON: `outputs/baseline_delay/metrics/baseline_delay_seed_sweep_summary.json`.
- Summary CSV: `outputs/baseline_delay/metrics/baseline_delay_seed_sweep.csv`.
- Combined figure: `outputs/baseline_delay/figures/baseline_delay_seed_sweep_delay_curves.png`.
- Per-seed checkpoints, metrics, and figures: `outputs/baseline_delay/seed_sweep/seed_101/` through `seed_104/`.

Next action under consideration:

- Run the hidden-state stability analysis on each of the four seed
  checkpoints to test whether settling ratio predicts long-delay accuracy
  within the `tanh` baseline, not just between `tanh` and `relu`.
- Revisit whether the still-open `tanh` + whole-delay-loss/randomized-delay
  combination (see the "superseded or worth testing" discussion) would
  reduce this seed-to-seed variability, since that combination directly
  rewards delay-period stability during training rather than relying on
  the bounded nonlinearity alone.

</details>

