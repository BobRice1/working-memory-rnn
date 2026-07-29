# Implementation Plan — Persistence Hidden-State Analysis Notebook

**Deliverable:** `notebooks/persistence_hidden_state_analysis.ipynb`
**Author of plan:** exploration/planning pass, 2026-07-29
**Scope:** Descriptive hidden-state mechanism evidence for the `state_persistence`
perturbation on the variable-timing circular working-memory RNN family. Approach
"B": core quantitative metrics + frozen-basis PCA visualization. **Not** a
matched-cost specificity claim vs Gaussian noise; do **not** introduce a Gaussian
comparator.

Downstream agent: this plan is authoritative for structure, data sources, and
function calls. Write the notebook code; do **not** change the experiment design.

---

## 1. Objective

For the 10 competent variable-timing circular checkpoints, compare the internal
(hidden-state) dynamics of the model under three carried-state persistence
settings:

- **persistence 1.00** — native baseline (the operator is a no-op at 1.00; see
  §3.4).
- **persistence 0.95** — the candidate strength.
- **persistence 0.90** — a stronger neighbour.

Produce, per seed and pooled: (1) clean-delay drift + step-speed/norm metrics,
(2) onset-aligned distractor attraction + recovery, and (3) a frozen-PCA
trajectory visualization of the mean delay-window dynamics. All conditions reuse
the **same frozen banks** (paired target angles, distractor angles, and
randomized per-trial distractor onsets) as the existing circular perturbation
runs, so baseline and perturbed differ only in the operator.

---

## 2. Environment & execution

- Python venv (CUDA): `C:\Users\Bob Rice\Documents\Obsidian Vault\Dissertation\working-memory-rnn\.venv\Scripts\python.exe`
- Imports require `PYTHONPATH=src` (set via `%env PYTHONPATH=src` or
  `sys.path.insert(0, "src")` in cell 1).
- Device: `"cuda"` available (RTX 3060, 6 GB). Load one checkpoint at a time and
  free it (`del model; torch.cuda.empty_cache()`) between seeds to stay under
  6 GB. Batch size 128 with 1,024 trials/condition ⇒ 8 batches/cell; hidden is
  64 units, ≤ 90 timesteps, so memory is trivial.
- Run notebook from the repo root (worktree
  `.worktrees/variable-timing-circular`). All repo-relative paths below assume
  that cwd.
- This analysis is **circular-only**. The N-back outputs are junctioned in but
  are out of scope; do not touch them.

---

## 3. Exact data sources (grounded in code)

### 3.1 Checkpoints & seeds

- Checkpoint loader: `full_candidate_perturbation_run.trained_distractor_checkpoints(repo_root, manifest_path)`
  → `tuple[FrozenCheckpoint(seed, path, sha256), ...]`.
  (`src/wm_rnn/full_candidate_perturbation_run.py:42`.)
- Manifest for the variable-timing family (pass explicitly, as
  `state_persistence_dense_run.py` does):
  `outputs/fixation_circular_variable_distractor_working_memory/metrics/fixation_circular_variable_distractor_working_memory_pool_summary.json`.
  Verified keys: `retained_checkpoint_seeds` =
  `[20260801,20260802,20260803,20260804,20260805,20260806,20260808,20260809,20260810,20260811]`
  (exactly the 10 requested seeds), `target_competent_checkpoints=10`,
  `results[i]` has `seed`, `checkpoint`, `checkpoint_sha256`, `competence_passed`.
- Base task config: `configs/fixation_circular_variable_distractor_working_memory.yaml`,
  with pinned SHA256 constant
  `C568B49FBF17504D6047454E150C00C54F3E8C9503CE9E4EDD50C2CDA5FA554D`
  (`full_candidate_variable_timing_run.py:25`, `state_persistence_dense_run.py:27`).
  The notebook should reuse these constants by importing them from
  `wm_rnn.state_persistence_dense_run` (`CONFIG`, `BASE_CONFIG`,
  `BASE_CONFIG_SHA256`, `POOL_MANIFEST`) rather than re-declaring them.

### 3.2 Task geometry (verified by running the config)

With the base config and `distractor_steps=5`, `delay_steps=20`, the phase index
is:

```
fixation (0,25)  cue (25,45)  delay (45,65)  distractor (53,58 nominal)  response (65,90)
```

- `n_tuned_units = 32`, hidden size = 64.
- Delay window = 20 steps (`delay.start=45`, `delay.stop=65`).
- Distractor duration = `min(distractor_steps, delay_steps) = 5`; valid relative
  starts within the delay are `0..15` (16 onsets), matching
  `distractor_valid_relative_starts` in the dense config.
- The nominal `phase_index["distractor"]` slice (53,58) is a fixed placeholder;
  under randomized onsets each trial's true onset is `delay.start + relative_start`.

### 3.3 Collection path (the core reuse)

Collect with `perturbation_experiment._collect_batches(...)`
(`src/wm_rnn/perturbation_experiment.py:378`). Its return dict (verified at
lines 448–476) contains:

| key | shape / type | notes |
|---|---|---|
| `predictions` | `[time, trials, out]` | readout; not needed for hidden analysis except fixation |
| `hidden_states` | `[time, trials, units]` | tanh states, in [-1,1]; concatenated on axis=1 |
| `angles` | `[trials]` radians | target cue angle |
| `distractor_angles` | `[trials]` or None | present for distractor condition |
| `phase_index` | dict[str,slice] | `delay`, `response`, etc. |
| `preferred_angles` | `[n_tuned_units]` | for population/decoder use |
| `distractor_relative_starts` | `[trials]` int or None | **per-trial onset**, present only when `randomize_distractor_onsets=True` |
| `angle_hashes` | list[str] | provenance |

Pairing/seed conventions (verified): batch seeds come from
`frozen_batch_seed(seed_base, condition_index, delay_steps, batch_index)`
(`:147`), with `seed_base=FINAL_SEED_BASE` (`202607300`). The randomized-onset
bank is `balanced_random_distractor_starts(delay_steps, distractor_steps, n, seed_base + 9_000_000 + delay_steps)`
(`:315`, `:404`). **Both** the batch seeds and the onset bank depend only on
`(seed_base, condition_index, delay_steps)` — **not** on the operator — so calling
`_collect_batches` with `forward_fn=state_persistence(model, gain=g)` for
g ∈ {1.00, 0.95, 0.90} yields identical target angles, distractor angles, and
per-trial onsets across conditions. This is the required paired/frozen design.

`randomize_distractor_onsets=True` is only valid for Family "A" + condition
`"distractor"` (`:399-403`); the clean condition must be collected with
`randomize_distractor_onsets=False`.

### 3.4 The persistence operator

`perturbation_operators.state_persistence(model, persistence_gain=g)`
(`src/wm_rnn/perturbation_operators.py:382`) returns a `ForwardFn` that returns
`(readout, hidden_states)` via `_run_explicit` — i.e. **hidden states are fully
exposed for the persistence forward pass** (no gap here). Coefficients:
`carried = model.rnn.oneminusalpha * g`, `drive = model.rnn.alpha`
(`state_persistence_coefficients`, `:118`). At `g == 1.0` the forward function
short-circuits to `model(inputs)` (`:394`), so persistence 1.00 == native
baseline collection. Build the forward via
`_operator_forward(model, task_config, operator="state_persistence", variant="carried_state_only", strength=g, condition=..., family="A", delay_steps=...)`
(`perturbation_experiment.py:996`, case at `:1051`) to stay consistent with the
runs.

### 3.5 Decoder & metric helpers

- `perturbation_experiment.fit_frozen_decoder(model, base_task, family="A", trials_per_delay=64, ridge_alpha=1.0)` → `[units, 2]` ridge weights
  (`:479`). Fit **once per checkpoint** on clean reference samples (delays
  10/20/40/80, late-delay pooled), matching the runs. Use this decoder for both
  clean and distractor decoding of that seed.
- `hidden_angle_decoder.decode_angles_from_hidden(hidden, weights)` → radians in
  [0,2π), shape = hidden.shape[:-1] (`src/wm_rnn/hidden_angle_decoder.py:62`).
  Input hidden `[time, trials, units]` ⇒ output `[time, trials]`.
- `perturbation_metrics.delay_decoding_error(hidden_states[time,trials,units], angles[trials], weights[units,2], window: slice)` → `{mean_error_degrees, median_error_degrees}` (`perturbation_metrics.py:182`). Shapes are asserted internally.
- `perturbation_metrics.distractor_drift_and_recovery(decoded_angles[time,trials], target_angles[trials], distractor_angles[trials], distractor_slice, post_distractor_slice)` → dict with `distractor_peak_attraction_fraction`, `distractor_recovery_fraction`, `distractor_peak_drift_degrees`, `distractor_end_attraction_fraction` (`perturbation_metrics.py:212`). **Takes a single fixed window pair — NOT per-trial onset aligned** (see §7 GAP-1).
- `perturbation_metrics.activation_slope_and_saturation(hidden[...,units])` → `{mean_activation_slope, saturation_fraction, mean_tanh_slope}` (`:355`). Requires hidden in [-1,1].
- Step-speed / norm helpers live in `baseline_competence_figures`:
  `_hidden_speed(hidden[time,trials,units]) -> [time-1, trials]` (`:129`) and norm
  via `np.linalg.norm(hidden, axis=-1) -> [time,trials]`. These are private; the
  notebook may import `_hidden_speed` or re-implement the 2-line body locally
  (prefer a local copy to avoid depending on a private symbol).

### 3.6 PCA precedent

`baseline_competence_figures._circular_dynamics_for_seed` (`:539`) is the
convention: `sklearn.decomposition.PCA(n_components=2)`, fit on
`hidden.reshape(-1, units)` (tanh states used **directly**, no arctanh), then
`.reshape(time, trials, 2)`; explained variance from
`pca.explained_variance_ratio_`; `plot_circular_pca` (`:350`) colours by cue
angle. The notebook adapts this but **freezes the basis on baseline** (see §5).

---

## 4. Notebook cell-by-cell outline

> Numbering = execution order. "md" = markdown cell, "code" = code cell. Keep the
> notebook deterministic and outcome-reproducible; no hidden global state between
> conditions beyond the explicitly frozen banks.

**Cell 1 (md) — Title & scope.** State objective, the three persistence
conditions, the descriptive claim boundary (no matched-cost Gaussian), and the
seed list.

**Cell 2 (code) — Environment.** `import sys, pathlib`; ensure repo root is cwd;
`sys.path.insert(0, "src")` if needed; imports:
`numpy as np`, `torch`, `pandas as pd`, `json`, `matplotlib.pyplot as plt`,
`from sklearn.decomposition import PCA`, `from dataclasses import replace`.
Repo imports: `load_config`; from `wm_rnn.perturbation_experiment`:
`_collect_batches`, `_load_checkpoint_model`, `_operator_forward`,
`fit_frozen_decoder`, `FINAL_SEED_BASE`; from `wm_rnn.hidden_angle_decoder`:
`decode_angles_from_hidden`; from `wm_rnn.perturbation_metrics`:
`delay_decoding_error`, `distractor_drift_and_recovery`,
`activation_slope_and_saturation`, `signed_circular_error`; from
`wm_rnn.training_utils`: `task_config_from_dict`; from
`wm_rnn.state_persistence_dense_run`: `CONFIG`, `BASE_CONFIG`,
`BASE_CONFIG_SHA256`, `POOL_MANIFEST`; from
`wm_rnn.full_candidate_perturbation_run`: `trained_distractor_checkpoints`.
Set `torch.manual_seed`/`np.random` not required (all seeding is explicit inside
collection). Define `DEVICE = torch.device("cuda")`.

**Cell 3 (code) — Constants / design.** Define:
`PERSISTENCE = (1.00, 0.95, 0.90)`; `FAMILY = "A"`;
`CLEAN_DELAY = 20`; `DISTRACTOR_DELAY = 20`; `DISTRACTOR_STEPS = 5`;
`TRIALS = 1024`; `BATCH_SIZE = 128`; `N_BATCHES = TRIALS // BATCH_SIZE` (= 8);
output dir `OUT = pathlib.Path("outputs/persistence_hidden_state_analysis")`
and figure/metrics/summary subpaths (see §6). Assert
`TRIALS % BATCH_SIZE == 0`.

**Cell 4 (code) — Load config + verify base SHA256.** `base_config = load_config(BASE_CONFIG)`; compute SHA256 of the base config file and assert equal to
`BASE_CONFIG_SHA256` (fail loudly on mismatch — mirrors `verify_frozen_inputs`).
Build `task_config = replace(task_config_from_dict(base_config), batch_size=BATCH_SIZE, distractor_steps=DISTRACTOR_STEPS)`.
Load `checkpoints = trained_distractor_checkpoints(".", POOL_MANIFEST)` and
assert the 10 seeds match `retained_checkpoint_seeds`.

**Cell 5 (md) — Collection helpers.** Explain the two collection wrappers and the
onset-alignment helper (GAP-1 mitigation).

**Cell 6 (code) — `collect_clean(model, task_config, gain)`.** Wrap
`_collect_batches` for `condition="clean"`, `condition_index=0`,
`delay_steps=CLEAN_DELAY`, `seed_base=FINAL_SEED_BASE`, `n_batches=N_BATCHES`,
`batch_size=BATCH_SIZE`, `forward_fn = None if gain==1.0 else _operator_forward(...operator="state_persistence", variant="carried_state_only", strength=gain, condition="clean", family="A", delay_steps=CLEAN_DELAY)`,
`randomize_distractor_onsets=False`. Return the collected dict.

**Cell 7 (code) — `collect_distractor(model, task_config, gain)`.** Same but
`condition="distractor"`, `condition_index=1`, `delay_steps=DISTRACTOR_DELAY`,
`randomize_distractor_onsets=True`, and forward built with
`condition="distractor"`. Assert `collected["distractor_relative_starts"] is not None`
and `collected["distractor_angles"] is not None`.

**Cell 8 (code) — Onset-aligned distractor helper (GAP-1).**
`onset_aligned_attraction(decoded[time,trials], targets[trials], distractors[trials], relative_starts[trials], delay_start:int, duration:int, post_len:int)`:
1. For each trial, build an aligned window of length `duration + post_len`
   starting at absolute time `delay_start + relative_starts[t]`, clipping/guarding
   the tail against `delay.stop`/array end (use the max feasible common
   `post_len` = `delay_steps - duration - max(relative_start)`; verified feasible
   because max start = 15, duration = 5, delay = 20 ⇒ some trials have 0 post
   steps inside the delay — so extend the post window into the **response**-free
   remaining delay only, and where trials run out, mask them out of the recovery
   statistic rather than padding). Concretely: compute per-trial
   `avail_post = delay.stop - (delay_start + rel + duration)` and set the common
   `post_len = min over trials`, or bucket trials by onset and compute recovery
   only on trials with `avail_post >= post_len_min` (document choice; recommend
   bucket-by-onset to keep all trials, then average the per-onset recovery).
2. Compute signed displacement `signed_circular_error(decoded, targets)` and the
   signed target→distractor arc, exactly as `distractor_drift_and_recovery` does
   internally (reuse `perturbation_metrics._signed_wrapped_radians` semantics via
   `signed_circular_error`), forming `attraction = displacement / arc`.
3. Return `{peak_attraction_fraction, end_attraction_fraction, recovery_fraction, peak_drift_degrees}` computed on the **aligned** trajectory (peak = max |mean attraction| over aligned steps; recovery = (peak - end)/peak).
Rationale: the stock `distractor_drift_and_recovery` assumes a single shared
window; with variable onsets we must align per trial before averaging. The
alignment can be done entirely in the notebook from data already returned by
`_collect_batches` (no src change required). Keep the arc/attraction math
identical to the metric module so numbers are comparable to fixed-timing runs.

**Cell 9 (code) — Clean-delay metric helper.**
`clean_delay_metrics(collected, weights)`:
- `delay = collected["phase_index"]["delay"]`; late window
  `slice(delay.stop - 10, delay.stop)` (mirrors runs' late-delay = last 10).
- `de = delay_decoding_error(hidden, angles, weights, delay)` → mean/median decode
  error across the full delay (this is the "decoded-angle drift" proxy). Also
  compute a **drift** measure: decode angle at `delay.start` vs `delay.stop-1`
  via `decode_angles_from_hidden` and `signed_circular_error`, mean |Δ| degrees.
- `hidden = collected["hidden_states"]`; `norm = np.linalg.norm(hidden[delay], axis=-1)` → mean over trials → per-step; report mean over delay.
- `speed = _hidden_speed(hidden)[delay.start:delay.stop-1]` (Δh norms within
  delay) → mean.
- `act = activation_slope_and_saturation(hidden[delay])`.
Return a flat dict of scalars.

**Cell 10 (code) — Distractor metric helper.**
`distractor_metrics(collected, weights)`: decode
`decoded = decode_angles_from_hidden(collected["hidden_states"], weights)`; call
the Cell-8 onset-aligned helper with `targets=collected["angles"]`,
`distractors=collected["distractor_angles"]`,
`relative_starts=collected["distractor_relative_starts"]`,
`delay_start=collected["phase_index"]["delay"].start`, `duration=5`. Return
scalars (peak attraction fraction, recovery fraction, peak drift degrees).

**Cell 11 (code) — Main loop over seeds × conditions.**
```
rows = []          # per-seed per-condition metric rows
pca_store = {}     # seed -> {gain -> mean_delay_trajectory [time_delay, units]}
for ckpt in checkpoints:
    model = _load_checkpoint_model(base_config, ckpt.path, DEVICE)
    weights = fit_frozen_decoder(model, task_config, FAMILY)  # once per seed
    for gain in PERSISTENCE:
        clean = collect_clean(model, task_config, gain)
        dist  = collect_distractor(model, task_config, gain)
        row = {"seed": ckpt.seed, "persistence": gain,
               **clean_delay_metrics(clean, weights),
               **distractor_metrics(dist, weights)}
        rows.append(row)
        # store mean clean delay-window hidden trajectory for PCA
        d = clean["phase_index"]["delay"]
        pca_store.setdefault(ckpt.seed, {})[gain] = \
            clean["hidden_states"][d].mean(axis=1)   # [delay_len, units]
    del model; torch.cuda.empty_cache()
metrics_df = pd.DataFrame(rows)
```
Memory: only one model resident at a time.

**Cell 12 (code) — Frozen-PCA fit & projection (see §5).**

**Cell 13 (code) — Save metrics.** Write `metrics_df` to CSV and a JSON summary
(pooled mean ± SD across seeds per condition) to the paths in §6.

**Cell 14 (code) — Figure 1: PCA trajectories.** (§5, spec in §6.)

**Cell 15 (code) — Figure 2: drift/recovery summary.** (spec in §6.)

**Cell 16 (md) — Results note + sanity checklist.** Inline the validation results
(§8) and point to the results-note path (§6). Restate the descriptive boundary.

---

## 5. Frozen-PCA procedure (exact)

1. **Basis fit — once, on baseline only.** Concatenate the **persistence-1.00**
   clean delay-window hidden states across seeds (or, if a single shared basis is
   preferred, per seed — recommend a **per-seed basis** because units are not
   aligned across seeds; pooling across seeds mixes bases and is not
   interpretable). For each seed: `X = clean_1.00["hidden_states"][delay].reshape(-1, units)` (shape `[delay_len*1024, 64]`); `pca = PCA(n_components=2).fit(X)`.
   Store `pca` and `pca.explained_variance_ratio_`.
2. **Freeze.** Do **not** refit for 0.95/0.90.
3. **Project mean trajectories.** For each gain, take the stored mean delay
   trajectory `pca_store[seed][gain]` (`[delay_len, 64]`) and
   `pca.transform(mean_traj)` → `[delay_len, 2]`. (Projecting the mean is exactly
   what the plan calls for; optionally also project a handful of individual
   baseline trials coloured by cue angle for context, using the same frozen
   `pca`.)
4. For the figure, either show one representative seed or a small grid of seeds;
   default: one representative seed (e.g. first seed 20260801) full detail, plus
   a pooled panel overlaying the three gains' mean trajectories for that seed.
   Keep the basis per-seed; never transform seed B's data with seed A's PCA.

Tanh handling: fit/transform on raw hidden (tanh) states directly, matching
`baseline_competence_figures` — **no arctanh**.

---

## 6. Outputs

Base dir: `outputs/persistence_hidden_state_analysis/`

- `metrics/persistence_hidden_state_metrics.csv` — one row per (seed, persistence)
  with all clean-delay + distractor scalars.
- `metrics/persistence_hidden_state_summary.json` — pooled mean/SD per condition,
  plus provenance: base config SHA256, manifest path, checkpoint seeds+sha256,
  `FINAL_SEED_BASE`, git commit (via `subprocess` like
  `perturbation_experiment._git_metadata`), and the interpretive/claim-boundary
  string.
- `figures/persistence_pca_delay_trajectories.png` — frozen-PCA figure.
- `figures/persistence_drift_recovery_summary.png` — drift/recovery figure.

Results note: `docs/reports/state_persistence_hidden_state_analysis_results.md`
(short; created/filled by the downstream agent after execution — the notebook
should print the exact numbers to paste). Also add a `docs/changelog.md` entry
per repo rules once the run is executed (Chronological Run Log + any commit).

### Figure specs

**Fig 1 (PCA):** `figsize≈(6.4,5)`; one panel overlaying the three mean delay
trajectories (1.00 solid, 0.95 dashed, 0.90 dotted) for a representative seed, in
the frozen baseline basis; mark delay start (open marker) and delay end (filled
marker); axis labels `PC1 (xx.x% var)` / `PC2 (xx.x% var)` from
`explained_variance_ratio_`; `ax.set_aspect("equal", adjustable="datalim")`.
Optional second panel: faint baseline single-trial trajectories coloured by cue
angle (hsv), reusing `plot_circular_pca` styling.

**Fig 2 (drift/recovery):** grouped summary across the three persistence values.
Left: clean-delay decode error (mean ± SD over seeds) and hidden step-speed vs
persistence. Right: distractor peak-attraction fraction and recovery fraction
(mean ± SD over seeds) vs persistence. Points per seed as thin overlaid markers;
connect condition means. Use the muted palette from
`baseline_competence_figures._setup_style` if convenient (optional).

---

## 7. Open risks / gaps

- **GAP-1 (must handle in notebook): distractor analysis is not onset-aligned in
  existing code.** `distractor_drift_and_recovery` takes a single fixed
  `distractor_slice`/`post_distractor_slice`, and `summarize_collected` **skips**
  the distractor metric entirely when `distractor_relative_starts is not None`
  (`perturbation_experiment.py:630-633`). Therefore the variable-timing distractor
  metric must be recomputed with per-trial alignment. **The needed inputs already
  exist** in `_collect_batches` output (`hidden_states`, `distractor_angles`,
  `distractor_relative_starts`, `phase_index`), so the fix is a **notebook-local
  helper** (Cell 8) that aligns each trial by its onset before averaging — **no
  `src/` change required**. Keep the attraction/arc math identical to
  `perturbation_metrics` for comparability.
- **Post-distractor window is short/variable.** With delay=20, duration=5, onsets
  0..15, late-onset trials have little/no in-delay recovery window. Recommended
  handling: bucket trials by onset and compute recovery only where a post window
  exists, then average across onsets (document the exact rule and the effective N
  per statistic). Do **not** silently pad. Flag any onset bucket dropped for
  insufficient post-window.
- **NO GAP for hidden-state exposure.** `state_persistence` returns hidden states
  via `_run_explicit`; `_collect_batches` returns them for every forward function.
  Persistence 1.00 is a genuine no-op (native baseline). Confirmed in code.
- **PCA basis across seeds.** Hidden units are not aligned across seeds, so a
  single pooled PCA basis is not interpretable. Plan uses a **per-seed frozen
  basis** (fit on that seed's baseline). If a single figure across seeds is later
  wanted, align via Procrustes — out of scope here.
- **Clean "drift" definition.** The plan uses delay decode error (mean over delay)
  plus a start-vs-end decoded-angle displacement as the drift measure. If the
  supervisor expects a specific drift definition, this is the adjustable knob;
  both are cheap to compute from the same arrays.
- **Determinism / provenance.** All seeding is internal to `_collect_batches`;
  record `FINAL_SEED_BASE`, config SHA256, and checkpoint SHA256 in the summary
  JSON. Verify base-config SHA256 before loading models (Cell 4).
- **Claim boundary.** Descriptive mechanism evidence only. Do not add a Gaussian
  matched-cost comparator or any specificity language.

---

## 8. Validation / sanity checklist (run inside the notebook)

1. **Pairing check:** for a fixed seed, assert `collect_distractor(model,·,1.00)`
   and `collect_distractor(model,·,0.90)` return identical `angles`,
   `distractor_angles`, and `distractor_relative_starts` (element-wise equal).
   Same for clean `angles`. This proves the frozen/paired banks.
2. **No-op check:** persistence 1.00 clean hidden states equal a plain
   `model(inputs)` collection (forward_fn=None) — assert allclose on a small
   batch.
3. **Shape contracts:** `hidden_states.shape == (T, 1024, 64)`;
   `weights.shape == (64, 2)`; `decode_angles_from_hidden(hidden, weights).shape == (T, 1024)`.
4. **Range check:** hidden states within [-1-1e-6, 1+1e-6] (tanh support), so
   `activation_slope_and_saturation` does not raise.
5. **Onset coverage:** `np.bincount(distractor_relative_starts)` is (near-)uniform
   over 0..15 (balanced bank) and count sums to 1024.
6. **PCA sanity:** `explained_variance_ratio_[:2].sum()` reported; frozen basis
   used unchanged for all three gains (assert same `pca` object id in projection).
7. **Monotonicity spot-check (descriptive only, not a claim):** print
   clean-delay decode error and distractor attraction for 1.00 < 0.95 < 0.90 to
   confirm the pipeline produces sensible ordering; do not gate on it.
8. **Row count:** `metrics_df` has exactly `10 seeds × 3 conditions = 30` rows.
