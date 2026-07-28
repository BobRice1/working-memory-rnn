# Repository Map

This map distinguishes current executable work from retained scientific
history and generated local data. It is the navigation source of truth for the
repository; `docs/changelog.md` remains the chronological source of truth.

## Current Scientific Path

```text
trained circular checkpoints ─┐
                              ├─> frozen perturbation operators
trained N-back checkpoints ───┘             │
                                             v
                              behavioural signature summaries
                                             │
                                             v
                                  scientific LaTeX report
```

### Circular family

- Current distractor-trained configuration:
  `configs/fixation_circular_distractor_working_memory.yaml`
- Prior clean-trained configuration:
  `configs/fixation_circular_working_memory.yaml`
- Task: `src/wm_rnn/tuned_task.py`
- Training: `src/wm_rnn/train.py`
- Distractor-trained checkpoint pool:
  `src/wm_rnn/circular_distractor_pool.py`
- Hidden-state memory decoding primitive:
  `src/wm_rnn/hidden_angle_decoder.py`
- Multi-delay and trained-distractor evaluation:
  `src/wm_rnn/circular_family_a_pilot.py`
- Baseline competence and hidden-state figures:
  `src/wm_rnn/baseline_competence_figures.py`

### N-back family

- Configuration: `configs/nback_working_memory_screened_final.yaml`
- Task: `src/wm_rnn/nback_task.py`
- Training: `src/wm_rnn/train_nback.py`
- Pool screening: `src/wm_rnn/nback_screened_pool.py`
- Evaluation: `src/wm_rnn/nback_evaluation.py`
- Baseline competence and hidden-state figures:
  `src/wm_rnn/baseline_competence_figures.py`

### Candidate perturbation evaluation

- Current configuration:
  `configs/full_candidate_perturbation_trained_distractor_1024.yaml`
- Prior clean-trained configuration:
  `configs/full_candidate_perturbation_1024.yaml`
- Circular operators and metrics:
  `src/wm_rnn/perturbation_operators.py`,
  `src/wm_rnn/perturbation_metrics.py`
- N-back operators and metrics:
  `src/wm_rnn/nback_perturbation.py`,
  `src/wm_rnn/nback_metrics.py`
- Runner: `src/wm_rnn/full_candidate_perturbation_run.py`
- Summaries and figures: `src/wm_rnn/exploratory_pilot_summary.py`,
  `src/wm_rnn/scientific_writeup_figures.py`
- Results: `docs/reports/full_candidate_perturbation_1024_results.md`
- Paper: `docs/reports/full_candidate_perturbation_scientific_writeup.tex`
- Current generated grid:
  `outputs/full_candidate_perturbation_trained_distractor_1024/`
- The N-back component was not retrained or recomputed because neither its
  checkpoints, task, operators, nor evaluation settings changed; the current
  cross-task summary records the reused source table explicitly.

## Retired Executable Work

Completed outputs and documentation are retained for the following work, but
their one-off executable modules have been removed from the active package:

- Gaussian and structured-noise control generation.
- Additive N-back calibration, cost-checking, and phased confirmatory execution.
- Failed two-slot Family B evaluation and rescue orchestration.
- Fixed-point, Jacobian, PCA, stability, delay-sweep, and movie utilities.
- Superseded perturbation scoring and assignment-sensitivity pipelines.

The exact implementations remain available in Git commit `d64b2cf`. Current
candidate evaluation does not import them.

## Historical Scientific Records

The following material is deliberately retained:

- `docs/preregistration/`: frozen plans, amendments, failure audits, and
  competence-gate records.
- `docs/changelog.md`: experiment chronology and interpretation history.
- Failed two-slot multicondition configurations and their tests.
- N-back training rescue configurations and audits.
- `docs/archive/` and `configs/archive/`: explicitly superseded model
  documentation and configurations.

Historical material may be reorganised only when old paths remain resolvable or
the affected reproduction instructions are migrated and verified.

## Generated and Local-Only Data

| Path | Git status | Treatment |
|---|---|---|
| `.venv/` | Ignored | Local Python/CUDA environment. Recreate only when necessary. |
| `outputs/` | Ignored | Checkpoints, metrics, arrays, figures, and rendered media. Review run provenance before deletion. |
| `tmp/` | Ignored | Disposable scratch output. Safe to remove after confirming no active process uses it. |
| `.pytest_cache/` | Ignored | Disposable test cache. |

The largest local output area is normally rendered media rather than trained
models or tracked source. Workspace size therefore should not be used as a
proxy for Git repository size.

## Active Package Boundary

The active `src/wm_rnn/` package now contains 26 substantive modules:

- shared configuration, device, I/O, model, and training utilities;
- current circular and N-back task/training code;
- current perturbation operators, metrics, and candidate evaluators;
- current checkpoint screening, result summaries, and report figures.

Historical scripts are recovered from Git when needed rather than kept beside
the current implementation.
