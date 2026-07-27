# Working-Memory RNN

This repository contains the executable modelling work for a dissertation on
working memory and psilocybin-related behavioural signatures. The project asks
whether controlled perturbations of trained recurrent neural networks can
reproduce selected human behavioural dissociations. It tests computational
sufficiency; it is not a literal pharmacological model of psilocybin.

## Current Experiment

The current task battery has two trained model families:

1. **Fixation-gated circular working memory** tests response settling,
   angular accuracy, and delay-length dependence.
2. **N-back working memory** tests the difference between a low-memory 0-back
   condition and a working-memory-dependent 2-back condition.

The completed 1,024-trial candidate screen applies synaptic-drive gain,
heterogeneous drive gain, sensory and distractor input gain, recurrent gain,
state persistence, and effective time-constant perturbations to frozen
checkpoints from both families. These are competing computational operators,
not biological claims about receptor action.

The present circular distractor condition was introduced only during
evaluation, so it is an exploratory out-of-distribution robustness test. A
separately trained, single-item distractor-capable circular family is the
planned route for a stronger distractor-filtering comparison.

## Canonical Files

| Purpose | Path |
|---|---|
| Circular baseline configuration | `configs/fixation_circular_working_memory.yaml` |
| N-back checkpoint-pool configuration | `configs/nback_working_memory_screened_final.yaml` |
| Completed 1,024-trial perturbation screen | `configs/full_candidate_perturbation_1024.yaml` |
| Circular task and training | `src/wm_rnn/tuned_task.py`, `src/wm_rnn/train.py` |
| N-back task and training | `src/wm_rnn/nback_task.py`, `src/wm_rnn/train_nback.py` |
| Perturbation operators | `src/wm_rnn/perturbation_operators.py`, `src/wm_rnn/nback_perturbation.py` |
| Full candidate runner | `src/wm_rnn/full_candidate_perturbation_run.py` |
| Current scientific write-up | `docs/reports/full_candidate_perturbation_scientific_writeup.tex` |
| Repository history and run log | `docs/changelog.md` |

See `docs/repository-map.md` for the active, supporting, historical, generated,
and local-only areas of the repository. See `configs/README.md` before choosing
a configuration: several root-level YAML files are frozen development or
preregistration records and are not current recommendations.

## Installation

Create an environment and install the package dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

CUDA-specific installation notes are recorded in `requirements-cuda.txt`.
The local `.venv/` is intentionally ignored by Git.

For module commands run from the repository root:

```powershell
$env:PYTHONPATH = "src"
```

## Common Commands

Run the test suite:

```powershell
$env:PYTHONPATH = "src"
python -m pytest
```

Train the circular baseline:

```powershell
python -m wm_rnn.train --config configs/fixation_circular_working_memory.yaml
```

Inspect or rebuild the screened N-back checkpoint pool:

```powershell
python -m wm_rnn.nback_screened_pool `
  --config configs/nback_working_memory_screened_final.yaml
```

Run the frozen full candidate screen:

```powershell
python -m wm_rnn.full_candidate_perturbation_run
```

The full screen is expensive and normally should not be rerun merely to inspect
the saved results.

## Repository Layout

- `src/wm_rnn/`: model, task, training, evaluation, analysis, and experiment
  code for the current task and candidate-evaluation path. Historical one-off
  runners are recovered from Git rather than retained in the active package.
- `tests/`: unit and integration tests.
- `configs/`: active and frozen experiment configurations, classified in
  `configs/README.md`.
- `docs/preregistration/`: preregistrations and outcome-independent audit
  records. These are retained at their original paths for provenance.
- `docs/reports/`: scientific reports, LaTeX sources, final figures, and PDFs.
- `docs/archive/`: explicitly superseded explanatory documents.
- `notebooks/`: thin interactive walkthroughs and figure-generation notebooks.
- `outputs/`: ignored checkpoints, metrics, arrays, and generated figures.
- `tmp/`: ignored disposable scratch space; safe to remove when no process is
  using it.

## Reproducibility Rules

- Treat independently trained checkpoints, rather than trials, as the
  inferential unit.
- Preserve preregistrations, audit records, checkpoint manifests, and hashes.
- Keep circular delay-memory analysis on hidden states because the circular
  output is intentionally silent before response.
- Keep generic disruption controls and candidate gain/persistence mechanisms
  conceptually separate from psilocybin.
- Record model changes, experiment runs, generated reports, and interpretation
  changes in `docs/changelog.md`.

## Generated Data

`outputs/`, `tmp/`, `.pytest_cache/`, and `.venv/` are ignored by Git. Their
presence can make the workspace appear much larger than the tracked
repository. Do not delete checkpoint or metric directories solely because they
are ignored: use `docs/repository-map.md` and the relevant run record to
determine whether an output is authoritative, reproducible, or disposable.
