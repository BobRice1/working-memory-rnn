# Circular Distractor-Timing Generalisation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible frozen-checkpoint evaluation that moves the same five-step circular distractor across five positions in a 20-step delay.

**Architecture:** Add one focused evaluation module that reuses the existing circular batch generator, checkpoint verification, frozen decoder and metric implementations. The module will resolve timing conditions, verify paired target/distractor banks, evaluate the five retained checkpoints, calculate checkpoint-level timing-minus-midpoint comparisons and save isolated CSV/JSON outputs.

**Tech Stack:** Python 3, PyTorch, NumPy, SciPy, pytest, existing `wm_rnn` task and analysis utilities.

---

## File structure

- Create `src/wm_rnn/circular_distractor_timing.py`: timing-condition resolution,
  paired-bank verification, frozen evaluation, summaries and CLI.
- Create `tests/test_circular_distractor_timing.py`: timing, pairing, comparison
  and execution-firewall tests.
- Create `configs/circular_distractor_timing_generalisation.yaml`: frozen
  checkpoint-independent evaluation settings.
- Create `docs/reports/circular_distractor_timing_generalisation.md`: result and
  interpretation after execution.
- Modify `configs/README.md`: register the completed robustness configuration.
- Modify `docs/changelog.md`: record implementation, run and result.
- Modify vault project-state notes after outcomes are verified.

### Task 1: Timing-condition resolution

**Files:**

- Create: `tests/test_circular_distractor_timing.py`
- Create: `src/wm_rnn/circular_distractor_timing.py`

- [ ] **Step 1: Write the failing timing-resolution test**

```python
from wm_rnn.circular_distractor_timing import (
    TIMING_CONDITIONS,
    resolve_timing_task,
)
from wm_rnn.tuned_task import TunedDelayTaskConfig


def test_timing_conditions_resolve_to_frozen_delay_starts():
    base = TunedDelayTaskConfig(
        pre_cue_steps=25,
        cue_steps=20,
        delay_steps=20,
        response_steps=25,
        batch_size=8,
        fixation_gated=True,
        distractor_steps=5,
    )
    expected = {
        "clean": (0, 8),
        "start": (5, 0),
        "quarter": (5, 4),
        "midpoint": (5, 8),
        "three_quarter": (5, 11),
        "end": (5, 15),
    }
    for label, (steps, relative_start) in expected.items():
        task = resolve_timing_task(base, label)
        batch_start = round(
            (task.delay_steps - task.distractor_steps)
            * task.distractor_onset_fraction
        )
        assert task.distractor_steps == steps
        assert batch_start == relative_start
    assert tuple(TIMING_CONDITIONS) == tuple(expected)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/test_circular_distractor_timing.py::test_timing_conditions_resolve_to_frozen_delay_starts -q
```

Expected: collection fails because
`wm_rnn.circular_distractor_timing` does not exist.

- [ ] **Step 3: Implement the minimal resolver**

```python
from dataclasses import replace

from wm_rnn.tuned_task import TunedDelayTaskConfig


TIMING_CONDITIONS = {
    "clean": None,
    "start": 0.00,
    "quarter": 0.25,
    "midpoint": 0.50,
    "three_quarter": 0.75,
    "end": 1.00,
}


def resolve_timing_task(
    base: TunedDelayTaskConfig,
    condition: str,
) -> TunedDelayTaskConfig:
    if condition not in TIMING_CONDITIONS:
        raise ValueError(f"unknown timing condition: {condition}")
    fraction = TIMING_CONDITIONS[condition]
    if fraction is None:
        return replace(base, distractor_steps=0)
    return replace(
        base,
        distractor_steps=5,
        distractor_onset_fraction=float(fraction),
    )
```

- [ ] **Step 4: Run the test and verify GREEN**

Run the Step 2 command.

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```powershell
git add src/wm_rnn/circular_distractor_timing.py tests/test_circular_distractor_timing.py
git commit -m "Add circular distractor timing conditions"
```

### Task 2: Paired-bank verification

**Files:**

- Modify: `tests/test_circular_distractor_timing.py`
- Modify: `src/wm_rnn/circular_distractor_timing.py`

- [ ] **Step 1: Write the failing paired-bank test**

```python
import numpy as np

from wm_rnn.circular_distractor_timing import generate_paired_banks


def test_paired_banks_hold_angles_constant_and_move_only_timing():
    base = TunedDelayTaskConfig(
        n_tuned_units=8,
        pre_cue_steps=3,
        cue_steps=4,
        delay_steps=20,
        response_steps=5,
        batch_size=16,
        seed=99,
        fixation_gated=True,
        distractor_steps=5,
    )
    banks = generate_paired_banks(base, seed=123)
    reference_targets = banks["clean"].angles
    reference_distractors = banks["midpoint"].distractor_angles
    for batch in banks.values():
        assert np.array_equal(batch.angles, reference_targets)
    for label in TIMING_CONDITIONS:
        if label != "clean":
            assert np.array_equal(
                banks[label].distractor_angles,
                reference_distractors,
            )
    assert banks["start"].phase_index["distractor"].start != (
        banks["midpoint"].phase_index["distractor"].start
    )
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/test_circular_distractor_timing.py::test_paired_banks_hold_angles_constant_and_move_only_timing -q
```

Expected: import fails because `generate_paired_banks` is absent.

- [ ] **Step 3: Implement paired generation and validation**

```python
import numpy as np

from wm_rnn.tuned_task import generate_tuned_delay_batch


def generate_paired_banks(
    base: TunedDelayTaskConfig,
    *,
    seed: int,
) -> dict[str, object]:
    banks = {
        label: generate_tuned_delay_batch(
            replace(resolve_timing_task(base, label), seed=int(seed))
        )
        for label in TIMING_CONDITIONS
    }
    reference_targets = banks["clean"].angles
    reference_distractors = banks["midpoint"].distractor_angles
    if not all(
        np.array_equal(batch.angles, reference_targets)
        for batch in banks.values()
    ):
        raise RuntimeError("target-angle banks are not paired")
    if not all(
        np.array_equal(banks[label].distractor_angles, reference_distractors)
        for label in TIMING_CONDITIONS
        if label != "clean"
    ):
        raise RuntimeError("distractor-angle banks are not paired")
    return banks
```

- [ ] **Step 4: Run all timing tests and verify GREEN**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/test_circular_distractor_timing.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```powershell
git add src/wm_rnn/circular_distractor_timing.py tests/test_circular_distractor_timing.py
git commit -m "Verify paired distractor timing banks"
```

### Task 3: Checkpoint-level comparisons and summaries

**Files:**

- Modify: `tests/test_circular_distractor_timing.py`
- Modify: `src/wm_rnn/circular_distractor_timing.py`

- [ ] **Step 1: Write the failing comparison test**

```python
from wm_rnn.circular_distractor_timing import checkpoint_comparisons


def test_checkpoint_comparisons_use_clean_cost_and_midpoint_reference():
    metrics = {
        "clean": {"mean_angular_error_degrees": 2.0},
        "start": {"mean_angular_error_degrees": 4.0},
        "quarter": {"mean_angular_error_degrees": 3.5},
        "midpoint": {"mean_angular_error_degrees": 3.0},
        "three_quarter": {"mean_angular_error_degrees": 3.25},
        "end": {"mean_angular_error_degrees": 4.5},
    }
    rows = checkpoint_comparisons(123, metrics)
    by_condition = {row["condition"]: row for row in rows}
    assert by_condition["midpoint"]["distractor_cost_degrees"] == 1.0
    assert by_condition["start"]["timing_minus_midpoint_degrees"] == 1.0
    assert by_condition["quarter"]["timing_minus_midpoint_degrees"] == 0.5
    assert by_condition["three_quarter"]["timing_minus_midpoint_degrees"] == 0.25
    assert by_condition["end"]["timing_minus_midpoint_degrees"] == 1.5
```

- [ ] **Step 2: Run the comparison test and verify RED**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/test_circular_distractor_timing.py::test_checkpoint_comparisons_use_clean_cost_and_midpoint_reference -q
```

Expected: import fails because `checkpoint_comparisons` is absent.

- [ ] **Step 3: Implement checkpoint comparisons**

```python
def checkpoint_comparisons(
    checkpoint_seed: int,
    metrics: dict[str, dict[str, float]],
) -> list[dict[str, float | int | str]]:
    clean_error = float(metrics["clean"]["mean_angular_error_degrees"])
    midpoint_cost = (
        float(metrics["midpoint"]["mean_angular_error_degrees"]) - clean_error
    )
    rows = []
    for condition in TIMING_CONDITIONS:
        if condition == "clean":
            continue
        error = float(metrics[condition]["mean_angular_error_degrees"])
        cost = error - clean_error
        rows.append(
            {
                "checkpoint_seed": int(checkpoint_seed),
                "condition": condition,
                "distractor_cost_degrees": cost,
                "timing_minus_midpoint_degrees": cost - midpoint_cost,
            }
        )
    return rows
```

- [ ] **Step 4: Add and test `summarize_comparisons`**

Add this test:

```python
import pytest

from wm_rnn.circular_distractor_timing import summarize_comparisons


def test_summarize_comparisons_uses_checkpoint_as_unit():
    rows = [
        {
            "checkpoint_seed": seed,
            "condition": "start",
            "distractor_cost_degrees": value + 2.0,
            "timing_minus_midpoint_degrees": value,
        }
        for seed, value in enumerate((1.0, 2.0, 3.0, 4.0, 5.0), start=1)
    ]
    summary = summarize_comparisons(rows)[0]
    assert summary["condition"] == "start"
    assert summary["n_checkpoints"] == 5
    assert summary["mean_timing_minus_midpoint_degrees"] == 3.0
    assert summary["sd_timing_minus_midpoint_degrees"] == pytest.approx(
        1.5811388300841898
    )
    assert summary["positive_checkpoints"] == 5
    assert summary["ci95_low"] < 3.0 < summary["ci95_high"]
```

Implement:

```python
import math

import numpy as np
from scipy.stats import t


def summarize_comparisons(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    summaries = []
    for condition in TIMING_CONDITIONS:
        if condition == "clean":
            continue
        selected = [row for row in rows if row["condition"] == condition]
        values = np.asarray(
            [
                float(row["timing_minus_midpoint_degrees"])
                for row in selected
            ],
            dtype=np.float64,
        )
        if values.size < 2:
            raise ValueError("at least two checkpoints are required")
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        half_width = float(
            t.ppf(0.975, values.size - 1)
            * sd
            / math.sqrt(values.size)
        )
        summaries.append(
            {
                "condition": condition,
                "n_checkpoints": int(values.size),
                "mean_timing_minus_midpoint_degrees": mean,
                "sd_timing_minus_midpoint_degrees": sd,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
                "positive_checkpoints": int(np.sum(values > 0.0)),
                "negative_checkpoints": int(np.sum(values < 0.0)),
            }
        )
    return summaries
```

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/test_circular_distractor_timing.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/wm_rnn/circular_distractor_timing.py tests/test_circular_distractor_timing.py
git commit -m "Summarize circular distractor timing effects"
```

### Task 4: Frozen evaluation runner and firewall

**Files:**

- Create: `configs/circular_distractor_timing_generalisation.yaml`
- Modify: `tests/test_circular_distractor_timing.py`
- Modify: `src/wm_rnn/circular_distractor_timing.py`

- [ ] **Step 1: Add the fixed configuration**

```yaml
evaluation:
  output_dir: outputs/circular_distractor_timing_generalisation
  trials_per_condition: 1024
  batch_size: 128
  delay_steps: 20
  distractor_steps: 5
  onset_fractions:
    clean: null
    start: 0.00
    quarter: 0.25
    midpoint: 0.50
    three_quarter: 0.75
    end: 1.00
  seed_base: 202607300

sources:
  circular_config: configs/fixation_circular_distractor_working_memory.yaml
  circular_manifest: outputs/fixation_circular_distractor_working_memory/metrics/fixation_circular_distractor_working_memory_pool_summary.json

interpretation:
  status: post_result_robustness_analysis
```

- [ ] **Step 2: Write failing config and execute-firewall tests**

```python
def test_design_loads_without_checkpoint_execution(tmp_path, monkeypatch):
    design = load_design(
        Path("configs/circular_distractor_timing_generalisation.yaml")
    )
    assert design["evaluation"]["trials_per_condition"] == 1024
    assert list(design["evaluation"]["onset_fractions"]) == list(
        TIMING_CONDITIONS
    )


def test_cli_requires_execute(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["circular_distractor_timing"],
    )
    main()
    assert "design_only" in capsys.readouterr().out
```

- [ ] **Step 3: Run firewall tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/test_circular_distractor_timing.py -q
```

Expected: failures because `load_design` and `main` are absent.

- [ ] **Step 4: Implement design validation and the frozen runner**

Use these imports and constants:

```python
import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from wm_rnn.circular_family_a_pilot import (
    _baseline_threshold,
    _collect_batches,
    fit_frozen_decoder,
    summarize_collected,
    verify_frozen_inputs,
)
from wm_rnn.config import load_config
from wm_rnn.device import select_device
from wm_rnn.full_candidate_perturbation_run import (
    BASE_CONFIG,
    BASE_CONFIG_SHA256,
    trained_distractor_checkpoints,
)
from wm_rnn.io import ensure_run_dirs, write_json
from wm_rnn.perturbation_experiment import (
    FINAL_SEED_BASE,
    _load_checkpoint_model,
)
from wm_rnn.training_utils import task_config_from_dict


DESIGN_PATH = Path(
    "configs/circular_distractor_timing_generalisation.yaml"
)
MIDPOINT_SOURCE = Path(
    "outputs/full_candidate_perturbation_trained_distractor_1024/"
    "circular_trained_distractor/metrics/"
    "circular_trained_distractor_grid.csv"
)
```

Validate the design without loading outcomes:

```python
def load_design(path: str | Path = DESIGN_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        design = yaml.safe_load(handle) or {}
    evaluation = design["evaluation"]
    if int(evaluation["trials_per_condition"]) != 1024:
        raise ValueError("trials_per_condition must remain 1024")
    if int(evaluation["batch_size"]) != 128:
        raise ValueError("batch_size must remain 128")
    if int(evaluation["delay_steps"]) != 20:
        raise ValueError("delay_steps must remain 20")
    if evaluation["onset_fractions"] != TIMING_CONDITIONS:
        raise ValueError("timing conditions differ from the frozen design")
    return design
```

Add a CSV helper:

```python
def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
```

Implement `run_evaluation` with the following complete data flow:

```python
def run_evaluation(
    repo_root: str | Path = ".",
    *,
    device: str = "auto",
    design_path: str | Path = DESIGN_PATH,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    design = load_design(root / design_path)
    checkpoints = trained_distractor_checkpoints(root)
    frozen = verify_frozen_inputs(
        root,
        checkpoints=checkpoints,
        config_path=root / BASE_CONFIG,
        expected_config_sha256=BASE_CONFIG_SHA256,
    )
    config = load_config(root / BASE_CONFIG)
    base_task = task_config_from_dict(config, batch_size=128)
    base_task = replace(
        base_task,
        pre_cue_steps=25,
        cue_steps=20,
        delay_steps=20,
        response_steps=25,
        distractor_steps=5,
    )
    selected = select_device(device)
    n_batches = 1024 // 128
    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    bank_checks: dict[str, Any] = {}

    for checkpoint in checkpoints:
        model = _load_checkpoint_model(
            config,
            root / checkpoint.path,
            selected.device,
        )
        decoder = fit_frozen_decoder(model, base_task, "A")
        threshold = _baseline_threshold(
            model,
            resolve_timing_task(base_task, "midpoint"),
            "A",
            "distractor",
            1,
            20,
        )
        collected_by_condition = {}
        metrics_by_condition = {}
        for label in TIMING_CONDITIONS:
            task = resolve_timing_task(base_task, label)
            condition = "clean" if label == "clean" else "distractor"
            collected = _collect_batches(
                model,
                task,
                "A",
                condition,
                1,
                20,
                seed_base=FINAL_SEED_BASE,
                n_batches=n_batches,
                batch_size=128,
            )
            collected_by_condition[label] = collected
            metric = summarize_collected(
                collected,
                decoder,
                threshold,
                family="A",
            )[0]
            metrics_by_condition[label] = metric
            metric_rows.append(
                {
                    "checkpoint_seed": checkpoint.seed,
                    "condition": label,
                    "onset_fraction": (
                        "" if label == "clean"
                        else TIMING_CONDITIONS[label]
                    ),
                    "delay_relative_start": (
                        ""
                        if label == "clean"
                        else collected["phase_index"]["distractor"].start
                        - collected["phase_index"]["delay"].start
                    ),
                    **metric,
                }
            )

        target_reference = collected_by_condition["clean"]["angles"]
        distractor_reference = collected_by_condition["midpoint"][
            "distractor_angles"
        ]
        targets_equal = all(
            np.array_equal(
                collected_by_condition[label]["angles"],
                target_reference,
            )
            for label in TIMING_CONDITIONS
        )
        distractors_equal = all(
            np.array_equal(
                collected_by_condition[label]["distractor_angles"],
                distractor_reference,
            )
            for label in TIMING_CONDITIONS
            if label != "clean"
        )
        if not targets_equal or not distractors_equal:
            raise RuntimeError("paired-bank verification failed")
        bank_checks[str(checkpoint.seed)] = {
            "targets_equal": targets_equal,
            "distractors_equal": distractors_equal,
            "angle_hashes": {
                label: collected_by_condition[label]["angle_hashes"]
                for label in TIMING_CONDITIONS
            },
        }
        comparison_rows.extend(
            checkpoint_comparisons(
                checkpoint.seed,
                metrics_by_condition,
            )
        )

    if len(metric_rows) != 30 or len(comparison_rows) != 25:
        raise RuntimeError("unexpected output row count")
    summary_rows = summarize_comparisons(comparison_rows)
    dirs = ensure_run_dirs(root / design["evaluation"]["output_dir"])
    metric_path = _write_csv(
        dirs["metrics"] / "timing_metrics.csv",
        metric_rows,
    )
    comparison_path = _write_csv(
        dirs["metrics"] / "timing_comparisons.csv",
        comparison_rows,
    )
    summary_path = write_json(
        dirs["metrics"] / "timing_summary.json",
        {
            "status": "post_result_robustness_analysis",
            "device": selected.description,
            "frozen_inputs": frozen,
            "bank_checks": bank_checks,
            "summary": summary_rows,
        },
    )
    return {
        "metrics": str(metric_path),
        "comparisons": str(comparison_path),
        "summary": str(summary_path),
    }
```

Before writing outputs, load `MIDPOINT_SOURCE`, select each checkpoint's native
`condition=distractor`, `delay_steps=20` baseline, and assert:

```python
np.testing.assert_allclose(
    metrics_by_condition["midpoint"]["mean_angular_error_degrees"],
    expected_midpoint_error,
    rtol=0.0,
    atol=1e-9,
)
```

The CLI prints the design without loading checkpoints unless `--execute` is
present:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "design_only", **load_design()}, indent=2))
        return
    print(
        json.dumps(
            run_evaluation(args.repo_root, device=args.device),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the focused tests**

Run:

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/test_circular_distractor_timing.py tests/test_circular_distractor_training.py tests/test_full_candidate_perturbation_run.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add configs/circular_distractor_timing_generalisation.yaml src/wm_rnn/circular_distractor_timing.py tests/test_circular_distractor_timing.py
git commit -m "Add circular distractor timing evaluator"
```

### Task 5: Execute, verify and report

**Files:**

- Create: `docs/reports/circular_distractor_timing_generalisation.md`
- Modify: `configs/README.md`
- Modify: `docs/changelog.md`
- Modify: `../wiki-hot-cache.md`
- Modify: `../wiki/experiments/psilocybin-signature-perturbation-experiment-plan.md`

- [ ] **Step 1: Run the frozen evaluation**

```powershell
$env:PYTHONPATH='src'
python -m wm_rnn.circular_distractor_timing --execute --device auto
```

Expected:

- 30 full metric rows: five checkpoints by six conditions;
- 25 comparison rows: five checkpoints by five distractor timings;
- paired-bank verification passes;
- midpoint-reproduction checks pass.

- [ ] **Step 2: Inspect all checkpoint-level outcomes**

Read:

```text
outputs/circular_distractor_timing_generalisation/metrics/timing_metrics.csv
outputs/circular_distractor_timing_generalisation/metrics/timing_comparisons.csv
outputs/circular_distractor_timing_generalisation/metrics/timing_summary.json
```

Report actual checkpoint values and do not infer generalisation from the mean
alone.

- [ ] **Step 3: Write the result report**

The report must include:

- question and post-result status;
- exact timing positions;
- pairing and midpoint-reproduction checks;
- per-checkpoint angular errors and distractor costs;
- timing-minus-midpoint mean, SD, 95% interval and direction count;
- fixation and settling validity;
- conclusion using only the frozen interpretation categories;
- limits on claims about exclusive timing strategies.

- [ ] **Step 4: Update durable project records**

Add the configuration to `configs/README.md`, a chronological run entry to
`docs/changelog.md`, and a concise evidence-calibrated project-state update to
the vault hot cache and experiment plan.

- [ ] **Step 5: Run final verification**

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/test_circular_distractor_timing.py tests/test_circular_distractor_training.py tests/test_full_candidate_perturbation_run.py -q
git diff --check
git status --short --branch
```

Expected: zero test failures and no whitespace errors.

- [ ] **Step 6: Commit the result record**

```powershell
git add configs/README.md docs/changelog.md docs/reports/circular_distractor_timing_generalisation.md
git commit -m "Report circular distractor timing generalisation"
```
