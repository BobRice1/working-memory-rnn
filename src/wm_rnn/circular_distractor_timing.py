"""Evaluate circular distractor filtering away from its trained timing."""

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t
import yaml

from wm_rnn.circular_family_a_pilot import verify_frozen_inputs
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
    _baseline_threshold,
    _collect_batches,
    _load_checkpoint_model,
    fit_frozen_decoder,
    summarize_collected,
)
from wm_rnn.training_utils import task_config_from_dict
from wm_rnn.tuned_task import (
    TunedDelayBatch,
    TunedDelayTaskConfig,
    generate_tuned_delay_batch,
)


TIMING_CONDITIONS = {
    "clean": None,
    "start": 0.00,
    "quarter": 0.25,
    "midpoint": 0.50,
    "three_quarter": 0.75,
    "end": 1.00,
}

DESIGN_PATH = Path(
    "configs/circular_distractor_timing_generalisation.yaml"
)
MIDPOINT_SOURCE = Path(
    "outputs/full_candidate_perturbation_trained_distractor_1024/"
    "circular_trained_distractor/metrics/"
    "circular_trained_distractor_grid.csv"
)


def resolve_timing_task(
    base: TunedDelayTaskConfig,
    condition: str,
) -> TunedDelayTaskConfig:
    """Return the frozen delay-20 task for one distractor timing."""
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


def generate_paired_banks(
    base: TunedDelayTaskConfig,
    *,
    seed: int,
) -> dict[str, TunedDelayBatch]:
    """Generate matched targets and distractors for every timing condition."""
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


def checkpoint_comparisons(
    checkpoint_seed: int,
    metrics: dict[str, dict[str, float]],
) -> list[dict[str, float | int | str]]:
    """Calculate distractor costs and deviations from trained midpoint timing."""
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


def summarize_comparisons(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    """Summarise paired timing contrasts across independently trained seeds."""
    summaries = []
    available = {str(row["condition"]) for row in rows}
    for condition in TIMING_CONDITIONS:
        if condition == "clean" or condition not in available:
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


def load_design(path: str | Path = DESIGN_PATH) -> dict[str, Any]:
    """Load and validate the fixed post-result robustness design."""
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


def load_midpoint_references(path: str | Path) -> dict[int, float]:
    """Read each checkpoint's native trained-midpoint distractor error."""
    references: dict[int, float] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["condition"] == "distractor"
                and int(row["delay_steps"]) == 20
                and row["operator"] == "state_persistence"
                and float(row["strength"]) == 1.0
            ):
                seed = int(row["checkpoint_seed"])
                if seed in references:
                    raise ValueError(f"duplicate midpoint reference for {seed}")
                references[seed] = float(
                    row["mean_angular_error_degrees"]
                )
    if not references:
        raise ValueError("no native midpoint references found")
    return references


def assert_midpoint_reproduction(
    observed: float,
    expected: float,
) -> None:
    """Require archived midpoint reproduction within GPU roundoff."""
    np.testing.assert_allclose(
        float(observed),
        float(expected),
        rtol=0.0,
        atol=1e-6,
    )


def relative_distractor_start(
    condition: str,
    phase_index: dict[str, slice],
) -> int | str:
    """Return the delay-relative onset, or blank when no distractor exists."""
    if condition == "clean":
        return ""
    return (
        phase_index["distractor"].start
        - phase_index["delay"].start
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write non-empty records using their insertion-ordered columns."""
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_evaluation(
    repo_root: str | Path = ".",
    *,
    device: str = "auto",
    design_path: str | Path = DESIGN_PATH,
) -> dict[str, Any]:
    """Run the paired frozen-checkpoint timing-generalisation evaluation."""
    root = Path(repo_root).resolve()
    resolved_design = Path(design_path)
    if not resolved_design.is_absolute():
        resolved_design = (Path.cwd() / resolved_design).resolve()
    design = load_design(resolved_design)
    evaluation = design["evaluation"]
    checkpoints = trained_distractor_checkpoints(root)
    frozen = verify_frozen_inputs(
        root,
        checkpoints=checkpoints,
        config_path=root / BASE_CONFIG,
        expected_config_sha256=BASE_CONFIG_SHA256,
    )
    midpoint_references = load_midpoint_references(root / MIDPOINT_SOURCE)
    checkpoint_seeds = {checkpoint.seed for checkpoint in checkpoints}
    if set(midpoint_references) != checkpoint_seeds:
        raise RuntimeError("midpoint reference checkpoints do not match")

    config = load_config(root / BASE_CONFIG)
    batch_size = int(evaluation["batch_size"])
    trials = int(evaluation["trials_per_condition"])
    n_batches, remainder = divmod(trials, batch_size)
    if remainder:
        raise ValueError("trials_per_condition must divide evenly by batch_size")
    base_task = task_config_from_dict(config, batch_size=batch_size)
    base_task = replace(
        base_task,
        delay_steps=int(evaluation["delay_steps"]),
        distractor_steps=int(evaluation["distractor_steps"]),
    )
    selected = select_device(device)
    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    bank_checks: dict[str, Any] = {}
    midpoint_checks: dict[str, Any] = {}

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
            int(evaluation["delay_steps"]),
        )
        collected_by_condition: dict[str, dict[str, Any]] = {}
        metrics_by_condition: dict[str, dict[str, Any]] = {}
        for label in TIMING_CONDITIONS:
            task = resolve_timing_task(base_task, label)
            condition = "clean" if label == "clean" else "distractor"
            collected = _collect_batches(
                model,
                task,
                "A",
                condition,
                1,
                int(evaluation["delay_steps"]),
                seed_base=int(evaluation["seed_base"]),
                n_batches=n_batches,
                batch_size=batch_size,
            )
            metric = summarize_collected(
                collected,
                decoder,
                threshold,
                family="A",
            )[0]
            collected_by_condition[label] = collected
            metrics_by_condition[label] = metric
            metric_rows.append(
                {
                    "checkpoint_seed": checkpoint.seed,
                    "condition": label,
                    "onset_fraction": (
                        "" if label == "clean" else TIMING_CONDITIONS[label]
                    ),
                    "delay_relative_start": relative_distractor_start(
                        label,
                        collected["phase_index"],
                    ),
                    **metric,
                }
            )

        observed_midpoint = float(
            metrics_by_condition["midpoint"][
                "mean_angular_error_degrees"
            ]
        )
        expected_midpoint = midpoint_references[checkpoint.seed]
        assert_midpoint_reproduction(
            observed_midpoint,
            expected_midpoint,
        )
        midpoint_checks[str(checkpoint.seed)] = {
            "expected_mean_angular_error_degrees": expected_midpoint,
            "observed_mean_angular_error_degrees": observed_midpoint,
            "absolute_difference_degrees": abs(
                observed_midpoint - expected_midpoint
            ),
        }

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
            "target_angle_hashes": {
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

    expected_metric_rows = len(checkpoints) * len(TIMING_CONDITIONS)
    expected_comparison_rows = len(checkpoints) * (
        len(TIMING_CONDITIONS) - 1
    )
    if (
        len(metric_rows) != expected_metric_rows
        or len(comparison_rows) != expected_comparison_rows
    ):
        raise RuntimeError("unexpected output row count")
    summary_rows = summarize_comparisons(comparison_rows)
    output_root = root / str(evaluation["output_dir"])
    dirs = ensure_run_dirs(output_root)
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
            "status": design["interpretation"]["status"],
            "device": selected.description,
            "design_path": str(resolved_design),
            "design": design,
            "frozen_inputs": frozen,
            "midpoint_source": str(root / MIDPOINT_SOURCE),
            "midpoint_checks": midpoint_checks,
            "bank_checks": bank_checks,
            "summary": summary_rows,
        },
    )
    return {
        "status": design["interpretation"]["status"],
        "metrics": str(metric_path),
        "comparisons": str(comparison_path),
        "summary": str(summary_path),
    }


def main() -> None:
    """Print the frozen design unless checkpoint execution is explicit."""
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
