"""Family B competence evaluation across the frozen 2 x 2 task conditions."""

from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wm_rnn.device import select_device
from wm_rnn.io import ensure_run_dirs, write_json
from wm_rnn.training_utils import (
    batch_to_tensors,
    fresh_model,
    generate_batch_for_task,
    task_config_from_dict,
    tuned_response_metrics,
)
from wm_rnn.tuned_task import circular_angular_error, decode_population_angle


CONDITIONS = (
    "load1_clean",
    "load1_distractor",
    "load2_clean",
    "load2_distractor",
)


@dataclass(frozen=True)
class FamilyBEvaluationResult:
    """Paths and rows for a Family B baseline-competence evaluation."""

    metrics_path: Path
    csv_path: Path
    rows: list[dict[str, Any]]
    acceptance: dict[str, Any]


def _condition_config(
    config: dict[str, Any], condition: str, seed_offset: int
):
    condition_config = deepcopy(config)
    condition_config["task"]["delay_steps"] = 20
    condition_config["task"]["n_items"] = (
        2 if condition.startswith("load2") else 1
    )
    condition_config["task"]["distractor_steps"] = (
        int(config["task"].get("distractor_steps", 0))
        if condition.endswith("distractor")
        else 0
    )
    return task_config_from_dict(condition_config, seed_offset=seed_offset)


def evaluate_family_b_conditions(
    config: dict[str, Any],
    checkpoint_path: str | Path,
) -> FamilyBEvaluationResult:
    """Evaluate all load/distractor cells and both serial positions at delay 20."""
    if not bool(config["task"].get("probe_gated", False)):
        raise ValueError("Family B evaluation requires task.probe_gated=true")
    device_info = select_device(config["training"].get("device", "auto"))
    model = fresh_model(config, device_info.device)
    checkpoint = torch.load(checkpoint_path, map_location=device_info.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    batches = int(config["evaluation"]["batches"])
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for condition_index, condition in enumerate(CONDITIONS):
            errors_by_position: dict[int, list[float]] = {0: [], 1: []}
            fixation_accuracies: list[float] = []
            for batch_index in range(batches):
                task_config = _condition_config(
                    config,
                    condition,
                    seed_offset=40000 + 1000 * condition_index + batch_index,
                )
                batch = generate_batch_for_task(task_config)
                inputs, targets, loss_mask = batch_to_tensors(
                    batch, device_info.device
                )
                predictions, _ = model(inputs)
                metrics = tuned_response_metrics(
                    predictions,
                    targets,
                    loss_mask,
                    batch.preferred_angles,
                    batch.angles,
                )
                fixation_accuracies.append(float(metrics["fixation_accuracy"]))
                response = batch.phase_index["response"]
                populations = (
                    predictions[
                        response, :, : len(batch.preferred_angles)
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )
                decoded = decode_population_angle(
                    populations, batch.preferred_angles
                )
                errors = np.degrees(
                    circular_angular_error(
                        decoded, batch.angles[np.newaxis, :]
                    )
                )
                for position in (0, 1):
                    selected = batch.probed_index == position
                    errors_by_position[position].extend(
                        errors[:, selected].reshape(-1).astype(float).tolist()
                    )

            pooled = errors_by_position[0] + errors_by_position[1]
            fixation_accuracy = float(np.mean(fixation_accuracies))
            for position, values in (
                ("pooled", pooled),
                ("first", errors_by_position[0]),
                ("second", errors_by_position[1]),
            ):
                error_array = np.asarray(values, dtype=np.float64)
                rows.append(
                    {
                        "condition": condition,
                        "position": position,
                        "delay_steps": 20,
                        "mean_angular_error_degrees": float(
                            np.mean(error_array)
                        ),
                        "median_angular_error_degrees": float(
                            np.median(error_array)
                        ),
                        "fixation_accuracy": fixation_accuracy,
                        "n_response_samples": int(error_array.size),
                    }
                )

    lookup = {
        (row["condition"], row["position"]): row for row in rows
    }
    checks: dict[str, bool] = {
        "load1_clean_under_10": (
            lookup[("load1_clean", "pooled")][
                "mean_angular_error_degrees"
            ]
            < 10.0
        ),
        "load2_clean_first_under_45": (
            lookup[("load2_clean", "first")][
                "mean_angular_error_degrees"
            ]
            < 45.0
        ),
        "load2_clean_second_under_45": (
            lookup[("load2_clean", "second")][
                "mean_angular_error_degrees"
            ]
            < 45.0
        ),
    }
    for condition in ("load1_distractor", "load2_distractor"):
        for position in ("first", "second"):
            checks[f"{condition}_{position}_under_45"] = (
                lookup[(condition, position)][
                    "mean_angular_error_degrees"
                ]
                < 45.0
            )
    checks["all_condition_fixation_at_least_0_94"] = all(
        lookup[(condition, "pooled")]["fixation_accuracy"] >= 0.94
        for condition in CONDITIONS
    )
    acceptance = {
        "passed": bool(all(checks.values())),
        "checks": checks,
    }

    dirs = ensure_run_dirs(config["paths"]["output_dir"])
    run_name = config["paths"].get(
        "run_name", "multicondition_working_memory"
    )
    csv_path = dirs["metrics"] / f"{run_name}_family_b_acceptance.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics_path = write_json(
        dirs["metrics"] / f"{run_name}_family_b_acceptance.json",
        {
            "checkpoint": str(checkpoint_path),
            "device": device_info.description,
            "batches": batches,
            "rows": rows,
            "acceptance": acceptance,
        },
    )
    return FamilyBEvaluationResult(
        metrics_path=metrics_path,
        csv_path=csv_path,
        rows=rows,
        acceptance=acceptance,
    )
