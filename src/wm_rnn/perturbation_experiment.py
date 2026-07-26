"""Deterministic harness for the psilocybin-signature perturbation experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from wm_rnn.config import load_config
from wm_rnn.hidden_angle_decoder import (
    decode_angles_from_hidden,
    fit_hidden_angle_decoder,
)
from wm_rnn.io import ensure_run_dirs, write_json
from wm_rnn.perturbation_metrics import (
    activation_slope_and_saturation,
    assess_settling_validity,
    delay_decoding_error,
    distractor_drift_and_recovery,
    marginal_state_entropy,
    response_geometry_measures,
    signed_circular_error,
    time_to_threshold,
)
from wm_rnn.perturbation_calibration import (
    calibrate_bidirectional,
    calibrate_strength,
    paired_bootstrap_proportional_cost,
    proportional_cost,
    required_cost_check_trials,
    round_up_to_batch,
)
from wm_rnn.perturbation_operators import (
    ForwardFn,
    distractor_input_gain,
    gaussian_state_noise,
    heterogeneous_drive_gain,
    recurrent_gain,
    sensory_input_gain,
    state_persistence,
    synaptic_drive_gain,
    time_constant,
)
from wm_rnn.training_utils import (
    batch_to_tensors,
    fresh_model,
    generate_batch_for_task,
    task_config_from_dict,
)
from wm_rnn.tuned_task import (
    TunedDelayBatch,
    TunedDelayTaskConfig,
    circular_angular_error,
    decode_population_angle,
)


DECODER_SEED_BASE = 202607100
CALIBRATION_SEED_BASE = 202607200
COST_CHECK_SEED_BASE = 202607250
FINAL_SEED_BASE = 202607300
P5_REPLICATES = (4101, 4102, 4103)
P2_VECTOR_SEEDS = (3101, 3102, 3103)
PREREGISTRATION_COMMIT = "5f888be"

MULTIPLICATIVE_GRID = (0.90, 0.95, 0.975, 1.0, 1.025, 1.05, 1.10, 1.15, 1.20)
P2_GRID = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15)
P5_GRID = (0.0, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10)
P7_GRID = (0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25)

SIGNATURE_COLUMNS = [
    "family",
    "operator",
    "variant",
    "strength",
    "strength_kind",
    "gain_vector_seed",
    "noise_replicate",
    "condition",
    "delay_steps",
    "seed",
    "n_trials",
    "baseline_mean_angular_error_degrees",
    "mean_angular_error_degrees",
    "median_angular_error_degrees",
    "delta_angular_error_degrees",
    "proportional_error_change",
    "proportional_clean_cost",
    "cost_match_valid",
    "population_mse",
    "fixation_accuracy",
    "fixation_valid",
    "baseline_median_settling_steps",
    "median_settling_steps",
    "delta_median_settling_steps",
    "baseline_restricted_mean_settling_steps",
    "restricted_mean_settling_steps",
    "delta_restricted_mean_settling_steps",
    "baseline_fraction_settled",
    "fraction_settled",
    "delta_fraction_settled",
    "baseline_failure_rate",
    "failure_rate",
    "delta_failure_rate",
    "latency_valid",
    "settling_validity_reason",
    "item_position",
    "mean_retention_steps",
    "delay_decode_error_degrees",
    "delta_delay_decode_error_degrees",
    "distractor_peak_drift_degrees",
    "distractor_recovery_fraction",
    "distractor_peak_attraction_fraction",
    "distractor_end_attraction_fraction",
    "mean_signed_error_degrees",
    "circular_bias_degrees",
    "mean_vector_length",
    "vector_length_cv",
    "mean_activation_slope",
    "saturation_fraction",
    "mean_late_delay_state_entropy",
    "delta_mean_late_delay_state_entropy",
    "delay_is_extrapolated",
]


@dataclass(frozen=True)
class ExperimentResult:
    """Paths and rows written by one harness invocation."""

    grid_path: Path
    metadata_path: Path
    rows: list[dict[str, Any]]


def frozen_batch_seed(
    seed_base: int,
    condition_index: int,
    delay_steps: int,
    batch_index: int,
) -> int:
    """Map a cell to a deterministic, documented batch seed."""
    if min(condition_index, delay_steps, batch_index) < 0:
        raise ValueError("seed offsets must be non-negative")
    return (
        int(seed_base)
        + 10_000 * int(condition_index)
        + 100 * int(delay_steps)
        + int(batch_index)
    )


def hash_angles(angles: np.ndarray) -> str:
    """Return the SHA256 of a contiguous angle array including shape and dtype."""
    values = np.ascontiguousarray(angles)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def condition_normalized_change(
    baseline_error: float, perturbed_error: float
) -> float:
    """Return the pre-registered condition-normalized error change."""
    baseline = float(baseline_error)
    if not np.isfinite(baseline) or baseline <= 0.0:
        raise ValueError("baseline_error must be finite and positive")
    if not np.isfinite(perturbed_error):
        raise ValueError("perturbed_error must be finite")
    return (float(perturbed_error) - baseline) / baseline


def compute_excess_constraints(
    baseline_errors: dict[str, float],
    candidate_errors: dict[str, float],
    p5_errors: dict[str, float],
    *,
    baseline_rmst: float,
    candidate_rmst: float,
    p5_rmst: float,
) -> dict[str, float]:
    """Construct C1-C3 and descriptive absolute DiDs from condition means."""
    required = {"load1_clean", "load2_clean", "load1_distractor"}
    for name, values in (
        ("baseline_errors", baseline_errors),
        ("candidate_errors", candidate_errors),
        ("p5_errors", p5_errors),
    ):
        missing = required - set(values)
        if missing:
            raise ValueError(f"{name} missing conditions: {sorted(missing)}")
    candidate_r = {
        condition: condition_normalized_change(
            baseline_errors[condition], candidate_errors[condition]
        )
        for condition in required
    }
    p5_r = {
        condition: condition_normalized_change(
            baseline_errors[condition], p5_errors[condition]
        )
        for condition in required
    }
    load_candidate = (
        candidate_r["load2_clean"] - candidate_r["load1_clean"]
    )
    load_p5 = p5_r["load2_clean"] - p5_r["load1_clean"]
    distractor_candidate = (
        candidate_r["load1_distractor"] - candidate_r["load1_clean"]
    )
    distractor_p5 = (
        p5_r["load1_distractor"] - p5_r["load1_clean"]
    )
    return {
        "x1": (candidate_rmst - baseline_rmst) - (p5_rmst - baseline_rmst),
        "x2": load_candidate - load_p5,
        "x3": distractor_candidate - distractor_p5,
        "candidate_load_proportional_did": load_candidate,
        "candidate_distractor_proportional_did": distractor_candidate,
        "candidate_load_absolute_did": (
            candidate_errors["load2_clean"]
            - baseline_errors["load2_clean"]
        )
        - (
            candidate_errors["load1_clean"]
            - baseline_errors["load1_clean"]
        ),
        "candidate_distractor_absolute_did": (
            candidate_errors["load1_distractor"]
            - baseline_errors["load1_distractor"]
        )
        - (
            candidate_errors["load1_clean"]
            - baseline_errors["load1_clean"]
        ),
    }


def average_p5_replicates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average nested P5 replicates within checkpoint/cell, never across seeds."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    grouping = (
        "family",
        "operator",
        "variant",
        "strength",
        "strength_kind",
        "condition",
        "delay_steps",
        "seed",
        "item_position",
    )
    for row in rows:
        groups[tuple(row.get(column) for column in grouping)].append(row)
    averaged: list[dict[str, Any]] = []
    for members in groups.values():
        record = members[0].copy()
        numeric_columns = [
            column
            for column in SIGNATURE_COLUMNS
            if all(
                isinstance(member.get(column), (int, float, np.number))
                and not isinstance(member.get(column), bool)
                for member in members
            )
        ]
        for column in numeric_columns:
            record[column] = float(
                np.mean([float(member[column]) for member in members])
            )
        record["noise_replicate"] = ""
        record["n_noise_replicates"] = len(members)
        averaged.append(record)
    return averaged


def _condition_task_config(
    base: TunedDelayTaskConfig,
    family: str,
    condition: str,
    delay_steps: int,
    seed: int,
    batch_size: int,
) -> TunedDelayTaskConfig:
    updates: dict[str, Any] = {
        "delay_steps": int(delay_steps),
        "seed": int(seed),
        "batch_size": int(batch_size),
    }
    if family == "B":
        updates["n_items"] = 2 if condition.startswith("load2") else 1
        updates["distractor_steps"] = (
            base.distractor_steps if condition.endswith("distractor") else 0
        )
    else:
        updates["distractor_steps"] = (
            base.distractor_steps if condition == "distractor" else 0
        )
    return replace(base, **updates)


def _collect_batches(
    model: torch.nn.Module,
    base_task: TunedDelayTaskConfig,
    family: str,
    condition: str,
    condition_index: int,
    delay_steps: int,
    *,
    seed_base: int,
    n_batches: int,
    batch_size: int,
    forward_fn: ForwardFn | None = None,
    forward_seed_factory: Callable[[int], ForwardFn] | None = None,
) -> dict[str, Any]:
    predictions: list[np.ndarray] = []
    hidden_states: list[np.ndarray] = []
    batches: list[TunedDelayBatch] = []
    angle_hashes: list[str] = []
    for batch_index in range(n_batches):
        batch_seed = frozen_batch_seed(
            seed_base, condition_index, delay_steps, batch_index
        )
        task_config = _condition_task_config(
            base_task,
            family,
            condition,
            delay_steps,
            batch_seed,
            batch_size,
        )
        batch = generate_batch_for_task(task_config)
        inputs, _, _ = batch_to_tensors(batch, next(model.parameters()).device)
        with torch.no_grad():
            resolved_forward = (
                forward_seed_factory(batch_seed)
                if forward_seed_factory is not None
                else forward_fn
            )
            output, hidden = (
                model(inputs)
                if resolved_forward is None
                else resolved_forward(inputs)
            )
        predictions.append(output.detach().cpu().numpy())
        hidden_states.append(hidden.detach().cpu().numpy())
        batches.append(batch)
        angle_hashes.append(hash_angles(batch.angles))
    return {
        "predictions": np.concatenate(predictions, axis=1),
        "hidden_states": np.concatenate(hidden_states, axis=1),
        "batches": batches,
        "angles": np.concatenate([batch.angles for batch in batches]),
        "probed_index": (
            np.concatenate([batch.probed_index for batch in batches])
            if family == "B"
            else None
        ),
        "retention": (
            np.concatenate([batch.probed_retention_steps for batch in batches])
            if family == "B"
            else None
        ),
        "distractor_angles": (
            np.concatenate([batch.distractor_angles for batch in batches])
            if batches[0].distractor_angles is not None
            else None
        ),
        "phase_index": batches[0].phase_index,
        "preferred_angles": batches[0].preferred_angles,
        "angle_hashes": angle_hashes,
    }


def fit_frozen_decoder(
    model: torch.nn.Module,
    base_task: TunedDelayTaskConfig,
    family: str,
    *,
    trials_per_delay: int = 64,
    ridge_alpha: float = 1.0,
) -> np.ndarray:
    """Fit one checkpoint decoder from equal clean reference samples per delay."""
    hidden_samples: list[np.ndarray] = []
    angle_samples: list[np.ndarray] = []
    clean_condition = "load1_clean" if family == "B" else "clean"
    for delay_index, delay_steps in enumerate((10, 20, 40, 80)):
        collected = _collect_batches(
            model,
            base_task,
            family,
            clean_condition,
            delay_index,
            delay_steps,
            seed_base=DECODER_SEED_BASE,
            n_batches=1,
            batch_size=trials_per_delay,
        )
        late_delay = collected["phase_index"]["delay"]
        start = late_delay.stop - min(10, delay_steps)
        selected = collected["hidden_states"][start : late_delay.stop]
        hidden_samples.append(selected.reshape(-1, selected.shape[-1]))
        angle_samples.append(
            np.tile(collected["angles"], selected.shape[0])
        )
    hidden = np.concatenate(hidden_samples, axis=0)
    angles = np.concatenate(angle_samples, axis=0)
    return fit_hidden_angle_decoder(hidden, angles, ridge_alpha)


def _vector_length(populations: np.ndarray, preferred: np.ndarray) -> np.ndarray:
    x = np.sum(populations * np.cos(preferred), axis=-1)
    y = np.sum(populations * np.sin(preferred), axis=-1)
    return np.hypot(x, y)


def _position_masks(
    family: str, probed_index: np.ndarray | None, n_trials: int
) -> list[tuple[str, np.ndarray]]:
    if family != "B":
        return [("", np.ones(n_trials, dtype=bool))]
    return [
        ("pooled", np.ones(n_trials, dtype=bool)),
        ("first", probed_index == 0),
        ("second", probed_index == 1),
    ]


def summarize_collected(
    collected: dict[str, Any],
    decoder_weights: np.ndarray,
    baseline_vector_length: float,
    *,
    family: str,
    fixation_floor: float = 0.90,
    baseline_fraction_settled: float | dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Compute every Phase 1 metric for pooled and serial-position rows."""
    predictions = collected["predictions"]
    hidden = collected["hidden_states"]
    preferred = collected["preferred_angles"]
    angles = collected["angles"]
    phases = collected["phase_index"]
    response = phases["response"]
    delay = phases["delay"]
    populations = predictions[..., : len(preferred)]
    decoded_output = decode_population_angle(populations, preferred)
    signed_errors = signed_circular_error(decoded_output, angles[np.newaxis, :])
    absolute_errors = np.abs(signed_errors)
    vector_lengths = _vector_length(populations, preferred)
    fixation_accuracy = (
        float(
            np.mean(
                (predictions[..., len(preferred)] >= 0.5)
                == (
                    np.concatenate(
                        (
                            np.ones(
                                (response.start, predictions.shape[1]),
                                dtype=bool,
                            ),
                            np.zeros(
                                (
                                    predictions.shape[0] - response.start,
                                    predictions.shape[1],
                                ),
                                dtype=bool,
                            ),
                        ),
                        axis=0,
                    )
                )
            )
        )
        if predictions.shape[-1] > len(preferred)
        else 1.0
    )
    decoded_hidden = decode_angles_from_hidden(hidden, decoder_weights)
    late_delay = slice(delay.stop - min(10, delay.stop - delay.start), delay.stop)
    rows: list[dict[str, Any]] = []
    for position, mask in _position_masks(
        family, collected["probed_index"], len(angles)
    ):
        response_errors = absolute_errors[response, :][:, mask]
        response_vectors = vector_lengths[response, :][:, mask]
        settling = time_to_threshold(
            response_errors,
            response_vectors,
            slice(0, response_errors.shape[0]),
            baseline_vector_length,
        )
        if baseline_fraction_settled is None:
            baseline_fraction = float(settling["fraction_settled"])
        elif isinstance(baseline_fraction_settled, dict):
            baseline_fraction = float(baseline_fraction_settled[position])
        else:
            baseline_fraction = float(baseline_fraction_settled)
        validity = assess_settling_validity(
            fixation_accuracy,
            baseline_fraction,
            float(settling["fraction_settled"]),
            fixation_floor=fixation_floor,
        )
        selected_hidden = hidden[:, mask, :]
        delay_metrics = delay_decoding_error(
            selected_hidden,
            angles[mask],
            decoder_weights,
            late_delay,
        )
        geometry = response_geometry_measures(
            signed_errors[:, mask],
            populations[:, mask, :],
            response,
        )
        activation = activation_slope_and_saturation(
            selected_hidden[late_delay]
        )
        entropy = marginal_state_entropy(selected_hidden[late_delay])
        distractor = {
            "distractor_peak_drift_degrees": np.nan,
            "distractor_recovery_fraction": np.nan,
            "distractor_peak_attraction_fraction": np.nan,
            "distractor_end_attraction_fraction": np.nan,
        }
        if collected["distractor_angles"] is not None:
            distractor_metrics = distractor_drift_and_recovery(
                decoded_hidden[:, mask],
                angles[mask],
                collected["distractor_angles"][mask],
                phases["distractor"],
                slice(phases["distractor"].stop, delay.stop),
            )
            distractor = {
                "distractor_peak_drift_degrees": distractor_metrics[
                    "distractor_peak_drift_degrees"
                ],
                "distractor_recovery_fraction": distractor_metrics[
                    "distractor_recovery_fraction"
                ],
                "distractor_peak_attraction_fraction": distractor_metrics[
                    "distractor_peak_attraction_fraction"
                ],
                "distractor_end_attraction_fraction": distractor_metrics[
                    "distractor_end_attraction_fraction"
                ],
            }
        response_population = populations[response, :][:, mask]
        response_targets = np.concatenate(
            [
                batch.targets[
                    batch.phase_index["response"],
                    :,
                    : len(preferred),
                ]
                for batch in collected["batches"]
            ],
            axis=1,
        )[:, mask]
        retention = collected["retention"]
        row = {
            "mean_angular_error_degrees": float(np.mean(response_errors)),
            "median_angular_error_degrees": float(np.median(response_errors)),
            "population_mse": float(
                np.mean((response_population - response_targets) ** 2)
            ),
            "fixation_accuracy": fixation_accuracy,
            "fixation_valid": fixation_accuracy >= fixation_floor,
            "median_settling_steps": settling["median_settling_steps"],
            "restricted_mean_settling_steps": settling[
                "restricted_mean_settling_steps"
            ],
            "fraction_settled": settling["fraction_settled"],
            "failure_rate": settling["failure_rate"],
            "latency_valid": validity["latency_valid"],
            "settling_validity_reason": validity[
                "settling_validity_reason"
            ],
            "item_position": position,
            "mean_retention_steps": (
                float(np.mean(retention[mask])) if retention is not None else ""
            ),
            "delay_decode_error_degrees": delay_metrics[
                "mean_error_degrees"
            ],
            **distractor,
            **geometry,
            "mean_activation_slope": activation["mean_activation_slope"],
            "saturation_fraction": activation["saturation_fraction"],
            "mean_late_delay_state_entropy": entropy["mean_entropy"],
            "n_trials": int(np.sum(mask)),
        }
        rows.append(row)
    return rows


def _baseline_threshold(
    model: torch.nn.Module,
    base_task: TunedDelayTaskConfig,
    family: str,
    condition: str,
    condition_index: int,
    delay_steps: int,
) -> float:
    reference = _collect_batches(
        model,
        base_task,
        family,
        condition,
        condition_index,
        delay_steps,
        seed_base=DECODER_SEED_BASE + 50_000,
        n_batches=1,
        batch_size=64,
    )
    response = reference["phase_index"]["response"]
    populations = reference["predictions"][
        response, :, : len(reference["preferred_angles"])
    ]
    return float(
        np.median(
            _vector_length(populations, reference["preferred_angles"])
        )
    )


def _empty_signature_row() -> dict[str, Any]:
    return {column: "" for column in SIGNATURE_COLUMNS}


def _merge_with_baseline(
    metric: dict[str, Any],
    baseline: dict[str, Any],
    *,
    family: str,
    operator: str,
    variant: str,
    strength: float,
    strength_kind: str,
    condition: str,
    delay_steps: int,
    seed: int,
    noise_replicate: int | str = "",
) -> dict[str, Any]:
    row = _empty_signature_row()
    row.update(
        {
            "family": family,
            "operator": operator,
            "variant": variant,
            "strength": float(strength),
            "strength_kind": strength_kind,
            "gain_vector_seed": "",
            "noise_replicate": noise_replicate,
            "condition": condition,
            "delay_steps": int(delay_steps),
            "seed": int(seed),
            "delay_is_extrapolated": bool(delay_steps == 160),
            **metric,
        }
    )
    paired = (
        "mean_angular_error_degrees",
        "median_settling_steps",
        "restricted_mean_settling_steps",
        "fraction_settled",
        "failure_rate",
        "delay_decode_error_degrees",
        "mean_late_delay_state_entropy",
    )
    for name in paired:
        row[f"baseline_{name}"] = baseline[name]
    row["delta_angular_error_degrees"] = (
        metric["mean_angular_error_degrees"]
        - baseline["mean_angular_error_degrees"]
    )
    row["proportional_error_change"] = condition_normalized_change(
        baseline["mean_angular_error_degrees"],
        metric["mean_angular_error_degrees"],
    )
    row["delta_median_settling_steps"] = (
        metric["median_settling_steps"] - baseline["median_settling_steps"]
    )
    row["delta_restricted_mean_settling_steps"] = (
        metric["restricted_mean_settling_steps"]
        - baseline["restricted_mean_settling_steps"]
    )
    row["delta_fraction_settled"] = (
        metric["fraction_settled"] - baseline["fraction_settled"]
    )
    row["delta_failure_rate"] = (
        metric["failure_rate"] - baseline["failure_rate"]
    )
    row["delta_delay_decode_error_degrees"] = (
        metric["delay_decode_error_degrees"]
        - baseline["delay_decode_error_degrees"]
    )
    row["delta_mean_late_delay_state_entropy"] = (
        metric["mean_late_delay_state_entropy"]
        - baseline["mean_late_delay_state_entropy"]
    )
    clean_condition = "load1_clean" if family == "B" else "clean"
    if condition == clean_condition:
        row["proportional_clean_cost"] = row["proportional_error_change"]
    return row


def _write_grid(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SIGNATURE_COLUMNS)
        writer.writeheader()
        writer.writerows(
            [{column: row.get(column, "") for column in SIGNATURE_COLUMNS} for row in rows]
        )
    return path


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        )
        return {"git_commit": commit, "worktree_clean": not bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "worktree_clean": False}


def run_smoke_experiment(
    model: torch.nn.Module,
    task_config: TunedDelayTaskConfig,
    output_dir: str | Path,
    *,
    family: str | None = None,
    checkpoint_seed: int = 0,
) -> ExperimentResult:
    """Run the frozen one-seed/two-delay/two-operator smoke grid."""
    model.eval()
    resolved_family = family or ("B" if task_config.probe_gated else "A")
    conditions = (
        ("load1_clean", "load1_distractor")
        if resolved_family == "B"
        else ("clean", "distractor")
    )
    decoder = fit_frozen_decoder(
        model,
        task_config,
        resolved_family,
        trials_per_delay=max(8, task_config.batch_size),
    )
    rows: list[dict[str, Any]] = []
    angle_hashes: dict[str, list[str]] = {}
    for condition_index, condition in enumerate(conditions):
        for delay_steps in (10, 20):
            threshold = _baseline_threshold(
                model,
                task_config,
                resolved_family,
                condition,
                condition_index,
                delay_steps,
            )
            baseline_collected = _collect_batches(
                model,
                task_config,
                resolved_family,
                condition,
                condition_index,
                delay_steps,
                seed_base=FINAL_SEED_BASE,
                n_batches=1,
                batch_size=task_config.batch_size,
            )
            angle_hashes[f"{condition}:{delay_steps}"] = baseline_collected[
                "angle_hashes"
            ]
            baseline_metrics = summarize_collected(
                baseline_collected,
                decoder,
                threshold,
                family=resolved_family,
            )
            baseline_by_position = {
                item["item_position"]: item for item in baseline_metrics
            }
            operator_specs: list[
                tuple[str, str, float, int | str, ForwardFn]
            ] = [
                (
                    "synaptic_drive_gain",
                    "bias_outside",
                    1.0,
                    "",
                    synaptic_drive_gain(
                        model, gain=1.0, bias_mode="bias_outside"
                    ),
                ),
                (
                    "gaussian_state_noise",
                    "generic_control",
                    0.0,
                    4101,
                    gaussian_state_noise(
                        model, sigma=0.0, generator_seed=4101
                    ),
                ),
            ]
            for operator, variant, strength, replicate, forward_fn in operator_specs:
                collected = _collect_batches(
                    model,
                    task_config,
                    resolved_family,
                    condition,
                    condition_index,
                    delay_steps,
                    seed_base=FINAL_SEED_BASE,
                    n_batches=1,
                    batch_size=task_config.batch_size,
                    forward_fn=forward_fn,
                )
                metrics = summarize_collected(
                    collected,
                    decoder,
                    threshold,
                    family=resolved_family,
                    baseline_fraction_settled={
                        item["item_position"]: item["fraction_settled"]
                        for item in baseline_metrics
                    },
                )
                for metric in metrics:
                    baseline = baseline_by_position[metric["item_position"]]
                    rows.append(
                        _merge_with_baseline(
                            metric,
                            baseline,
                            family=resolved_family,
                            operator=operator,
                            variant=variant,
                            strength=strength,
                            strength_kind="grid",
                            condition=condition,
                            delay_steps=delay_steps,
                            seed=checkpoint_seed,
                            noise_replicate=replicate,
                        )
                    )

    dirs = ensure_run_dirs(output_dir)
    grid_path = _write_grid(dirs["metrics"] / "signature_grid.csv", rows)
    metadata = {
        **_git_metadata(),
        "smoke": True,
        "family": resolved_family,
        "checkpoint_seed": int(checkpoint_seed),
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "implementation_commit": _git_metadata()["git_commit"],
        "seed_scheme": {
            "decoder_reference": DECODER_SEED_BASE,
            "calibration": CALIBRATION_SEED_BASE,
            "cost_check": COST_CHECK_SEED_BASE,
            "final_evaluation": FINAL_SEED_BASE,
            "offset": "base + 10000*condition_index + 100*delay_steps + batch_index",
        },
        "angle_hashes": angle_hashes,
        "operators": ["synaptic_drive_gain", "gaussian_state_noise"],
        "delays": [10, 20],
        "n_batches": 1,
    }
    metadata_path = write_json(
        dirs["metrics"] / "signature_run_metadata.json", metadata
    )
    return ExperimentResult(grid_path, metadata_path, rows)


def _load_checkpoint_model(
    config: dict[str, Any], checkpoint_path: str | Path, device: torch.device
) -> torch.nn.Module:
    model = fresh_model(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _operator_forward(
    model: torch.nn.Module,
    task_config: TunedDelayTaskConfig,
    *,
    operator: str,
    variant: str,
    strength: float,
    condition: str,
    family: str,
    delay_steps: int,
    gain_vector_seed: int | None = None,
    noise_replicate: int | None = None,
) -> ForwardFn:
    if operator == "synaptic_drive_gain":
        return synaptic_drive_gain(
            model, gain=strength, bias_mode=variant
        )
    if operator == "heterogeneous_drive_gain":
        return heterogeneous_drive_gain(
            model,
            log_std=strength,
            vector_seed=int(gain_vector_seed or P2_VECTOR_SEEDS[0]),
            bias_mode=variant,
        )
    if operator == "sensory_input_gain":
        return sensory_input_gain(
            model,
            gain=strength,
            n_tuned_units=task_config.n_tuned_units,
        )
    if operator == "distractor_input_gain":
        probe = _condition_task_config(
            task_config, family, condition, delay_steps, 1, 1
        )
        distractor_slice = generate_batch_for_task(probe).phase_index[
            "distractor"
        ]
        return distractor_input_gain(
            model,
            gain=strength,
            n_tuned_units=task_config.n_tuned_units,
            distractor_slice=distractor_slice,
        )
    if operator == "recurrent_gain":
        return recurrent_gain(model, gain=strength)
    if operator == "gaussian_state_noise":
        return gaussian_state_noise(
            model,
            sigma=strength,
            generator_seed=int(noise_replicate or P5_REPLICATES[0]),
        )
    if operator == "state_persistence":
        return state_persistence(model, persistence_gain=strength)
    if operator == "time_constant":
        return time_constant(model, tau_scale=strength)
    raise KeyError(f"unknown operator: {operator}")


def _grid_settings() -> list[dict[str, Any]]:
    settings: list[dict[str, Any]] = []
    for bias_mode in ("bias_outside", "bias_inside"):
        for strength in MULTIPLICATIVE_GRID:
            settings.append(
                {
                    "operator": "synaptic_drive_gain",
                    "variant": bias_mode,
                    "strength": strength,
                    "strength_kind": "grid",
                }
            )
        for strength in P2_GRID:
            for vector_seed in P2_VECTOR_SEEDS:
                settings.append(
                    {
                        "operator": "heterogeneous_drive_gain",
                        "variant": bias_mode,
                        "strength": strength,
                        "strength_kind": "grid",
                        "gain_vector_seed": vector_seed,
                    }
                )
    for operator, variant, grid in (
        ("sensory_input_gain", "tuned_only", MULTIPLICATIVE_GRID),
        ("distractor_input_gain", "distractor_only", MULTIPLICATIVE_GRID),
        ("recurrent_gain", "weights_only", MULTIPLICATIVE_GRID),
        ("state_persistence", "carried_state_only", MULTIPLICATIVE_GRID),
        ("time_constant", "conserved_integrator", P7_GRID),
    ):
        for strength in grid:
            settings.append(
                {
                    "operator": operator,
                    "variant": variant,
                    "strength": strength,
                    "strength_kind": "grid",
                }
            )
    for strength in P5_GRID:
        for replicate in P5_REPLICATES:
            settings.append(
                {
                    "operator": "gaussian_state_noise",
                    "variant": "generic_control",
                    "strength": strength,
                    "strength_kind": "grid",
                    "noise_replicate": replicate,
                }
            )
    return settings


def _response_trial_errors(collected: dict[str, Any]) -> np.ndarray:
    response = collected["phase_index"]["response"]
    populations = collected["predictions"][
        response, :, : len(collected["preferred_angles"])
    ]
    decoded = decode_population_angle(
        populations, collected["preferred_angles"]
    )
    errors = np.degrees(
        circular_angular_error(decoded, collected["angles"][np.newaxis, :])
    )
    return np.mean(errors, axis=0)


def _setting_trial_errors(
    model: torch.nn.Module,
    task_config: TunedDelayTaskConfig,
    family: str,
    checkpoint_seed: int,
    operator: str,
    variant: str,
    strength: float,
    *,
    seed_base: int,
    n_batches: int,
    batch_size: int = 64,
) -> np.ndarray:
    clean_condition = "load1_clean" if family == "B" else "clean"
    if operator == "heterogeneous_drive_gain":
        replicate_errors = []
        for vector_seed in P2_VECTOR_SEEDS:
            forward = _operator_forward(
                model,
                task_config,
                operator=operator,
                variant=variant,
                strength=strength,
                condition=clean_condition,
                family=family,
                delay_steps=20,
                gain_vector_seed=vector_seed,
            )
            collected = _collect_batches(
                model,
                task_config,
                family,
                clean_condition,
                0,
                20,
                seed_base=seed_base,
                n_batches=n_batches,
                batch_size=batch_size,
                forward_fn=forward,
            )
            replicate_errors.append(_response_trial_errors(collected))
        return np.mean(replicate_errors, axis=0)
    if operator == "gaussian_state_noise":
        replicate_errors = []
        for replicate in P5_REPLICATES:
            def forward_factory(
                batch_seed: int, replicate_id: int = replicate
            ) -> ForwardFn:
                generator_seed = (
                    int(batch_seed)
                    + 1_000_000 * int(checkpoint_seed)
                    + int(replicate_id)
                )
                return _operator_forward(
                    model,
                    task_config,
                    operator=operator,
                    variant=variant,
                    strength=strength,
                    condition=clean_condition,
                    family=family,
                    delay_steps=20,
                    noise_replicate=generator_seed,
                )

            collected = _collect_batches(
                model,
                task_config,
                family,
                clean_condition,
                0,
                20,
                seed_base=seed_base,
                n_batches=n_batches,
                batch_size=batch_size,
                forward_seed_factory=forward_factory,
            )
            replicate_errors.append(_response_trial_errors(collected))
        return np.mean(replicate_errors, axis=0)
    forward = _operator_forward(
        model,
        task_config,
        operator=operator,
        variant=variant,
        strength=strength,
        condition=clean_condition,
        family=family,
        delay_steps=20,
    )
    collected = _collect_batches(
        model,
        task_config,
        family,
        clean_condition,
        0,
        20,
        seed_base=seed_base,
        n_batches=n_batches,
        batch_size=batch_size,
        forward_fn=forward,
    )
    return _response_trial_errors(collected)


def calibrate_checkpoint(
    model: torch.nn.Module,
    task_config: TunedDelayTaskConfig,
    *,
    family: str,
    checkpoint_seed: int,
    n_batches: int = 4,
) -> list[dict[str, Any]]:
    """Calibrate every clean-matchable operator for one checkpoint."""
    clean_condition = "load1_clean" if family == "B" else "clean"
    baseline_collected = _collect_batches(
        model,
        task_config,
        family,
        clean_condition,
        0,
        20,
        seed_base=CALIBRATION_SEED_BASE,
        n_batches=n_batches,
        batch_size=64,
    )
    baseline_errors = _response_trial_errors(baseline_collected)
    baseline_mean = float(np.mean(baseline_errors))
    records: list[dict[str, Any]] = []
    specifications = [
        ("synaptic_drive_gain", "bias_outside", MULTIPLICATIVE_GRID, True),
        ("synaptic_drive_gain", "bias_inside", MULTIPLICATIVE_GRID, True),
        ("heterogeneous_drive_gain", "bias_outside", P2_GRID, False),
        ("heterogeneous_drive_gain", "bias_inside", P2_GRID, False),
        ("sensory_input_gain", "tuned_only", MULTIPLICATIVE_GRID, True),
        ("recurrent_gain", "weights_only", MULTIPLICATIVE_GRID, True),
        ("gaussian_state_noise", "generic_control", P5_GRID, False),
        ("state_persistence", "carried_state_only", MULTIPLICATIVE_GRID, True),
        ("time_constant", "conserved_integrator", P7_GRID, True),
    ]
    for operator, variant, grid, bidirectional in specifications:
        def cost_function(strength: float) -> float:
            trial_errors = _setting_trial_errors(
                model,
                task_config,
                family,
                checkpoint_seed,
                operator,
                variant,
                strength,
                seed_base=CALIBRATION_SEED_BASE,
                n_batches=n_batches,
            )
            return proportional_cost(
                baseline_mean, float(np.mean(trial_errors))
            )

        results = (
            calibrate_bidirectional(
                cost_function,
                grid,
                neutral_strength=1.0,
                target_proportional_cost=0.30,
            )
            if bidirectional
            else {
                "above_neutral": calibrate_strength(
                    cost_function,
                    grid,
                    target_proportional_cost=0.30,
                )
            }
        )
        for branch, result in results.items():
            absolute_delta = baseline_mean * result.achieved_proportional_cost
            records.append(
                {
                    "family": family,
                    "operator": operator,
                    "variant": variant,
                    "seed": int(checkpoint_seed),
                    "branch": branch,
                    "target_delta_degrees": baseline_mean * 0.30,
                    "target_proportional_cost": 0.30,
                    "strength": result.strength,
                    "achieved_delta_degrees": absolute_delta,
                    "achieved_proportional_cost": result.achieved_proportional_cost,
                    "converged": result.converged,
                    "n_iterations": result.n_iterations,
                    "note": result.note,
                }
            )
    p3a_match = next(
        row
        for row in records
        if row["operator"] == "sensory_input_gain"
        and row["branch"] == "above_neutral"
    )
    distractor_condition = (
        "load1_distractor" if family == "B" else "distractor"
    )
    decoder = fit_frozen_decoder(model, task_config, family)

    def attraction(forward_fn: ForwardFn | None) -> float:
        collected = _collect_batches(
            model,
            task_config,
            family,
            distractor_condition,
            1,
            20,
            seed_base=CALIBRATION_SEED_BASE,
            n_batches=n_batches,
            batch_size=64,
            forward_fn=forward_fn,
        )
        decoded = decode_angles_from_hidden(
            collected["hidden_states"], decoder
        )
        phases = collected["phase_index"]
        metric = distractor_drift_and_recovery(
            decoded,
            collected["angles"],
            collected["distractor_angles"],
            phases["distractor"],
            slice(phases["distractor"].stop, phases["delay"].stop),
        )
        return float(metric["distractor_peak_attraction_fraction"])

    baseline_attraction = attraction(None)
    p3a_forward = _operator_forward(
        model,
        task_config,
        operator="sensory_input_gain",
        variant="tuned_only",
        strength=float(p3a_match["strength"]),
        condition=distractor_condition,
        family=family,
        delay_steps=20,
    )
    target_attraction_change = attraction(p3a_forward) - baseline_attraction

    def p3b_attraction_change(strength: float) -> float:
        forward = _operator_forward(
            model,
            task_config,
            operator="distractor_input_gain",
            variant="distractor_only",
            strength=strength,
            condition=distractor_condition,
            family=family,
            delay_steps=20,
        )
        return attraction(forward) - baseline_attraction

    p3b_result = calibrate_strength(
        p3b_attraction_change,
        MULTIPLICATIVE_GRID,
        target_proportional_cost=target_attraction_change,
        tolerance=0.01,
    )
    records.append(
        {
            "family": family,
            "operator": "distractor_input_gain",
            "variant": "distractor_only",
            "seed": int(checkpoint_seed),
            "branch": "matched_distractor",
            "target_delta_degrees": "",
            "target_proportional_cost": target_attraction_change,
            "strength": p3b_result.strength,
            "achieved_delta_degrees": "",
            "achieved_proportional_cost": p3b_result.achieved_proportional_cost,
            "converged": p3b_result.converged,
            "n_iterations": p3b_result.n_iterations,
            "note": f"matched_P3a_distractor_attraction:{p3b_result.note}",
        }
    )
    return records


CALIBRATION_COLUMNS = [
    "family",
    "operator",
    "variant",
    "seed",
    "branch",
    "target_delta_degrees",
    "target_proportional_cost",
    "strength",
    "achieved_delta_degrees",
    "achieved_proportional_cost",
    "converged",
    "n_iterations",
    "note",
]

COST_CHECK_COLUMNS = [
    "family",
    "operator",
    "variant",
    "branch",
    "seed",
    "strength",
    "n_trials",
    "n_noise_replicates",
    "baseline_mean_angular_error_degrees",
    "mean_angular_error_degrees",
    "delta_angular_error_degrees",
    "proportional_clean_cost",
    "paired_difference_sd_degrees",
    "paired_se_degrees",
    "bootstrap_ci_lower_proportional",
    "bootstrap_ci_upper_proportional",
    "bootstrap_ci_half_width",
    "cost_precision_valid",
    "band_lower",
    "band_upper",
    "cost_match_valid",
    "p5_proportional_clean_cost",
    "p5_cost_gap",
    "p5_cost_gap_valid",
]


def _write_records(
    path: Path, rows: list[dict[str, Any]], columns: list[str]
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            [{column: row.get(column, "") for column in columns} for row in rows]
        )
    return path


def run_cost_checks(
    model: torch.nn.Module,
    task_config: TunedDelayTaskConfig,
    calibration_rows: list[dict[str, Any]],
    *,
    family: str,
    checkpoint_seed: int,
    n_trials: int,
    bootstrap_draws: int = 10_000,
) -> list[dict[str, Any]]:
    """Run the dedicated high-precision D7 checks without moving strengths."""
    if n_trials % 64:
        raise ValueError("n_trials must be a multiple of 64")
    clean_condition = "load1_clean" if family == "B" else "clean"
    baseline_collected = _collect_batches(
        model,
        task_config,
        family,
        clean_condition,
        0,
        20,
        seed_base=COST_CHECK_SEED_BASE,
        n_batches=n_trials // 64,
        batch_size=64,
    )
    baseline = _response_trial_errors(baseline_collected)
    relevant = [
        row
        for row in calibration_rows
        if row["seed"] == checkpoint_seed
        and row["operator"] != "distractor_input_gain"
        and bool(row["converged"])
    ]
    p5_row = next(
        row
        for row in relevant
        if row["operator"] == "gaussian_state_noise"
    )
    p5_errors = _setting_trial_errors(
        model,
        task_config,
        family,
        checkpoint_seed,
        p5_row["operator"],
        p5_row["variant"],
        float(p5_row["strength"]),
        seed_base=COST_CHECK_SEED_BASE,
        n_batches=n_trials // 64,
    )
    p5_point, p5_lower, p5_upper, p5_half = (
        paired_bootstrap_proportional_cost(
            baseline,
            p5_errors,
            draws=bootstrap_draws,
            bootstrap_seed=COST_CHECK_SEED_BASE + checkpoint_seed,
        )
    )
    output: list[dict[str, Any]] = []
    for calibration in relevant:
        if calibration["operator"] == "gaussian_state_noise":
            perturbed = p5_errors
            point, lower, upper, half_width = (
                p5_point,
                p5_lower,
                p5_upper,
                p5_half,
            )
        else:
            perturbed = _setting_trial_errors(
                model,
                task_config,
                family,
                checkpoint_seed,
                calibration["operator"],
                calibration["variant"],
                float(calibration["strength"]),
                seed_base=COST_CHECK_SEED_BASE,
                n_batches=n_trials // 64,
            )
            point, lower, upper, half_width = (
                paired_bootstrap_proportional_cost(
                    baseline,
                    perturbed,
                    draws=bootstrap_draws,
                    bootstrap_seed=(
                        COST_CHECK_SEED_BASE
                        + checkpoint_seed
                        + len(output)
                        + 1
                    ),
                )
            )
        differences = perturbed - baseline
        band_valid = 0.20 <= point <= 0.40
        precision_valid = half_width <= 0.10
        p5_gap = (
            "" if calibration["operator"] == "gaussian_state_noise" else point - p5_point
        )
        p5_gap_valid = (
            ""
            if calibration["operator"] == "gaussian_state_noise"
            else abs(float(p5_gap)) <= 0.05
        )
        output.append(
            {
                "family": family,
                "operator": calibration["operator"],
                "variant": calibration["variant"],
                "branch": calibration["branch"],
                "seed": checkpoint_seed,
                "strength": calibration["strength"],
                "n_trials": n_trials,
                "n_noise_replicates": (
                    3
                    if calibration["operator"] == "gaussian_state_noise"
                    else 1
                ),
                "baseline_mean_angular_error_degrees": float(np.mean(baseline)),
                "mean_angular_error_degrees": float(np.mean(perturbed)),
                "delta_angular_error_degrees": float(np.mean(differences)),
                "proportional_clean_cost": point,
                "paired_difference_sd_degrees": float(np.std(differences, ddof=1)),
                "paired_se_degrees": float(
                    np.std(differences, ddof=1) / np.sqrt(n_trials)
                ),
                "bootstrap_ci_lower_proportional": lower,
                "bootstrap_ci_upper_proportional": upper,
                "bootstrap_ci_half_width": half_width,
                "cost_precision_valid": precision_valid,
                "band_lower": 0.20,
                "band_upper": 0.40,
                "cost_match_valid": bool(
                    band_valid
                    and precision_valid
                    and (p5_gap_valid is not False)
                ),
                "p5_proportional_clean_cost": (
                    "" if calibration["operator"] == "gaussian_state_noise" else p5_point
                ),
                "p5_cost_gap": p5_gap,
                "p5_cost_gap_valid": p5_gap_valid,
            }
        )
    return output


def selected_cost_check_count(
    baseline_trial_errors_by_seed: dict[int, np.ndarray],
    *,
    family: str,
) -> int:
    """Apply the frozen Family A count or blinded Family B precision adaptation."""
    if family == "A":
        return 1024
    requirements = []
    for errors in baseline_trial_errors_by_seed.values():
        values = np.asarray(errors, dtype=np.float64)
        requirements.append(
            required_cost_check_trials(
                float(np.mean(values)), float(np.std(values, ddof=1))
            )
        )
    return max(1024, round_up_to_batch(max(requirements), 64))


def _matched_settings(
    calibration_rows: list[dict[str, Any]], checkpoint_seed: int
) -> list[dict[str, Any]]:
    settings = []
    for row in calibration_rows:
        if row["seed"] != checkpoint_seed or not bool(row["converged"]):
            continue
        settings.append(
            {
                "operator": row["operator"],
                "variant": row["variant"],
                "strength": float(row["strength"]),
                "strength_kind": (
                    "matched_distractor"
                    if row["branch"] == "matched_distractor"
                    else (
                        "matched_below"
                        if row["branch"] == "below_neutral"
                        else "matched_above"
                    )
                ),
                "branch": row["branch"],
            }
        )
    return settings


def run_checkpoint_grid(
    model: torch.nn.Module,
    task_config: TunedDelayTaskConfig,
    *,
    family: str,
    checkpoint_seed: int,
    calibration_rows: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
    n_batches: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Run the D3 plus calibrated grid for one checkpoint."""
    conditions = (
        ("load1_clean", "load1_distractor", "load2_clean", "load2_distractor")
        if family == "B"
        else ("clean", "distractor")
    )
    delays = (10, 20, 40, 80, 160) if family == "B" else (10, 20, 40, 80)
    decoder = fit_frozen_decoder(model, task_config, family)
    cost_lookup = {
        (row["operator"], row["variant"], row["branch"]): row
        for row in cost_rows
        if row["seed"] == checkpoint_seed
    }
    settings = _grid_settings() + _matched_settings(
        calibration_rows, checkpoint_seed
    )
    rows: list[dict[str, Any]] = []
    angle_hashes: dict[str, list[str]] = {}
    for condition_index, condition in enumerate(conditions):
        for delay_steps in delays:
            threshold = _baseline_threshold(
                model,
                task_config,
                family,
                condition,
                condition_index,
                delay_steps,
            )
            baseline_collected = _collect_batches(
                model,
                task_config,
                family,
                condition,
                condition_index,
                delay_steps,
                seed_base=FINAL_SEED_BASE,
                n_batches=n_batches,
                batch_size=64,
            )
            angle_hashes[f"{condition}:{delay_steps}"] = baseline_collected[
                "angle_hashes"
            ]
            baseline_metrics = summarize_collected(
                baseline_collected, decoder, threshold, family=family
            )
            baseline_lookup = {
                row["item_position"]: row for row in baseline_metrics
            }
            baseline_fractions = {
                key: value["fraction_settled"]
                for key, value in baseline_lookup.items()
            }
            for setting in settings:
                operator = setting["operator"]
                variant = setting["variant"]
                strength = float(setting["strength"])
                vector_seeds = (
                    P2_VECTOR_SEEDS
                    if operator == "heterogeneous_drive_gain"
                    and "gain_vector_seed" not in setting
                    else (setting.get("gain_vector_seed"),)
                )
                noise_replicates = (
                    P5_REPLICATES
                    if operator == "gaussian_state_noise"
                    and "noise_replicate" not in setting
                    else (setting.get("noise_replicate"),)
                )
                for vector_seed in vector_seeds:
                    for noise_replicate in noise_replicates:
                        forward_factory = None
                        if operator == "gaussian_state_noise":
                            def forward_factory(
                                batch_seed: int,
                                replicate_id: int = int(noise_replicate),
                            ) -> ForwardFn:
                                return _operator_forward(
                                    model,
                                    task_config,
                                    operator=operator,
                                    variant=variant,
                                    strength=strength,
                                    condition=condition,
                                    family=family,
                                    delay_steps=delay_steps,
                                    noise_replicate=(
                                        batch_seed
                                        + 1_000_000 * checkpoint_seed
                                        + replicate_id
                                    ),
                                )
                            forward = None
                        else:
                            forward = _operator_forward(
                                model,
                                task_config,
                                operator=operator,
                                variant=variant,
                                strength=strength,
                                condition=condition,
                                family=family,
                                delay_steps=delay_steps,
                                gain_vector_seed=vector_seed,
                            )
                        collected = _collect_batches(
                            model,
                            task_config,
                            family,
                            condition,
                            condition_index,
                            delay_steps,
                            seed_base=FINAL_SEED_BASE,
                            n_batches=n_batches,
                            batch_size=64,
                            forward_fn=forward,
                            forward_seed_factory=forward_factory,
                        )
                        metrics = summarize_collected(
                            collected,
                            decoder,
                            threshold,
                            family=family,
                            baseline_fraction_settled=baseline_fractions,
                        )
                        for metric in metrics:
                            row = _merge_with_baseline(
                                metric,
                                baseline_lookup[metric["item_position"]],
                                family=family,
                                operator=operator,
                                variant=variant,
                                strength=strength,
                                strength_kind=setting["strength_kind"],
                                condition=condition,
                                delay_steps=delay_steps,
                                seed=checkpoint_seed,
                                noise_replicate=(
                                    noise_replicate
                                    if noise_replicate is not None
                                    else ""
                                ),
                            )
                            row["gain_vector_seed"] = (
                                vector_seed if vector_seed is not None else ""
                            )
                            if setting["strength_kind"].startswith("matched"):
                                branch = setting.get(
                                    "branch",
                                    "above_neutral",
                                )
                                cost = cost_lookup.get(
                                    (operator, variant, branch)
                                )
                                row["cost_match_valid"] = (
                                    cost["cost_match_valid"] if cost else False
                                )
                            rows.append(row)
    return rows, angle_hashes


def main() -> None:
    """Run smoke, calibration/cost checks, or the complete checkpoint grid."""
    parser = argparse.ArgumentParser(
        description="Run the psilocybin-signature perturbation experiment."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--family", choices=["A", "B"], required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoints", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    args = parser.parse_args()
    config = load_config(args.config)
    config["training"]["device"] = args.device
    device = torch.device(
        "cuda"
        if args.device in {"auto", "cuda"} and torch.cuda.is_available()
        else "cpu"
    )
    task_config = task_config_from_dict(config)
    output = (
        Path(config["paths"]["output_dir"])
        / "perturbation_experiments"
        / "signature"
    )
    if args.smoke:
        checkpoint = args.checkpoint or (
            args.checkpoints[0] if args.checkpoints else None
        )
        if not checkpoint:
            raise SystemExit(
                "--checkpoint is required for CLI smoke verification"
            )
        model = _load_checkpoint_model(config, checkpoint, device)
        result = run_smoke_experiment(
            model, task_config, output, family=args.family
        )
        print(f"grid={result.grid_path}")
        print(f"metadata={result.metadata_path}")
        return

    if not (args.calibrate or args.full):
        raise SystemExit("choose --smoke, --calibrate, or --full")
    checkpoints = list(args.checkpoints or [])
    if args.checkpoint:
        checkpoints.insert(0, args.checkpoint)
    seeds = list(args.seeds or [])
    if not checkpoints or len(checkpoints) != len(seeds):
        raise SystemExit(
            "--checkpoints and --seeds are required with equal lengths"
        )
    expected = 5 if args.family == "A" else 10
    if args.full and len(checkpoints) != expected:
        raise SystemExit(
            f"full Family {args.family} run requires {expected} checkpoints"
        )

    dirs = ensure_run_dirs(output)
    calibration_dir = output / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    calibration_rows: list[dict[str, Any]] = []
    baseline_reference_errors: dict[int, np.ndarray] = {}
    loaded: list[tuple[int, torch.nn.Module]] = []
    for checkpoint_seed, checkpoint in zip(seeds, checkpoints):
        model = _load_checkpoint_model(config, checkpoint, device)
        loaded.append((checkpoint_seed, model))
        calibration_rows.extend(
            calibrate_checkpoint(
                model,
                task_config,
                family=args.family,
                checkpoint_seed=checkpoint_seed,
            )
        )
        clean_condition = (
            "load1_clean" if args.family == "B" else "clean"
        )
        reference = _collect_batches(
            model,
            task_config,
            args.family,
            clean_condition,
            0,
            20,
            seed_base=DECODER_SEED_BASE + 70_000,
            n_batches=16,
            batch_size=64,
        )
        baseline_reference_errors[checkpoint_seed] = _response_trial_errors(
            reference
        )
    calibration_path = _write_records(
        calibration_dir / "matched_cost_strengths.csv",
        calibration_rows,
        CALIBRATION_COLUMNS,
    )
    n_cost_check = selected_cost_check_count(
        baseline_reference_errors, family=args.family
    )
    cost_rows: list[dict[str, Any]] = []
    for checkpoint_seed, model in loaded:
        cost_rows.extend(
            run_cost_checks(
                model,
                task_config,
                calibration_rows,
                family=args.family,
                checkpoint_seed=checkpoint_seed,
                n_trials=n_cost_check,
                bootstrap_draws=args.bootstrap_draws,
            )
        )
    cost_path = _write_records(
        dirs["metrics"] / "cost_match_check.csv",
        cost_rows,
        COST_CHECK_COLUMNS,
    )
    if args.calibrate and not args.full:
        print(f"calibration={calibration_path}")
        print(f"cost_check={cost_path}")
        print(f"n_cost_check={n_cost_check}")
        return

    all_rows: list[dict[str, Any]] = []
    all_hashes: dict[str, Any] = {}
    for checkpoint_seed, model in loaded:
        checkpoint_rows, angle_hashes = run_checkpoint_grid(
            model,
            task_config,
            family=args.family,
            checkpoint_seed=checkpoint_seed,
            calibration_rows=calibration_rows,
            cost_rows=cost_rows,
        )
        all_rows.extend(checkpoint_rows)
        all_hashes[str(checkpoint_seed)] = angle_hashes
    grid_path = _write_grid(
        dirs["metrics"] / "signature_grid.csv", all_rows
    )
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    git = _git_metadata()
    metadata_path = write_json(
        dirs["metrics"] / "signature_run_metadata.json",
        {
            **git,
            "family": args.family,
            "device": str(device),
            "config_hash": config_hash,
            "checkpoint_paths": checkpoints,
            "checkpoint_seeds": seeds,
            "grid_definition": {
                "settings": _grid_settings(),
                "conditions": (
                    [
                        "load1_clean",
                        "load1_distractor",
                        "load2_clean",
                        "load2_distractor",
                    ]
                    if args.family == "B"
                    else ["clean", "distractor"]
                ),
                "delays": (
                    [10, 20, 40, 80, 160]
                    if args.family == "B"
                    else [10, 20, 40, 80]
                ),
                "batches_per_cell": 4,
                "trials_per_batch": 64,
            },
            "seed_scheme": {
                "decoder_reference": DECODER_SEED_BASE,
                "calibration": CALIBRATION_SEED_BASE,
                "cost_check": COST_CHECK_SEED_BASE,
                "final_evaluation": FINAL_SEED_BASE,
                "offset": "base + 10000*condition_index + 100*delay_steps + batch_index",
            },
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "implementation_commit": git["git_commit"],
            "worktree_clean": git["worktree_clean"],
            "angle_hashes": all_hashes,
            "n_cost_check": n_cost_check,
            "calibration_csv": str(calibration_path),
            "cost_check_csv": str(cost_path),
            "signature_grid_csv": str(grid_path),
        },
    )
    print(f"calibration={calibration_path}")
    print(f"cost_check={cost_path}")
    print(f"grid={grid_path}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
