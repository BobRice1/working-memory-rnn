"""Standalone fixed-grid exploratory pilot for the trained N-back RNNs.

This module deliberately does not import or call the registered phased runner,
calibration, held-out cost gates, or confirmatory task banks.  It uses three
frozen competent checkpoints, a separate task/noise seed namespace, and paired
native-versus-perturbed batches to describe 0-back and 2-back effects.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wm_rnn.config import load_config
from wm_rnn.device import SelectedDevice, select_device
from wm_rnn.nback_additive_calibration import (
    P2_VECTOR_SEEDS,
    PROFILE_BY_ID,
    OperatorProfile,
)
from wm_rnn.nback_additive_cost_precision import (
    DEFAULT_MANIFEST,
    RetainedCheckpoint,
    load_retained_checkpoints,
)
from wm_rnn.nback_additive_outcomes import (
    aggregate_three_replicate_condition_metrics,
    pool_condition_batch_metrics,
)
from wm_rnn.nback_metrics import nback_metrics
from wm_rnn.nback_perturbation import (
    build_nback_operator,
    condition_normalized_discriminability_impairment,
)
from wm_rnn.nback_perturbation_state import atomic_write_json, sha256_file
from wm_rnn.nback_task import (
    NBackBatch,
    NBackTaskConfig,
    generate_nback_batch,
)
from wm_rnn.training_utils import (
    batch_to_tensors,
    fresh_model,
    task_config_from_dict,
)


PILOT_CHECKPOINT_SEEDS = (20260912, 20260913, 20260914)
PILOT_PROFILE_IDS = (1, 4, 7, 9, 10, 12, 14)
DEFAULT_CONFIG = Path("configs/exploratory_psilocybin_signature_pilot.yaml")
DEFAULT_PRECISION_SUMMARY = Path(
    "outputs/nback_additive_cost_precision/metrics/"
    "nback_additive_cost_precision_summary.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/exploratory_psilocybin_signature_pilot/nback"
)
PILOT_OPERATOR_NAMES = (
    "synaptic_drive_gain",
    "heterogeneous_drive_gain",
    "sensory_input_gain",
    "recurrent_gain",
    "gaussian_state_noise",
    "state_persistence",
    "time_constant",
)


@dataclass(frozen=True)
class PilotDesign:
    """Immutable exploratory scope and collision-free seed addressing."""

    checkpoint_seeds: tuple[int, ...] = PILOT_CHECKPOINT_SEEDS
    profile_ids: tuple[int, ...] = PILOT_PROFILE_IDS
    batch_size: int = 128
    n_batches: int = 2
    task_seed_base: int = 151_000_000
    noise_seed_base: int = 152_000_000
    probability_threshold: float = 0.80
    margin_threshold: float = 0.60
    consecutive_steps: int = 3

    @property
    def sequences_per_cell(self) -> int:
        return self.batch_size * self.n_batches

    def validate(self) -> None:
        if self.checkpoint_seeds != PILOT_CHECKPOINT_SEEDS:
            raise ValueError("pilot requires seeds 20260912-20260914")
        if (
            not self.profile_ids
            or len(set(self.profile_ids)) != len(self.profile_ids)
            or any(profile_id not in PROFILE_BY_ID for profile_id in self.profile_ids)
        ):
            raise ValueError("profile_ids must be unique registered profiles")
        if min(
            self.batch_size,
            self.n_batches,
            self.task_seed_base,
            self.noise_seed_base,
            self.consecutive_steps,
        ) <= 0:
            raise ValueError("pilot counts and seed bases must be positive")
        if self.sequences_per_cell != 256:
            raise ValueError("pilot requires exactly 256 sequences per cell")
        if self.task_seed_base == self.noise_seed_base:
            raise ValueError("task and noise seed namespaces must differ")
        if not 0.0 <= self.probability_threshold <= 1.0:
            raise ValueError("probability_threshold must lie in [0, 1]")
        if not 0.0 <= self.margin_threshold <= 1.0:
            raise ValueError("margin_threshold must lie in [0, 1]")

    def task_seed(
        self,
        checkpoint_ordinal: int,
        condition_code: int,
        batch_index: int,
    ) -> int:
        self.validate()
        if condition_code not in (0, 1):
            raise ValueError("condition_code must be 0 or 1")
        if not 0 <= batch_index < self.n_batches:
            raise ValueError("batch_index is outside the pilot bank")
        return (
            self.task_seed_base
            + 10_000 * int(checkpoint_ordinal)
            + 1_000 * condition_code
            + batch_index
        )

    def noise_seed(
        self,
        checkpoint_ordinal: int,
        condition_code: int,
        replicate_ordinal: int,
        batch_index: int,
    ) -> int:
        self.validate()
        if condition_code not in (0, 1):
            raise ValueError("condition_code must be 0 or 1")
        if not 0 <= replicate_ordinal < 3:
            raise ValueError("replicate_ordinal must lie in [0, 2]")
        if not 0 <= batch_index < self.n_batches:
            raise ValueError("batch_index is outside the pilot bank")
        return (
            self.noise_seed_base
            + 1_000_000 * int(checkpoint_ordinal)
            + 10_000 * condition_code
            + 1_000 * replicate_ordinal
            + batch_index
        )


FROZEN_PILOT_DESIGN = PilotDesign()


def load_pilot_operator_grids(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    repo_root: str | Path = ".",
) -> dict[str, tuple[float, ...]]:
    """Load the exact frozen N-back grids from the shared pilot config."""
    root = Path(repo_root).resolve()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    config = load_config(path)
    pilot = config.get("pilot", {})
    if (
        tuple(int(seed) for seed in pilot.get("nback_seeds", ()))
        != PILOT_CHECKPOINT_SEEDS
        or int(pilot.get("nback_sequences_per_cell", -1)) != 256
        or int(pilot.get("batch_size", -1)) != 128
        or int(pilot.get("stochastic_replicates", -1)) != 3
    ):
        raise ValueError("shared pilot config does not match frozen N-back scope")
    raw_operators = config.get("operators")
    if not isinstance(raw_operators, Mapping):
        raise ValueError("shared pilot config lacks operators")
    grids: dict[str, tuple[float, ...]] = {}
    for operator in PILOT_OPERATOR_NAMES:
        raw_grid = raw_operators.get(operator)
        if (
            isinstance(raw_grid, (str, bytes))
            or not isinstance(raw_grid, Sequence)
            or len(raw_grid) < 2
        ):
            raise ValueError(f"pilot operator grid is invalid: {operator}")
        grid = tuple(float(value) for value in raw_grid)
        if (
            len(set(grid)) != len(grid)
            or not np.all(np.isfinite(np.asarray(grid)))
        ):
            raise ValueError(
                f"pilot operator grid must be unique and finite: {operator}"
            )
        grids[operator] = grid
    return grids


def _load_pilot_source_paths(
    config_path: str | Path,
    *,
    repo_root: str | Path,
) -> tuple[Path, Path, Path]:
    root = Path(repo_root).resolve()
    pilot_path = Path(config_path)
    if not pilot_path.is_absolute():
        pilot_path = root / pilot_path
    pilot_config = load_config(pilot_path)
    sources = pilot_config.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("shared pilot config lacks sources")
    nback_config = Path(str(sources.get("nback_config", "")))
    manifest = Path(str(sources.get("nback_manifest", "")))
    if not nback_config.is_absolute():
        nback_config = root / nback_config
    if not manifest.is_absolute():
        manifest = root / manifest
    return pilot_path.resolve(), nback_config.resolve(), manifest.resolve()


def select_pilot_checkpoints(
    checkpoints: Sequence[RetainedCheckpoint],
    *,
    design: PilotDesign = FROZEN_PILOT_DESIGN,
) -> tuple[RetainedCheckpoint, ...]:
    """Select the three named checkpoints without changing their ordinals."""
    design.validate()
    by_seed = {checkpoint.seed: checkpoint for checkpoint in checkpoints}
    if len(by_seed) != len(checkpoints):
        raise ValueError("checkpoint seeds must be unique")
    missing = [
        seed for seed in design.checkpoint_seeds if seed not in by_seed
    ]
    if missing:
        raise ValueError(f"pilot checkpoints are missing: {missing}")
    selected = tuple(by_seed[seed] for seed in design.checkpoint_seeds)
    if tuple(checkpoint.ordinal for checkpoint in selected) != (0, 1, 2):
        raise ValueError("pilot checkpoints must retain frozen ordinals 0, 1, 2")
    return selected


def load_pilot_checkpoints(
    *,
    repo_root: str | Path,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    precision_summary_path: str | Path = DEFAULT_PRECISION_SUMMARY,
    design: PilotDesign = FROZEN_PILOT_DESIGN,
) -> tuple[RetainedCheckpoint, ...]:
    """Load the screened family, select the pilot seeds, and verify hashes."""
    root = Path(repo_root).resolve()
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = root / manifest
    precision_path = Path(precision_summary_path)
    if not precision_path.is_absolute():
        precision_path = root / precision_path
    checkpoints = load_retained_checkpoints(manifest, repo_root=root)
    selected = select_pilot_checkpoints(checkpoints, design=design)
    with precision_path.open(encoding="utf-8") as handle:
        precision = json.load(handle)
    hashes = precision.get("checkpoint_sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("precision summary lacks checkpoint_sha256")
    for checkpoint in selected:
        expected = hashes.get(str(checkpoint.seed))
        if not isinstance(expected, str) or sha256_file(checkpoint.path) != expected:
            raise ValueError(
                f"pilot checkpoint hash mismatch: {checkpoint.seed}"
            )
    return selected


def _load_checkpoint_model(
    config: dict[str, Any],
    checkpoint: RetainedCheckpoint,
    device: torch.device,
) -> torch.nn.Module:
    saved = torch.load(checkpoint.path, map_location=device)
    embedded_seed = saved.get("config", {}).get("task", {}).get("seed")
    if int(embedded_seed) != checkpoint.seed:
        raise ValueError("checkpoint seed does not match retained manifest")
    model = fresh_model(config, device)
    model.load_state_dict(saved["model_state"])
    model.eval()
    if (
        model.config.input_size != 8
        or model.config.output_size != 2
        or model.config.hidden_size != 64
    ):
        raise ValueError("pilot requires the frozen 8-input, 2-output N-back RNN")
    return model


def pilot_task_batches(
    config: dict[str, Any],
    checkpoint: RetainedCheckpoint,
    condition_code: int,
    *,
    design: PilotDesign = FROZEN_PILOT_DESIGN,
) -> tuple[NBackBatch, ...]:
    """Generate one paired pilot bank for native and all perturbations."""
    design.validate()
    base = task_config_from_dict(config, batch_size=design.batch_size)
    if not isinstance(base, NBackTaskConfig):
        raise ValueError("pilot requires task_type: n_back")
    return tuple(
        generate_nback_batch(
            replace(
                base,
                n_back=condition_code * 2,
                seed=design.task_seed(
                    checkpoint.ordinal,
                    condition_code,
                    batch_index,
                ),
            )
        )
        for batch_index in range(design.n_batches)
    )


def pilot_profile_parameters(
    profile: OperatorProfile,
    strength: float,
    *,
    checkpoint_ordinal: int,
    condition_code: int,
    replicate_ordinal: int,
    batch_index: int,
    design: PilotDesign = FROZEN_PILOT_DESIGN,
) -> dict[str, object]:
    """Map one registered profile to the exploratory pilot seed namespace."""
    if profile.operator == "synaptic_drive_gain":
        return {"gain": strength, "bias_mode": profile.variant}
    if profile.operator == "heterogeneous_drive_gain":
        return {
            "log_std": strength,
            "vector_seed": P2_VECTOR_SEEDS[replicate_ordinal],
            "bias_mode": profile.variant,
        }
    if profile.operator == "sensory_input_gain":
        return {"gain": strength}
    if profile.operator == "recurrent_gain":
        return {"gain": strength}
    if profile.operator == "state_persistence":
        return {"persistence_gain": strength}
    if profile.operator == "time_constant":
        return {"tau_scale": strength}
    if profile.operator == "gaussian_state_noise":
        return {
            "sigma": strength,
            "generator_seed": design.noise_seed(
                checkpoint_ordinal,
                condition_code,
                replicate_ordinal,
                batch_index,
            ),
        }
    raise KeyError(f"unsupported pilot operator: {profile.operator}")


def _replicate_count(profile: OperatorProfile) -> int:
    return (
        3
        if profile.operator
        in {"heterogeneous_drive_gain", "gaussian_state_noise"}
        else 1
    )


def _batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    batch: NBackBatch,
    design: PilotDesign,
) -> dict[str, Any]:
    if not torch.isfinite(logits).all():
        raise ValueError("pilot operator produced nonfinite logits")
    return nback_metrics(
        logits,
        targets,
        loss_mask,
        batch,
        probability_threshold=design.probability_threshold,
        margin_threshold=design.margin_threshold,
        consecutive_steps=design.consecutive_steps,
    )


@torch.no_grad()
def evaluate_checkpoint_cells(
    config: dict[str, Any],
    checkpoint: RetainedCheckpoint,
    device: torch.device,
    *,
    operator_grids: Mapping[str, Sequence[float]],
    design: PilotDesign = FROZEN_PILOT_DESIGN,
) -> list[dict[str, Any]]:
    """Evaluate native baseline and every fixed profile grid at one checkpoint."""
    design.validate()
    model = _load_checkpoint_model(config, checkpoint, device)
    rows: list[dict[str, Any]] = []
    for condition_code in (0, 1):
        batches = pilot_task_batches(
            config, checkpoint, condition_code, design=design
        )
        tensor_batches = [
            (*batch_to_tensors(batch, device), batch) for batch in batches
        ]
        native_batch_metrics = []
        for inputs, targets, loss_mask, batch in tensor_batches:
            logits, _ = model(inputs)
            native_batch_metrics.append(
                _batch_metrics(logits, targets, loss_mask, batch, design)
            )
        rows.append(
            {
                "checkpoint_seed": checkpoint.seed,
                "checkpoint_ordinal": checkpoint.ordinal,
                "profile_id": None,
                "operator": "native",
                "variant": "native",
                "strength": None,
                "condition_code": condition_code,
                "metrics": pool_condition_batch_metrics(
                    native_batch_metrics
                ),
            }
        )
        for profile_id in design.profile_ids:
            profile = PROFILE_BY_ID[profile_id]
            raw_grid = operator_grids.get(profile.operator)
            if raw_grid is None:
                raise ValueError(
                    f"pilot grid is missing for {profile.operator}"
                )
            for strength in tuple(float(value) for value in raw_grid):
                replicate_metrics = []
                for replicate in range(_replicate_count(profile)):
                    batch_metrics = []
                    static_forward = None
                    if profile.operator != "gaussian_state_noise":
                        static_forward = build_nback_operator(
                            model,
                            profile.operator,
                            **pilot_profile_parameters(
                                profile,
                                strength,
                                checkpoint_ordinal=checkpoint.ordinal,
                                condition_code=condition_code,
                                replicate_ordinal=replicate,
                                batch_index=0,
                                design=design,
                            ),
                        )
                    for batch_index, (
                        inputs,
                        targets,
                        loss_mask,
                        batch,
                    ) in enumerate(tensor_batches):
                        forward = static_forward
                        if profile.operator == "gaussian_state_noise":
                            forward = build_nback_operator(
                                model,
                                profile.operator,
                                **pilot_profile_parameters(
                                    profile,
                                    strength,
                                    checkpoint_ordinal=checkpoint.ordinal,
                                    condition_code=condition_code,
                                    replicate_ordinal=replicate,
                                    batch_index=batch_index,
                                    design=design,
                                ),
                            )
                        if forward is None:
                            raise RuntimeError("pilot forward was not constructed")
                        logits, _ = forward(inputs)
                        batch_metrics.append(
                            _batch_metrics(
                                logits, targets, loss_mask, batch, design
                            )
                        )
                    replicate_metrics.append(
                        pool_condition_batch_metrics(batch_metrics)
                    )
                pooled = (
                    replicate_metrics[0]
                    if len(replicate_metrics) == 1
                    else aggregate_three_replicate_condition_metrics(
                        replicate_metrics
                    )
                )
                rows.append(
                    {
                        "checkpoint_seed": checkpoint.seed,
                        "checkpoint_ordinal": checkpoint.ordinal,
                        "profile_id": profile.profile_id,
                        "operator": profile.operator,
                        "variant": profile.variant,
                        "strength": float(strength),
                        "condition_code": condition_code,
                        "replicate_count": len(replicate_metrics),
                        "metrics": pooled,
                    }
                )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def _finite_metric(metrics: Mapping[str, Any], key: str) -> float:
    value = metrics.get(key)
    if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
        raise ValueError(f"pilot metric {key} must be finite")
    return float(value)


def _settling_value(metrics: Mapping[str, Any]) -> float | None:
    if not bool(metrics.get("settling_valid")):
        return None
    settling = metrics.get("settling_all")
    if not isinstance(settling, Mapping):
        return None
    value = settling.get("restricted_mean_settling_steps")
    return (
        float(value)
        if isinstance(value, (int, float)) and np.isfinite(float(value))
        else None
    )


def summarize_checkpoint_signatures(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create paired baseline-versus-perturbation 0/2-back signatures."""
    baseline = {
        int(row["condition_code"]): row["metrics"]
        for row in rows
        if row.get("profile_id") is None
    }
    if set(baseline) != {0, 1}:
        raise ValueError("each checkpoint requires native 0-back and 2-back")
    grouped: dict[tuple[int, float], dict[int, Mapping[str, Any]]] = {}
    metadata: dict[tuple[int, float], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("profile_id") is None:
            continue
        key = (int(row["profile_id"]), float(row["strength"]))
        grouped.setdefault(key, {})[int(row["condition_code"])] = row["metrics"]
        metadata[key] = row
    signatures = []
    for key in sorted(grouped):
        conditions = grouped[key]
        if set(conditions) != {0, 1}:
            raise ValueError("every profile-strength requires both conditions")
        impairments = {
            condition: condition_normalized_discriminability_impairment(
                _finite_metric(baseline[condition], "discriminability"),
                _finite_metric(conditions[condition], "discriminability"),
            )
            for condition in (0, 1)
        }
        ce_changes = {
            condition: (
                _finite_metric(conditions[condition], "mean_cross_entropy")
                - _finite_metric(baseline[condition], "mean_cross_entropy")
            )
            for condition in (0, 1)
        }
        accuracy_changes = {
            condition: (
                _finite_metric(conditions[condition], "accuracy")
                - _finite_metric(baseline[condition], "accuracy")
            )
            for condition in (0, 1)
        }
        failure_changes = {
            condition: (
                _finite_metric(conditions[condition], "failure_rate")
                - _finite_metric(baseline[condition], "failure_rate")
            )
            for condition in (0, 1)
        }
        settling_changes: dict[int, float | None] = {}
        for condition in (0, 1):
            native_settling = _settling_value(baseline[condition])
            perturbed_settling = _settling_value(conditions[condition])
            settling_changes[condition] = (
                None
                if native_settling is None or perturbed_settling is None
                else perturbed_settling - native_settling
            )
        row = metadata[key]
        signatures.append(
            {
                "checkpoint_seed": int(row["checkpoint_seed"]),
                "checkpoint_ordinal": int(row["checkpoint_ordinal"]),
                "profile_id": key[0],
                "operator": str(row["operator"]),
                "variant": str(row["variant"]),
                "strength": key[1],
                "zero_back_discriminability_impairment": impairments[0],
                "two_back_discriminability_impairment": impairments[1],
                "load_selectivity": impairments[1] - impairments[0],
                "zero_back_additive_ce_change": ce_changes[0],
                "two_back_additive_ce_change": ce_changes[1],
                "ce_load_interaction": ce_changes[1] - ce_changes[0],
                "zero_back_accuracy_change": accuracy_changes[0],
                "two_back_accuracy_change": accuracy_changes[1],
                "accuracy_load_interaction": (
                    accuracy_changes[1] - accuracy_changes[0]
                ),
                "zero_back_failure_rate_change": failure_changes[0],
                "two_back_failure_rate_change": failure_changes[1],
                "failure_rate_load_interaction": (
                    failure_changes[1] - failure_changes[0]
                ),
                "zero_back_settling_change": settling_changes[0],
                "two_back_settling_change": settling_changes[1],
                "settling_load_interaction": (
                    None
                    if settling_changes[0] is None
                    or settling_changes[1] is None
                    else settling_changes[1] - settling_changes[0]
                ),
            }
        )
    return signatures


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    result.pop("sequence_cross_entropies", None)
    return result


def _atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def run_pilot(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    repo_root: str | Path = ".",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    device_override: str | None = None,
    design: PilotDesign = FROZEN_PILOT_DESIGN,
) -> dict[str, Any]:
    """Execute the isolated exploratory pilot and save descriptive artifacts."""
    design.validate()
    root = Path(repo_root).resolve()
    pilot_config_file, config_file, manifest_file = (
        _load_pilot_source_paths(config_path, repo_root=root)
    )
    operator_grids = load_pilot_operator_grids(
        pilot_config_file, repo_root=root
    )
    config = load_config(config_file)
    selected: SelectedDevice = select_device(
        device_override or config["training"].get("device", "auto")
    )
    checkpoints = load_pilot_checkpoints(
        repo_root=root,
        manifest_path=manifest_file,
        design=design,
    )
    all_rows: list[dict[str, Any]] = []
    signatures: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        checkpoint_rows = evaluate_checkpoint_cells(
            config,
            checkpoint,
            selected.device,
            operator_grids=operator_grids,
            design=design,
        )
        all_rows.extend(checkpoint_rows)
        signatures.extend(summarize_checkpoint_signatures(checkpoint_rows))
    resolved_output = Path(output_dir)
    if not resolved_output.is_absolute():
        resolved_output = root / resolved_output
    metrics_path = resolved_output / "pilot_cells.json"
    signatures_path = resolved_output / "pilot_signatures.csv"
    payload = {
        "status": "exploratory_descriptive_only",
        "claim_boundary": (
            "fixed strengths are not cost matched; three checkpoints are not "
            "a confirmatory inferential family"
        ),
        "design": asdict(design),
        "operator_grids": {
            name: list(grid) for name, grid in operator_grids.items()
        },
        "device": selected.description,
        "pilot_config_path": str(pilot_config_file),
        "pilot_config_sha256": sha256_file(pilot_config_file),
        "config_path": str(config_file.resolve()),
        "config_sha256": sha256_file(config_file),
        "cells": [
            {
                **{key: value for key, value in row.items() if key != "metrics"},
                "metrics": _compact_metrics(row["metrics"]),
            }
            for row in all_rows
        ],
    }
    atomic_write_json(metrics_path, payload)
    _atomic_write_csv(signatures_path, signatures)
    return {
        "metrics_path": str(metrics_path.resolve()),
        "signatures_path": str(signatures_path.resolve()),
        "n_cells": len(all_rows),
        "n_signatures": len(signatures),
        "status": payload["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the isolated three-checkpoint fixed-grid N-back pilot."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    result = run_pilot(
        config_path=args.config,
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        device_override=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
