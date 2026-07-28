"""Compact exploratory perturbation pilot for the frozen Family A RNNs.

The module deliberately reuses the frozen task, operator, decoder, and metric
implementations.  Importing it is outcome-free; checkpoint evaluation occurs
only through :func:`run_pilot` or the explicit ``--execute`` CLI flag.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml

from wm_rnn.config import load_config
from wm_rnn.device import select_device
from wm_rnn.io import ensure_run_dirs, write_json
from wm_rnn.perturbation_experiment import (
    FINAL_SEED_BASE,
    P2_VECTOR_SEEDS,
    P5_REPLICATES,
    _baseline_threshold,
    _collect_batches,
    _load_checkpoint_model,
    _operator_forward,
    fit_frozen_decoder,
    summarize_collected,
)
from wm_rnn.training_utils import task_config_from_dict
from wm_rnn.tuned_task import TunedDelayTaskConfig


CONFIG_PATH = Path("configs/fixation_circular_working_memory.yaml")
CONFIG_SHA256 = "8B7CC5AF773674D064684F81DC4C82BEE72BBD5DDD0313EE8FB89DD333D51DB9"
PILOT_CONFIG_PATH = Path("configs/exploratory_psilocybin_signature_pilot.yaml")
DEFAULT_OUTPUT_DIR = Path(
    "outputs/exploratory_psilocybin_signature_pilot/circular_family_a"
)

PILOT_SEEDS = (20260714, 20260715, 20260716)
TRIALS_PER_CELL = 256
DELAYS = (10, 20, 40, 80)
DISTRACTOR_DELAY = 20
DISTRACTOR_STEPS = 5
FAMILY = "A"


@dataclass(frozen=True)
class FrozenCheckpoint:
    seed: int
    path: str
    sha256: str


@dataclass(frozen=True)
class PilotCell:
    condition: str
    delay_steps: int


@dataclass(frozen=True)
class OperatorSetting:
    operator: str
    variant: str
    strength: float
    gain_vector_seed: int | None = None
    noise_replicate: int | None = None


FROZEN_CHECKPOINTS = (
    FrozenCheckpoint(
        20260714,
        "outputs/fixation_circular_working_memory/seed_sweep/"
        "seed_20260714/checkpoints/"
        "fixation_circular_working_memory_seed_20260714.pt",
        "C5DB705243C7CE3E9B699E6FB6F7EECEAB3713BD262650DF44D8E54C67DA1CA3",
    ),
    FrozenCheckpoint(
        20260715,
        "outputs/fixation_circular_working_memory/seed_sweep/"
        "seed_20260715/checkpoints/"
        "fixation_circular_working_memory_seed_20260715.pt",
        "77793E5087A70BEE377A20D969A3145803E476BA5D7560495608143104947839",
    ),
    FrozenCheckpoint(
        20260716,
        "outputs/fixation_circular_working_memory/seed_sweep/"
        "seed_20260716/checkpoints/"
        "fixation_circular_working_memory_seed_20260716.pt",
        "89ADA9F6B5D28951E876E295EC2E5D70D0DEB667524721CDC2220A7BCB781CA0",
    ),
)

FULL_CHECKPOINTS = FROZEN_CHECKPOINTS + (
    FrozenCheckpoint(
        20260717,
        "outputs/fixation_circular_working_memory/seed_sweep/"
        "seed_20260717/checkpoints/"
        "fixation_circular_working_memory_seed_20260717.pt",
        "8421AE02CFC8000EFD0771ABD5B3968DEED3866F9598AF728BBB8E2A9BB3D3BF",
    ),
    FrozenCheckpoint(
        20260718,
        "outputs/fixation_circular_working_memory/seed_sweep/"
        "seed_20260718/checkpoints/"
        "fixation_circular_working_memory_seed_20260718.pt",
        "B91AC1D26B6F28C7BB3B51B4E982505D2BD23D81D3B86F4EF614F72066F3601E",
    ),
)

PILOT_CELLS = tuple(PilotCell("clean", delay) for delay in DELAYS) + (
    PilotCell("distractor", DISTRACTOR_DELAY),
)

OPERATOR_VARIANTS = {
    "synaptic_drive_gain": "bias_outside",
    "heterogeneous_drive_gain": "bias_outside",
    "sensory_input_gain": "sensory_only",
    "distractor_input_gain": "distractor_only",
    "recurrent_gain": "recurrent_only",
    "gaussian_state_noise": "generic_control",
    "state_persistence": "carried_state_only",
    "time_constant": "conserved_integrator",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_pilot_config(
    path: str | Path,
    *,
    checkpoint_seeds: tuple[int, ...] = PILOT_SEEDS,
    trials_per_cell: int = TRIALS_PER_CELL,
    required_operators: set[str] | None = None,
) -> dict[str, Any]:
    """Load and validate the fixed exploratory circular-pilot design."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    pilot = config.get("pilot", {})
    expected = {
        "circular_seeds": list(checkpoint_seeds),
        "circular_trials_per_cell": trials_per_cell,
        "circular_delays": list(DELAYS),
        "distractor_delay": DISTRACTOR_DELAY,
        "distractor_steps": DISTRACTOR_STEPS,
    }
    for key, value in expected.items():
        if pilot.get(key) != value:
            raise ValueError(f"pilot.{key} must remain fixed at {value!r}")
    if int(pilot["batch_size"]) <= 0:
        raise ValueError("pilot.batch_size must be positive")
    if trials_per_cell % int(pilot["batch_size"]) != 0:
        raise ValueError("circular_trials_per_cell must divide evenly by batch_size")
    operators = config.get("operators", {})
    expected_operators = required_operators or set(OPERATOR_VARIANTS)
    if set(operators) != expected_operators:
        raise ValueError("operator grids do not match the frozen circular pilot")
    if any(not values for values in operators.values()):
        raise ValueError("every circular pilot operator requires a non-empty grid")
    return config


def verify_frozen_inputs(
    repo_root: str | Path,
    *,
    checkpoints: tuple[FrozenCheckpoint, ...] = FROZEN_CHECKPOINTS,
    config_path: str | Path = CONFIG_PATH,
    expected_config_sha256: str | None = CONFIG_SHA256,
) -> dict[str, Any]:
    """Verify exact config and checkpoint identities before any model loading."""
    root = Path(repo_root).resolve()
    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_absolute():
        resolved_config_path = root / resolved_config_path
    observed_config_hash = _sha256(resolved_config_path)
    if (
        expected_config_sha256 is not None
        and observed_config_hash != expected_config_sha256
    ):
        raise RuntimeError(
            f"config hash mismatch: {observed_config_hash} != "
            f"{expected_config_sha256}"
        )
    verified = []
    for checkpoint in checkpoints:
        path = root / checkpoint.path
        observed_hash = _sha256(path)
        if observed_hash != checkpoint.sha256:
            raise RuntimeError(
                f"checkpoint {checkpoint.seed} hash mismatch: "
                f"{observed_hash} != {checkpoint.sha256}"
            )
        verified.append({**asdict(checkpoint), "resolved_path": str(path)})
    return {
        "config_path": str(resolved_config_path),
        "config_sha256": observed_config_hash,
        "checkpoints": verified,
    }


def build_operator_settings(
    operator_grids: dict[str, Iterable[float]],
) -> tuple[OperatorSetting, ...]:
    """Expand fixed grids and registered stochastic/vector replicates."""
    settings: list[OperatorSetting] = []
    for operator, strengths in operator_grids.items():
        variant = OPERATOR_VARIANTS[operator]
        for raw_strength in strengths:
            strength = float(raw_strength)
            vector_seeds = (
                P2_VECTOR_SEEDS
                if operator == "heterogeneous_drive_gain"
                else (None,)
            )
            noise_replicates = (
                P5_REPLICATES if operator == "gaussian_state_noise" else (None,)
            )
            for vector_seed in vector_seeds:
                for noise_replicate in noise_replicates:
                    settings.append(
                        OperatorSetting(
                            operator,
                            variant,
                            strength,
                            vector_seed,
                            noise_replicate,
                        )
                    )
    return tuple(settings)


def settings_for_cell(
    settings: Iterable[OperatorSetting], cell: PilotCell
) -> tuple[OperatorSetting, ...]:
    """Keep distractor-only input gain confined to the distractor cell."""
    return tuple(
        setting
        for setting in settings
        if setting.operator != "distractor_input_gain"
        or cell.condition == "distractor"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("pilot produced no rows")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _run_setting(
    model: torch.nn.Module,
    task_config: TunedDelayTaskConfig,
    checkpoint_seed: int,
    cell: PilotCell,
    setting: OperatorSetting,
    *,
    n_batches: int,
    batch_size: int,
    randomize_distractor_onsets: bool = False,
) -> dict[str, Any]:
    condition_index = 0 if cell.condition == "clean" else 1
    forward_factory = None
    if setting.operator == "gaussian_state_noise":

        def forward_factory(batch_seed: int):
            return _operator_forward(
                model,
                task_config,
                operator=setting.operator,
                variant=setting.variant,
                strength=setting.strength,
                condition=cell.condition,
                family=FAMILY,
                delay_steps=cell.delay_steps,
                noise_replicate=(
                    batch_seed
                    + 1_000_000 * checkpoint_seed
                    + int(setting.noise_replicate or 0)
                ),
            )

        forward = None
    else:
        forward = _operator_forward(
            model,
            task_config,
            operator=setting.operator,
            variant=setting.variant,
            strength=setting.strength,
            condition=cell.condition,
            family=FAMILY,
            delay_steps=cell.delay_steps,
            gain_vector_seed=setting.gain_vector_seed,
            randomize_distractor_onsets=randomize_distractor_onsets,
        )
    return _collect_batches(
        model,
        task_config,
        FAMILY,
        cell.condition,
        condition_index,
        cell.delay_steps,
        seed_base=FINAL_SEED_BASE,
        n_batches=n_batches,
        batch_size=batch_size,
        forward_fn=forward,
        forward_seed_factory=forward_factory,
        randomize_distractor_onsets=randomize_distractor_onsets,
    )


def run_pilot(
    repo_root: str | Path,
    *,
    pilot_config_path: str | Path = PILOT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    device: str = "auto",
    checkpoints: tuple[FrozenCheckpoint, ...] = FROZEN_CHECKPOINTS,
    trials_per_cell: int = TRIALS_PER_CELL,
    required_operators: set[str] | None = None,
    base_config_path: str | Path = CONFIG_PATH,
    expected_base_config_sha256: str | None = CONFIG_SHA256,
    output_stem: str = "circular_family_a",
    interpretive_limit: str | None = None,
    randomize_distractor_onsets: bool = False,
) -> tuple[Path, Path]:
    """Run a fixed one-item circular exploratory perturbation grid."""
    root = Path(repo_root).resolve()
    frozen = verify_frozen_inputs(
        root,
        checkpoints=checkpoints,
        config_path=base_config_path,
        expected_config_sha256=expected_base_config_sha256,
    )
    pilot_path = Path(pilot_config_path)
    if not pilot_path.is_absolute():
        pilot_path = root / pilot_path
    checkpoint_seeds = tuple(checkpoint.seed for checkpoint in checkpoints)
    pilot_config = load_pilot_config(
        pilot_path,
        checkpoint_seeds=checkpoint_seeds,
        trials_per_cell=trials_per_cell,
        required_operators=required_operators,
    )
    resolved_base_config = Path(base_config_path)
    if not resolved_base_config.is_absolute():
        resolved_base_config = root / resolved_base_config
    base_config = load_config(resolved_base_config)
    task_config = task_config_from_dict(base_config)
    if not isinstance(task_config, TunedDelayTaskConfig):
        raise TypeError("Family A pilot requires TunedDelayTaskConfig")
    batch_size = int(pilot_config["pilot"]["batch_size"])
    task_config = replace(
        task_config,
        batch_size=batch_size,
        distractor_steps=DISTRACTOR_STEPS,
    )
    n_batches = trials_per_cell // batch_size
    settings = build_operator_settings(pilot_config["operators"])
    selected = select_device(device)
    rows: list[dict[str, Any]] = []
    angle_hashes: dict[str, Any] = {}
    distractor_timing_hashes: dict[str, str] = {}

    for checkpoint in checkpoints:
        checkpoint_path = root / checkpoint.path
        model = _load_checkpoint_model(
            base_config, checkpoint_path, selected.device
        )
        decoder = fit_frozen_decoder(model, task_config, FAMILY)
        for cell in PILOT_CELLS:
            condition_index = 0 if cell.condition == "clean" else 1
            threshold = _baseline_threshold(
                model,
                task_config,
                FAMILY,
                cell.condition,
                condition_index,
                cell.delay_steps,
            )
            baseline = _collect_batches(
                model,
                task_config,
                FAMILY,
                cell.condition,
                condition_index,
                cell.delay_steps,
                seed_base=FINAL_SEED_BASE,
                n_batches=n_batches,
                batch_size=batch_size,
                randomize_distractor_onsets=(
                    randomize_distractor_onsets
                    and cell.condition == "distractor"
                ),
            )
            angle_hashes[
                f"{checkpoint.seed}:{cell.condition}:{cell.delay_steps}"
            ] = baseline["angle_hashes"]
            if baseline["distractor_relative_starts"] is not None:
                timing_key = (
                    f"{checkpoint.seed}:{cell.condition}:"
                    f"{cell.delay_steps}"
                )
                distractor_timing_hashes[timing_key] = hashlib.sha256(
                    baseline["distractor_relative_starts"].tobytes()
                ).hexdigest().upper()
            baseline_metric = summarize_collected(
                baseline, decoder, threshold, family=FAMILY
            )[0]
            for setting in settings_for_cell(settings, cell):
                collected = _run_setting(
                    model,
                    task_config,
                    checkpoint.seed,
                    cell,
                    setting,
                    n_batches=n_batches,
                    batch_size=batch_size,
                    randomize_distractor_onsets=(
                        randomize_distractor_onsets
                        and cell.condition == "distractor"
                    ),
                )
                metric = summarize_collected(
                    collected,
                    decoder,
                    threshold,
                    family=FAMILY,
                    baseline_fraction_settled=baseline_metric[
                        "fraction_settled"
                    ],
                )[0]
                row = {
                    "checkpoint_seed": checkpoint.seed,
                    "condition": cell.condition,
                    "delay_steps": cell.delay_steps,
                    **asdict(setting),
                    **metric,
                }
                for key in (
                    "mean_angular_error_degrees",
                    "restricted_mean_settling_steps",
                    "fraction_settled",
                    "failure_rate",
                    "delay_decode_error_degrees",
                    "mean_late_delay_state_entropy",
                ):
                    row[f"baseline_{key}"] = baseline_metric[key]
                    row[f"delta_{key}"] = metric[key] - baseline_metric[key]
                rows.append(row)

    resolved_output = Path(output_dir)
    if not resolved_output.is_absolute():
        resolved_output = root / resolved_output
    dirs = ensure_run_dirs(resolved_output)
    csv_path = _write_csv(dirs["metrics"] / f"{output_stem}_grid.csv", rows)
    metadata = {
        "exploratory": True,
        "family": FAMILY,
        "device": selected.description,
        "frozen_inputs": frozen,
        "pilot_config_path": str(pilot_path),
        "pilot_config_sha256": _sha256(pilot_path),
        "seeds": list(checkpoint_seeds),
        "trials_per_cell": trials_per_cell,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "cells": [asdict(cell) for cell in PILOT_CELLS],
        "operator_settings": [asdict(setting) for setting in settings],
        "angle_hashes": angle_hashes,
        "distractor_timing_mode": (
            "per_trial_stratified_uniform_all_valid_starts"
            if randomize_distractor_onsets
            else "fixed"
        ),
        "distractor_timing_hashes": distractor_timing_hashes,
        "interpretive_limit": interpretive_limit
        or (
            "Exploratory three-checkpoint/OOD-distractor pilot; not "
            "confirmatory inference and not a biological psilocybin model."
        ),
    }
    metadata_path = write_json(
        dirs["metrics"] / f"{output_stem}_metadata.json", metadata
    )
    return csv_path, metadata_path


def design_summary(pilot_config_path: str | Path) -> dict[str, Any]:
    """Return the outcome-free expanded design for inspection and tests."""
    config = load_pilot_config(pilot_config_path)
    settings = build_operator_settings(config["operators"])
    return {
        "seeds": list(PILOT_SEEDS),
        "trials_per_cell": TRIALS_PER_CELL,
        "cells": [asdict(cell) for cell in PILOT_CELLS],
        "operator_settings": [asdict(setting) for setting in settings],
        "planned_cells": sum(
            len(settings_for_cell(settings, cell)) for cell in PILOT_CELLS
        )
        * len(PILOT_SEEDS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--pilot-config", default=str(PILOT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Load checkpoints and run outcomes; otherwise print design only.",
    )
    args = parser.parse_args()
    pilot_path = Path(args.pilot_config)
    if not pilot_path.is_absolute():
        pilot_path = Path(args.repo_root).resolve() / pilot_path
    if not args.execute:
        print(json.dumps(design_summary(pilot_path), indent=2))
        return
    csv_path, metadata_path = run_pilot(
        args.repo_root,
        pilot_config_path=pilot_path,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(csv_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
