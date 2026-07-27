"""Baseline-only precision planning for additive N-back log-loss cost."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from wm_rnn.config import load_config
from wm_rnn.device import select_device
from wm_rnn.io import ensure_run_dirs, write_json
from wm_rnn.nback_metrics import per_sequence_cross_entropy
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


FROZEN_RETAINED_SEEDS = (
    20260912,
    20260913,
    20260914,
    20260915,
    20260916,
    20260917,
    20260918,
    20260919,
    20260920,
    20260921,
)
DEFAULT_MANIFEST = Path(
    "outputs/nback_working_memory_screened_final/metrics/"
    "nback_working_memory_screened_final_screened_pool_summary.json"
)
DEFAULT_OUTPUT_DIR = Path("outputs/nback_additive_cost_precision")
DEFAULT_RUN_NAME = "nback_additive_cost_precision"


@dataclass(frozen=True)
class PrecisionDesign:
    """Frozen precision-reference and planning constants."""

    retained_seeds: tuple[int, ...] = FROZEN_RETAINED_SEEDS
    bank_base: int = 131000000
    n_batches: int = 64
    batch_size: int = 128
    bootstrap_seed: int = 135000000
    bootstrap_draws: int = 10000
    bootstrap_chunk_size: int = 32
    bootstrap_percentile: float = 95.0
    kappa: float = 2.0
    z_value: float = 1.96
    half_width: float = 0.005
    minimum_cost_check: int = 1024
    maximum_cost_check: int = 8192
    cost_check_multiple: int = 128

    @property
    def sequences_per_checkpoint(self) -> int:
        return self.n_batches * self.batch_size

    def validate(self) -> None:
        if not self.retained_seeds:
            raise ValueError("retained_seeds must not be empty")
        if len(set(self.retained_seeds)) != len(self.retained_seeds):
            raise ValueError("retained_seeds must be unique")
        if min(
            self.bank_base,
            self.n_batches,
            self.batch_size,
            self.bootstrap_draws,
            self.bootstrap_chunk_size,
            self.minimum_cost_check,
            self.maximum_cost_check,
            self.cost_check_multiple,
        ) <= 0:
            raise ValueError("precision design counts and bases must be positive")
        if not 0.0 < self.bootstrap_percentile < 100.0:
            raise ValueError("bootstrap_percentile must lie between 0 and 100")
        if min(self.kappa, self.z_value, self.half_width) <= 0.0:
            raise ValueError("planning constants must be positive")
        if self.minimum_cost_check > self.maximum_cost_check:
            raise ValueError(
                "minimum_cost_check must not exceed maximum_cost_check"
            )


FROZEN_DESIGN = PrecisionDesign()


@dataclass(frozen=True)
class RetainedCheckpoint:
    """One manifest-validated retained checkpoint."""

    ordinal: int
    seed: int
    path: Path


@dataclass(frozen=True)
class AdditiveCostPrecisionResult:
    """Paths and decision from one baseline-only precision run."""

    summary_path: Path
    descriptions_csv_path: Path
    seed_map_csv_path: Path
    arrays_path: Path
    passed: bool
    n_cost_check: int | None


CollectFunction = Callable[
    [
        dict[str, Any],
        RetainedCheckpoint,
        torch.device,
        PrecisionDesign,
    ],
    np.ndarray,
]


def precision_task_seed(
    checkpoint_ordinal: int,
    batch_index: int,
    *,
    design: PrecisionDesign = FROZEN_DESIGN,
) -> int:
    """Return the frozen baseline-reference 0-back task seed."""
    design.validate()
    ordinal = int(checkpoint_ordinal)
    batch = int(batch_index)
    if not 0 <= ordinal < len(design.retained_seeds):
        raise ValueError("checkpoint_ordinal is outside the retained family")
    if not 0 <= batch < design.n_batches:
        raise ValueError("batch_index is outside the precision bank")
    return design.bank_base + 10_000 * ordinal + batch


def precision_task_config(
    config: dict[str, Any],
    checkpoint_ordinal: int,
    batch_index: int,
    *,
    design: PrecisionDesign = FROZEN_DESIGN,
) -> NBackTaskConfig:
    """Build one frozen homogeneous 0-back precision batch config."""
    task = task_config_from_dict(config, batch_size=design.batch_size)
    if not isinstance(task, NBackTaskConfig):
        raise ValueError("precision planning requires task_type: n_back")
    return replace(
        task,
        n_back=0,
        seed=precision_task_seed(
            checkpoint_ordinal,
            batch_index,
            design=design,
        ),
    )


def sequence_log_loss_units(
    logits: torch.Tensor,
    batch: NBackBatch,
) -> np.ndarray:
    """Return one unweighted natural-log CE observation per sequence."""
    if (
        logits.ndim != 3
        or logits.shape[:2] != batch.targets.shape
        or logits.shape[-1] != 2
    ):
        raise ValueError(
            "logits must have shape [time, batch, 2] matching the batch"
        )
    targets = torch.as_tensor(
        batch.targets, dtype=torch.long, device=logits.device
    )
    loss_mask = torch.as_tensor(
        batch.loss_mask, dtype=logits.dtype, device=logits.device
    )
    return per_sequence_cross_entropy(
        logits,
        targets,
        loss_mask,
    ).astype(np.float64, copy=False)


def _resolve_checkpoint_path(path: str, repo_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def load_retained_checkpoints(
    manifest_path: str | Path,
    *,
    repo_root: str | Path,
    design: PrecisionDesign = FROZEN_DESIGN,
) -> list[RetainedCheckpoint]:
    """Load and strictly validate the screened-pool retained manifest."""
    design.validate()
    manifest = Path(manifest_path)
    with manifest.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = list(design.retained_seeds)
    if payload.get("retained_seeds") != expected:
        raise ValueError("manifest retained seed order does not match registration")
    if payload.get("attempted_seeds") != expected:
        raise ValueError("manifest attempted seeds do not match retained family")
    if payload.get("failed_seeds") != []:
        raise ValueError("precision manifest must contain no failed retained seeds")
    if not bool(payload.get("passed")):
        raise ValueError("screened-pool manifest did not pass")
    if payload.get("stop_reason") != "target_reached":
        raise ValueError("screened-pool manifest did not reach its target")
    paths = payload.get("retained_checkpoints")
    if not isinstance(paths, list) or len(paths) != len(expected):
        raise ValueError("manifest checkpoint list is incomplete")
    root = Path(repo_root).resolve()
    records: list[RetainedCheckpoint] = []
    for ordinal, (seed, raw_path) in enumerate(zip(expected, paths)):
        checkpoint_path = _resolve_checkpoint_path(str(raw_path), root)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"retained checkpoint does not exist: {checkpoint_path}"
            )
        records.append(
            RetainedCheckpoint(
                ordinal=ordinal,
                seed=int(seed),
                path=checkpoint_path,
            )
        )
    return records


@torch.no_grad()
def collect_checkpoint_sequence_units(
    config: dict[str, Any],
    checkpoint: RetainedCheckpoint,
    device: torch.device,
    design: PrecisionDesign = FROZEN_DESIGN,
) -> np.ndarray:
    """Collect the complete native 0-back precision bank for one checkpoint."""
    saved = torch.load(checkpoint.path, map_location=device)
    checkpoint_config = saved.get("config", {})
    saved_seed = checkpoint_config.get("task", {}).get("seed")
    if int(saved_seed) != checkpoint.seed:
        raise ValueError("checkpoint task seed does not match retained manifest")
    model = fresh_model(config, device)
    model.load_state_dict(saved["model_state"])
    model.eval()
    collected: list[np.ndarray] = []
    for batch_index in range(design.n_batches):
        task = precision_task_config(
            config,
            checkpoint.ordinal,
            batch_index,
            design=design,
        )
        batch = generate_nback_batch(task)
        inputs, _, _ = batch_to_tensors(batch, device)
        logits, _ = model(inputs)
        units = sequence_log_loss_units(logits, batch)
        if units.shape != (design.batch_size,):
            raise ValueError(
                "each precision batch must yield one unit per sequence"
            )
        collected.append(units)
    values = np.concatenate(collected).astype(np.float64, copy=False)
    validate_sequence_units(
        values,
        expected_count=design.sequences_per_checkpoint,
    )
    return values


def validate_sequence_units(
    values: np.ndarray,
    *,
    expected_count: int,
) -> np.ndarray:
    """Validate one checkpoint's complete sequence-level CE vector."""
    units = np.asarray(values, dtype=np.float64)
    if units.shape != (int(expected_count),):
        raise ValueError(
            f"expected exactly {expected_count} sequence log-loss units"
        )
    if not np.all(np.isfinite(units)):
        raise ValueError("sequence log-loss units must all be finite")
    if np.any(units < 0.0):
        raise ValueError("sequence log-loss units must all be non-negative")
    return units


def describe_sequence_units(values: np.ndarray) -> dict[str, float | int]:
    """Return all registered checkpoint-level descriptive statistics."""
    units = np.asarray(values, dtype=np.float64)
    validate_sequence_units(units, expected_count=units.size)
    if units.size < 2:
        raise ValueError("at least two sequence units are required")
    q1, median, q3, p90, p95, p99 = np.percentile(
        units,
        [25.0, 50.0, 75.0, 90.0, 95.0, 99.0],
        method="linear",
    )
    return {
        "n_sequences": int(units.size),
        "mean": float(np.mean(units)),
        "sample_sd": float(np.std(units, ddof=1)),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99),
        "maximum": float(np.max(units)),
    }


def bootstrap_family_max_sd(
    checkpoint_units: np.ndarray,
    *,
    draws: int,
    seed: int,
    chunk_size: int,
) -> np.ndarray:
    """Bootstrap the family maximum sample SD in memory-safe chunks."""
    units = np.asarray(checkpoint_units, dtype=np.float64)
    if units.ndim != 2 or units.shape[0] == 0 or units.shape[1] < 2:
        raise ValueError(
            "checkpoint_units must be [checkpoint, sequence] with n >= 2"
        )
    if not np.all(np.isfinite(units)) or np.any(units < 0.0):
        raise ValueError("checkpoint_units must be finite and non-negative")
    resolved_draws = int(draws)
    resolved_chunk = int(chunk_size)
    if resolved_draws <= 0 or resolved_chunk <= 0:
        raise ValueError("draws and chunk_size must be positive")

    maximum_sds = np.full(resolved_draws, -np.inf, dtype=np.float64)
    child_seeds = np.random.SeedSequence(int(seed)).spawn(units.shape[0])
    for checkpoint_index, child_seed in enumerate(child_seeds):
        rng = np.random.default_rng(child_seed)
        values = units[checkpoint_index]
        n_sequences = values.size
        for start in range(0, resolved_draws, resolved_chunk):
            stop = min(start + resolved_chunk, resolved_draws)
            indices = rng.integers(
                0,
                n_sequences,
                size=(stop - start, n_sequences),
                dtype=np.int32,
            )
            sampled_sds = np.std(values[indices], axis=1, ddof=1)
            np.maximum(
                maximum_sds[start:stop],
                sampled_sds,
                out=maximum_sds[start:stop],
            )
    if not np.all(np.isfinite(maximum_sds)):
        raise ValueError("bootstrap maximum SD values must be finite")
    return maximum_sds


def round_up_to_multiple(value: float, multiple: int) -> int:
    """Round a finite non-negative value upward to a complete batch."""
    resolved = float(value)
    if not np.isfinite(resolved) or resolved < 0.0:
        raise ValueError("value must be finite and non-negative")
    if int(multiple) <= 0:
        raise ValueError("multiple must be positive")
    return int(np.ceil(resolved / int(multiple)) * int(multiple))


def derive_cost_check_size(
    sigma_upper: float,
    *,
    design: PrecisionDesign = FROZEN_DESIGN,
) -> tuple[float, int]:
    """Apply the frozen conservative additive-cost planning formula."""
    design.validate()
    sigma = float(sigma_upper)
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma_upper must be finite and non-negative")
    n_required = (
        design.z_value
        * sigma
        * np.sqrt(1.0 + design.kappa**2)
        / design.half_width
    ) ** 2
    rounded = round_up_to_multiple(
        n_required,
        design.cost_check_multiple,
    )
    return float(n_required), max(design.minimum_cost_check, rounded)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _seed_map_rows(
    checkpoints: list[RetainedCheckpoint],
    design: PrecisionDesign,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    task_seeds = np.empty(
        (len(checkpoints), design.n_batches),
        dtype=np.int64,
    )
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        for batch_index in range(design.n_batches):
            task_seed = precision_task_seed(
                checkpoint.ordinal,
                batch_index,
                design=design,
            )
            task_seeds[checkpoint.ordinal, batch_index] = task_seed
            rows.append(
                {
                    "checkpoint_ordinal": checkpoint.ordinal,
                    "checkpoint_seed": checkpoint.seed,
                    "condition": "0-back",
                    "condition_code": 0,
                    "batch_index": batch_index,
                    "batch_size": design.batch_size,
                    "task_seed": task_seed,
                    "bank_base": design.bank_base,
                }
            )
    return rows, task_seeds


def run_nback_additive_cost_precision(
    config: dict[str, Any],
    manifest_path: str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    run_name: str = DEFAULT_RUN_NAME,
    repo_root: str | Path = ".",
    design: PrecisionDesign = FROZEN_DESIGN,
    collect_fn: CollectFunction | None = None,
) -> AdditiveCostPrecisionResult:
    """Run and persist the frozen baseline-only additive precision phase."""
    design.validate()
    root = Path(repo_root).resolve()
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = root / manifest
    manifest = manifest.resolve()
    checkpoints = load_retained_checkpoints(
        manifest,
        repo_root=root,
        design=design,
    )
    device_info = select_device(config["training"].get("device", "auto"))
    collect = collect_fn or collect_checkpoint_sequence_units
    unit_rows: list[np.ndarray] = []
    descriptions: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        units = validate_sequence_units(
            collect(config, checkpoint, device_info.device, design),
            expected_count=design.sequences_per_checkpoint,
        )
        unit_rows.append(units)
        descriptions.append(
            {
                "checkpoint_ordinal": checkpoint.ordinal,
                "checkpoint_seed": checkpoint.seed,
                "checkpoint_path": str(checkpoint.path),
                **describe_sequence_units(units),
                "first_task_seed": precision_task_seed(
                    checkpoint.ordinal, 0, design=design
                ),
                "last_task_seed": precision_task_seed(
                    checkpoint.ordinal,
                    design.n_batches - 1,
                    design=design,
                ),
            }
        )
    unit_matrix = np.stack(unit_rows)
    expected_shape = (
        len(design.retained_seeds),
        design.sequences_per_checkpoint,
    )
    if unit_matrix.shape != expected_shape:
        raise ValueError(f"precision units must have shape {expected_shape}")

    maximum_sds = bootstrap_family_max_sd(
        unit_matrix,
        draws=design.bootstrap_draws,
        seed=design.bootstrap_seed,
        chunk_size=design.bootstrap_chunk_size,
    )
    sigma_upper = float(
        np.percentile(
            maximum_sds,
            design.bootstrap_percentile,
            method="linear",
        )
    )
    n_required, n_cost_check = derive_cost_check_size(
        sigma_upper,
        design=design,
    )
    seed_rows, task_seeds = _seed_map_rows(checkpoints, design)
    complete_seed_map = (
        len(seed_rows) == len(checkpoints) * design.n_batches
        and len(np.unique(task_seeds)) == task_seeds.size
    )

    dirs = ensure_run_dirs(output_dir)
    arrays_path = dirs["arrays"] / f"{run_name}_arrays.npz"
    np.savez_compressed(
        arrays_path,
        checkpoint_seeds=np.asarray(
            design.retained_seeds, dtype=np.int64
        ),
        sequence_log_loss_units=unit_matrix,
        task_seeds=task_seeds,
        bootstrap_maximum_sds=maximum_sds,
    )
    descriptions_path = _write_csv(
        dirs["metrics"] / f"{run_name}_checkpoint_descriptions.csv",
        descriptions,
        list(descriptions[0]),
    )
    seed_map_path = _write_csv(
        dirs["metrics"] / f"{run_name}_seed_map.csv",
        seed_rows,
        list(seed_rows[0]),
    )

    checks = {
        "checkpoint_order_exact": [
            checkpoint.seed for checkpoint in checkpoints
        ]
        == list(design.retained_seeds),
        "all_checkpoints_have_exact_sequence_count": all(
            row["n_sequences"] == design.sequences_per_checkpoint
            for row in descriptions
        ),
        "all_sequence_units_finite": bool(np.all(np.isfinite(unit_matrix))),
        "all_sequence_units_nonnegative": bool(np.all(unit_matrix >= 0.0)),
        "sigma_upper_finite": bool(np.isfinite(sigma_upper)),
        "n_required_finite": bool(np.isfinite(n_required)),
        "bootstrap_draw_count_exact": (
            maximum_sds.shape == (design.bootstrap_draws,)
        ),
        "bootstrap_values_finite": bool(np.all(np.isfinite(maximum_sds))),
        "n_cost_check_meets_minimum": (
            n_cost_check >= design.minimum_cost_check
        ),
        "n_cost_check_is_complete_batch_multiple": (
            n_cost_check % design.cost_check_multiple == 0
        ),
        "n_cost_check_within_maximum": (
            n_cost_check <= design.maximum_cost_check
        ),
        "complete_seed_map": bool(complete_seed_map),
        "audit_arrays_persisted": arrays_path.is_file(),
        "descriptions_persisted": descriptions_path.is_file(),
        "seed_map_persisted": seed_map_path.is_file(),
    }
    passed = bool(all(checks.values()))
    checkpoint_hashes = {
        str(checkpoint.seed): _sha256(checkpoint.path)
        for checkpoint in checkpoints
    }
    payload = {
        "phase": "baseline_only_additive_cost_precision",
        "device": device_info.description,
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "design": asdict(design),
        "sequence_unit": (
            "mean natural-log cross-entropy across registered scored "
            "timepoints within one complete 0-back sequence"
        ),
        "task_seed_mapping": {
            "formula": (
                "bank_base + 10000 * checkpoint_ordinal "
                "+ 1000 * condition_code + batch_index"
            ),
            "condition": "0-back",
            "condition_code": 0,
        },
        "percentile_method": "numpy_linear",
        "checkpoint_descriptions": descriptions,
        "bootstrap": {
            "rng": "numpy SeedSequence spawn plus default_rng PCG64",
            "seed": design.bootstrap_seed,
            "draws": design.bootstrap_draws,
            "chunk_size": design.bootstrap_chunk_size,
            "family_statistic": "maximum checkpoint sample SD",
            "sigma_upper_percentile": design.bootstrap_percentile,
            "sigma_upper": sigma_upper,
        },
        "planning": {
            "kappa": design.kappa,
            "z_value": design.z_value,
            "half_width": design.half_width,
            "n_required": n_required,
            "n_cost_check": n_cost_check,
            "minimum_cost_check": design.minimum_cost_check,
            "maximum_cost_check": design.maximum_cost_check,
            "rounding_multiple": design.cost_check_multiple,
        },
        "checks": checks,
        "passed": passed,
        "checkpoint_sha256": checkpoint_hashes,
        "artifacts": {
            "arrays_npz": str(arrays_path.resolve()),
            "arrays_sha256": _sha256(arrays_path),
            "checkpoint_descriptions_csv": str(
                descriptions_path.resolve()
            ),
            "checkpoint_descriptions_sha256": _sha256(descriptions_path),
            "seed_map_csv": str(seed_map_path.resolve()),
            "seed_map_sha256": _sha256(seed_map_path),
        },
    }
    summary_path = write_json(
        dirs["metrics"] / f"{run_name}_summary.json",
        payload,
    )
    return AdditiveCostPrecisionResult(
        summary_path=summary_path,
        descriptions_csv_path=descriptions_path,
        seed_map_csv_path=seed_map_path,
        arrays_path=arrays_path,
        passed=passed,
        n_cost_check=n_cost_check,
    )


def main() -> None:
    """Run the frozen baseline-only N-back additive-cost precision phase."""
    parser = argparse.ArgumentParser(
        description="Plan additive N-back cost-check precision."
    )
    parser.add_argument(
        "--config",
        default="configs/nback_working_memory_screened_final.yaml",
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device
    result = run_nback_additive_cost_precision(
        config,
        args.manifest,
        output_dir=args.output_dir,
        run_name=args.run_name,
        repo_root=Path.cwd(),
    )
    print(f"summary={result.summary_path}")
    print(f"arrays={result.arrays_path}")
    print(f"n_cost_check={result.n_cost_check}")
    print(f"passed={result.passed}")
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
