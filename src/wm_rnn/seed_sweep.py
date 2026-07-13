"""Train and evaluate working-memory models across multiple random seeds."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wm_rnn.config import load_config
from wm_rnn.cross_temporal_decoder import run_cross_temporal_decoder
from wm_rnn.delay_sweep import run_delay_sweep
from wm_rnn.evaluate import evaluate_model
from wm_rnn.io import ensure_run_dirs, write_json
from wm_rnn.train import train_model
from wm_rnn.training_utils import task_config_from_dict


@dataclass(frozen=True)
class SeedSweepResult:
    """Paths and metrics produced by multi-seed training.

    Attributes:
        summary_path: JSON file containing all seed-run metadata.
        csv_path: CSV file containing one row per seed.
        results: In-memory per-seed records.
    """

    summary_path: Path
    csv_path: Path
    results: list[dict[str, Any]]


def run_seed_sweep(config: dict[str, Any], seeds: list[int], delays: list[int] | None = None) -> SeedSweepResult:
    """Train and evaluate independent baseline models across task seeds.

    Each seed gets its own output directory below
    ``<output_dir>/seed_sweep/seed_<seed>``. If ``delays`` are supplied, each
    trained checkpoint is also evaluated with a frozen-weight delay sweep.

    Args:
        config: Experiment configuration dictionary.
        seeds: Random seeds used for task sampling and torch initialization.
        delays: Optional delay lengths to evaluate after each seed run.

    Returns:
        ``SeedSweepResult`` with summary output paths and per-seed metrics.

    Raises:
        ValueError: If no seeds are supplied.
    """
    if not seeds:
        raise ValueError("at least one seed is required")

    base_output_dir = Path(config["paths"]["output_dir"])
    base_run_name = config["paths"].get("run_name", "working_memory_model")
    task_type = str(config["task"].get("task_type", "categorical"))
    summary_dirs = ensure_run_dirs(base_output_dir)
    results: list[dict[str, Any]] = []

    for seed in [int(value) for value in seeds]:
        seed_config = deepcopy(config)
        seed_config["task"]["seed"] = seed
        seed_config["paths"]["output_dir"] = str(base_output_dir / "seed_sweep" / f"seed_{seed}")
        seed_config["paths"]["run_name"] = f"{base_run_name}_seed_{seed}"

        train_result = train_model(seed_config)
        eval_result = evaluate_model(seed_config, train_result.checkpoint_path)

        final_history = train_result.history[-1]
        row: dict[str, Any] = {
            "seed": seed,
            "task_type": task_type,
            "output_dir": seed_config["paths"]["output_dir"],
            "checkpoint": str(train_result.checkpoint_path),
            "train_final_loss": final_history["loss"],
            "train_final_accuracy": final_history.get("accuracy", ""),
            "train_final_population_mse": final_history.get("population_mse", ""),
            "eval_accuracy": eval_result.metrics.get("accuracy", ""),
            "eval_mean_angular_error_degrees": eval_result.metrics.get("mean_angular_error_degrees", ""),
            "eval_population_mse": eval_result.metrics.get("population_mse", ""),
            "eval_fixation_accuracy": eval_result.metrics.get("fixation_accuracy", ""),
            "decoder_summary": "",
            "decoder_mean_diagonal_error_degrees": "",
            "decoder_mean_delay_diagonal_error_degrees": "",
            "delay_sweep_metrics": "",
            "delay_sweep_csv": "",
            "delay_sweep_figure": "",
        }

        if task_type == "tuned":
            decoder_result = run_cross_temporal_decoder(seed_config, train_result.checkpoint_path)
            decoder_task = task_config_from_dict(seed_config)
            delay_start = int(getattr(decoder_task, "pre_cue_steps", 0)) + int(decoder_task.cue_steps)
            delay_end = delay_start + int(decoder_task.delay_steps)
            diagonal = np.diag(decoder_result.mean_error_degrees)
            row.update(
                {
                    "decoder_summary": str(decoder_result.summary_path),
                    "decoder_mean_diagonal_error_degrees": float(diagonal.mean()),
                    "decoder_mean_delay_diagonal_error_degrees": float(
                        diagonal[delay_start:delay_end].mean()
                    ),
                }
            )

        if delays:
            delay_result = run_delay_sweep(seed_config, train_result.checkpoint_path, delays)
            row.update(
                {
                    "delay_sweep_metrics": str(delay_result.metrics_path),
                    "delay_sweep_csv": str(delay_result.csv_path),
                    "delay_sweep_figure": str(delay_result.figure_path),
                }
            )

        results.append(row)

    csv_path = _write_seed_sweep_csv(summary_dirs["metrics"] / f"{base_run_name}_seed_sweep.csv", results)
    summary_path = write_json(
        summary_dirs["metrics"] / f"{base_run_name}_seed_sweep_summary.json",
        {
            "base_output_dir": str(base_output_dir),
            "base_run_name": base_run_name,
            "task_type": task_type,
            "trained_delay_steps": int(config["task"]["delay_steps"]),
            "chance_accuracy": (
                1.0 / int(config["task"]["n_classes"])
                if task_type == "categorical"
                else None
            ),
            "seeds": [int(value) for value in seeds],
            "delays": [int(value) for value in delays] if delays else [],
            "results": results,
        },
    )
    return SeedSweepResult(summary_path=summary_path, csv_path=csv_path, results=results)


def _write_seed_sweep_csv(path: str | Path, results: list[dict[str, Any]]) -> Path:
    """Write per-seed summary rows to CSV and return the output path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "task_type",
        "output_dir",
        "checkpoint",
        "train_final_loss",
        "train_final_accuracy",
        "train_final_population_mse",
        "eval_accuracy",
        "eval_mean_angular_error_degrees",
        "eval_population_mse",
        "eval_fixation_accuracy",
        "decoder_summary",
        "decoder_mean_diagonal_error_degrees",
        "decoder_mean_delay_diagonal_error_degrees",
        "delay_sweep_metrics",
        "delay_sweep_csv",
        "delay_sweep_figure",
    ]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    return target


def main() -> None:
    """Parse command-line arguments and run multi-seed training."""
    parser = argparse.ArgumentParser(description="Train and evaluate working-memory RNNs across seeds.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="Training/task seeds to run.")
    parser.add_argument("--delays", nargs="*", type=int, help="Optional delay lengths to sweep after each seed run.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Override training/evaluation device.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device

    result = run_seed_sweep(config, args.seeds, delays=args.delays)
    print(f"summary={result.summary_path}")
    print(f"csv={result.csv_path}")
    for row in result.results:
        if row["task_type"] == "tuned":
            print(
                f"seed={row['seed']} mean_angular_error_degrees="
                f"{row['eval_mean_angular_error_degrees']:.3f} "
                f"delay_decoder_error_degrees="
                f"{row['decoder_mean_delay_diagonal_error_degrees']:.3f}"
            )
        else:
            print(
                f"seed={row['seed']} train_accuracy={row['train_final_accuracy']:.3f} "
                f"eval_accuracy={row['eval_accuracy']:.3f}"
            )


if __name__ == "__main__":
    main()
