"""Delay-length sweep for trained working-memory RNN checkpoints."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from wm_rnn.config import load_config
from wm_rnn.device import select_device
from wm_rnn.io import ensure_run_dirs, write_json
from wm_rnn.training_utils import (
    batch_to_tensors,
    fresh_model,
    generate_batch_for_task,
    response_accuracy,
    task_config_from_dict,
    tuned_response_metrics,
)


@dataclass(frozen=True)
class DelaySweepResult:
    """Paths and metrics produced by delay-length evaluation.

    Attributes:
        metrics_path: JSON file containing sweep metadata and results.
        csv_path: CSV file containing one row per tested delay length.
        figure_path: PNG figure showing task performance across delay lengths.
        results: In-memory delay-length metric records.
    """

    metrics_path: Path
    csv_path: Path
    figure_path: Path
    results: list[dict[str, float | int]]


def run_delay_sweep(config: dict[str, Any], checkpoint_path: str | Path, delays: list[int]) -> DelaySweepResult:
    """Evaluate a frozen checkpoint across multiple delay lengths.

    Args:
        config: Experiment configuration dictionary.
        checkpoint_path: Path to a checkpoint produced by ``train_model``.
        delays: Delay lengths, in task time steps, to evaluate.

    Returns:
        ``DelaySweepResult`` with JSON/CSV output paths and per-delay metrics.

    Raises:
        ValueError: If no delays are supplied or any delay is non-positive.
    """
    if not delays:
        raise ValueError("at least one delay length is required")
    if any(int(delay) <= 0 for delay in delays):
        raise ValueError("delay lengths must be positive")

    device_info = select_device(config["training"].get("device", "auto"))
    dirs = ensure_run_dirs(config["paths"]["output_dir"])
    model = fresh_model(config, device_info.device)
    checkpoint = torch.load(checkpoint_path, map_location=device_info.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    task_type = str(config["task"].get("task_type", "categorical"))
    batches = int(config["evaluation"]["batches"])
    results: list[dict[str, float | int]] = []

    with torch.no_grad():
        for delay in [int(value) for value in delays]:
            sweep_config = deepcopy(config)
            sweep_config["task"]["delay_steps"] = delay
            accuracies: list[float] = []
            angular_errors: list[float] = []
            population_errors: list[float] = []
            for batch_idx in range(batches):
                task_config = task_config_from_dict(sweep_config, seed_offset=30000 + delay * 100 + batch_idx)
                batch = generate_batch_for_task(task_config)
                inputs, targets, loss_mask = batch_to_tensors(batch, device_info.device)
                logits, _ = model(inputs)
                if task_type == "tuned":
                    metrics = tuned_response_metrics(
                        logits,
                        targets,
                        loss_mask,
                        batch.preferred_angles,
                        batch.angles,
                    )
                    angular_errors.extend(metrics["angular_errors_degrees"])
                    population_errors.extend(metrics["population_squared_errors"])
                else:
                    accuracies.append(response_accuracy(logits, targets, loss_mask))

            if task_type == "tuned":
                angle_array = np.asarray(angular_errors, dtype=np.float64)
                mse_array = np.asarray(population_errors, dtype=np.float64)
                results.append(
                    {
                        "delay_steps": delay,
                        "mean_angular_error_degrees": float(angle_array.mean()),
                        "median_angular_error_degrees": float(np.median(angle_array)),
                        "p95_angular_error_degrees": float(np.percentile(angle_array, 95)),
                        "max_angular_error_degrees": float(angle_array.max()),
                        "population_mse": float(mse_array.mean()),
                        "batches": batches,
                    }
                )
            else:
                results.append({
                    "delay_steps": delay,
                    "accuracy": float(np.mean(accuracies)),
                    "batches": batches,
                })

    run_name = config["paths"].get("run_name", "working_memory_model")
    figure_path = _plot_delay_sweep(
        dirs["figures"] / f"{run_name}_delay_sweep.png",
        results,
        trained_delay_steps=int(config["task"]["delay_steps"]),
        task_type=task_type,
        chance_accuracy=1.0 / int(config["task"]["n_classes"]) if task_type == "categorical" else None,
    )
    metrics_path = write_json(
        dirs["metrics"] / f"{run_name}_delay_sweep_metrics.json",
        {
            "device": device_info.description,
            "checkpoint": str(checkpoint_path),
            "task_type": task_type,
            "trained_delay_steps": int(config["task"]["delay_steps"]),
            "figure": str(figure_path),
            "results": results,
        },
    )
    csv_path = _write_delay_sweep_csv(dirs["metrics"] / f"{run_name}_delay_sweep.csv", results)
    return DelaySweepResult(metrics_path=metrics_path, csv_path=csv_path, figure_path=figure_path, results=results)


def _write_delay_sweep_csv(path: str | Path, results: list[dict[str, float | int]]) -> Path:
    """Write delay-sweep result rows to CSV and return the output path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys()) if results else []
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    return target


def _plot_delay_sweep(
    path: str | Path,
    results: list[dict[str, float | int]],
    trained_delay_steps: int,
    task_type: str,
    chance_accuracy: float | None,
) -> Path:
    """Plot task performance across delay lengths and return the output path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sorted_results = sorted(results, key=lambda row: int(row["delay_steps"]))
    delays = [int(row["delay_steps"]) for row in sorted_results]

    fig, ax = plt.subplots(figsize=(7.3, 4.8))
    if task_type == "tuned":
        mean_errors = [float(row["mean_angular_error_degrees"]) for row in sorted_results]
        p95_errors = [float(row["p95_angular_error_degrees"]) for row in sorted_results]
        ax.plot(delays, mean_errors, marker="o", linewidth=2, color="#1f77b4", label="mean decoded-angle error")
        ax.plot(delays, p95_errors, marker="s", linewidth=1.5, color="#ff7f0e", label="95th-percentile error")
        ax.set_ylabel("Angular error (degrees)")
        ax.set_title("Frozen-model tuned delay-length sweep")
    else:
        accuracies = [float(row["accuracy"]) for row in sorted_results]
        ax.plot(delays, accuracies, marker="o", linewidth=2, color="#1f77b4", label="response accuracy")
        ax.axhline(chance_accuracy, linestyle="--", linewidth=1.25, color="#666666", label="chance accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Response accuracy")
        ax.set_title("Frozen-model delay-length sweep")
    ax.axvline(trained_delay_steps, linestyle=":", linewidth=1.5, color="#444444", label="configured reference delay")
    ax.set_xlabel("Delay length (time steps)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(target, dpi=160)
    plt.close(fig)
    return target


def main() -> None:
    """Parse command-line arguments and run a delay-length sweep."""
    parser = argparse.ArgumentParser(description="Evaluate a trained working-memory RNN across delay lengths.")
    parser.add_argument("--config", default="configs/categorical_working_memory.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint produced by training.")
    parser.add_argument("--delays", nargs="+", type=int, required=True, help="Delay lengths to evaluate.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Override evaluation device.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device
    result = run_delay_sweep(config, args.checkpoint, args.delays)
    print(f"metrics={result.metrics_path}")
    print(f"csv={result.csv_path}")
    print(f"figure={result.figure_path}")
    for row in result.results:
        if "accuracy" in row:
            print(f"delay_steps={row['delay_steps']} accuracy={row['accuracy']:.3f}")
        else:
            print(
                f"delay_steps={row['delay_steps']} "
                f"mean_angular_error_degrees={row['mean_angular_error_degrees']:.3f} "
                f"p95_angular_error_degrees={row['p95_angular_error_degrees']:.3f} "
                f"population_mse={row['population_mse']:.6f}"
            )


if __name__ == "__main__":
    main()
