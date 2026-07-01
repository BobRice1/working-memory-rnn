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
from wm_rnn.task import generate_delay_batch
from wm_rnn.training_utils import batch_to_tensors, fresh_model, response_accuracy, task_config_from_dict


@dataclass(frozen=True)
class DelaySweepResult:
    """Paths and metrics produced by delay-length evaluation.

    Attributes:
        metrics_path: JSON file containing sweep metadata and results.
        csv_path: CSV file containing one row per tested delay length.
        figure_path: PNG figure showing accuracy across delay lengths.
        results: In-memory delay-length accuracy records.
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

    batches = int(config["evaluation"]["batches"])
    results: list[dict[str, float | int]] = []

    with torch.no_grad():
        for delay in [int(value) for value in delays]:
            sweep_config = deepcopy(config)
            sweep_config["task"]["delay_steps"] = delay
            accuracies = []
            for batch_idx in range(batches):
                task_config = task_config_from_dict(sweep_config, seed_offset=30000 + delay * 100 + batch_idx)
                batch = generate_delay_batch(task_config)
                inputs, targets, loss_mask = batch_to_tensors(batch, device_info.device)
                logits, _ = model(inputs)
                accuracies.append(response_accuracy(logits, targets, loss_mask))

            results.append(
                {
                    "delay_steps": delay,
                    "accuracy": float(np.mean(accuracies)),
                    "batches": batches,
                }
            )

    run_name = config["paths"].get("run_name", "baseline_delay")
    figure_path = _plot_delay_sweep(
        dirs["figures"] / f"{run_name}_delay_sweep.png",
        results,
        trained_delay_steps=int(config["task"]["delay_steps"]),
        chance_accuracy=1.0 / int(config["task"]["n_classes"]),
    )
    metrics_path = write_json(
        dirs["metrics"] / f"{run_name}_delay_sweep_metrics.json",
        {
            "device": device_info.description,
            "checkpoint": str(checkpoint_path),
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
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["delay_steps", "accuracy", "batches"])
        writer.writeheader()
        writer.writerows(results)
    return target


def _plot_delay_sweep(
    path: str | Path,
    results: list[dict[str, float | int]],
    trained_delay_steps: int,
    chance_accuracy: float,
) -> Path:
    """Plot response accuracy across delay lengths and return the output path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sorted_results = sorted(results, key=lambda row: int(row["delay_steps"]))
    delays = [int(row["delay_steps"]) for row in sorted_results]
    accuracies = [float(row["accuracy"]) for row in sorted_results]

    plt.figure(figsize=(7, 4.5))
    plt.plot(delays, accuracies, marker="o", linewidth=2, color="#1f77b4")
    plt.axhline(chance_accuracy, linestyle="--", linewidth=1.25, color="#666666", label="chance")
    plt.axvline(trained_delay_steps, linestyle=":", linewidth=1.5, color="#444444", label="trained delay")
    plt.ylim(0.0, 1.05)
    plt.xlabel("Delay length (time steps)")
    plt.ylabel("Response accuracy")
    plt.title("Frozen-model delay-length sweep")
    plt.legend()
    plt.tight_layout()
    plt.savefig(target, dpi=160)
    plt.close()
    return target


def main() -> None:
    """Parse command-line arguments and run a delay-length sweep."""
    parser = argparse.ArgumentParser(description="Evaluate a trained working-memory RNN across delay lengths.")
    parser.add_argument("--config", default="configs/baseline_delay.yaml", help="Path to YAML config.")
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
        print(f"delay_steps={row['delay_steps']} accuracy={row['accuracy']:.3f}")


if __name__ == "__main__":
    main()
