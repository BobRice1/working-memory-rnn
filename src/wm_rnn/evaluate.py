"""Evaluation entry point for trained working-memory RNN checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wm_rnn.config import load_config
from wm_rnn.device import select_device
from wm_rnn.io import ensure_run_dirs, write_json
from wm_rnn.training_utils import (
    batch_to_tensors,
    confusion_matrix,
    fresh_model,
    generate_batch_for_task,
    response_accuracy,
    task_config_from_dict,
    tuned_response_metrics,
)


@dataclass(frozen=True)
class EvalResult:
    """Paths and metrics produced by checkpoint evaluation.

    Attributes:
        metrics_path: JSON file containing aggregate evaluation metrics.
        confusion_path: CSV file containing the class confusion matrix, or
            ``None`` for tuned continuous evaluations.
        metrics: In-memory copy of aggregate evaluation metrics.
    """

    metrics_path: Path
    confusion_path: Path | None
    metrics: dict[str, Any]


def aggregate_tuned_metrics(batch_metrics: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate tuned response metrics over all scored response samples."""
    angular_errors = np.array(
        [value for metrics in batch_metrics for value in metrics["angular_errors_degrees"]],
        dtype=np.float64,
    )
    population_errors = np.array(
        [value for metrics in batch_metrics for value in metrics["population_squared_errors"]],
        dtype=np.float64,
    )
    if angular_errors.size == 0 or population_errors.size == 0:
        return {
            "mean_angular_error_degrees": 0.0,
            "median_angular_error_degrees": 0.0,
            "population_mse": 0.0,
        }
    metrics = {
        "mean_angular_error_degrees": float(np.mean(angular_errors)),
        "median_angular_error_degrees": float(np.median(angular_errors)),
        "population_mse": float(np.mean(population_errors)),
    }
    if batch_metrics and "fixation_mse" in batch_metrics[0]:
        metrics["fixation_mse"] = float(np.mean([item["fixation_mse"] for item in batch_metrics]))
        metrics["fixation_accuracy"] = float(
            np.mean([item["fixation_accuracy"] for item in batch_metrics])
        )
    return metrics


def evaluate_model(config: dict[str, Any], checkpoint_path: str | Path) -> EvalResult:
    """Evaluate a saved checkpoint on freshly generated delay-task batches.

    Args:
        config: Experiment configuration dictionary.
        checkpoint_path: Path to a checkpoint produced by ``train_model``.

    Returns:
        ``EvalResult`` with aggregate accuracy and output paths.
    """
    device_info = select_device(config["training"].get("device", "auto"))
    dirs = ensure_run_dirs(config["paths"]["output_dir"])
    model = fresh_model(config, device_info.device)
    checkpoint = torch.load(checkpoint_path, map_location=device_info.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    task_type = str(config["task"].get("task_type", "categorical"))
    batches = int(config["evaluation"]["batches"])
    run_name = config["paths"].get("run_name", "baseline_delay")

    if task_type == "tuned":
        tuned_metrics = []
        with torch.no_grad():
            for batch_idx in range(batches):
                task_config = task_config_from_dict(config, seed_offset=10000 + batch_idx)
                batch = generate_batch_for_task(task_config)
                inputs, targets, loss_mask = batch_to_tensors(batch, device_info.device)
                logits, _ = model(inputs)
                batch_metrics = tuned_response_metrics(
                    logits,
                    targets,
                    loss_mask,
                    batch.preferred_angles,
                    batch.angles,
                )
                tuned_metrics.append(batch_metrics)

        metrics = {
            "device": device_info.description,
            **aggregate_tuned_metrics(tuned_metrics),
            "batches": batches,
            "checkpoint": str(checkpoint_path),
        }
        metrics_path = write_json(dirs["metrics"] / f"{run_name}_eval_metrics.json", metrics)
        return EvalResult(metrics_path=metrics_path, confusion_path=None, metrics=metrics)

    n_classes = int(config["task"]["n_classes"])
    accuracies = []
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    with torch.no_grad():
        for batch_idx in range(batches):
            task_config = task_config_from_dict(config, seed_offset=10000 + batch_idx)
            batch = generate_batch_for_task(task_config)
            inputs, targets, loss_mask = batch_to_tensors(batch, device_info.device)
            logits, _ = model(inputs)
            accuracies.append(response_accuracy(logits, targets, loss_mask))
            matrix += confusion_matrix(logits, targets, loss_mask, n_classes)

    metrics = {
        "device": device_info.description,
        "accuracy": float(np.mean(accuracies)),
        "batches": batches,
        "checkpoint": str(checkpoint_path),
    }
    metrics_path = write_json(dirs["metrics"] / f"{run_name}_eval_metrics.json", metrics)
    confusion_path = dirs["metrics"] / f"{run_name}_confusion_matrix.csv"
    np.savetxt(confusion_path, matrix, fmt="%d", delimiter=",")
    return EvalResult(metrics_path=metrics_path, confusion_path=confusion_path, metrics=metrics)


def main() -> None:
    """Parse command-line arguments and evaluate a trained checkpoint."""
    parser = argparse.ArgumentParser(description="Evaluate a trained working-memory RNN checkpoint.")
    parser.add_argument("--config", default="configs/baseline_delay.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint produced by training.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Override evaluation device.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device
    result = evaluate_model(config, args.checkpoint)
    print(f"metrics={result.metrics_path}")
    if "accuracy" in result.metrics:
        print(f"accuracy={result.metrics['accuracy']:.3f}")
    else:
        print(f"mean_angular_error_degrees={result.metrics['mean_angular_error_degrees']:.3f}")
        print(f"population_mse={result.metrics['population_mse']:.3f}")


if __name__ == "__main__":
    main()
