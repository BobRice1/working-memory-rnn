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
from wm_rnn.task import generate_delay_batch
from wm_rnn.training_utils import (
    batch_to_tensors,
    confusion_matrix,
    fresh_model,
    response_accuracy,
    task_config_from_dict,
)


@dataclass(frozen=True)
class EvalResult:
    metrics_path: Path
    confusion_path: Path
    metrics: dict[str, Any]


def evaluate_model(config: dict[str, Any], checkpoint_path: str | Path) -> EvalResult:
    device_info = select_device(config["training"].get("device", "auto"))
    dirs = ensure_run_dirs(config["paths"]["output_dir"])
    model = fresh_model(config, device_info.device)
    checkpoint = torch.load(checkpoint_path, map_location=device_info.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    n_classes = int(config["task"]["n_classes"])
    batches = int(config["evaluation"]["batches"])
    accuracies = []
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    with torch.no_grad():
        for batch_idx in range(batches):
            task_config = task_config_from_dict(config, seed_offset=10000 + batch_idx)
            batch = generate_delay_batch(task_config)
            inputs, targets, loss_mask = batch_to_tensors(batch, device_info.device)
            logits, _ = model(inputs)
            accuracies.append(response_accuracy(logits, targets, loss_mask))
            matrix += confusion_matrix(logits, targets, loss_mask, n_classes)

    run_name = config["paths"].get("run_name", "baseline_delay")
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
    print(f"accuracy={result.metrics['accuracy']:.3f}")


if __name__ == "__main__":
    main()
