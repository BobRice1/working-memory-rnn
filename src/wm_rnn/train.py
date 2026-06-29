from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from wm_rnn.config import load_config
from wm_rnn.device import select_device
from wm_rnn.io import ensure_run_dirs, write_history_csv, write_json
from wm_rnn.task import generate_delay_batch
from wm_rnn.training_utils import (
    batch_to_tensors,
    fresh_model,
    masked_cross_entropy,
    response_accuracy,
    task_config_from_dict,
)


@dataclass(frozen=True)
class TrainResult:
    checkpoint_path: Path
    metrics_path: Path
    history_path: Path
    history: list[dict[str, Any]]


def train_model(config: dict[str, Any]) -> TrainResult:
    device_info = select_device(config["training"].get("device", "auto"))
    print(device_info.description)

    torch.manual_seed(int(config["task"].get("seed") or 0))
    dirs = ensure_run_dirs(config["paths"]["output_dir"])
    model = fresh_model(config, device_info.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))

    steps = int(config["training"]["steps"])
    log_every = int(config["training"].get("log_every", 50))
    history: list[dict[str, Any]] = []

    for step in range(1, steps + 1):
        task_config = task_config_from_dict(config, seed_offset=step)
        batch = generate_delay_batch(task_config)
        inputs, targets, loss_mask = batch_to_tensors(batch, device_info.device)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(inputs)
        loss = masked_cross_entropy(logits, targets, loss_mask)
        loss.backward()
        optimizer.step()

        accuracy = response_accuracy(logits.detach(), targets, loss_mask)
        row = {"step": step, "loss": float(loss.item()), "accuracy": accuracy}
        history.append(row)
        if step == 1 or step % log_every == 0 or step == steps:
            print(f"step={step} loss={row['loss']:.4f} accuracy={accuracy:.3f}")

    run_name = config["paths"].get("run_name", "baseline_delay")
    checkpoint_path = dirs["checkpoints"] / f"{run_name}.pt"
    history_path = dirs["metrics"] / f"{run_name}_train_history.csv"
    metrics_path = dirs["metrics"] / f"{run_name}_train_metrics.json"

    torch.save({"model_state": model.state_dict(), "config": config, "history": history}, checkpoint_path)
    write_history_csv(history_path, history)
    write_json(
        metrics_path,
        {
            "device": device_info.description,
            "steps": steps,
            "final_loss": history[-1]["loss"],
            "final_accuracy": history[-1]["accuracy"],
            "checkpoint": str(checkpoint_path),
        },
    )
    return TrainResult(checkpoint_path=checkpoint_path, metrics_path=metrics_path, history_path=history_path, history=history)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline delayed-response working-memory RNN.")
    parser.add_argument("--config", default="configs/baseline_delay.yaml", help="Path to YAML config.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Override training device.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device
    result = train_model(config)
    print(f"checkpoint={result.checkpoint_path}")
    print(f"metrics={result.metrics_path}")


if __name__ == "__main__":
    main()
