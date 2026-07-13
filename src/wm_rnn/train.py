"""Training entry point for the baseline delayed-response RNN."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wm_rnn.config import load_config
from wm_rnn.device import select_device
from wm_rnn.io import ensure_run_dirs, write_history_csv, write_json
from wm_rnn.training_utils import (
    batch_to_tensors,
    fresh_model,
    generate_batch_for_task,
    masked_cross_entropy,
    masked_mse,
    response_accuracy,
    task_config_from_dict,
    with_delay_steps,
    weighted_tuned_mse,
)


@dataclass(frozen=True)
class TrainResult:
    """Paths and in-memory metrics produced by a training run.

    Attributes:
        checkpoint_path: Saved PyTorch checkpoint path.
        metrics_path: JSON summary path for final training metrics.
        history_path: CSV path for per-step training history.
        history: Per-step loss and task-specific metric records.
    """

    checkpoint_path: Path
    metrics_path: Path
    history_path: Path
    history: list[dict[str, Any]]


def train_model(config: dict[str, Any]) -> TrainResult:
    """Train a baseline working-memory RNN from a nested config dictionary.

    Two behaviors are opt-in and disabled unless configured, so existing
    configs keep training exactly as before:

    - Randomized training delay length: if ``task.delay_steps_min`` and
      ``task.delay_steps_max`` are both set, each training batch samples its
      delay length uniformly from that inclusive range instead of always
      using the fixed ``task.delay_steps``. This is intended to discourage
      solutions that are only correct at one specific, fixed delay length.
    - Whole-delay loss scoring: if ``training.score_delay_period`` is
      ``true``, the training loss is computed over the delay period as well
      as the response period, instead of the response period only. This is
      intended to reward a hidden state that stays decodable throughout the
      delay, not only at the final response window.
    - Whole-trial loss scoring: if ``training.score_all_periods`` is ``true``,
      the loss is computed over every phase. Fixation-gated tuned tasks use
      this to train their silent pre-response circular readout and fixation output.

    Categorical runs log response-period accuracy using the response-only mask.
    Tuned runs log population MSE instead of categorical accuracy.

    Args:
        config: Experiment configuration with task, model, training, and path
            sections.

    Returns:
        ``TrainResult`` containing output paths and training history.
    """
    device_info = select_device(config["training"].get("device", "auto"))
    print(device_info.description)

    base_seed = int(config["task"].get("seed") or 0)
    torch.manual_seed(base_seed)
    dirs = ensure_run_dirs(config["paths"]["output_dir"])
    model = fresh_model(config, device_info.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))

    steps = int(config["training"]["steps"])
    log_every = int(config["training"].get("log_every", 50))
    task_type = str(config["task"].get("task_type", "categorical"))
    score_delay_period = bool(config["training"].get("score_delay_period", False))
    score_all_periods = bool(config["training"].get("score_all_periods", False))
    yang_weighted_loss = bool(config["training"].get("yang_weighted_loss", False))
    input_noise_std = float(config["training"].get("input_noise_std", 0.0))
    gradient_clip_value = float(config["training"].get("gradient_clip_value", 0.0))

    delay_min = config["task"].get("delay_steps_min")
    delay_max = config["task"].get("delay_steps_max")
    delay_choices = config["task"].get("delay_steps_choices")
    pre_cue_choices = config["task"].get("pre_cue_steps_choices")
    cue_choices = config["task"].get("cue_steps_choices")
    randomize_delay = (delay_min is not None and delay_max is not None) or delay_choices is not None
    delay_rng = np.random.default_rng(base_seed + 777777) if randomize_delay else None

    history: list[dict[str, Any]] = []

    for step in range(1, steps + 1):
        task_config = task_config_from_dict(config, seed_offset=step)
        if randomize_delay:
            sampled_delay = (
                int(delay_rng.choice(delay_choices))
                if delay_choices is not None
                else int(delay_rng.integers(int(delay_min), int(delay_max) + 1))
            )
            task_config = with_delay_steps(task_config, sampled_delay)
        if pre_cue_choices is not None or cue_choices is not None:
            task_config = replace(
                task_config,
                pre_cue_steps=(
                    int(delay_rng.choice(pre_cue_choices))
                    if pre_cue_choices is not None
                    else task_config.pre_cue_steps
                ),
                cue_steps=(
                    int(delay_rng.choice(cue_choices))
                    if cue_choices is not None
                    else task_config.cue_steps
                ),
            )
        batch = generate_batch_for_task(task_config)
        inputs, targets, loss_mask = batch_to_tensors(batch, device_info.device)

        if input_noise_std > 0:
            inputs = inputs + torch.randn_like(inputs) * input_noise_std

        if yang_weighted_loss:
            training_mask = torch.zeros_like(loss_mask)
            response_slice = batch.phase_index["response"]
            ignore_initial = int(config["training"].get("ignore_initial_steps", 5))
            transition_steps = int(config["training"].get("response_transition_steps", 5))
            response_weight = float(config["training"].get("response_weight", 5.0))
            training_mask[ignore_initial : response_slice.start, :] = 1.0
            training_mask[response_slice.start + transition_steps : response_slice.stop, :] = response_weight
        elif score_all_periods:
            training_mask = torch.ones_like(loss_mask)
        elif score_delay_period:
            full_mask = batch.loss_mask.copy()
            full_mask[batch.phase_index["delay"], :] = 1.0
            training_mask = torch.from_numpy(full_mask).float().to(device_info.device)
        else:
            training_mask = loss_mask

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(inputs)
        if task_type == "tuned":
            if yang_weighted_loss:
                loss = weighted_tuned_mse(
                    logits,
                    targets,
                    training_mask,
                    fixation_weight=float(config["training"].get("fixation_weight", 2.0)),
                )
            else:
                loss = masked_mse(logits, targets, training_mask)
        else:
            loss = masked_cross_entropy(logits, targets, training_mask)
        loss.backward()
        if gradient_clip_value > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), gradient_clip_value)
        optimizer.step()

        row = {
            "step": step,
            "loss": float(loss.item()),
            "delay_steps": task_config.delay_steps,
            "pre_cue_steps": getattr(task_config, "pre_cue_steps", 0),
            "cue_steps": task_config.cue_steps,
        }
        if task_type == "tuned":
            row["population_mse"] = row["loss"]
            metric_text = f"population_mse={row['population_mse']:.4f}"
        else:
            row["accuracy"] = response_accuracy(logits.detach(), targets, loss_mask)
            metric_text = f"accuracy={row['accuracy']:.3f}"
        history.append(row)
        if step == 1 or step % log_every == 0 or step == steps:
            print(f"step={step} loss={row['loss']:.4f} {metric_text} delay_steps={task_config.delay_steps}")

    run_name = config["paths"].get("run_name", "working_memory_model")
    checkpoint_path = dirs["checkpoints"] / f"{run_name}.pt"
    history_path = dirs["metrics"] / f"{run_name}_train_history.csv"
    metrics_path = dirs["metrics"] / f"{run_name}_train_metrics.json"

    torch.save({"model_state": model.state_dict(), "config": config, "history": history}, checkpoint_path)
    write_history_csv(history_path, history)
    metrics = {
        "device": device_info.description,
        "steps": steps,
        "final_loss": history[-1]["loss"],
        "score_delay_period": score_delay_period,
        "score_all_periods": score_all_periods,
        "yang_weighted_loss": yang_weighted_loss,
        "input_noise_std": input_noise_std,
        "gradient_clip_value": gradient_clip_value,
        "randomize_delay": randomize_delay,
        "delay_steps_range": (
            [int(delay_min), int(delay_max)] if delay_choices is None and randomize_delay else None
        ),
        "delay_steps_choices": [int(value) for value in delay_choices] if delay_choices is not None else None,
        "checkpoint": str(checkpoint_path),
    }
    if task_type == "tuned":
        metrics["final_population_mse"] = history[-1]["population_mse"]
    else:
        metrics["final_accuracy"] = history[-1]["accuracy"]
    write_json(metrics_path, metrics)
    return TrainResult(checkpoint_path=checkpoint_path, metrics_path=metrics_path, history_path=history_path, history=history)


def main() -> None:
    """Parse command-line arguments and run baseline training."""
    parser = argparse.ArgumentParser(description="Train the baseline delayed-response working-memory RNN.")
    parser.add_argument("--config", default="configs/categorical_working_memory.yaml", help="Path to YAML config.")
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
