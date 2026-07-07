"""Shared training and evaluation utilities for the working-memory RNN."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wm_rnn.model import RNNConfig, WorkingMemoryRNN
from wm_rnn.task import DelayBatch, DelayTaskConfig, generate_delay_batch
from wm_rnn.tuned_task import (
    TunedDelayBatch,
    TunedDelayTaskConfig,
    circular_angular_error,
    decode_population_angle,
    generate_tuned_delay_batch,
)


TaskConfig = DelayTaskConfig | TunedDelayTaskConfig
TaskBatch = DelayBatch | TunedDelayBatch


def generate_batch_for_task(task_config: TaskConfig) -> TaskBatch:
    """Generate the appropriate batch type for a task config."""
    if isinstance(task_config, TunedDelayTaskConfig):
        return generate_tuned_delay_batch(task_config)
    if isinstance(task_config, DelayTaskConfig):
        return generate_delay_batch(task_config)
    raise TypeError(f"unsupported task config type: {type(task_config).__name__}")


def task_config_from_dict(config: dict[str, Any], seed_offset: int = 0, batch_size: int | None = None) -> TaskConfig:
    """Build a typed task config from the nested experiment config.

    Args:
        config: Experiment configuration dictionary.
        seed_offset: Value added to the configured seed for deterministic
            independent batches.
        batch_size: Optional batch-size override for analysis or evaluation.

    Returns:
        Task config ready for batch generation.
    """
    task = config["task"]
    task_type = str(task.get("task_type", "categorical"))
    seed = task.get("seed")
    if seed is not None:
        seed = int(seed) + seed_offset
    resolved_batch_size = int(batch_size if batch_size is not None else task["batch_size"])

    if task_type == "categorical":
        return DelayTaskConfig(
            n_classes=int(task["n_classes"]),
            cue_steps=int(task["cue_steps"]),
            delay_steps=int(task["delay_steps"]),
            response_steps=int(task["response_steps"]),
            batch_size=resolved_batch_size,
            seed=seed,
        )
    if task_type == "tuned":
        return TunedDelayTaskConfig(
            n_tuned_units=int(task["n_tuned_units"]),
            tuning_kappa=float(task["tuning_kappa"]),
            cue_steps=int(task["cue_steps"]),
            delay_steps=int(task["delay_steps"]),
            response_steps=int(task["response_steps"]),
            batch_size=resolved_batch_size,
            seed=seed,
        )
    raise ValueError(f"unknown task_type: {task_type}")


def model_config_from_dict(config: dict[str, Any]) -> RNNConfig:
    """Build a typed model config from the nested experiment config."""
    task_config = task_config_from_dict(config)
    model = config["model"]
    return RNNConfig(
        input_size=task_config.input_size,
        hidden_size=int(model["hidden_size"]),
        output_size=task_config.output_size if isinstance(task_config, TunedDelayTaskConfig) else task_config.n_classes,
        dt=float(model["dt"]),
        tau=float(model["tau"]),
        activation=str(model.get("activation", "tanh")),
    )


def batch_to_tensors(batch: TaskBatch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move a generated NumPy batch onto a torch device."""
    inputs = torch.from_numpy(batch.inputs).float().to(device)
    target_tensor = torch.from_numpy(batch.targets)
    if np.issubdtype(batch.targets.dtype, np.integer):
        targets = target_tensor.long().to(device)
    else:
        targets = target_tensor.float().to(device)
    loss_mask = torch.from_numpy(batch.loss_mask).float().to(device)
    return inputs, targets, loss_mask


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy only on time points selected by ``loss_mask``."""
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
    mask = loss_mask.reshape(-1)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def masked_mse(predictions: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Compute population MSE only on time points selected by ``loss_mask``."""
    loss = F.mse_loss(predictions, targets, reduction="none").mean(dim=-1)
    return (loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)


def tuned_response_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    preferred_angles: np.ndarray,
    target_angles: np.ndarray,
) -> dict[str, Any]:
    """Return circular decoding and population-error metrics for tuned outputs."""
    mask_np = loss_mask.detach().cpu().numpy().astype(bool)
    if not mask_np.any():
        return {
            "mean_angular_error_degrees": 0.0,
            "median_angular_error_degrees": 0.0,
            "population_mse": 0.0,
            "angular_errors_degrees": [],
            "population_squared_errors": [],
        }

    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    decoded_angles = decode_population_angle(pred_np, preferred_angles)
    target_angle_values = np.asarray(target_angles, dtype=np.float32).reshape(1, -1)
    repeated_targets = np.broadcast_to(target_angle_values, mask_np.shape)
    angular_errors = circular_angular_error(decoded_angles, repeated_targets)[mask_np]
    angular_error_degrees = np.degrees(angular_errors)
    population_squared_errors = ((pred_np - target_np) ** 2).mean(axis=-1)[mask_np]
    return {
        "mean_angular_error_degrees": float(np.nan_to_num(angular_error_degrees.mean(), nan=0.0)),
        "median_angular_error_degrees": float(np.nan_to_num(np.median(angular_error_degrees), nan=0.0)),
        "population_mse": float(np.nan_to_num(population_squared_errors.mean(), nan=0.0, posinf=0.0, neginf=0.0)),
        "angular_errors_degrees": [float(x) for x in angular_error_degrees],
        "population_squared_errors": [float(x) for x in population_squared_errors],
    }


def response_accuracy(logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor) -> float:
    """Return classification accuracy over response-period time points."""
    predictions = logits.argmax(dim=-1)
    scored = loss_mask.bool()
    correct = (predictions[scored] == targets[scored]).float()
    if correct.numel() == 0:
        return 0.0
    return float(correct.mean().item())


def confusion_matrix(logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor, n_classes: int) -> np.ndarray:
    """Build a class-by-class confusion matrix over scored response steps."""
    predictions = logits.argmax(dim=-1).detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    mask_np = loss_mask.detach().cpu().numpy().astype(bool)
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_class, pred_class in zip(target_np[mask_np].ravel(), predictions[mask_np].ravel()):
        matrix[int(true_class), int(pred_class)] += 1
    return matrix


def fresh_model(config: dict[str, Any], device: torch.device) -> WorkingMemoryRNN:
    """Instantiate a new model from config and move it to ``device``."""
    model = WorkingMemoryRNN(model_config_from_dict(config))
    return model.to(device)


def with_batch_size(task_config: TaskConfig, batch_size: int) -> TaskConfig:
    """Return a copy of a task config with a different batch size."""
    return replace(task_config, batch_size=batch_size)


def with_delay_steps(task_config: TaskConfig, delay_steps: int) -> TaskConfig:
    """Return a copy of a task config with a different delay length."""
    return replace(task_config, delay_steps=delay_steps)
