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
    normalize_population_output,
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
            pre_cue_steps=int(task.get("pre_cue_steps", 0)),
            cue_steps=int(task["cue_steps"]),
            delay_steps=int(task["delay_steps"]),
            response_steps=int(task["response_steps"]),
            batch_size=resolved_batch_size,
            seed=seed,
            fixation_gated=bool(task.get("fixation_gated", False)),
            distractor_steps=int(task.get("distractor_steps", 0)),
            distractor_onset_fraction=float(
                task.get("distractor_onset_fraction", 0.5)
            ),
            distractor_angle_mode=str(
                task.get("distractor_angle_mode", "random")
            ),
            distractor_offset=float(task.get("distractor_offset", np.pi / 2)),
            n_items=int(task.get("n_items", 1)),
            probe_gated=bool(task.get("probe_gated", False)),
            stimulus_role_channel=bool(
                task.get("stimulus_role_channel", False)
            ),
            serial_item_cue_steps=int(task.get("serial_item_cue_steps", 8)),
            item_gap_steps=int(task.get("item_gap_steps", 2)),
            min_item_separation=float(
                task.get("min_item_separation", np.pi / 6)
            ),
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
        recurrent_noise_std=float(model.get("recurrent_noise_std", 0.0)),
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


def weighted_tuned_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    time_weights: torch.Tensor,
    fixation_weight: float = 2.0,
) -> torch.Tensor:
    """Compute Yang-style time-weighted MSE with extra fixation emphasis."""
    squared_error = (predictions - targets).square()
    output_weights = torch.ones(predictions.size(-1), device=predictions.device)
    if predictions.size(-1) > 1:
        output_weights[-1] = fixation_weight
    weights = time_weights.unsqueeze(-1) * output_weights
    return (squared_error * weights).sum() / weights.sum().clamp_min(1.0)


def circular_distribution_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    response_mask: torch.Tensor,
    fixation_mask: torch.Tensor,
    *,
    n_tuned_units: int,
    fixation_weight: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, circular cross-entropy, and separate fixation MSE."""
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise ValueError(
            "predictions and targets must be shape-matched (time, batch, output)"
        )
    expected_mask_shape = predictions.shape[:2]
    if (
        response_mask.shape != expected_mask_shape
        or fixation_mask.shape != expected_mask_shape
    ):
        raise ValueError("loss masks must match the time and batch dimensions")
    if not 0 < n_tuned_units < predictions.size(-1):
        raise ValueError(
            "circular-distribution loss requires tuned and fixation outputs"
        )
    if fixation_weight < 0.0:
        raise ValueError("fixation_weight must be non-negative")
    if not (
        torch.isfinite(predictions).all()
        and torch.isfinite(targets).all()
        and torch.isfinite(response_mask).all()
        and torch.isfinite(fixation_mask).all()
    ):
        raise ValueError("loss inputs must be finite")

    target_activity = targets[..., :n_tuned_units]
    target_mass = target_activity.sum(dim=-1)
    scored_response = response_mask > 0.0
    if not torch.any(scored_response):
        raise ValueError("response_mask must score at least one sample")
    if torch.any(target_mass[scored_response] <= 0.0):
        raise ValueError(
            "every scored circular target must have positive mass"
        )
    target_probabilities = target_activity / target_mass.unsqueeze(-1).clamp_min(
        torch.finfo(target_activity.dtype).tiny
    )
    log_probabilities = F.log_softmax(
        predictions[..., :n_tuned_units], dim=-1
    )
    per_sample_cross_entropy = -torch.sum(
        target_probabilities * log_probabilities, dim=-1
    )
    circular_loss = (
        per_sample_cross_entropy * response_mask
    ).sum() / response_mask.sum().clamp_min(1.0)

    if not torch.any(fixation_mask > 0.0):
        raise ValueError("fixation_mask must score at least one sample")
    fixation_squared_error = (
        predictions[..., n_tuned_units] - targets[..., n_tuned_units]
    ).square()
    fixation_loss = (
        fixation_squared_error * fixation_mask
    ).sum() / fixation_mask.sum().clamp_min(1.0)
    total_loss = circular_loss + float(fixation_weight) * fixation_loss
    if not torch.isfinite(total_loss):
        raise ValueError("circular-distribution loss must be finite")
    return total_loss, circular_loss, fixation_loss


def population_normalization_from_config(config: dict[str, Any]) -> str:
    """Resolve tuned-output decoding without changing legacy configurations."""
    tuned_loss = str(config["training"].get("tuned_loss", "legacy"))
    return "softmax" if tuned_loss == "circular_distribution" else "none"


def tuned_response_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    preferred_angles: np.ndarray,
    target_angles: np.ndarray,
    population_normalization: str = "none",
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
            "response_cross_entropies": [],
            "population_resultant_lengths": [],
        }

    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    n_tuned_units = len(preferred_angles)
    population_predictions = pred_np[..., :n_tuned_units]
    population_targets = target_np[..., :n_tuned_units]
    normalized_predictions = normalize_population_output(
        population_predictions, normalization=population_normalization
    )
    decoded_angles = decode_population_angle(
        normalized_predictions, preferred_angles
    )
    target_angle_values = np.asarray(target_angles, dtype=np.float32).reshape(1, -1)
    repeated_targets = np.broadcast_to(target_angle_values, mask_np.shape)
    angular_errors = circular_angular_error(decoded_angles, repeated_targets)[mask_np]
    angular_error_degrees = np.degrees(angular_errors)
    if population_normalization == "softmax":
        target_mass = population_targets.sum(axis=-1, keepdims=True)
        normalized_targets = np.divide(
            population_targets,
            target_mass,
            out=np.zeros_like(population_targets),
            where=target_mass > 0.0,
        )
        comparison_predictions = normalized_predictions
        comparison_targets = normalized_targets
    else:
        comparison_predictions = population_predictions
        comparison_targets = population_targets
    population_squared_errors = (
        (comparison_predictions - comparison_targets) ** 2
    ).mean(axis=-1)[mask_np]
    clipped_probabilities = np.clip(
        normalize_population_output(
            population_predictions, normalization="softmax"
        ),
        np.finfo(np.float32).tiny,
        1.0,
    )
    target_mass = population_targets.sum(axis=-1, keepdims=True)
    target_probabilities = np.divide(
        population_targets,
        target_mass,
        out=np.zeros_like(population_targets),
        where=target_mass > 0.0,
    )
    response_cross_entropies = -np.sum(
        target_probabilities * np.log(clipped_probabilities), axis=-1
    )[mask_np]
    preferred_complex = np.exp(1j * np.asarray(preferred_angles))
    population_resultant_lengths = np.abs(
        np.sum(
            normalize_population_output(
                population_predictions, normalization="softmax"
            )
            * preferred_complex,
            axis=-1,
        )
    )[mask_np]
    metrics = {
        "mean_angular_error_degrees": float(np.nan_to_num(angular_error_degrees.mean(), nan=0.0)),
        "median_angular_error_degrees": float(np.nan_to_num(np.median(angular_error_degrees), nan=0.0)),
        "population_mse": float(np.nan_to_num(population_squared_errors.mean(), nan=0.0, posinf=0.0, neginf=0.0)),
        "angular_errors_degrees": [float(x) for x in angular_error_degrees],
        "population_squared_errors": [float(x) for x in population_squared_errors],
        "mean_response_cross_entropy": float(
            np.mean(response_cross_entropies)
        ),
        "response_cross_entropies": [
            float(x) for x in response_cross_entropies
        ],
        "mean_population_resultant_length": float(
            np.mean(population_resultant_lengths)
        ),
        "population_resultant_lengths": [
            float(x) for x in population_resultant_lengths
        ],
        "population_normalization": population_normalization,
    }
    if pred_np.shape[-1] > n_tuned_units:
        gate_predictions = pred_np[..., n_tuned_units]
        gate_targets = target_np[..., n_tuned_units]
        metrics["fixation_mse"] = float(np.mean((gate_predictions - gate_targets) ** 2))
        metrics["fixation_accuracy"] = float(
            np.mean((gate_predictions >= 0.5) == (gate_targets >= 0.5))
        )
    return metrics


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
