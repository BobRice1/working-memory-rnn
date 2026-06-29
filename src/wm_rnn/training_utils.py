"""Shared training and evaluation utilities for the working-memory RNN."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wm_rnn.model import RNNConfig, WorkingMemoryRNN
from wm_rnn.task import DelayBatch, DelayTaskConfig


def task_config_from_dict(config: dict[str, Any], seed_offset: int = 0, batch_size: int | None = None) -> DelayTaskConfig:
    """Build a typed task config from the nested experiment config.

    Args:
        config: Experiment configuration dictionary.
        seed_offset: Value added to the configured seed for deterministic
            independent batches.
        batch_size: Optional batch-size override for analysis or evaluation.

    Returns:
        ``DelayTaskConfig`` ready for batch generation.
    """
    task = config["task"]
    seed = task.get("seed")
    if seed is not None:
        seed = int(seed) + seed_offset
    task_config = DelayTaskConfig(
        n_classes=int(task["n_classes"]),
        cue_steps=int(task["cue_steps"]),
        delay_steps=int(task["delay_steps"]),
        response_steps=int(task["response_steps"]),
        batch_size=int(batch_size if batch_size is not None else task["batch_size"]),
        seed=seed,
    )
    return task_config


def model_config_from_dict(config: dict[str, Any]) -> RNNConfig:
    """Build a typed model config from the nested experiment config."""
    task_config = task_config_from_dict(config)
    model = config["model"]
    return RNNConfig(
        input_size=task_config.input_size,
        hidden_size=int(model["hidden_size"]),
        output_size=task_config.n_classes,
        dt=float(model["dt"]),
        tau=float(model["tau"]),
    )


def batch_to_tensors(batch: DelayBatch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move a generated NumPy batch onto a torch device."""
    inputs = torch.from_numpy(batch.inputs).float().to(device)
    targets = torch.from_numpy(batch.targets).long().to(device)
    loss_mask = torch.from_numpy(batch.loss_mask).float().to(device)
    return inputs, targets, loss_mask


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy only on time points selected by ``loss_mask``."""
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
    mask = loss_mask.reshape(-1)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


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


def with_batch_size(task_config: DelayTaskConfig, batch_size: int) -> DelayTaskConfig:
    """Return a copy of a task config with a different batch size."""
    return replace(task_config, batch_size=batch_size)
