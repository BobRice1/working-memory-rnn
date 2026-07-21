"""Sine/cosine ridge decoding of circular angle from hidden states.

Used when the model readout is not a valid memory code (e.g. Yang fixation-gated
delay, where the circular population is trained to stay silent).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.linear_model import Ridge

from wm_rnn.tuned_task import circular_angular_error


def fit_hidden_angle_decoder(
    hidden: np.ndarray | torch.Tensor,
    angles: np.ndarray,
    ridge_alpha: float = 1.0,
) -> np.ndarray:
    """Fit a ridge map from hidden activity to (cos theta, sin theta).

    Parameters
    ----------
    hidden:
        Array shaped ``[trials, units]`` or ``[time, trials, units]``.
        If 3-D, all time steps are pooled for fitting.
    angles:
        Target angles in radians, one per trial.
    ridge_alpha:
        Ridge regularization strength.

    Returns
    -------
    weights:
        Array shaped ``[units, 2]`` mapping hidden state to cos/sin.
    """
    hidden_np = _as_numpy(hidden)
    angles_np = np.asarray(angles, dtype=np.float64)
    if hidden_np.ndim == 3:
        n_time, n_trials, n_units = hidden_np.shape
        if angles_np.shape != (n_trials,):
            raise ValueError("angles must have shape [trials] matching hidden")
        x = hidden_np.reshape(n_time * n_trials, n_units)
        y = np.column_stack((np.cos(angles_np), np.sin(angles_np)))
        y = np.tile(y, (n_time, 1))
    elif hidden_np.ndim == 2:
        if angles_np.shape != (hidden_np.shape[0],):
            raise ValueError("angles must have shape [trials] matching hidden")
        x = hidden_np
        y = np.column_stack((np.cos(angles_np), np.sin(angles_np)))
    else:
        raise ValueError("hidden must be 2-D [trials, units] or 3-D [time, trials, units]")

    decoder = Ridge(alpha=float(ridge_alpha), fit_intercept=False)
    decoder.fit(x, y)
    return np.asarray(decoder.coef_.T, dtype=np.float64)


def decode_angles_from_hidden(
    hidden: np.ndarray | torch.Tensor,
    weights: np.ndarray,
) -> np.ndarray:
    """Decode circular angles (radians) from hidden states with fitted weights."""
    hidden_np = _as_numpy(hidden)
    weights_np = np.asarray(weights, dtype=np.float64)
    if hidden_np.shape[-1] != weights_np.shape[0]:
        raise ValueError("hidden unit count does not match decoder weights")
    vectors = hidden_np @ weights_np
    return np.mod(np.arctan2(vectors[..., 1], vectors[..., 0]), 2.0 * np.pi).astype(np.float32)


def angle_error_degrees(
    predicted_angles: np.ndarray,
    target_angles: np.ndarray,
) -> np.ndarray:
    """Absolute circular error in degrees."""
    return np.degrees(circular_angular_error(predicted_angles, target_angles)).astype(np.float32)


def resolve_angle_decode_method(config: dict[str, Any]) -> str:
    """Return ``hidden_ridge`` for fixation-gated tuned models, else ``population_readout``."""
    task = config.get("task", {})
    if bool(task.get("fixation_gated", False)) and str(task.get("task_type", "")) == "tuned":
        return "hidden_ridge"
    return "population_readout"


def _as_numpy(values: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(values, dtype=np.float64)
