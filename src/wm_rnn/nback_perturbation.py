"""Outcome-free N-back adapters for the registered perturbation operators.

This module deliberately contains no checkpoint loading, strength calibration,
or experiment runner.  It only fixes the N-back channel boundary, exposes
sequence-level log-loss observations for later calibration, and defines the
registered candidate-versus-P5 load contrast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from wm_rnn.model import WorkingMemoryRNN
from wm_rnn.nback_metrics import per_sequence_cross_entropy
from wm_rnn.nback_task import NBackBatch
from wm_rnn.perturbation_operators import (
    OPERATORS,
    ForwardFn,
    neutral_parameters,
)


NBACK_STIMULUS_CHANNELS = 6
NBACK_CONTEXT_CHANNELS = 2
NBACK_INPUT_SIZE = NBACK_STIMULUS_CHANNELS + NBACK_CONTEXT_CHANNELS
NBACK_OUTPUT_SIZE = 2

# P3b is omitted because the registered N-back task has no distractor window.
NBACK_OPERATOR_NAMES = (
    "synaptic_drive_gain",
    "heterogeneous_drive_gain",
    "sensory_input_gain",
    "recurrent_gain",
    "gaussian_state_noise",
    "state_persistence",
    "time_constant",
)


@dataclass(frozen=True)
class SequenceLogLossCost:
    """Matched sequence-level summary used by later strength calibration."""

    n_sequences: int
    baseline_mean_log_loss: float
    perturbed_mean_log_loss: float
    additive_cost: float


@dataclass(frozen=True)
class NBackLoadContrast:
    """Candidate-versus-P5 condition-normalized load contrast."""

    candidate_zero_back_impairment: float
    candidate_two_back_impairment: float
    candidate_load_selectivity: float
    p5_zero_back_impairment: float
    p5_two_back_impairment: float
    p5_load_selectivity: float
    c2_nback: float


def _validate_nback_model(model: WorkingMemoryRNN) -> None:
    if not isinstance(model, WorkingMemoryRNN):
        raise TypeError("model must be a WorkingMemoryRNN")
    if (
        model.config.input_size != NBACK_INPUT_SIZE
        or model.config.output_size != NBACK_OUTPUT_SIZE
    ):
        raise ValueError(
            "N-back perturbations require an 8-input, 2-output model"
        )


def build_nback_operator(
    model: WorkingMemoryRNN,
    operator: str,
    **parameters: object,
) -> ForwardFn:
    """Build one existing operator with the N-back stimulus boundary fixed.

    The first six input channels are stimulus identities.  The final two are
    rule contexts and therefore remain outside P3a sensory gain.
    """
    _validate_nback_model(model)
    if operator not in NBACK_OPERATOR_NAMES:
        raise ValueError(
            f"operator must be one of {list(NBACK_OPERATOR_NAMES)}"
        )
    resolved = dict(parameters)
    if operator == "sensory_input_gain":
        supplied = resolved.pop("n_tuned_units", NBACK_STIMULUS_CHANNELS)
        if int(supplied) != NBACK_STIMULUS_CHANNELS:
            raise ValueError(
                "N-back sensory gain must target exactly six stimulus channels"
            )
        resolved["n_tuned_units"] = NBACK_STIMULUS_CHANNELS
    return OPERATORS[operator](model, **resolved)


def build_neutral_nback_operator(
    model: WorkingMemoryRNN,
    operator: str,
) -> ForwardFn:
    """Build the registered neutral setting for an N-back operator."""
    return build_nback_operator(
        model,
        operator,
        **neutral_parameters(operator),
    )


def sequence_log_loss_units(
    logits: torch.Tensor,
    batch: NBackBatch,
) -> np.ndarray:
    """Return one calibration observation per generated N-back sequence."""
    if (
        logits.ndim != 3
        or logits.shape[:2] != batch.targets.shape
        or logits.shape[-1] != NBACK_OUTPUT_SIZE
    ):
        raise ValueError(
            "logits must have shape [time, batch, 2] matching the batch"
        )
    targets = torch.as_tensor(
        batch.targets, dtype=torch.long, device=logits.device
    )
    loss_mask = torch.as_tensor(
        batch.loss_mask, dtype=logits.dtype, device=logits.device
    )
    units = per_sequence_cross_entropy(
        logits,
        targets,
        loss_mask,
    ).astype(np.float64, copy=False)
    if units.shape != (batch.inputs.shape[1],):
        raise RuntimeError("sequence log-loss units must have shape [batch]")
    return units


def summarize_sequence_log_loss_cost(
    baseline_units: np.ndarray,
    perturbed_units: np.ndarray,
) -> SequenceLogLossCost:
    """Summarize paired sequence log losses without collapsing to events."""
    baseline = np.asarray(baseline_units, dtype=np.float64)
    perturbed = np.asarray(perturbed_units, dtype=np.float64)
    if baseline.ndim != 1 or perturbed.ndim != 1:
        raise ValueError("sequence log-loss units must be one-dimensional")
    if baseline.size == 0 or baseline.shape != perturbed.shape:
        raise ValueError(
            "baseline and perturbed units must be non-empty and paired"
        )
    if not np.all(np.isfinite(baseline)) or np.any(baseline < 0.0):
        raise ValueError("baseline units must be finite and non-negative")
    if not np.all(np.isfinite(perturbed)) or np.any(perturbed < 0.0):
        raise ValueError("perturbed units must be finite and non-negative")
    baseline_mean = float(np.mean(baseline))
    perturbed_mean = float(np.mean(perturbed))
    return SequenceLogLossCost(
        n_sequences=int(baseline.size),
        baseline_mean_log_loss=baseline_mean,
        perturbed_mean_log_loss=perturbed_mean,
        additive_cost=perturbed_mean - baseline_mean,
    )


def condition_normalized_discriminability_impairment(
    baseline_discriminability: float,
    perturbed_discriminability: float,
) -> float:
    """Return ``(baseline - perturbed) / baseline`` for one condition."""
    baseline = float(baseline_discriminability)
    perturbed = float(perturbed_discriminability)
    if not np.isfinite(baseline) or baseline <= 0.0:
        raise ValueError(
            "baseline_discriminability must be finite and positive"
        )
    if not np.isfinite(perturbed):
        raise ValueError("perturbed_discriminability must be finite")
    return (baseline - perturbed) / baseline


def candidate_vs_p5_load_contrast(
    *,
    baseline_zero_back: float,
    baseline_two_back: float,
    candidate_zero_back: float,
    candidate_two_back: float,
    p5_zero_back: float,
    p5_two_back: float,
) -> NBackLoadContrast:
    """Compute registered C2_NBACK; positive means candidate selectivity.

    Each perturbation's load selectivity is its condition-normalized 2-back
    discriminability impairment minus its normalized 0-back impairment.
    C2_NBACK subtracts the matched Gaussian (P5) load selectivity.
    """
    candidate_zero = condition_normalized_discriminability_impairment(
        baseline_zero_back,
        candidate_zero_back,
    )
    candidate_two = condition_normalized_discriminability_impairment(
        baseline_two_back,
        candidate_two_back,
    )
    p5_zero = condition_normalized_discriminability_impairment(
        baseline_zero_back,
        p5_zero_back,
    )
    p5_two = condition_normalized_discriminability_impairment(
        baseline_two_back,
        p5_two_back,
    )
    candidate_load = candidate_two - candidate_zero
    p5_load = p5_two - p5_zero
    return NBackLoadContrast(
        candidate_zero_back_impairment=candidate_zero,
        candidate_two_back_impairment=candidate_two,
        candidate_load_selectivity=candidate_load,
        p5_zero_back_impairment=p5_zero,
        p5_two_back_impairment=p5_two,
        p5_load_selectivity=p5_load,
        c2_nback=candidate_load - p5_load,
    )
