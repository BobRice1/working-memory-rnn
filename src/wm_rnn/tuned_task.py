"""Tuned circular-location delayed-response task generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TunedDelayTaskConfig:
    """Configuration for a tuned continuous circular-location task batch."""

    n_tuned_units: int = 32
    tuning_kappa: float = 8.0
    pre_cue_steps: int = 0
    cue_steps: int = 5
    delay_steps: int = 20
    response_steps: int = 5
    batch_size: int = 64
    seed: int | None = None
    fixation_gated: bool = False

    @property
    def input_size(self) -> int:
        """Return tuned input channels plus the fixation/context channel."""
        return self.n_tuned_units + 1

    @property
    def output_size(self) -> int:
        """Return tuned output channels and, when enabled, a fixation channel."""
        return self.n_tuned_units + int(self.fixation_gated)

    @property
    def seq_len(self) -> int:
        """Return total trial length in time steps."""
        return self.pre_cue_steps + self.cue_steps + self.delay_steps + self.response_steps


@dataclass(frozen=True)
class TunedDelayBatch:
    """Generated tuned circular-location delayed-response trials."""

    inputs: np.ndarray
    targets: np.ndarray
    loss_mask: np.ndarray
    angles: np.ndarray
    preferred_angles: np.ndarray
    phase_index: dict[str, slice]


def circular_preferred_angles(n_tuned_units: int) -> np.ndarray:
    """Return evenly spaced preferred angles in ``[0, 2*pi)``."""
    if n_tuned_units < 3:
        raise ValueError("n_tuned_units must be at least 3")

    return np.linspace(0.0, 2.0 * np.pi, n_tuned_units, endpoint=False, dtype=np.float32)


def encode_circular_population(
    angles: np.ndarray,
    preferred_angles: np.ndarray,
    tuning_kappa: float,
) -> np.ndarray:
    """Encode angles as circular population bumps over preferred angles."""
    if tuning_kappa <= 0:
        raise ValueError("tuning_kappa must be positive")

    angle_values = np.asarray(angles, dtype=np.float32)[..., np.newaxis]
    preferred_values = np.asarray(preferred_angles, dtype=np.float32).reshape(1, -1)
    encoded = np.exp(tuning_kappa * (np.cos(angle_values - preferred_values) - 1.0))
    return encoded.astype(np.float32, copy=False)


def decode_population_angle(populations: np.ndarray, preferred_angles: np.ndarray) -> np.ndarray:
    """Decode circular population activity with vector averaging."""
    population_values = np.asarray(populations, dtype=np.float32)
    preferred_values = np.asarray(preferred_angles, dtype=np.float32)
    if population_values.shape[-1] < preferred_values.size:
        raise ValueError("population has fewer channels than preferred angles")
    population_values = population_values[..., : preferred_values.size]

    x = np.sum(population_values * np.cos(preferred_values), axis=-1)
    y = np.sum(population_values * np.sin(preferred_values), axis=-1)
    return np.mod(np.arctan2(y, x), 2.0 * np.pi).astype(np.float32, copy=False)


def circular_angular_error(predicted_angles: np.ndarray, target_angles: np.ndarray) -> np.ndarray:
    """Return the shortest absolute wrapped angular error in radians."""
    predicted = np.asarray(predicted_angles, dtype=np.float32)
    target = np.asarray(target_angles, dtype=np.float32)
    wrapped = (predicted - target + np.pi) % (2.0 * np.pi) - np.pi
    return np.abs(wrapped).astype(np.float32, copy=False)


def generate_tuned_delay_batch(config: TunedDelayTaskConfig) -> TunedDelayBatch:
    """Generate a tuned continuous circular-location delayed-response batch."""
    if min(config.cue_steps, config.delay_steps, config.response_steps, config.batch_size) <= 0:
        raise ValueError("cue_steps, delay_steps, response_steps, and batch_size must be positive")

    preferred_angles = circular_preferred_angles(config.n_tuned_units)
    rng = np.random.default_rng(config.seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=config.batch_size).astype(np.float32)
    encoded_targets = encode_circular_population(
        angles,
        preferred_angles,
        config.tuning_kappa,
    )

    inputs = np.zeros((config.seq_len, config.batch_size, config.input_size), dtype=np.float32)
    targets = np.broadcast_to(
        encoded_targets[np.newaxis, :, :],
        (config.seq_len, config.batch_size, config.n_tuned_units),
    ).copy()
    loss_mask = np.zeros((config.seq_len, config.batch_size), dtype=np.float32)

    fixation_slice = slice(0, config.pre_cue_steps)
    cue_slice = slice(config.pre_cue_steps, config.pre_cue_steps + config.cue_steps)
    delay_slice = slice(cue_slice.stop, cue_slice.stop + config.delay_steps)
    response_slice = slice(delay_slice.stop, config.seq_len)

    inputs[cue_slice, :, : config.n_tuned_units] = encoded_targets[np.newaxis, :, :]
    if config.fixation_gated:
        inputs[: response_slice.start, :, -1] = 1.0
        targets = np.concatenate(
            (
                np.zeros_like(targets),
                np.zeros((config.seq_len, config.batch_size, 1), dtype=np.float32),
            ),
            axis=-1,
        )
        targets[response_slice, :, : config.n_tuned_units] = encoded_targets[np.newaxis, :, :]
        targets[: response_slice.start, :, -1] = 1.0
    else:
        inputs[:, :, -1] = 1.0
    loss_mask[response_slice, :] = 1.0

    return TunedDelayBatch(
        inputs=inputs,
        targets=targets,
        loss_mask=loss_mask,
        angles=angles,
        preferred_angles=preferred_angles,
        phase_index={
            "fixation": fixation_slice,
            "cue": cue_slice,
            "delay": delay_slice,
            "response": response_slice,
        },
    )
