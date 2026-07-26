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
    distractor_steps: int = 0
    distractor_onset_fraction: float = 0.5
    distractor_angle_mode: str = "random"
    distractor_offset: float = np.pi / 2
    n_items: int = 1
    probe_gated: bool = False
    serial_item_cue_steps: int = 8
    item_gap_steps: int = 2
    min_item_separation: float = np.pi / 6

    @property
    def input_size(self) -> int:
        """Return tuned, fixation/context, and optional probe channels."""
        return self.n_tuned_units + 1 + int(self.probe_gated)

    @property
    def output_size(self) -> int:
        """Return tuned output channels and, when enabled, a fixation channel."""
        return self.n_tuned_units + int(self.fixation_gated)

    @property
    def seq_len(self) -> int:
        """Return total trial length in time steps."""
        cue_block_steps = (
            2 * self.serial_item_cue_steps + self.item_gap_steps
            if self.probe_gated
            else self.cue_steps
        )
        return (
            self.pre_cue_steps
            + cue_block_steps
            + self.delay_steps
            + self.response_steps
        )


@dataclass(frozen=True)
class TunedDelayBatch:
    """Generated tuned circular-location delayed-response trials."""

    inputs: np.ndarray
    targets: np.ndarray
    loss_mask: np.ndarray
    angles: np.ndarray
    preferred_angles: np.ndarray
    phase_index: dict[str, slice]
    distractor_angles: np.ndarray | None = None
    item_angles: np.ndarray | None = None
    item_present_mask: np.ndarray | None = None
    item_retention_steps: np.ndarray | None = None
    probed_retention_steps: np.ndarray | None = None
    probed_index: np.ndarray | None = None


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


def _balanced_binary_indices(batch_size: int, rng: np.random.Generator) -> np.ndarray:
    """Return shuffled zero/one indices with counts differing by at most one."""
    indices = np.arange(batch_size, dtype=np.int64) % 2
    rng.shuffle(indices)
    return indices


def _draw_separated_item_angles(
    batch_size: int,
    min_separation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw two circular angles per trial with a minimum wrapped separation."""
    first = rng.uniform(0.0, 2.0 * np.pi, size=batch_size)
    second = rng.uniform(0.0, 2.0 * np.pi, size=batch_size)
    separation = np.abs((second - first + np.pi) % (2.0 * np.pi) - np.pi)
    pending = separation < min_separation
    while np.any(pending):
        second[pending] = rng.uniform(0.0, 2.0 * np.pi, size=int(np.sum(pending)))
        separation = np.abs((second - first + np.pi) % (2.0 * np.pi) - np.pi)
        pending = separation < min_separation
    return np.stack((first, second), axis=1).astype(np.float32)


def generate_tuned_delay_batch(config: TunedDelayTaskConfig) -> TunedDelayBatch:
    """Generate a tuned continuous circular-location delayed-response batch."""
    if min(config.cue_steps, config.delay_steps, config.response_steps, config.batch_size) <= 0:
        raise ValueError("cue_steps, delay_steps, response_steps, and batch_size must be positive")
    if config.distractor_steps < 0:
        raise ValueError("distractor_steps must be non-negative")
    if not 0.0 <= config.distractor_onset_fraction <= 1.0:
        raise ValueError("distractor_onset_fraction must lie in [0, 1]")
    if config.distractor_angle_mode not in {"random", "fixed_offset"}:
        raise ValueError("distractor_angle_mode must be 'random' or 'fixed_offset'")
    if config.n_items not in {1, 2}:
        raise ValueError("n_items must be 1 or 2")
    if config.probe_gated and config.serial_item_cue_steps <= 0:
        raise ValueError("serial_item_cue_steps must be positive")
    if config.item_gap_steps < 0:
        raise ValueError("item_gap_steps must be non-negative")
    if not 0.0 < config.min_item_separation <= np.pi:
        raise ValueError("min_item_separation must lie in (0, pi]")
    if not config.probe_gated and config.n_items != 1:
        raise ValueError("n_items=2 requires probe_gated=True")

    preferred_angles = circular_preferred_angles(config.n_tuned_units)
    rng = np.random.default_rng(config.seed)
    item_angles: np.ndarray | None = None
    item_present_mask: np.ndarray | None = None
    item_retention_steps: np.ndarray | None = None
    probed_retention_steps: np.ndarray | None = None
    probed_index: np.ndarray | None = None

    if config.probe_gated:
        if config.n_items == 1:
            probed_index = _balanced_binary_indices(config.batch_size, rng)
            item_present_mask = np.zeros((config.batch_size, 2), dtype=bool)
            item_present_mask[np.arange(config.batch_size), probed_index] = True
            item_angles = np.full((config.batch_size, 2), np.nan, dtype=np.float32)
            occupied_angles = rng.uniform(
                0.0, 2.0 * np.pi, size=config.batch_size
            ).astype(np.float32)
            item_angles[np.arange(config.batch_size), probed_index] = occupied_angles
        else:
            item_present_mask = np.ones((config.batch_size, 2), dtype=bool)
            item_angles = _draw_separated_item_angles(
                config.batch_size, config.min_item_separation, rng
            )
            probed_index = _balanced_binary_indices(config.batch_size, rng)
        angles = item_angles[np.arange(config.batch_size), probed_index]
    else:
        angles = rng.uniform(0.0, 2.0 * np.pi, size=config.batch_size).astype(
            np.float32
        )

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
    if config.probe_gated:
        first_item_slice = slice(
            config.pre_cue_steps,
            config.pre_cue_steps + config.serial_item_cue_steps,
        )
        item_gap_slice = slice(
            first_item_slice.stop,
            first_item_slice.stop + config.item_gap_steps,
        )
        second_item_slice = slice(
            item_gap_slice.stop,
            item_gap_slice.stop + config.serial_item_cue_steps,
        )
        cue_slice = slice(first_item_slice.start, second_item_slice.stop)
    else:
        cue_slice = slice(config.pre_cue_steps, config.pre_cue_steps + config.cue_steps)
        first_item_slice = cue_slice
        item_gap_slice = slice(cue_slice.stop, cue_slice.stop)
        second_item_slice = slice(cue_slice.stop, cue_slice.stop)
    delay_slice = slice(cue_slice.stop, cue_slice.stop + config.delay_steps)
    response_slice = slice(delay_slice.stop, config.seq_len)

    if config.probe_gated:
        safe_item_angles = np.nan_to_num(item_angles, nan=0.0)
        encoded_items = encode_circular_population(
            safe_item_angles, preferred_angles, config.tuning_kappa
        )
        first_present = item_present_mask[:, 0].astype(np.float32)
        second_present = item_present_mask[:, 1].astype(np.float32)
        inputs[first_item_slice, :, : config.n_tuned_units] = (
            encoded_items[np.newaxis, :, 0, :]
            * first_present[np.newaxis, :, np.newaxis]
        )
        inputs[second_item_slice, :, : config.n_tuned_units] = (
            encoded_items[np.newaxis, :, 1, :]
            * second_present[np.newaxis, :, np.newaxis]
        )
        item_retention_steps = np.full(
            (config.batch_size, 2), np.nan, dtype=np.float32
        )
        item_retention_steps[:, 0] = float(response_slice.start - first_item_slice.stop)
        item_retention_steps[:, 1] = float(response_slice.start - second_item_slice.stop)
        item_retention_steps[~item_present_mask] = np.nan
        probed_retention_steps = item_retention_steps[
            np.arange(config.batch_size), probed_index
        ].copy()
    else:
        inputs[cue_slice, :, : config.n_tuned_units] = encoded_targets[
            np.newaxis, :, :
        ]

    distractor_duration = min(config.distractor_steps, config.delay_steps)
    distractor_relative_start = round(
        (config.delay_steps - distractor_duration)
        * config.distractor_onset_fraction
    )
    distractor_slice = slice(
        delay_slice.start + distractor_relative_start,
        delay_slice.start + distractor_relative_start + distractor_duration,
    )
    distractor_angles: np.ndarray | None = None
    if distractor_duration > 0:
        if config.distractor_angle_mode == "fixed_offset":
            distractor_angles = np.mod(
                angles + config.distractor_offset, 2.0 * np.pi
            ).astype(np.float32)
        else:
            distractor_angles = rng.uniform(
                0.0, 2.0 * np.pi, size=config.batch_size
            ).astype(np.float32)
        encoded_distractors = encode_circular_population(
            distractor_angles, preferred_angles, config.tuning_kappa
        )
        inputs[distractor_slice, :, : config.n_tuned_units] += (
            encoded_distractors[np.newaxis, :, :]
        )

    fixation_channel = config.n_tuned_units
    if config.fixation_gated:
        inputs[: response_slice.start, :, fixation_channel] = 1.0
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
        inputs[:, :, fixation_channel] = 1.0
    if config.probe_gated:
        probe_values = np.where(probed_index == 0, -1.0, 1.0).astype(np.float32)
        inputs[response_slice, :, -1] = probe_values[np.newaxis, :]
    loss_mask[response_slice, :] = 1.0

    phase_index = {
        "fixation": fixation_slice,
        "cue": cue_slice,
        "delay": delay_slice,
        "distractor": distractor_slice,
        "response": response_slice,
    }
    if config.probe_gated:
        phase_index.update(
            {
                "item1": first_item_slice,
                "item_gap": item_gap_slice,
                "item2": second_item_slice,
            }
        )

    return TunedDelayBatch(
        inputs=inputs,
        targets=targets,
        loss_mask=loss_mask,
        angles=angles,
        preferred_angles=preferred_angles,
        phase_index=phase_index,
        distractor_angles=distractor_angles,
        item_angles=item_angles,
        item_present_mask=item_present_mask,
        item_retention_steps=item_retention_steps,
        probed_retention_steps=probed_retention_steps,
        probed_index=probed_index,
    )
