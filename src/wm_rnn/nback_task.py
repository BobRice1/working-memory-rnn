"""Deterministic categorical 0-back and 2-back task generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NBackTaskConfig:
    """Configuration for homogeneous-rule N-back sequence batches."""

    n_stimuli: int = 6
    n_back: int = 0
    sequence_items: int = 20
    stimulus_steps: int = 3
    interstimulus_steps: int = 6
    scored_start_item: int = 2
    target_identity: int = 0
    matches_per_sequence: int = 6
    min_one_back_lures: int = 3
    batch_size: int = 128
    seed: int | None = None

    @property
    def input_size(self) -> int:
        """Return stimulus plus two rule-context channels."""
        return self.n_stimuli + 2

    @property
    def output_size(self) -> int:
        """Return non-match/match readout size."""
        return 2

    @property
    def event_steps(self) -> int:
        """Return time steps occupied by one stimulus event."""
        return self.stimulus_steps + self.interstimulus_steps

    @property
    def seq_len(self) -> int:
        """Return total time steps in a sequence."""
        return self.sequence_items * self.event_steps


@dataclass(frozen=True)
class NBackBatch:
    """Generated N-back sequences and item-level audit metadata."""

    inputs: np.ndarray
    targets: np.ndarray
    loss_mask: np.ndarray
    stimuli: np.ndarray
    item_labels: np.ndarray
    item_scored: np.ndarray
    one_back_lures: np.ndarray
    event_onsets: np.ndarray
    event_steps: int
    n_back: int

    @property
    def condition(self) -> str:
        """Return the human-readable task rule."""
        return f"{self.n_back}-back"


def _validate_config(config: NBackTaskConfig) -> None:
    if config.n_back not in {0, 2}:
        raise ValueError("n_back must be 0 or 2")
    if config.n_stimuli < 3:
        raise ValueError("n_stimuli must be at least 3")
    if config.sequence_items <= config.scored_start_item:
        raise ValueError("sequence_items must exceed scored_start_item")
    if config.scored_start_item < 2:
        raise ValueError("scored_start_item must be at least 2")
    if min(
        config.stimulus_steps,
        config.interstimulus_steps,
        config.batch_size,
    ) <= 0:
        raise ValueError("timing values and batch_size must be positive")
    scored_items = config.sequence_items - config.scored_start_item
    if not 0 < config.matches_per_sequence < scored_items:
        raise ValueError(
            "matches_per_sequence must be between zero and scored item count"
        )
    if not 0 <= config.target_identity < config.n_stimuli:
        raise ValueError("target_identity is outside the stimulus set")
    nonmatches = scored_items - config.matches_per_sequence
    if not 0 <= config.min_one_back_lures <= nonmatches:
        raise ValueError("min_one_back_lures exceeds non-match count")


def _generate_zero_back_sequence(
    config: NBackTaskConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stimuli = rng.integers(
        0, config.n_stimuli, size=config.sequence_items, dtype=np.int64
    )
    scored_positions = np.arange(
        config.scored_start_item, config.sequence_items
    )
    match_positions = rng.choice(
        scored_positions,
        size=config.matches_per_sequence,
        replace=False,
    )
    is_match_position = np.zeros(config.sequence_items, dtype=bool)
    is_match_position[match_positions] = True
    non_target_identities = np.delete(
        np.arange(config.n_stimuli, dtype=np.int64),
        config.target_identity,
    )
    for item_idx in scored_positions:
        if is_match_position[item_idx]:
            stimuli[item_idx] = config.target_identity
        else:
            stimuli[item_idx] = rng.choice(non_target_identities)
    labels = (stimuli == config.target_identity).astype(np.int64)
    lures = np.zeros(config.sequence_items, dtype=bool)
    return stimuli, labels, lures


def _generate_two_back_sequence(
    config: NBackTaskConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scored_positions = np.arange(
        config.scored_start_item, config.sequence_items
    )
    for _ in range(10_000):
        match_positions = rng.choice(
            scored_positions,
            size=config.matches_per_sequence,
            replace=False,
        )
        is_match_position = np.zeros(config.sequence_items, dtype=bool)
        is_match_position[match_positions] = True
        stimuli = np.empty(config.sequence_items, dtype=np.int64)
        stimuli[:2] = rng.integers(
            0, config.n_stimuli, size=2, dtype=np.int64
        )
        for item_idx in scored_positions:
            two_back = stimuli[item_idx - 2]
            if is_match_position[item_idx]:
                stimuli[item_idx] = two_back
                continue
            previous = stimuli[item_idx - 1]
            if previous != two_back and rng.random() < 0.60:
                stimuli[item_idx] = previous
                continue
            choices = np.delete(
                np.arange(config.n_stimuli, dtype=np.int64), two_back
            )
            stimuli[item_idx] = rng.choice(choices)

        labels = np.zeros(config.sequence_items, dtype=np.int64)
        labels[2:] = (stimuli[2:] == stimuli[:-2]).astype(np.int64)
        lures = np.zeros(config.sequence_items, dtype=bool)
        lures[2:] = (
            (stimuli[2:] == stimuli[1:-1])
            & (stimuli[2:] != stimuli[:-2])
        )
        if int(lures[config.scored_start_item :].sum()) >= (
            config.min_one_back_lures
        ):
            return stimuli, labels, lures
    raise RuntimeError("could not generate a 2-back sequence with enough lures")


def generate_nback_batch(config: NBackTaskConfig) -> NBackBatch:
    """Generate exact-balance 0-back or 2-back sequence batches."""
    _validate_config(config)
    rng = np.random.default_rng(config.seed)
    stimuli = np.empty(
        (config.sequence_items, config.batch_size), dtype=np.int64
    )
    item_labels = np.empty_like(stimuli)
    one_back_lures = np.zeros_like(stimuli, dtype=bool)

    generator = (
        _generate_zero_back_sequence
        if config.n_back == 0
        else _generate_two_back_sequence
    )
    for batch_idx in range(config.batch_size):
        sequence, labels, lures = generator(config, rng)
        stimuli[:, batch_idx] = sequence
        item_labels[:, batch_idx] = labels
        one_back_lures[:, batch_idx] = lures

    item_scored = np.zeros_like(stimuli, dtype=bool)
    item_scored[config.scored_start_item :, :] = True
    event_onsets = np.arange(config.sequence_items) * config.event_steps
    inputs = np.zeros(
        (config.seq_len, config.batch_size, config.input_size),
        dtype=np.float32,
    )
    targets = np.zeros(
        (config.seq_len, config.batch_size), dtype=np.int64
    )
    loss_mask = np.zeros(
        (config.seq_len, config.batch_size), dtype=np.float32
    )

    context_channel = config.n_stimuli + (0 if config.n_back == 0 else 1)
    inputs[:, :, context_channel] = 1.0
    batch_indices = np.arange(config.batch_size)
    for item_idx, onset in enumerate(event_onsets):
        stimulus_stop = onset + config.stimulus_steps
        event_stop = onset + config.event_steps
        inputs[
            onset:stimulus_stop,
            batch_indices,
            stimuli[item_idx],
        ] = 1.0
        targets[onset:event_stop, :] = item_labels[item_idx]
        if item_idx >= config.scored_start_item:
            loss_mask[onset:event_stop, :] = 1.0

    return NBackBatch(
        inputs=inputs,
        targets=targets,
        loss_mask=loss_mask,
        stimuli=stimuli,
        item_labels=item_labels,
        item_scored=item_scored,
        one_back_lures=one_back_lures,
        event_onsets=event_onsets.astype(np.int64),
        event_steps=config.event_steps,
        n_back=config.n_back,
    )
