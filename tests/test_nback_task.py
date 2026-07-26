"""Tests for exact-balance 0-back and 2-back generation."""

from __future__ import annotations

import numpy as np
import pytest

from wm_rnn.nback_task import NBackTaskConfig, generate_nback_batch


@pytest.mark.parametrize("n_back", [0, 2])
def test_nback_batch_shapes_contexts_and_warmup(n_back: int) -> None:
    config = NBackTaskConfig(n_back=n_back, batch_size=8, seed=123)
    batch = generate_nback_batch(config)

    assert batch.inputs.shape == (180, 8, 8)
    assert batch.targets.shape == (180, 8)
    assert batch.loss_mask.shape == (180, 8)
    assert batch.stimuli.shape == (20, 8)
    assert not batch.item_scored[:2].any()
    assert batch.item_scored[2:].all()
    assert not batch.loss_mask[: 2 * config.event_steps].any()
    assert batch.loss_mask[2 * config.event_steps :].all()

    expected_context = config.n_stimuli + (0 if n_back == 0 else 1)
    other_context = config.n_stimuli + (1 if n_back == 0 else 0)
    assert np.all(batch.inputs[:, :, expected_context] == 1.0)
    assert np.all(batch.inputs[:, :, other_context] == 0.0)


@pytest.mark.parametrize("n_back", [0, 2])
def test_nback_generation_is_deterministic_and_exactly_balanced(
    n_back: int,
) -> None:
    config = NBackTaskConfig(n_back=n_back, batch_size=16, seed=91)
    first = generate_nback_batch(config)
    second = generate_nback_batch(config)

    assert np.array_equal(first.inputs, second.inputs)
    assert np.array_equal(first.stimuli, second.stimuli)
    assert np.all(first.item_labels[2:].sum(axis=0) == 6)
    assert np.all((first.item_labels[2:] == 0).sum(axis=0) == 12)


def test_zero_back_labels_use_only_fixed_target() -> None:
    config = NBackTaskConfig(n_back=0, batch_size=12, seed=17)
    batch = generate_nback_batch(config)

    expected = batch.stimuli == config.target_identity
    assert np.array_equal(batch.item_labels.astype(bool), expected)
    assert not batch.one_back_lures.any()


def test_two_back_labels_and_lures_match_actual_sequence() -> None:
    config = NBackTaskConfig(n_back=2, batch_size=32, seed=88)
    batch = generate_nback_batch(config)

    expected_labels = (
        batch.stimuli[2:] == batch.stimuli[:-2]
    ).astype(np.int64)
    assert np.array_equal(batch.item_labels[2:], expected_labels)
    expected_lures = (
        (batch.stimuli[2:] == batch.stimuli[1:-1])
        & (batch.stimuli[2:] != batch.stimuli[:-2])
    )
    assert np.array_equal(batch.one_back_lures[2:], expected_lures)
    assert np.all(batch.one_back_lures[2:].sum(axis=0) >= 3)
    assert not np.any(
        batch.item_labels.astype(bool) & batch.one_back_lures
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_back": 1},
        {"n_stimuli": 2},
        {"scored_start_item": 1},
        {"matches_per_sequence": 18},
        {"min_one_back_lures": 13},
    ],
)
def test_invalid_nback_configs_fail(overrides: dict[str, int]) -> None:
    values = {
        "n_back": 2,
        "n_stimuli": 6,
        "scored_start_item": 2,
        "matches_per_sequence": 6,
        "min_one_back_lures": 3,
    }
    values.update(overrides)
    with pytest.raises(ValueError):
        generate_nback_batch(NBackTaskConfig(**values))
