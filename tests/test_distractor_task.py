"""Tests for the additive delay-period distractor extension."""

from __future__ import annotations

import numpy as np

from wm_rnn.tuned_task import (
    TunedDelayTaskConfig,
    circular_preferred_angles,
    encode_circular_population,
    generate_tuned_delay_batch,
)


def _legacy_default_arrays(config: TunedDelayTaskConfig) -> tuple[np.ndarray, ...]:
    """Reproduce the pre-Phase-2 generator for the byte-regression guard."""
    preferred = circular_preferred_angles(config.n_tuned_units)
    rng = np.random.default_rng(config.seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=config.batch_size).astype(
        np.float32
    )
    encoded = encode_circular_population(
        angles, preferred, config.tuning_kappa
    )
    inputs = np.zeros(
        (config.seq_len, config.batch_size, config.n_tuned_units + 1),
        dtype=np.float32,
    )
    targets = np.broadcast_to(
        encoded[np.newaxis, :, :],
        (config.seq_len, config.batch_size, config.n_tuned_units),
    ).copy()
    loss_mask = np.zeros(
        (config.seq_len, config.batch_size), dtype=np.float32
    )
    cue = slice(config.pre_cue_steps, config.pre_cue_steps + config.cue_steps)
    delay = slice(cue.stop, cue.stop + config.delay_steps)
    response = slice(delay.stop, config.seq_len)
    inputs[cue, :, : config.n_tuned_units] = encoded[np.newaxis, :, :]
    if config.fixation_gated:
        inputs[: response.start, :, -1] = 1.0
        targets = np.concatenate(
            (
                np.zeros_like(targets),
                np.zeros(
                    (config.seq_len, config.batch_size, 1), dtype=np.float32
                ),
            ),
            axis=-1,
        )
        targets[response, :, : config.n_tuned_units] = encoded[
            np.newaxis, :, :
        ]
        targets[: response.start, :, -1] = 1.0
    else:
        inputs[:, :, -1] = 1.0
    loss_mask[response, :] = 1.0
    return inputs, targets, loss_mask, angles


def test_new_option_defaults_preserve_legacy_batch_bytes() -> None:
    config = TunedDelayTaskConfig(
        n_tuned_units=8,
        pre_cue_steps=3,
        cue_steps=4,
        delay_steps=10,
        response_steps=5,
        batch_size=7,
        seed=1234,
        fixation_gated=True,
    )

    batch = generate_tuned_delay_batch(config)
    legacy_inputs, legacy_targets, legacy_mask, legacy_angles = (
        _legacy_default_arrays(config)
    )

    np.testing.assert_array_equal(batch.inputs, legacy_inputs)
    np.testing.assert_array_equal(batch.targets, legacy_targets)
    np.testing.assert_array_equal(batch.loss_mask, legacy_mask)
    np.testing.assert_array_equal(batch.angles, legacy_angles)
    assert batch.distractor_angles is None
    assert batch.item_angles is None


def test_distractor_is_input_only_and_window_stays_inside_delay() -> None:
    base = dict(
        n_tuned_units=8,
        cue_steps=3,
        delay_steps=10,
        response_steps=4,
        batch_size=6,
        seed=77,
        fixation_gated=True,
    )
    clean = generate_tuned_delay_batch(TunedDelayTaskConfig(**base))
    distracted = generate_tuned_delay_batch(
        TunedDelayTaskConfig(
            **base,
            distractor_steps=5,
            distractor_onset_fraction=0.5,
            distractor_angle_mode="fixed_offset",
            distractor_offset=np.pi / 2,
        )
    )

    distractor_slice = distracted.phase_index["distractor"]
    delay_slice = distracted.phase_index["delay"]
    assert delay_slice.start <= distractor_slice.start
    assert distractor_slice.stop <= delay_slice.stop
    assert distractor_slice.stop - distractor_slice.start == 5
    np.testing.assert_array_equal(distracted.targets, clean.targets)
    np.testing.assert_array_equal(distracted.loss_mask, clean.loss_mask)
    np.testing.assert_array_equal(
        distracted.inputs[:, :, 8], clean.inputs[:, :, 8]
    )

    encoded = encode_circular_population(
        distracted.distractor_angles,
        distracted.preferred_angles,
        8.0,
    )
    difference = distracted.inputs[:, :, :8] - clean.inputs[:, :, :8]
    np.testing.assert_allclose(
        difference[distractor_slice],
        np.broadcast_to(encoded, difference[distractor_slice].shape),
    )
    outside = np.ones(difference.shape[0], dtype=bool)
    outside[distractor_slice] = False
    np.testing.assert_array_equal(difference[outside], 0.0)


def test_distractor_duration_is_clipped_to_delay() -> None:
    batch = generate_tuned_delay_batch(
        TunedDelayTaskConfig(
            delay_steps=4,
            distractor_steps=20,
            distractor_onset_fraction=1.0,
            seed=3,
        )
    )

    assert batch.phase_index["distractor"] == batch.phase_index["delay"]

