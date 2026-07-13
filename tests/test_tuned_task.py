import math

import numpy as np

from wm_rnn.tuned_task import (
    TunedDelayTaskConfig,
    circular_angular_error,
    circular_preferred_angles,
    decode_population_angle,
    encode_circular_population,
    generate_tuned_delay_batch,
)


def test_circular_preferred_angles_are_evenly_spaced():
    angles = circular_preferred_angles(4)

    np.testing.assert_allclose(angles, [0.0, math.pi / 2, math.pi, 3 * math.pi / 2])


def test_circular_population_encoding_peaks_at_nearest_preferred_angle():
    preferred = circular_preferred_angles(8)
    encoded = encode_circular_population(np.array([math.pi / 2]), preferred, tuning_kappa=8.0)

    assert encoded.shape == (1, 8)
    assert int(encoded[0].argmax()) == 2
    assert encoded[0, 2] == 1.0


def test_circular_population_encoding_preserves_angle_grid_shape():
    preferred = circular_preferred_angles(8)
    angles = np.array(
        [
            [0.0, math.pi / 2],
            [math.pi, 3 * math.pi / 2],
        ]
    )

    encoded = encode_circular_population(angles, preferred, tuning_kappa=8.0)

    assert encoded.shape == (2, 2, 8)
    assert int(encoded[0, 1].argmax()) == 2
    assert encoded[0, 1, 2] == 1.0


def test_circular_population_encoding_wraps_around_zero_boundary():
    preferred = circular_preferred_angles(8)
    near_zero = encode_circular_population(np.array([2 * math.pi - 0.01]), preferred, tuning_kappa=8.0)

    assert int(near_zero[0].argmax()) == 0
    assert near_zero[0, 7] > near_zero[0, 4]


def test_generate_tuned_delay_batch_shapes_and_phases():
    config = TunedDelayTaskConfig(
        n_tuned_units=16,
        cue_steps=2,
        delay_steps=3,
        response_steps=4,
        batch_size=5,
        seed=123,
    )
    batch = generate_tuned_delay_batch(config)

    assert batch.inputs.shape == (9, 5, 17)
    assert batch.targets.shape == (9, 5, 16)
    assert batch.loss_mask.shape == (9, 5)
    assert batch.angles.shape == (5,)
    assert np.all(batch.angles >= 0.0)
    assert np.all(batch.angles < 2 * math.pi)
    assert batch.phase_index["cue"] == slice(0, 2)
    assert batch.phase_index["delay"] == slice(2, 5)
    assert batch.phase_index["response"] == slice(5, 9)
    encoded_targets = encode_circular_population(
        batch.angles, batch.preferred_angles, config.tuning_kappa
    )
    np.testing.assert_allclose(batch.inputs[:, :, -1], 1.0)
    np.testing.assert_allclose(batch.loss_mask[:5, :], 0.0)
    np.testing.assert_allclose(batch.loss_mask[5:, :], 1.0)
    np.testing.assert_allclose(
        batch.targets[batch.phase_index["response"], :, :],
        np.broadcast_to(
            encoded_targets[np.newaxis, :, :],
            (config.response_steps, config.batch_size, config.n_tuned_units),
        ),
    )
    np.testing.assert_allclose(
        batch.inputs[: config.cue_steps, :, :-1],
        np.broadcast_to(
            encoded_targets[np.newaxis, :, :],
            (config.cue_steps, config.batch_size, config.n_tuned_units),
        ),
    )
    assert np.all(batch.inputs[:2, :, :-1] > 0.0)
    np.testing.assert_allclose(batch.inputs[2:, :, :-1], 0.0)

    repeat_batch = generate_tuned_delay_batch(config)

    np.testing.assert_allclose(repeat_batch.angles, batch.angles)
    np.testing.assert_allclose(repeat_batch.inputs, batch.inputs)
    np.testing.assert_allclose(repeat_batch.targets, batch.targets)
    np.testing.assert_allclose(repeat_batch.loss_mask, batch.loss_mask)


def test_fixation_gated_tuned_batch_holds_fixation_until_response():
    config = TunedDelayTaskConfig(
        n_tuned_units=8,
        cue_steps=2,
        delay_steps=3,
        response_steps=2,
        batch_size=4,
        seed=123,
        fixation_gated=True,
    )
    batch = generate_tuned_delay_batch(config)

    assert config.input_size == 9
    assert config.output_size == 9
    np.testing.assert_allclose(batch.inputs[:5, :, -1], 1.0)
    np.testing.assert_allclose(batch.inputs[5:, :, -1], 0.0)
    np.testing.assert_allclose(batch.targets[:5, :, :8], 0.0)
    np.testing.assert_allclose(batch.targets[:5, :, -1], 1.0)
    np.testing.assert_allclose(batch.targets[5:, :, -1], 0.0)
    np.testing.assert_allclose(batch.loss_mask[:5, :], 0.0)
    np.testing.assert_allclose(batch.loss_mask[5:, :], 1.0)


def test_fixation_gated_batch_supports_pre_cue_fixation_period():
    config = TunedDelayTaskConfig(
        n_tuned_units=8,
        pre_cue_steps=3,
        cue_steps=2,
        delay_steps=4,
        response_steps=5,
        batch_size=2,
        seed=321,
        fixation_gated=True,
    )
    batch = generate_tuned_delay_batch(config)

    assert batch.inputs.shape == (14, 2, 9)
    assert batch.phase_index["fixation"] == slice(0, 3)
    assert batch.phase_index["cue"] == slice(3, 5)
    assert batch.phase_index["delay"] == slice(5, 9)
    assert batch.phase_index["response"] == slice(9, 14)
    np.testing.assert_allclose(batch.inputs[:9, :, -1], 1.0)
    np.testing.assert_allclose(batch.inputs[9:, :, -1], 0.0)
    np.testing.assert_allclose(batch.inputs[:3, :, :8], 0.0)
    assert np.all(batch.inputs[3:5, :, :8] > 0.0)


def test_decode_population_angle_recovers_known_preferred_angle():
    preferred = circular_preferred_angles(8)
    populations = np.zeros((2, 8), dtype=np.float32)
    populations[0, 0] = 1.0
    populations[1, 2] = 1.0

    decoded = decode_population_angle(populations, preferred)

    np.testing.assert_allclose(decoded, [0.0, math.pi / 2], atol=1e-6)


def test_circular_angular_error_uses_shortest_wrapped_distance():
    predicted = np.array([0.01])
    target = np.array([2 * math.pi - 0.01])

    error = circular_angular_error(predicted, target)

    np.testing.assert_allclose(error, [0.02], atol=1e-6)
