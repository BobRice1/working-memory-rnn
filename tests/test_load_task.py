"""Tests for the matched-timing, probe-gated load manipulation."""

from __future__ import annotations

import numpy as np
import pytest

from wm_rnn.tuned_task import (
    TunedDelayTaskConfig,
    circular_angular_error,
    decode_population_angle,
    generate_tuned_delay_batch,
)
from wm_rnn.training_utils import model_config_from_dict, task_config_from_dict


def _probe_config(n_items: int, *, delay_steps: int = 20) -> TunedDelayTaskConfig:
    return TunedDelayTaskConfig(
        n_tuned_units=32,
        pre_cue_steps=2,
        cue_steps=20,
        delay_steps=delay_steps,
        response_steps=5,
        batch_size=64,
        seed=20260726 + n_items + delay_steps,
        fixation_gated=True,
        n_items=n_items,
        probe_gated=True,
        serial_item_cue_steps=8,
        item_gap_steps=2,
        min_item_separation=np.pi / 6,
    )


def test_probe_flag_controls_input_size_without_changing_output_size() -> None:
    ordinary = TunedDelayTaskConfig(n_tuned_units=32, probe_gated=False)
    probed = TunedDelayTaskConfig(
        n_tuned_units=32, probe_gated=True, fixation_gated=True
    )

    assert ordinary.input_size == 33
    assert probed.input_size == 34
    assert probed.output_size == 33


def test_role_flag_adds_exactly_one_distinct_input_channel() -> None:
    legacy = _probe_config(2)
    role_enabled = TunedDelayTaskConfig(
        **{
            **legacy.__dict__,
            "stimulus_role_channel": True,
        }
    )

    assert role_enabled.input_size == legacy.input_size + 1
    assert role_enabled.fixation_input_index == 32
    assert role_enabled.probe_input_index == 33
    assert role_enabled.stimulus_role_input_index == 34
    assert role_enabled.output_size == legacy.output_size


@pytest.mark.parametrize("n_items", [1, 2])
def test_role_channel_marks_present_items_and_distractors_only(
    n_items: int,
) -> None:
    base = _probe_config(n_items)
    config = TunedDelayTaskConfig(
        **{
            **base.__dict__,
            "stimulus_role_channel": True,
            "distractor_steps": 5,
        }
    )
    batch = generate_tuned_delay_batch(config)
    role = batch.inputs[:, :, config.stimulus_role_input_index]
    item1 = batch.phase_index["item1"]
    item2 = batch.phase_index["item2"]
    distractor = batch.phase_index["distractor"]
    response = batch.phase_index["response"]

    np.testing.assert_array_equal(
        role[item1],
        np.broadcast_to(
            batch.item_present_mask[:, 0].astype(np.float32),
            role[item1].shape,
        ),
    )
    np.testing.assert_array_equal(
        role[item2],
        np.broadcast_to(
            batch.item_present_mask[:, 1].astype(np.float32),
            role[item2].shape,
        ),
    )
    np.testing.assert_array_equal(role[distractor], -1.0)

    expected = np.zeros_like(role)
    expected[item1] = batch.item_present_mask[:, 0].astype(np.float32)
    expected[item2] = batch.item_present_mask[:, 1].astype(np.float32)
    expected[distractor] = -1.0
    np.testing.assert_array_equal(role, expected)

    probe = batch.inputs[:, :, config.probe_input_index]
    np.testing.assert_array_equal(probe[: response.start], 0.0)
    np.testing.assert_array_equal(
        probe[response],
        np.broadcast_to(
            np.where(batch.probed_index == 0, -1.0, 1.0),
            probe[response].shape,
        ),
    )
    np.testing.assert_array_equal(role[response], 0.0)


def test_load_levels_share_two_slot_timeline_and_sequence_length() -> None:
    load1 = generate_tuned_delay_batch(_probe_config(1))
    load2 = generate_tuned_delay_batch(_probe_config(2))

    assert load1.inputs.shape[0] == load2.inputs.shape[0]
    assert load1.phase_index["cue"] == load2.phase_index["cue"]
    assert load1.phase_index["item1"] == load2.phase_index["item1"]
    assert load1.phase_index["item_gap"] == load2.phase_index["item_gap"]
    assert load1.phase_index["item2"] == load2.phase_index["item2"]
    assert load1.phase_index["cue"].stop - load1.phase_index["cue"].start == 18
    assert load1.phase_index["item1"].stop - load1.phase_index["item1"].start == 8
    assert load1.phase_index["item_gap"].stop - load1.phase_index["item_gap"].start == 2
    assert load1.phase_index["item2"].stop - load1.phase_index["item2"].start == 8


def test_two_item_cues_are_sequential_and_probe_selects_target() -> None:
    batch = generate_tuned_delay_batch(_probe_config(2))
    tuned_drive = np.sum(batch.inputs[:, :, :32], axis=-1)
    item1 = batch.phase_index["item1"]
    gap = batch.phase_index["item_gap"]
    item2 = batch.phase_index["item2"]
    response = batch.phase_index["response"]

    assert np.all(tuned_drive[item1] > 0.0)
    assert np.all(tuned_drive[item2] > 0.0)
    np.testing.assert_array_equal(tuned_drive[gap], 0.0)
    assert item1.stop <= gap.start < gap.stop <= item2.start

    probe = batch.inputs[response.start, :, -1]
    np.testing.assert_array_equal(probe, np.where(batch.probed_index == 0, -1.0, 1.0))
    assert set(np.unique(probe)) == {-1.0, 1.0}
    np.testing.assert_array_equal(
        batch.angles,
        batch.item_angles[np.arange(64), batch.probed_index],
    )

    decoded_targets = decode_population_angle(
        batch.targets[response.start, :, :32], batch.preferred_angles
    )
    error = circular_angular_error(decoded_targets, batch.angles)
    np.testing.assert_allclose(error, 0.0, atol=1e-6)


def test_one_item_trials_have_one_randomized_occupied_slot() -> None:
    batch = generate_tuned_delay_batch(_probe_config(1))

    np.testing.assert_array_equal(batch.item_present_mask.sum(axis=1), 1)
    assert set(np.unique(batch.probed_index)) == {0, 1}
    np.testing.assert_array_equal(
        batch.probed_index, np.argmax(batch.item_present_mask, axis=1)
    )
    assert np.all(
        np.isnan(batch.item_angles[~batch.item_present_mask])
    )
    assert np.all(
        np.isfinite(batch.item_angles[batch.item_present_mask])
    )


@pytest.mark.parametrize("delay_steps", [10, 20, 40, 80])
@pytest.mark.parametrize("n_items", [1, 2])
def test_retention_and_shapes_are_correct_across_trained_delays(
    delay_steps: int, n_items: int
) -> None:
    config = _probe_config(n_items, delay_steps=delay_steps)
    batch = generate_tuned_delay_batch(config)

    assert batch.inputs.shape == (2 + 18 + delay_steps + 5, 64, 34)
    assert batch.targets.shape == (2 + 18 + delay_steps + 5, 64, 33)
    assert batch.loss_mask.shape == (2 + 18 + delay_steps + 5, 64)
    assert batch.phase_index["delay"].stop - batch.phase_index["delay"].start == delay_steps
    expected = np.where(
        batch.probed_index == 0, delay_steps + 10, delay_steps
    )
    np.testing.assert_array_equal(batch.probed_retention_steps, expected)
    for position, retention in enumerate((delay_steps + 10, delay_steps)):
        present = batch.item_present_mask[:, position]
        np.testing.assert_array_equal(
            batch.item_retention_steps[present, position], retention
        )
        assert np.all(np.isnan(batch.item_retention_steps[~present, position]))


def test_two_item_angles_respect_minimum_separation() -> None:
    config = _probe_config(2)
    batch = generate_tuned_delay_batch(config)
    separation = circular_angular_error(
        batch.item_angles[:, 0], batch.item_angles[:, 1]
    )

    assert np.all(separation >= config.min_item_separation - 1e-6)


def test_training_config_passes_new_task_options() -> None:
    config = {
        "task": {
            "task_type": "tuned",
            "n_tuned_units": 32,
            "tuning_kappa": 8.0,
            "cue_steps": 20,
            "delay_steps": 40,
            "response_steps": 25,
            "batch_size": 64,
            "seed": 7,
            "fixation_gated": True,
            "distractor_steps": 5,
            "distractor_onset_fraction": 0.25,
            "distractor_angle_mode": "fixed_offset",
            "distractor_offset": 1.0,
            "n_items": 2,
            "probe_gated": True,
            "stimulus_role_channel": True,
            "serial_item_cue_steps": 8,
            "item_gap_steps": 2,
            "min_item_separation": 0.4,
        }
    }

    resolved = task_config_from_dict(config)

    assert isinstance(resolved, TunedDelayTaskConfig)
    assert resolved.distractor_steps == 5
    assert resolved.distractor_onset_fraction == 0.25
    assert resolved.distractor_angle_mode == "fixed_offset"
    assert resolved.distractor_offset == 1.0
    assert resolved.n_items == 2
    assert resolved.probe_gated is True
    assert resolved.stimulus_role_channel is True
    assert resolved.serial_item_cue_steps == 8
    assert resolved.item_gap_steps == 2
    assert resolved.min_item_separation == 0.4


def test_model_input_size_follows_role_enabled_task_config() -> None:
    config = {
        "task": {
            "task_type": "tuned",
            "n_tuned_units": 32,
            "tuning_kappa": 8.0,
            "cue_steps": 20,
            "delay_steps": 20,
            "response_steps": 25,
            "batch_size": 64,
            "fixation_gated": True,
            "probe_gated": True,
            "stimulus_role_channel": True,
        },
        "model": {
            "hidden_size": 64,
            "dt": 20.0,
            "tau": 100.0,
        },
    }

    assert model_config_from_dict(config).input_size == 35
