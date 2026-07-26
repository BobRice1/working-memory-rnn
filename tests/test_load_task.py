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
from wm_rnn.training_utils import task_config_from_dict


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
    assert resolved.serial_item_cue_steps == 8
    assert resolved.item_gap_steps == 2
    assert resolved.min_item_separation == 0.4
