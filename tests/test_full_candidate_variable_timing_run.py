"""Tests for the 10-seed variable-timing circular perturbation rerun."""

from __future__ import annotations

import json

import numpy as np

from wm_rnn.config import load_config
from wm_rnn.full_candidate_perturbation_run import (
    trained_distractor_checkpoints,
)
from wm_rnn.full_candidate_variable_timing_run import (
    BASE_CONFIG_SHA256,
    CONFIG,
)
from wm_rnn.perturbation_experiment import (
    _relocate_distractors,
    balanced_random_distractor_starts,
)
from wm_rnn.training_utils import task_config_from_dict
from wm_rnn.tuned_task import generate_tuned_delay_batch


def test_variable_timing_sweep_freezes_only_family_change() -> None:
    config = load_config(CONFIG)
    assert config["pilot"]["circular_seeds"] == [
        20260801,
        20260802,
        20260803,
        20260804,
        20260805,
        20260806,
        20260808,
        20260809,
        20260810,
        20260811,
    ]
    assert config["pilot"]["circular_trials_per_cell"] == 1024
    assert config["pilot"]["circular_delays"] == [10, 20, 40, 80]
    assert config["pilot"]["distractor_timing"] == (
        "per_trial_stratified_uniform_all_valid_starts"
    )
    assert config["pilot"]["distractor_valid_relative_starts"] == list(
        range(16)
    )
    assert config["interpretation"][
        "distractor_outcome_updated_for_timing_robustness"
    ]
    assert len(BASE_CONFIG_SHA256) == 64


def test_checkpoint_loader_accepts_explicit_manifest(tmp_path) -> None:
    manifest = {
        "target_competent_checkpoints": 1,
        "retained_checkpoint_seeds": [101],
        "results": [
            {
                "seed": 101,
                "checkpoint": "checkpoint.pt",
                "checkpoint_sha256": "A" * 64,
                "competence_passed": True,
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    checkpoints = trained_distractor_checkpoints(
        tmp_path, manifest_path=path
    )
    assert checkpoints[0].seed == 101


def test_random_onset_bank_balances_every_valid_start() -> None:
    first = balanced_random_distractor_starts(20, 5, 1024, 91)
    second = balanced_random_distractor_starts(20, 5, 1024, 91)
    counts = np.bincount(first, minlength=16)
    assert np.array_equal(first, second)
    assert counts.tolist() == [64] * 16
    assert not np.array_equal(first, np.sort(first))


def test_relocated_distractors_use_each_trials_frozen_start() -> None:
    config = load_config(
        "configs/fixation_circular_variable_distractor_working_memory.yaml"
    )
    config["task"]["batch_size"] = 16
    config["task"]["delay_steps"] = 20
    config["task"]["distractor_steps"] = 5
    task = task_config_from_dict(config)
    original = generate_tuned_delay_batch(task)
    starts = balanced_random_distractor_starts(20, 5, 16, 7)
    relocated = _relocate_distractors(original, task, starts)
    delay = relocated.phase_index["delay"]

    assert np.array_equal(original.angles, relocated.angles)
    assert np.array_equal(
        original.distractor_angles, relocated.distractor_angles
    )
    for trial, relative_start in enumerate(starts):
        active = np.flatnonzero(
            np.any(
                relocated.inputs[
                    delay, trial, : task.n_tuned_units
                ] > 0.0,
                axis=1,
            )
        )
        assert active.tolist() == list(
            range(int(relative_start), int(relative_start) + 5)
        )
