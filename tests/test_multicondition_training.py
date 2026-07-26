"""Tests for balanced homogeneous Family B training batches."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from wm_rnn.config import load_config
from wm_rnn.family_b_evaluation import (
    CONDITIONS,
    evaluate_family_b_conditions,
)
from wm_rnn.train import (
    apply_trial_type,
    draw_balanced_trial_type_block,
    train_model,
)
from wm_rnn.tuned_task import TunedDelayTaskConfig
from wm_rnn.training_utils import fresh_model


EXPECTED_TYPES = {
    "load1_clean",
    "load1_distractor",
    "load2_clean",
    "load2_distractor",
}


def test_hidden128_rescue_config_changes_only_capacity_and_paths() -> None:
    role = load_config(
        "configs/multicondition_working_memory_distribution_role.yaml"
    )
    hidden128 = load_config(
        "configs/multicondition_working_memory_distribution_role_h128.yaml"
    )

    assert role["model"]["hidden_size"] == 64
    assert hidden128["model"]["hidden_size"] == 128
    assert hidden128["paths"]["output_dir"].endswith("_h128")
    assert hidden128["paths"]["run_name"].endswith("_h128")

    role["model"]["hidden_size"] = hidden128["model"]["hidden_size"]
    role["paths"] = hidden128["paths"]
    assert role == hidden128


def test_balanced_block_contains_every_trial_type_once() -> None:
    first_rng = np.random.default_rng(123)
    second_rng = np.random.default_rng(123)

    first = draw_balanced_trial_type_block(first_rng)
    repeat = draw_balanced_trial_type_block(second_rng)

    assert set(first) == EXPECTED_TYPES
    assert len(first) == 4
    assert first == repeat


def test_balanced_block_accepts_deterministic_curriculum_subset() -> None:
    first_rng = np.random.default_rng(321)
    second_rng = np.random.default_rng(321)
    subset = ("load1_clean", "load2_clean")

    assert draw_balanced_trial_type_block(first_rng, subset) == (
        draw_balanced_trial_type_block(second_rng, subset)
    )
    assert set(draw_balanced_trial_type_block(first_rng, subset)) == set(
        subset
    )


def test_trial_type_maps_to_homogeneous_task_config() -> None:
    base = TunedDelayTaskConfig(
        probe_gated=True,
        n_items=1,
        distractor_steps=5,
    )

    for trial_type in EXPECTED_TYPES:
        resolved = apply_trial_type(base, trial_type, distractor_steps=5)
        assert resolved.n_items == (2 if trial_type.startswith("load2") else 1)
        assert resolved.distractor_steps == (
            5 if trial_type.endswith("distractor") else 0
        )

    ordinary = replace(base, probe_gated=False)
    try:
        apply_trial_type(ordinary, "load1_clean", distractor_steps=5)
    except ValueError as error:
        assert "probe_gated" in str(error)
    else:
        raise AssertionError("ordinary task unexpectedly accepted Family B sampling")


def test_training_uses_balanced_four_batch_blocks(tmp_path) -> None:
    config = load_config("configs/multicondition_working_memory.yaml")
    config["task"]["n_tuned_units"] = 8
    config["task"]["batch_size"] = 4
    config["task"]["pre_cue_steps_choices"] = [2]
    config["task"]["delay_steps_choices"] = [4]
    config["task"]["serial_item_cue_steps"] = 2
    config["task"]["item_gap_steps"] = 1
    config["task"]["response_steps"] = 4
    config["model"]["hidden_size"] = 8
    config["model"]["recurrent_noise_std"] = 0.0
    config["training"]["steps"] = 8
    config["training"]["curriculum"] = [
        {"until_step": 2, "trial_types": ["load1_clean"]},
        {
            "until_step": 4,
            "trial_types": ["load1_clean", "load2_clean"],
        },
        {
            "until_step": 8,
            "learning_rate": 0.0001,
            "trial_type_counts": {
                "load1_clean": 1,
                "load1_distractor": 1,
                "load2_clean": 1,
                "load2_distractor": 1,
            },
        },
    ]
    config["training"]["log_every"] = 8
    config["training"]["device"] = "cpu"
    config["paths"]["output_dir"] = str(tmp_path / "family_b")
    config["paths"]["run_name"] = "family_b_test"

    result = train_model(config)

    trial_types = [row["trial_type"] for row in result.history]
    assert trial_types[:2] == ["load1_clean", "load1_clean"]
    assert set(trial_types[2:4]) == {"load1_clean", "load2_clean"}
    assert set(trial_types[4:8]) == EXPECTED_TYPES
    assert [row["curriculum_stage"] for row in result.history] == [
        1,
        1,
        2,
        2,
        3,
        3,
        3,
        3,
    ]
    assert [row["learning_rate"] for row in result.history] == [
        0.001,
        0.001,
        0.001,
        0.001,
        0.0001,
        0.0001,
        0.0001,
        0.0001,
    ]
    for row in result.history:
        assert row["n_items"] == (
            2 if row["trial_type"].startswith("load2") else 1
        )
        assert row["distractor_steps"] == (
            5 if row["trial_type"].endswith("distractor") else 0
        )
        assert row["cue_steps"] == 20


def test_weighted_block_encodes_35_percent_distractor_rate() -> None:
    block = draw_balanced_trial_type_block(
        np.random.default_rng(99),
        (
            *("load1_clean",) * 13,
            *("load1_distractor",) * 7,
            *("load2_clean",) * 13,
            *("load2_distractor",) * 7,
        ),
    )

    assert len(block) == 40
    assert sum(value.endswith("distractor") for value in block) == 14
    assert sum(value.startswith("load2") for value in block) == 20


def test_family_b_evaluation_reports_every_condition_and_position(
    tmp_path,
) -> None:
    config = load_config("configs/multicondition_working_memory.yaml")
    config["task"]["n_tuned_units"] = 8
    config["task"]["batch_size"] = 4
    config["task"]["pre_cue_steps"] = 2
    config["task"]["serial_item_cue_steps"] = 2
    config["task"]["item_gap_steps"] = 1
    config["task"]["response_steps"] = 4
    config["model"]["hidden_size"] = 8
    config["model"]["recurrent_noise_std"] = 0.0
    config["training"]["device"] = "cpu"
    config["evaluation"]["batches"] = 1
    config["paths"]["output_dir"] = str(tmp_path / "family_b_eval")
    config["paths"]["run_name"] = "family_b_eval"
    model = fresh_model(config, torch.device("cpu"))
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state": model.state_dict(), "config": config}, checkpoint)

    result = evaluate_family_b_conditions(config, checkpoint)

    assert len(result.rows) == 12
    assert {row["condition"] for row in result.rows} == set(CONDITIONS)
    assert {row["position"] for row in result.rows} == {
        "pooled",
        "first",
        "second",
    }
    assert result.metrics_path.exists()
    assert result.csv_path.exists()
    assert set(result.acceptance) == {"passed", "checks"}


def test_distribution_loss_training_smoke_records_components(
    tmp_path,
) -> None:
    config = load_config(
        "configs/multicondition_working_memory_distribution_loss.yaml"
    )
    config["task"]["n_tuned_units"] = 8
    config["task"]["batch_size"] = 4
    config["task"]["pre_cue_steps_choices"] = [2]
    config["task"]["delay_steps_choices"] = [4]
    config["task"]["serial_item_cue_steps"] = 2
    config["task"]["item_gap_steps"] = 1
    config["task"]["response_steps"] = 4
    config["model"]["hidden_size"] = 8
    config["model"]["recurrent_noise_std"] = 0.0
    config["training"]["steps"] = 8
    config["training"]["response_transition_steps"] = 1
    config["training"]["curriculum"] = [
        {"until_step": 2, "trial_types": ["load1_clean"]},
        {
            "until_step": 4,
            "trial_types": ["load1_clean", "load2_clean"],
        },
        {
            "until_step": 8,
            "learning_rate": 0.0001,
            "trial_type_counts": {
                "load1_clean": 1,
                "load1_distractor": 1,
                "load2_clean": 1,
                "load2_distractor": 1,
            },
        },
    ]
    config["training"]["log_every"] = 8
    config["training"]["device"] = "cpu"
    config["paths"]["output_dir"] = str(tmp_path / "distribution_loss")
    config["paths"]["run_name"] = "distribution_loss_test"

    result = train_model(config)

    assert all("response_cross_entropy" in row for row in result.history)
    assert all("fixation_loss" in row for row in result.history)
    assert all(np.isfinite(row["loss"]) for row in result.history)
    metrics = result.metrics_path.read_text(encoding="utf-8")
    assert '"tuned_loss": "circular_distribution"' in metrics
    assert '"population_normalization": "softmax"' in metrics


def test_distribution_role_training_smoke_uses_extra_channel(tmp_path) -> None:
    config = load_config(
        "configs/multicondition_working_memory_distribution_role.yaml"
    )
    config["task"]["n_tuned_units"] = 8
    config["task"]["batch_size"] = 4
    config["task"]["pre_cue_steps_choices"] = [2]
    config["task"]["delay_steps_choices"] = [4]
    config["task"]["serial_item_cue_steps"] = 2
    config["task"]["item_gap_steps"] = 1
    config["task"]["response_steps"] = 4
    config["model"]["hidden_size"] = 8
    config["model"]["recurrent_noise_std"] = 0.0
    config["training"]["steps"] = 4
    config["training"]["response_transition_steps"] = 1
    config["training"]["curriculum"] = [
        {
            "until_step": 4,
            "trial_type_counts": {
                "load1_clean": 1,
                "load1_distractor": 1,
                "load2_clean": 1,
                "load2_distractor": 1,
            },
        }
    ]
    config["training"]["log_every"] = 4
    config["training"]["device"] = "cpu"
    config["paths"]["output_dir"] = str(tmp_path / "distribution_role")
    config["paths"]["run_name"] = "distribution_role_test"

    result = train_model(config)
    checkpoint = torch.load(
        result.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint["model_state"]["rnn.input2h.weight"].shape[1] == 11
    assert all("response_cross_entropy" in row for row in result.history)
    assert all(np.isfinite(row["loss"]) for row in result.history)


def test_hidden128_rescue_training_smoke_preserves_dimensions(tmp_path) -> None:
    config = load_config(
        "configs/multicondition_working_memory_distribution_role_h128.yaml"
    )
    config["task"]["n_tuned_units"] = 8
    config["task"]["batch_size"] = 4
    config["task"]["pre_cue_steps_choices"] = [2]
    config["task"]["delay_steps_choices"] = [4]
    config["task"]["serial_item_cue_steps"] = 2
    config["task"]["item_gap_steps"] = 1
    config["task"]["response_steps"] = 4
    config["model"]["recurrent_noise_std"] = 0.0
    config["training"]["steps"] = 4
    config["training"]["response_transition_steps"] = 1
    config["training"]["curriculum"] = [
        {
            "until_step": 4,
            "trial_type_counts": {
                "load1_clean": 1,
                "load1_distractor": 1,
                "load2_clean": 1,
                "load2_distractor": 1,
            },
        }
    ]
    config["training"]["log_every"] = 4
    config["training"]["device"] = "cpu"
    config["paths"]["output_dir"] = str(tmp_path / "hidden128")
    config["paths"]["run_name"] = "hidden128_test"

    result = train_model(config)
    checkpoint = torch.load(
        result.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint["model_state"]["rnn.input2h.weight"].shape == (128, 11)
    assert checkpoint["model_state"]["readout.weight"].shape == (9, 128)
    assert all(np.isfinite(row["loss"]) for row in result.history)
