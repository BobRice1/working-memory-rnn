"""Tests for trained one-item circular distractor sampling and evaluation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wm_rnn.circular_distractor_pool import evaluate_condition
from wm_rnn.config import load_config
from wm_rnn.train import (
    apply_circular_distractor_trial_type,
    draw_circular_distractor_block,
    train_model,
)
from wm_rnn.training_utils import fresh_model, task_config_from_dict


def _tiny_config(tmp_path):
    config = load_config(
        "configs/fixation_circular_distractor_working_memory.yaml"
    )
    config["task"].update(
        {
            "n_tuned_units": 8,
            "pre_cue_steps": 3,
            "pre_cue_steps_choices": [3],
            "cue_steps": 3,
            "cue_steps_choices": [3],
            "delay_steps": 4,
            "delay_steps_choices": [4],
            "response_steps": 4,
            "batch_size": 4,
            "distractor_steps": 2,
        }
    )
    config["model"]["hidden_size"] = 8
    config["training"].update(
        {
            "steps": 2,
            "log_every": 1,
            "response_transition_steps": 1,
            "ignore_initial_steps": 1,
            "device": "cpu",
        }
    )
    config["evaluation"].update(
        {
            "trials_per_condition": 8,
            "batch_size": 4,
            "seed_base": 901,
        }
    )
    config["paths"].update(
        {
            "output_dir": str(tmp_path / "circular_distractor"),
            "run_name": "tiny_circular_distractor",
        }
    )
    return config


def test_circular_distractor_block_contains_one_batch_per_condition() -> None:
    block = draw_circular_distractor_block(np.random.default_rng(7))
    assert sorted(block) == ["clean", "distractor"]


def test_trial_type_only_changes_distractor_presence() -> None:
    config = load_config(
        "configs/fixation_circular_distractor_working_memory.yaml"
    )
    task = task_config_from_dict(config)
    clean = apply_circular_distractor_trial_type(task, "clean", 5)
    distracted = apply_circular_distractor_trial_type(
        task, "distractor", 5
    )

    assert clean.distractor_steps == 0
    assert distracted.distractor_steps == 5
    assert clean.n_items == distracted.n_items == 1
    assert not clean.probe_gated and not distracted.probe_gated


def test_balanced_training_logs_clean_and_distractor_batches(tmp_path) -> None:
    result = train_model(_tiny_config(tmp_path))
    assert {row["trial_type"] for row in result.history} == {
        "clean",
        "distractor",
    }
    assert sorted(row["distractor_steps"] for row in result.history) == [0, 2]


@pytest.mark.parametrize("condition", ["clean", "distractor"])
def test_condition_evaluation_returns_finite_metrics(
    tmp_path,
    condition: str,
) -> None:
    config = _tiny_config(tmp_path)
    model = fresh_model(config, torch.device("cpu"))
    model.eval()
    metrics = evaluate_condition(model, config, condition)

    assert metrics.condition == condition
    assert metrics.trials == 8
    assert np.isfinite(metrics.mean_angular_error_degrees)
    assert np.isfinite(metrics.median_angular_error_degrees)
    assert 0.0 <= metrics.fixation_accuracy <= 1.0
