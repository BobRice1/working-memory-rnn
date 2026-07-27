"""Smoke and device tests for competence-gated N-back training."""

from __future__ import annotations

import torch

from wm_rnn.config import load_config
from wm_rnn.nback_evaluation import evaluate_nback_checkpoint
from wm_rnn.nback_seed_sweep import config_for_nback_seed
from wm_rnn.nback_task import NBackTaskConfig, generate_nback_batch
from wm_rnn.train_nback import (
    draw_balanced_nback_block,
    draw_nback_rule_block,
    train_nback_model,
)
from wm_rnn.training_utils import (
    batch_to_tensors,
    fresh_model,
    masked_cross_entropy,
    task_config_from_dict,
)


def _tiny_config(tmp_path) -> dict:
    config = load_config("configs/nback_working_memory.yaml")
    config["task"].update(
        {
            "sequence_items": 6,
            "stimulus_steps": 2,
            "interstimulus_steps": 2,
            "matches_per_sequence": 2,
            "min_one_back_lures": 1,
            "batch_size": 4,
            "seed": 1234,
        }
    )
    config["model"]["hidden_size"] = 8
    config["training"].update(
        {
            "stage1_min_steps": 2,
            "stage1_max_steps": 2,
            "stage2_max_steps": 2,
            "stage1_validate_every": 1,
            "stage2_validate_every": 1,
            "required_consecutive_passes": 2,
            "log_every": 10,
            "device": "cpu",
        }
    )
    config["validation"]["sequences_per_condition"] = 4
    config["evaluation"]["sequences_per_condition"] = 4
    config["stage1_competence"].update(
        {"accuracy": 0.0, "discriminability": -1.0}
    )
    config["competence"].update(
        {
            "accuracy": 0.0,
            "discriminability": -1.0,
            "lure_accuracy": 0.0,
        }
    )
    config["paths"].update(
        {
            "output_dir": str(tmp_path / "nback"),
            "run_name": "nback_smoke",
        }
    )
    return config


def test_nback_config_builds_eight_input_two_output_model() -> None:
    config = load_config("configs/nback_working_memory.yaml")
    task = task_config_from_dict(config)
    model = fresh_model(config, torch.device("cpu"))

    assert isinstance(task, NBackTaskConfig)
    assert model.config.input_size == 8
    assert model.config.output_size == 2


def test_balanced_nback_block_is_reproducible() -> None:
    import numpy as np

    first = draw_balanced_nback_block(np.random.default_rng(44))
    second = draw_balanced_nback_block(np.random.default_rng(44))
    assert first == second
    assert set(first) == {0, 2}


def test_rescue_rule_block_contains_one_zero_and_three_two_back() -> None:
    import numpy as np

    first = draw_nback_rule_block(
        np.random.default_rng(51), (0, 2, 2, 2)
    )
    second = draw_nback_rule_block(
        np.random.default_rng(51), (0, 2, 2, 2)
    )
    assert first == second
    assert first.count(0) == 1
    assert first.count(2) == 3


def test_tiny_cpu_training_completes_both_stages(tmp_path) -> None:
    config = _tiny_config(tmp_path)
    result = train_nback_model(config)
    evaluation = evaluate_nback_checkpoint(
        config, result.checkpoint_path
    )

    assert result.passed
    assert evaluation.passed
    assert result.checkpoint_path.exists()
    assert result.metrics_path.exists()
    assert result.history_path.exists()
    assert evaluation.metrics_path.exists()
    assert set(evaluation.metrics["conditions"]) == {"0-back", "2-back"}
    assert {row["stage"] for row in result.history} == {1, 2}


def test_cuda_forward_and_loss_stay_on_gpu_when_available() -> None:
    if not torch.cuda.is_available():
        return
    config = load_config("configs/nback_working_memory.yaml")
    config["task"]["batch_size"] = 2
    config["model"]["hidden_size"] = 8
    task = task_config_from_dict(config)
    batch = generate_nback_batch(task)
    device = torch.device("cuda")
    model = fresh_model(config, device)
    inputs, targets, loss_mask = batch_to_tensors(batch, device)
    logits, states = model(inputs)
    loss = masked_cross_entropy(logits, targets, loss_mask)
    weighted_loss = masked_cross_entropy(
        logits,
        targets,
        loss_mask,
        class_weights=torch.tensor([1.0, 2.0], device=device),
    )

    assert next(model.parameters()).is_cuda
    assert inputs.is_cuda and targets.is_cuda and loss_mask.is_cuda
    assert (
        logits.is_cuda
        and states.is_cuda
        and loss.is_cuda
        and weighted_loss.is_cuda
    )


def test_rescue_config_changes_only_registered_training_fields() -> None:
    base = load_config("configs/nback_working_memory.yaml")
    rescue = load_config(
        "configs/nback_working_memory_balance_rescue.yaml"
    )
    assert rescue["training"]["stage2_rule_block"] == [0, 2, 2, 2]
    assert rescue["training"]["stage2_class_weights"] == {
        2: [1.0, 2.0]
    }
    assert rescue["task"]["seed"] == 20260824
    assert rescue["validation"]["seed_offset"] == 300000
    assert rescue["evaluation"]["seed_offset"] == 400000
    assert rescue["paths"] != base["paths"]

    for config in (base, rescue):
        config["task"]["seed"] = 0
        config["validation"]["seed_offset"] = 0
        config["evaluation"]["seed_offset"] = 0
        config["paths"] = {}
    rescue["training"].pop("stage2_rule_block")
    rescue["training"].pop("stage2_class_weights")
    assert rescue == base


def test_per_seed_config_isolates_outputs_without_mutating_base() -> None:
    config = load_config(
        "configs/nback_working_memory_balance_rescue.yaml"
    )
    original_seed = config["task"]["seed"]
    seeded = config_for_nback_seed(config, 20260825)

    assert config["task"]["seed"] == original_seed
    assert seeded["task"]["seed"] == 20260825
    assert seeded["paths"]["output_dir"].endswith(
        "seed_sweep\\seed_20260825"
    ) or seeded["paths"]["output_dir"].endswith(
        "seed_sweep/seed_20260825"
    )
    assert seeded["paths"]["run_name"].endswith("_seed_20260825")
