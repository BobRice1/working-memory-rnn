"""Tests for perturbation-blind N-back checkpoint screening."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from wm_rnn.config import load_config
from wm_rnn.nback_evaluation import resolve_nback_bank_seed
from wm_rnn.nback_screened_pool import (
    config_for_nback_seed,
    run_nback_screened_pool,
)


def _train_result(tmp_path: Path, seed: int, passed: bool = True):
    checkpoint = tmp_path / f"seed_{seed}.pt"
    checkpoint.touch()
    return SimpleNamespace(
        passed=passed,
        checkpoint_path=checkpoint,
        metrics_path=tmp_path / f"seed_{seed}_training.json",
        history=[{"step": 1}],
    )


def _eval_result(tmp_path: Path, seed: int, passed: bool):
    checks = {
        "zero_back_accuracy": True,
        "two_back_discriminability": passed,
    }
    conditions = {
        "0-back": {"accuracy": 1.0, "discriminability": 1.0},
        "2-back": {
            "accuracy": 0.96,
            "discriminability": 0.91 if passed else 0.89,
            "one_back_lure_accuracy": 0.99,
        },
    }
    return SimpleNamespace(
        passed=passed,
        metrics_path=tmp_path / f"seed_{seed}.json",
        metrics={
            "acceptance": {"passed": passed, "checks": checks},
            "conditions": conditions,
        },
    )


def test_screened_pool_continues_after_failures_and_stops_at_target(
    tmp_path,
) -> None:
    config = load_config(
        "configs/nback_working_memory_screened_final.yaml"
    )
    config["paths"] = {
        "output_dir": str(tmp_path / "pool"),
        "run_name": "screened",
    }
    seeds = [10, 11, 12, 13, 14]
    passing = {10, 12, 13}
    attempted: list[int] = []

    def train_fn(seed_config):
        seed = seed_config["task"]["seed"]
        attempted.append(seed)
        return _train_result(tmp_path, seed)

    def evaluate_fn(seed_config, _checkpoint):
        seed = seed_config["task"]["seed"]
        return _eval_result(tmp_path, seed, seed in passing)

    result = run_nback_screened_pool(
        config,
        seeds,
        target_count=3,
        train_fn=train_fn,
        evaluate_fn=evaluate_fn,
    )

    assert result.passed
    assert result.selected_seeds == [10, 12, 13]
    assert attempted == [10, 11, 12, 13]
    assert [row["status"] for row in result.results] == [
        "selected",
        "competence_failed",
        "selected",
        "selected",
    ]
    assert result.summary_path.read_text(encoding="utf-8")


def test_screened_pool_stops_as_soon_as_target_is_impossible(
    tmp_path,
) -> None:
    config = load_config(
        "configs/nback_working_memory_screened_final.yaml"
    )
    config["paths"] = {
        "output_dir": str(tmp_path / "pool"),
        "run_name": "screened",
    }
    attempted: list[int] = []

    def train_fn(seed_config):
        seed = seed_config["task"]["seed"]
        attempted.append(seed)
        return _train_result(tmp_path, seed)

    def evaluate_fn(seed_config, _checkpoint):
        seed = seed_config["task"]["seed"]
        return _eval_result(tmp_path, seed, False)

    result = run_nback_screened_pool(
        config,
        [20, 21, 22, 23, 24],
        target_count=3,
        train_fn=train_fn,
        evaluate_fn=evaluate_fn,
    )

    assert not result.passed
    assert result.selected_seeds == []
    assert attempted == [20, 21, 22]
    assert all(
        row["failed_checks"] == ["two_back_discriminability"]
        for row in result.results
    )


def test_training_failure_skips_evaluation_and_pool_continues(
    tmp_path,
) -> None:
    config = load_config(
        "configs/nback_working_memory_screened_final.yaml"
    )
    config["paths"] = {
        "output_dir": str(tmp_path / "pool"),
        "run_name": "screened",
    }
    evaluated: list[int] = []

    def train_fn(seed_config):
        seed = seed_config["task"]["seed"]
        return _train_result(tmp_path, seed, passed=seed != 31)

    def evaluate_fn(seed_config, _checkpoint):
        seed = seed_config["task"]["seed"]
        evaluated.append(seed)
        return _eval_result(tmp_path, seed, True)

    result = run_nback_screened_pool(
        config,
        [30, 31, 32],
        target_count=2,
        train_fn=train_fn,
        evaluate_fn=evaluate_fn,
    )

    assert result.passed
    assert result.selected_seeds == [30, 32]
    assert evaluated == [30, 32]
    assert result.results[1]["failure_stage"] == "training"


def test_infrastructure_exception_preserves_progress_for_resume(
    tmp_path,
) -> None:
    config = load_config(
        "configs/nback_working_memory_screened_final.yaml"
    )
    config["paths"] = {
        "output_dir": str(tmp_path / "pool"),
        "run_name": "screened",
    }
    attempts: list[int] = []
    interrupt_once = True

    def train_fn(seed_config):
        nonlocal interrupt_once
        seed = seed_config["task"]["seed"]
        attempts.append(seed)
        if seed == 41 and interrupt_once:
            interrupt_once = False
            raise RuntimeError("simulated infrastructure interruption")
        return _train_result(tmp_path, seed)

    def evaluate_fn(seed_config, _checkpoint):
        seed = seed_config["task"]["seed"]
        return _eval_result(tmp_path, seed, True)

    with pytest.raises(
        RuntimeError, match="infrastructure interruption"
    ):
        run_nback_screened_pool(
            config,
            [40, 41, 42],
            target_count=3,
            train_fn=train_fn,
            evaluate_fn=evaluate_fn,
        )

    result = run_nback_screened_pool(
        config,
        [40, 41, 42],
        target_count=3,
        train_fn=train_fn,
        evaluate_fn=evaluate_fn,
    )

    assert result.passed
    assert attempts == [40, 41, 41, 42]
    assert result.selected_seeds == [40, 41, 42]


@pytest.mark.parametrize(
    ("seeds", "target", "message"),
    [
        ([], 1, "at least one"),
        ([1, 1], 1, "unique"),
        ([2, 1], 1, "ascending"),
        ([1, 2], 3, "between one"),
    ],
)
def test_screened_pool_rejects_invalid_specs(
    tmp_path, seeds, target, message
) -> None:
    config = load_config(
        "configs/nback_working_memory_screened_final.yaml"
    )
    config["paths"]["output_dir"] = str(tmp_path)
    with pytest.raises(ValueError, match=message):
        run_nback_screened_pool(
            config,
            seeds,
            target_count=target,
        )


def test_screened_bank_seeds_are_disjoint_across_adjacent_checkpoints() -> None:
    config = load_config(
        "configs/nback_working_memory_screened_final.yaml"
    )
    first = config_for_nback_seed(config, 20260912)
    second = config_for_nback_seed(config, 20260913)

    first_base = resolve_nback_bank_seed(first, "evaluation")
    second_base = resolve_nback_bank_seed(second, "evaluation")
    assert first_base + 1 < second_base


def test_historical_configs_keep_unit_checkpoint_seed_stride() -> None:
    config = load_config(
        "configs/nback_working_memory_budget_final.yaml"
    )
    assert resolve_nback_bank_seed(config, "evaluation") == (
        config["task"]["seed"] + config["evaluation"]["seed_offset"]
    )
