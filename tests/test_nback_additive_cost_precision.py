"""Focused tests for baseline-only additive N-back precision planning."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import wm_rnn.nback_additive_cost_precision as precision_module
from wm_rnn.config import load_config
from wm_rnn.nback_additive_cost_precision import (
    PrecisionDesign,
    bootstrap_family_max_sd,
    derive_cost_check_size,
    describe_sequence_units,
    precision_task_config,
    precision_task_seed,
    run_nback_additive_cost_precision,
)


def _small_design() -> PrecisionDesign:
    return PrecisionDesign(
        retained_seeds=(11, 12),
        bank_base=1000,
        n_batches=2,
        batch_size=3,
        bootstrap_seed=2000,
        bootstrap_draws=25,
        bootstrap_chunk_size=4,
        bootstrap_percentile=95.0,
        kappa=2.0,
        z_value=1.96,
        half_width=0.5,
        minimum_cost_check=4,
        maximum_cost_check=100,
        cost_check_multiple=2,
    )


def test_precision_seed_mapping_and_task_are_frozen_zero_back() -> None:
    design = _small_design()
    config = load_config("configs/nback_working_memory_screened_final.yaml")

    assert precision_task_seed(0, 0, design=design) == 1000
    assert precision_task_seed(1, 1, design=design) == 11001
    task = precision_task_config(config, 1, 1, design=design)

    assert task.n_back == 0
    assert task.batch_size == 3
    assert task.seed == 11001
    with pytest.raises(ValueError, match="batch_index"):
        precision_task_seed(0, 2, design=design)


def test_descriptions_use_sample_sd_and_registered_percentiles() -> None:
    values = np.asarray([0.0, 1.0, 2.0, 3.0])

    result = describe_sequence_units(values)

    assert result["n_sequences"] == 4
    assert result["mean"] == pytest.approx(1.5)
    assert result["sample_sd"] == pytest.approx(np.std(values, ddof=1))
    assert result["median"] == pytest.approx(1.5)
    assert result["q1"] == pytest.approx(0.75)
    assert result["q3"] == pytest.approx(2.25)
    assert result["iqr"] == pytest.approx(1.5)
    assert result["p90"] == pytest.approx(2.7)
    assert result["p95"] == pytest.approx(2.85)
    assert result["p99"] == pytest.approx(2.97)
    assert result["maximum"] == 3.0


def test_family_bootstrap_is_deterministic_and_chunk_invariant() -> None:
    units = np.asarray(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4],
            [0.0, 0.2, 0.4, 0.6, 0.8],
        ]
    )

    first = bootstrap_family_max_sd(
        units, draws=31, seed=77, chunk_size=1
    )
    second = bootstrap_family_max_sd(
        units, draws=31, seed=77, chunk_size=7
    )
    repeat = bootstrap_family_max_sd(
        units, draws=31, seed=77, chunk_size=7
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(second, repeat)
    assert first.shape == (31,)
    assert np.all(first >= 0.0)


def test_cost_check_formula_rounds_and_enforces_frozen_minimum() -> None:
    design = _small_design()

    n_required, n_cost_check = derive_cost_check_size(
        0.1, design=design
    )

    expected = (1.96 * 0.1 * np.sqrt(5.0) / 0.5) ** 2
    assert n_required == pytest.approx(expected)
    assert n_cost_check == 4


def test_small_runner_persists_complete_audit_artifacts(
    tmp_path: Path,
) -> None:
    design = _small_design()
    checkpoint_paths = []
    for seed in design.retained_seeds:
        checkpoint = tmp_path / f"seed_{seed}.pt"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        checkpoint_paths.append(str(checkpoint))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "attempted_seeds": list(design.retained_seeds),
                "retained_seeds": list(design.retained_seeds),
                "retained_checkpoints": checkpoint_paths,
                "failed_seeds": [],
                "passed": True,
                "stop_reason": "target_reached",
            }
        ),
        encoding="utf-8",
    )
    config = load_config("configs/nback_working_memory_screened_final.yaml")
    config["training"]["device"] = "cpu"
    calls: list[tuple[int, int, torch.device]] = []

    def fake_collect(
        _config: dict[str, object],
        checkpoint: object,
        device: torch.device,
        received_design: PrecisionDesign,
    ) -> np.ndarray:
        calls.append(
            (
                checkpoint.ordinal,  # type: ignore[attr-defined]
                checkpoint.seed,  # type: ignore[attr-defined]
                device,
            )
        )
        base = np.linspace(0.01, 0.02, received_design.sequences_per_checkpoint)
        return base + 0.001 * checkpoint.ordinal  # type: ignore[attr-defined]

    result = run_nback_additive_cost_precision(
        config,
        manifest,
        output_dir=tmp_path / "precision",
        run_name="tiny_precision",
        repo_root=tmp_path,
        design=design,
        collect_fn=fake_collect,
    )

    assert result.passed
    assert calls == [
        (0, 11, torch.device("cpu")),
        (1, 12, torch.device("cpu")),
    ]
    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert all(payload["checks"].values())
    assert len(payload["checkpoint_descriptions"]) == 2
    assert payload["planning"]["n_cost_check"] == 4
    assert payload["task_seed_mapping"]["condition_code"] == 0
    assert payload["checks"]["bootstrap_draw_count_exact"] is True
    assert (
        payload["checks"]["n_cost_check_is_complete_batch_multiple"] is True
    )
    with np.load(result.arrays_path) as arrays:
        assert arrays["sequence_log_loss_units"].shape == (2, 6)
        assert arrays["task_seeds"].tolist() == [
            [1000, 1001],
            [11000, 11001],
        ]
        assert arrays["bootstrap_maximum_sds"].shape == (25,)
    assert len(result.seed_map_csv_path.read_text().splitlines()) == 5
    assert len(result.descriptions_csv_path.read_text().splitlines()) == 3


def test_module_has_no_operator_or_perturbation_imports() -> None:
    source = Path(precision_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)

    assert not any(
        module.startswith("wm_rnn.perturbation")
        or module == "wm_rnn.nback_perturbation"
        for module in imported_modules
    )
