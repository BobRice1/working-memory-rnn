"""Focused outcome-free tests for the standalone fixed-grid N-back pilot."""

from __future__ import annotations

from pathlib import Path

import pytest

from wm_rnn.config import load_config
from wm_rnn.nback_additive_calibration import PROFILE_BY_ID
from wm_rnn.nback_additive_cost_precision import RetainedCheckpoint
from wm_rnn.nback_exploratory_pilot import (
    FROZEN_PILOT_DESIGN,
    PILOT_CHECKPOINT_SEEDS,
    PILOT_PROFILE_IDS,
    PilotDesign,
    load_pilot_operator_grids,
    pilot_profile_parameters,
    pilot_task_batches,
    select_pilot_checkpoints,
    summarize_checkpoint_signatures,
)


def test_exact_shared_pilot_operator_grids_are_loaded() -> None:
    grids = load_pilot_operator_grids()

    assert grids == {
        "synaptic_drive_gain": (0.90, 0.95, 1.00, 1.05, 1.10),
        "heterogeneous_drive_gain": (0.00, 0.10, 0.20, 0.30),
        "sensory_input_gain": (0.80, 0.90, 1.00, 1.10, 1.20),
        "recurrent_gain": (0.90, 0.95, 1.00, 1.05, 1.10),
        "gaussian_state_noise": (0.00, 0.025, 0.050, 0.075, 0.100),
        "state_persistence": (0.90, 0.95, 1.00, 1.05, 1.10),
        "time_constant": (0.80, 0.90, 1.00, 1.10, 1.25),
    }
    assert min(grids["synaptic_drive_gain"]) < 1.0 < max(
        grids["synaptic_drive_gain"]
    )
    assert min(grids["sensory_input_gain"]) < 1.0 < max(
        grids["sensory_input_gain"]
    )
    assert min(grids["recurrent_gain"]) < 1.0 < max(
        grids["recurrent_gain"]
    )
    assert min(grids["state_persistence"]) < 1.0 < max(
        grids["state_persistence"]
    )
    assert min(grids["time_constant"]) < 1.0 < max(
        grids["time_constant"]
    )


def test_frozen_pilot_scope_and_seed_namespaces() -> None:
    design = FROZEN_PILOT_DESIGN
    design.validate()

    assert design.checkpoint_seeds == (
        20260912,
        20260913,
        20260914,
    )
    assert design.profile_ids == (1, 4, 7, 9, 10, 12, 14)
    assert design.sequences_per_cell == 256
    task_seeds = {
        design.task_seed(checkpoint, condition, batch)
        for checkpoint in range(3)
        for condition in (0, 1)
        for batch in range(2)
    }
    noise_seeds = {
        design.noise_seed(checkpoint, condition, replicate, batch)
        for checkpoint in range(3)
        for condition in (0, 1)
        for replicate in range(3)
        for batch in range(2)
    }
    assert len(task_seeds) == 12
    assert len(noise_seeds) == 36
    assert task_seeds.isdisjoint(noise_seeds)


def test_pilot_selects_three_named_checkpoints_without_reindexing(
    tmp_path: Path,
) -> None:
    checkpoints = [
        RetainedCheckpoint(
            ordinal=index,
            seed=seed,
            path=tmp_path / f"{seed}.pt",
        )
        for index, seed in enumerate(range(20260912, 20260922))
    ]

    selected = select_pilot_checkpoints(checkpoints)

    assert tuple(item.seed for item in selected) == PILOT_CHECKPOINT_SEEDS
    assert tuple(item.ordinal for item in selected) == (0, 1, 2)
    reindexed = [
        RetainedCheckpoint(
            ordinal=index + 1,
            seed=item.seed,
            path=item.path,
        )
        for index, item in enumerate(selected)
    ]
    with pytest.raises(ValueError, match="ordinals"):
        select_pilot_checkpoints(reindexed)


def test_pilot_task_bank_has_two_paired_128_sequence_batches(
    tmp_path: Path,
) -> None:
    config = load_config("configs/nback_additive_perturbation.yaml")
    checkpoint = RetainedCheckpoint(
        ordinal=0,
        seed=20260912,
        path=tmp_path / "unused.pt",
    )

    zero = pilot_task_batches(config, checkpoint, 0)
    two = pilot_task_batches(config, checkpoint, 1)

    assert len(zero) == len(two) == 2
    assert sum(batch.inputs.shape[1] for batch in zero) == 256
    assert sum(batch.inputs.shape[1] for batch in two) == 256
    assert {batch.n_back for batch in zero} == {0}
    assert {batch.n_back for batch in two} == {2}
    assert not (zero[0].inputs == zero[1].inputs).all()
    assert not (zero[0].inputs == two[0].inputs).all()


def test_fixed_grid_parameters_reuse_common_random_numbers() -> None:
    p2 = PROFILE_BY_ID[4]
    p5 = PROFILE_BY_ID[14]

    p2_low = pilot_profile_parameters(
        p2,
        p2.ordered_grid[1],
        checkpoint_ordinal=0,
        condition_code=0,
        replicate_ordinal=1,
        batch_index=0,
    )
    p2_high = pilot_profile_parameters(
        p2,
        p2.ordered_grid[-1],
        checkpoint_ordinal=2,
        condition_code=1,
        replicate_ordinal=1,
        batch_index=1,
    )
    p5_low = pilot_profile_parameters(
        p5,
        p5.ordered_grid[1],
        checkpoint_ordinal=1,
        condition_code=1,
        replicate_ordinal=2,
        batch_index=1,
    )
    p5_high = pilot_profile_parameters(
        p5,
        p5.ordered_grid[-1],
        checkpoint_ordinal=1,
        condition_code=1,
        replicate_ordinal=2,
        batch_index=1,
    )

    assert p2_low["vector_seed"] == p2_high["vector_seed"] == 3102
    assert p5_low["generator_seed"] == p5_high["generator_seed"]
    assert p5_low["sigma"] != p5_high["sigma"]
    assert PILOT_PROFILE_IDS == tuple(
        profile_id
        for profile_id in PILOT_PROFILE_IDS
        if PROFILE_BY_ID[profile_id].ordered_grid
    )


def _metrics(
    *,
    discriminability: float,
    accuracy: float,
    cross_entropy: float,
    settling: float,
    failure_rate: float,
    settling_valid: bool = True,
) -> dict[str, object]:
    return {
        "discriminability": discriminability,
        "accuracy": accuracy,
        "mean_cross_entropy": cross_entropy,
        "failure_rate": failure_rate,
        "settling_valid": settling_valid,
        "settling_all": {
            "restricted_mean_settling_steps": settling,
        },
    }


def test_signature_summary_is_baseline_paired_and_load_selective() -> None:
    rows = [
        {
            "checkpoint_seed": 20260912,
            "checkpoint_ordinal": 0,
            "profile_id": None,
            "condition_code": 0,
            "metrics": _metrics(
                discriminability=1.0,
                accuracy=1.0,
                cross_entropy=0.01,
                settling=1.0,
                failure_rate=0.0,
            ),
        },
        {
            "checkpoint_seed": 20260912,
            "checkpoint_ordinal": 0,
            "profile_id": None,
            "condition_code": 1,
            "metrics": _metrics(
                discriminability=0.9,
                accuracy=0.97,
                cross_entropy=0.02,
                settling=1.5,
                failure_rate=0.01,
            ),
        },
        {
            "checkpoint_seed": 20260912,
            "checkpoint_ordinal": 0,
            "profile_id": 1,
            "operator": "synaptic_drive_gain",
            "variant": "bias_outside",
            "strength": 1.05,
            "condition_code": 0,
            "metrics": _metrics(
                discriminability=0.95,
                accuracy=0.98,
                cross_entropy=0.03,
                settling=1.2,
                failure_rate=0.02,
            ),
        },
        {
            "checkpoint_seed": 20260912,
            "checkpoint_ordinal": 0,
            "profile_id": 1,
            "operator": "synaptic_drive_gain",
            "variant": "bias_outside",
            "strength": 1.05,
            "condition_code": 1,
            "metrics": _metrics(
                discriminability=0.72,
                accuracy=0.90,
                cross_entropy=0.08,
                settling=2.0,
                failure_rate=0.08,
            ),
        },
    ]

    signature = summarize_checkpoint_signatures(rows)[0]

    assert signature["zero_back_discriminability_impairment"] == pytest.approx(
        0.05
    )
    assert signature["two_back_discriminability_impairment"] == pytest.approx(
        0.20
    )
    assert signature["load_selectivity"] == pytest.approx(0.15)
    assert signature["ce_load_interaction"] == pytest.approx(0.04)
    assert signature["settling_load_interaction"] == pytest.approx(0.3)
    assert signature["failure_rate_load_interaction"] == pytest.approx(0.05)


def test_invalid_settling_propagates_latency_na() -> None:
    rows = []
    for condition in (0, 1):
        rows.append(
            {
                "checkpoint_seed": 20260912,
                "checkpoint_ordinal": 0,
                "profile_id": None,
                "condition_code": condition,
                "metrics": _metrics(
                    discriminability=0.9,
                    accuracy=0.97,
                    cross_entropy=0.02,
                    settling=1.0,
                    failure_rate=0.01,
                ),
            }
        )
        rows.append(
            {
                "checkpoint_seed": 20260912,
                "checkpoint_ordinal": 0,
                "profile_id": 14,
                "operator": "gaussian_state_noise",
                "variant": "generic_control",
                "strength": 0.05,
                "condition_code": condition,
                "metrics": _metrics(
                    discriminability=0.8,
                    accuracy=0.9,
                    cross_entropy=0.07,
                    settling=9.0,
                    failure_rate=0.3,
                    settling_valid=(condition == 0),
                ),
            }
        )

    signature = summarize_checkpoint_signatures(rows)[0]

    assert signature["zero_back_settling_change"] == 8.0
    assert signature["two_back_settling_change"] is None
    assert signature["settling_load_interaction"] is None


def test_design_rejects_any_non_256_cell_size() -> None:
    changed = PilotDesign(batch_size=64, n_batches=2)
    with pytest.raises(ValueError, match="256 sequences"):
        changed.validate()
