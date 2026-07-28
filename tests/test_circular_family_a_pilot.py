from __future__ import annotations

from pathlib import Path

import pytest

from wm_rnn.circular_family_a_pilot import (
    DELAYS,
    DISTRACTOR_DELAY,
    FROZEN_CHECKPOINTS,
    PILOT_CELLS,
    PILOT_SEEDS,
    TRIALS_PER_CELL,
    build_operator_settings,
    design_summary,
    load_pilot_config,
    settings_for_cell,
    verify_frozen_inputs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_CONFIG = REPO_ROOT / "configs/exploratory_psilocybin_signature_pilot.yaml"


def test_frozen_family_a_manifest_is_exact_three_seed_subset() -> None:
    assert tuple(item.seed for item in FROZEN_CHECKPOINTS) == PILOT_SEEDS
    assert PILOT_SEEDS == (20260714, 20260715, 20260716)
    assert all(len(item.sha256) == 64 for item in FROZEN_CHECKPOINTS)
    assert all(item.path.endswith(f"seed_{item.seed}.pt") for item in FROZEN_CHECKPOINTS)


def test_fixed_cells_are_four_clean_delays_plus_one_distractor() -> None:
    clean = [cell.delay_steps for cell in PILOT_CELLS if cell.condition == "clean"]
    distractor = [
        cell.delay_steps for cell in PILOT_CELLS if cell.condition == "distractor"
    ]
    assert tuple(clean) == DELAYS
    assert distractor == [DISTRACTOR_DELAY]
    assert TRIALS_PER_CELL == 256


def test_operator_expansion_uses_three_registered_replicates() -> None:
    config = load_pilot_config(PILOT_CONFIG)
    settings = build_operator_settings(config["operators"])
    p2 = [item for item in settings if item.operator == "heterogeneous_drive_gain"]
    p5 = [item for item in settings if item.operator == "gaussian_state_noise"]
    assert len(p2) == len(config["operators"]["heterogeneous_drive_gain"]) * 3
    assert len(p5) == len(config["operators"]["gaussian_state_noise"]) * 3
    assert {item.gain_vector_seed for item in p2} == {3101, 3102, 3103}
    assert {item.noise_replicate for item in p5} == {4101, 4102, 4103}


def test_distractor_only_gain_is_not_run_on_clean_cells() -> None:
    config = load_pilot_config(PILOT_CONFIG)
    settings = build_operator_settings(config["operators"])
    clean = settings_for_cell(settings, PILOT_CELLS[0])
    distractor = settings_for_cell(settings, PILOT_CELLS[-1])
    assert not any(item.operator == "distractor_input_gain" for item in clean)
    assert any(item.operator == "distractor_input_gain" for item in distractor)


def test_design_summary_is_outcome_free_and_counts_expanded_cells() -> None:
    summary = design_summary(PILOT_CONFIG)
    assert summary["seeds"] == list(PILOT_SEEDS)
    assert summary["trials_per_cell"] == 256
    assert len(summary["cells"]) == 5
    assert summary["planned_cells"] > 0
    assert "outcomes" not in summary


def test_changed_fixed_seed_list_is_rejected(tmp_path: Path) -> None:
    text = PILOT_CONFIG.read_text(encoding="utf-8").replace(
        "circular_seeds: [20260714, 20260715, 20260716]",
        "circular_seeds: [1, 2, 3]",
    )
    changed = tmp_path / "changed.yaml"
    changed.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="circular_seeds"):
        load_pilot_config(changed)


def test_input_verification_accepts_explicit_config_and_checkpoint(
    tmp_path: Path,
) -> None:
    from hashlib import sha256

    from wm_rnn.circular_family_a_pilot import FrozenCheckpoint

    config = tmp_path / "task.yaml"
    checkpoint = tmp_path / "model.pt"
    config.write_bytes(b"task: tuned\n")
    checkpoint.write_bytes(b"checkpoint")
    frozen = FrozenCheckpoint(
        seed=7,
        path=str(checkpoint),
        sha256=sha256(checkpoint.read_bytes()).hexdigest().upper(),
    )

    verified = verify_frozen_inputs(
        tmp_path,
        checkpoints=(frozen,),
        config_path=config,
        expected_config_sha256=sha256(config.read_bytes()).hexdigest().upper(),
    )

    assert verified["config_path"] == str(config)
    assert verified["checkpoints"][0]["seed"] == 7
