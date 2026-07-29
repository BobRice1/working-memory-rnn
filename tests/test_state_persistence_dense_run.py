"""Outcome-free checks for the dense persistence neighbourhood runner."""

from __future__ import annotations

from pathlib import Path

from wm_rnn.config import load_config
from wm_rnn.state_persistence_dense_run import (
    CONFIG,
    PERSISTENCE_STRENGTHS,
    _validate_config,
    dense_nback_design,
)


def test_dense_persistence_config_matches_frozen_grid() -> None:
    config = load_config(CONFIG)
    _validate_config(config)
    assert tuple(config["operators"]["state_persistence"]) == PERSISTENCE_STRENGTHS
    assert config["pilot"]["circular_trials_per_cell"] == 1024
    assert config["pilot"]["nback_sequences_per_cell"] == 1024
    assert len(config["pilot"]["circular_seeds"]) == 10
    assert len(config["pilot"]["nback_seeds"]) == 10


def test_dense_nback_design_is_persistence_only_1024() -> None:
    design = dense_nback_design()
    design.validate()
    assert design.profile_ids == (10,)
    assert design.sequences_per_cell == 1024
    assert design.checkpoint_seeds == tuple(range(20260912, 20260922))


def test_preregistration_note_exists() -> None:
    path = Path(
        "docs/preregistration/state_persistence_dense_variable_timing_1024.md"
    )
    text = path.read_text(encoding="utf-8")
    assert "0.80, 0.85, 0.88" in text
    assert "Frozen before outcome inspection" in text
