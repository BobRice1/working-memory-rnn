"""Tests for the variable-timing circular checkpoint pool."""

from __future__ import annotations

import pytest

from wm_rnn.circular_variable_distractor_pool import (
    validate_variable_timing_config,
)
from wm_rnn.config import load_config


def _config():
    return load_config(
        "configs/fixation_circular_variable_distractor_working_memory.yaml"
    )


def test_frozen_variable_timing_schedule_is_valid() -> None:
    seeds, fractions, target = validate_variable_timing_config(_config())
    assert seeds == list(range(20260801, 20260816))
    assert fractions == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert target == 10


def test_training_and_evaluation_timings_must_match() -> None:
    config = _config()
    config["evaluation"]["distractor_onset_fractions"] = [0.0, 0.5, 1.0]
    with pytest.raises(ValueError, match="must match"):
        validate_variable_timing_config(config)
