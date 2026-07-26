"""Tests for pre-registered proportional matched-cost calibration."""

from __future__ import annotations

import numpy as np
import pytest

from wm_rnn.perturbation_calibration import (
    calibrate_bidirectional,
    calibrate_strength,
    paired_bootstrap_proportional_cost,
    proportional_cost,
    required_cost_check_trials,
    round_up_to_batch,
    validate_cost_match,
)


def test_bisection_converges_on_monotone_cost_function() -> None:
    result = calibrate_strength(
        lambda strength: 0.6 * strength,
        [0.0, 0.25, 0.75, 1.0],
        target_proportional_cost=0.30,
        tolerance=1e-5,
    )

    assert result.converged is True
    assert result.strength == pytest.approx(0.5, abs=1e-4)
    assert result.achieved_proportional_cost == pytest.approx(0.30, abs=1e-5)
    assert result.note == "bisection_converged"


def test_u_shaped_cost_yields_separate_branch_solutions() -> None:
    results = calibrate_bidirectional(
        lambda strength: 1.2 * (strength - 1.0) ** 2,
        [0.4, 0.7, 1.0, 1.3, 1.6],
        neutral_strength=1.0,
        target_proportional_cost=0.30,
        tolerance=1e-4,
    )

    assert results["below_neutral"].converged
    assert results["above_neutral"].converged
    assert results["below_neutral"].strength == pytest.approx(0.5, abs=1e-4)
    assert results["above_neutral"].strength == pytest.approx(1.5, abs=1e-4)


def test_unreachable_target_returns_closest_grid_value_without_extrapolation() -> None:
    result = calibrate_strength(
        lambda strength: 0.1 * strength,
        [0.0, 0.5, 1.0],
        target_proportional_cost=0.30,
    )

    assert result.converged is False
    assert result.strength == 1.0
    assert result.achieved_proportional_cost == pytest.approx(0.1)
    assert result.note == "target_unreachable_no_extrapolation"
    assert result.bracket_lower is None
    assert result.bracket_upper is None


def test_proportional_matching_is_checkpoint_normalized() -> None:
    low_baseline = proportional_cost(4.0, 5.2)
    high_baseline = proportional_cost(8.0, 10.4)

    assert low_baseline == pytest.approx(0.30)
    assert high_baseline == pytest.approx(0.30)
    assert 5.2 - 4.0 != 10.4 - 8.0


def test_phase_zero_precision_formula_and_batch_rounding() -> None:
    required = required_cost_check_trials(7.013, 6.144)

    assert required == pytest.approx(590.3, rel=0.01)
    assert round_up_to_batch(required, 64) == 640


def test_paired_bootstrap_is_deterministic_and_precise_for_fixed_effect() -> None:
    baseline = np.linspace(3.0, 5.0, 128)
    perturbed = baseline * 1.3

    first = paired_bootstrap_proportional_cost(
        baseline, perturbed, draws=500, bootstrap_seed=9
    )
    repeat = paired_bootstrap_proportional_cost(
        baseline, perturbed, draws=500, bootstrap_seed=9
    )

    assert first == repeat
    assert first[0] == pytest.approx(0.30)
    assert first[3] < 1e-12


def test_cost_match_checks_band_precision_and_p5_gap() -> None:
    baseline = np.linspace(3.0, 5.0, 128)
    perturbed = baseline * 1.3

    valid = validate_cost_match(
        baseline,
        perturbed,
        p5_proportional_cost=0.28,
        draws=500,
    )
    mismatch = validate_cost_match(
        baseline,
        perturbed,
        p5_proportional_cost=0.20,
        draws=500,
    )

    assert valid.cost_match_valid is True
    assert valid.p5_cost_gap == pytest.approx(0.02)
    assert valid.invalid_reason is None
    assert mismatch.cost_match_valid is False
    assert mismatch.p5_cost_gap_valid is False
    assert mismatch.invalid_reason == "p5_cost_mismatch"


def test_calibration_rejects_duplicate_grid_points() -> None:
    with pytest.raises(ValueError, match="unique"):
        calibrate_strength(lambda strength: strength, [0.0, 0.5, 0.5, 1.0])
