"""Tests for result-contingent P2 assignment sensitivity."""

from __future__ import annotations

import numpy as np
import pytest

from wm_rnn.perturbation_assignment_sensitivity import (
    gain_strength_correlations,
    permute_gain_multiset,
    within_checkpoint_slope,
)


def test_assignment_preserves_gain_multiset_exactly() -> None:
    gains = np.array([0.8, 0.9, 1.0, 1.1, 1.2])

    assigned = permute_gain_multiset(gains, assignment_seed=17)

    np.testing.assert_array_equal(np.sort(assigned), np.sort(gains))


def test_assignment_is_deterministic_by_seed() -> None:
    gains = np.linspace(0.8, 1.2, 20)

    first = permute_gain_multiset(gains, assignment_seed=4)
    repeat = permute_gain_multiset(gains, assignment_seed=4)
    other = permute_gain_multiset(gains, assignment_seed=5)

    np.testing.assert_array_equal(first, repeat)
    assert not np.array_equal(first, other)


def test_gain_strength_correlation_recovers_known_alignment() -> None:
    gains = np.array([0.8, 0.9, 1.1, 1.2])
    weights = np.diag(gains)

    correlations = gain_strength_correlations(gains, weights)

    assert correlations["gain_in_strength_correlation"] == pytest.approx(1.0)
    assert correlations["gain_out_strength_correlation"] == pytest.approx(1.0)
    assert correlations["gain_total_strength_correlation"] == pytest.approx(1.0)


def test_within_checkpoint_regression_recovers_known_slope() -> None:
    alignment = np.linspace(-1.0, 1.0, 50)
    outcome = 2.5 * alignment + 0.4

    slope = within_checkpoint_slope(alignment, outcome)

    assert slope == pytest.approx(2.5)
