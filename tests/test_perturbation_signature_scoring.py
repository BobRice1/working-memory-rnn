"""Tests for the frozen confirmatory signature-scoring rules."""

from __future__ import annotations

import numpy as np
import pytest

from wm_rnn.perturbation_experiment import (
    compute_excess_constraints,
    condition_normalized_change,
)
from wm_rnn.perturbation_signature_scoring import (
    classify_profile,
    cost_gap_valid,
    holm_adjust,
    intersection_union_pvalue,
    minimum_detectable_dz,
)


def test_condition_normalization_removes_baseline_headroom_effect() -> None:
    easy = condition_normalized_change(4.0, 5.2)
    hard = condition_normalized_change(20.0, 26.0)

    assert easy == pytest.approx(0.30)
    assert hard == pytest.approx(0.30)
    assert 5.2 - 4.0 != 26.0 - 20.0


def test_subtracting_p5_recovers_known_excess_c1_c3() -> None:
    contrasts = compute_excess_constraints(
        {
            "load1_clean": 10.0,
            "load2_clean": 20.0,
            "load1_distractor": 30.0,
        },
        {
            "load1_clean": 13.0,
            "load2_clean": 32.0,
            "load1_distractor": 57.0,
        },
        {
            "load1_clean": 13.0,
            "load2_clean": 26.0,
            "load1_distractor": 39.0,
        },
        baseline_rmst=4.0,
        candidate_rmst=9.0,
        p5_rmst=6.0,
    )

    assert contrasts["x1"] == pytest.approx(3.0)
    assert contrasts["x2"] == pytest.approx(0.30)
    assert contrasts["x3"] == pytest.approx(0.60)


def test_iut_is_exactly_maximum_component_pvalue() -> None:
    assert intersection_union_pvalue([0.001, 0.02, 0.005]) == 0.02


def test_holm_is_applied_across_six_profile_pvalues() -> None:
    p_values = [0.001, 0.01, 0.02, 0.03, 0.04, 0.2]
    adjusted = holm_adjust(p_values)

    assert len(adjusted) == 6
    assert adjusted[0] == pytest.approx(0.006)
    assert adjusted[1] == pytest.approx(0.05)
    assert all(0.0 <= value <= 1.0 for value in adjusted)


def test_invalid_p5_gate_yields_not_testable_not_tested_null() -> None:
    label, reason = classify_profile(
        profile_class="primary",
        component_means=[1.0, 1.0, 1.0],
        sign_fractions=[1.0, 1.0, 1.0],
        adjusted_iut_pvalue=0.001,
        all_cost_checks_valid=False,
        all_metric_gates_valid=True,
        invalid_reason="p5_reference_invalid",
    )

    assert label == "not_testable_validity"
    assert reason == "p5_reference_invalid"


def test_p5_cost_gap_above_tolerance_is_invalid() -> None:
    assert cost_gap_valid(0.05)
    assert not cost_gap_valid(0.050001)


def test_frozen_noncentral_t_mde_values_are_reproduced() -> None:
    component = minimum_detectable_dz(10, 0.05 / 6, 0.80)
    complete_independence = minimum_detectable_dz(
        10, 0.05 / 6, 0.80 ** (1 / 3)
    )

    assert component == pytest.approx(1.2235, abs=0.001)
    assert complete_independence == pytest.approx(1.4655, abs=0.001)


def test_confirmatory_match_requires_every_substantive_criterion() -> None:
    match, _ = classify_profile(
        profile_class="primary",
        component_means=[0.2, 0.3, 0.4],
        sign_fractions=[0.8, 0.9, 1.0],
        adjusted_iut_pvalue=0.049,
        all_cost_checks_valid=True,
        all_metric_gates_valid=True,
    )
    null, _ = classify_profile(
        profile_class="primary",
        component_means=[0.2, 0.3, 0.4],
        sign_fractions=[0.8, 0.7, 1.0],
        adjusted_iut_pvalue=0.049,
        all_cost_checks_valid=True,
        all_metric_gates_valid=True,
    )

    assert match == "confirmatory_match"
    assert null == "tested_null"
