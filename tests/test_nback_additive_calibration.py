"""Synthetic tests for frozen additive N-back calibration utilities."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import wm_rnn.nback_additive_calibration as calibration_module
from wm_rnn.nback_additive_calibration import (
    CONFIRMATORY_PROFILE_IDS,
    OPERATOR_PROFILES,
    P2_VECTOR_SEEDS,
    P5_REPLICATE_LABELS,
    average_replicate_sequence_units,
    calibrate_additive_branch,
    cost_check_bootstrap_seed,
    p2_vector_seed,
    p5_generator_seed,
    paired_bootstrap_additive_cost,
    summarize_additive_cost,
    task_seed,
    validate_heldout_additive_cost,
)


def test_profile_manifest_is_exact_and_excludes_p3b() -> None:
    assert [profile.profile_id for profile in OPERATOR_PROFILES] == list(
        range(15)
    )
    assert CONFIRMATORY_PROFILE_IDS == (1, 4, 7, 9, 10, 12)
    assert [profile.profile_class for profile in OPERATOR_PROFILES].count(
        "confirmatory"
    ) == 6
    assert OPERATOR_PROFILES[6].variant == "six_sensory_channels"
    assert OPERATOR_PROFILES[10].branch == "below"
    assert OPERATOR_PROFILES[12].ordered_grid == (
        1.0,
        0.95,
        0.90,
        0.80,
    )
    assert OPERATOR_PROFILES[14].profile_class == "comparator"
    assert OPERATOR_PROFILES[14].ordered_grid == (
        0.0,
        0.01,
        0.02,
        0.035,
        0.05,
        0.075,
        0.10,
    )
    assert not any(
        "distractor" in profile.operator for profile in OPERATOR_PROFILES
    )


def test_frozen_seed_maps_are_exact_unique_and_disjoint() -> None:
    task_seeds = []
    p5_seeds = []
    for checkpoint in range(10):
        for phase, conditions, batches in (
            ("calibration", (0,), 4),
            ("cost_check", (0,), 8),
            ("confirmatory", (0, 1), 8),
        ):
            for condition in conditions:
                for batch in range(batches):
                    task_seeds.append(
                        task_seed(phase, checkpoint, condition, batch)
                    )
                    for replicate in range(3):
                        p5_seeds.append(
                            p5_generator_seed(
                                phase,
                                checkpoint,
                                condition,
                                replicate,
                                batch,
                            )
                        )
    assert len(task_seeds) == len(set(task_seeds))
    assert len(p5_seeds) == len(set(p5_seeds))
    assert set(task_seeds).isdisjoint(p5_seeds)
    assert task_seed("calibration", 9, 0, 3) == 132090003
    assert task_seed("confirmatory", 9, 1, 7) == 134091007
    assert (
        p5_generator_seed("confirmatory", 9, 1, 2, 7)
        == 147212007
    )
    bootstrap = {
        cost_check_bootstrap_seed(checkpoint, profile)
        for checkpoint in range(10)
        for profile in range(15)
    }
    assert len(bootstrap) == 150
    assert set(bootstrap).isdisjoint(task_seeds)
    assert set(bootstrap).isdisjoint(p5_seeds)


def test_p2_uses_literal_invariant_seeds() -> None:
    assert tuple(p2_vector_seed(index) for index in range(3)) == (
        P2_VECTOR_SEEDS
    )
    assert P2_VECTOR_SEEDS == (3101, 3102, 3103)
    assert P5_REPLICATE_LABELS == (4101, 4102, 4103)
    with pytest.raises(ValueError, match="P2"):
        p2_vector_seed(3)


def test_additive_cost_and_sequencewise_replicate_average() -> None:
    baseline = np.asarray([0.01, 0.02, 0.03, 0.04])
    replicates = np.asarray(
        [
            baseline + 0.04,
            baseline + 0.05,
            baseline + 0.06,
        ]
    )

    averaged = average_replicate_sequence_units(replicates)
    summary = summarize_additive_cost(baseline, averaged)

    assert averaged.shape == (4,)
    np.testing.assert_allclose(averaged, baseline + 0.05)
    assert summary.n_sequences == 4
    assert summary.baseline_mean_ce == pytest.approx(0.025)
    assert summary.perturbed_mean_ce == pytest.approx(0.075)
    assert summary.additive_cost == pytest.approx(0.05)
    assert summary.paired_difference_sd == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(ValueError, match=r"\[3, n_sequences\]"):
        average_replicate_sequence_units(replicates[:2])
    contaminated = replicates.copy()
    contaminated[1, 0] = np.nan
    with pytest.raises(ValueError, match="all three"):
        average_replicate_sequence_units(contaminated)


def test_grid_selection_tolerates_registered_small_inversion() -> None:
    costs = {1.0: 0.0, 1.1: 0.030, 1.2: 0.029, 1.3: 0.051}
    result = calibrate_additive_branch(
        costs.__getitem__,
        tuple(costs),
    )

    assert result.converged
    assert result.selected_strength == 1.3
    assert result.note == "grid_point_within_tolerance"
    assert result.n_iterations == 0


def test_hard_grid_inversion_is_nonmonotone() -> None:
    costs = {1.0: 0.0, 1.1: 0.030, 1.2: 0.020, 1.3: 0.060}
    result = calibrate_additive_branch(costs.__getitem__, tuple(costs))

    assert not result.converged
    assert result.selected_strength is None
    assert result.note == "nonmonotone_calibration"


def test_bisection_converges_and_caches_every_strength() -> None:
    calls: Counter[float] = Counter()

    def cost(strength: float) -> float:
        calls[strength] += 1
        return 0.08 * strength

    result = calibrate_additive_branch(
        cost,
        (0.0, 0.5, 1.0),
        target=0.05,
        tolerance=1e-8,
    )

    assert result.converged
    assert result.selected_strength == pytest.approx(0.625)
    assert result.achieved_additive_cost == pytest.approx(0.05)
    assert result.note == "bisection_converged"
    assert result.n_iterations == 2
    assert all(count == 1 for count in calls.values())
    assert len(result.evaluations) == len(calls)


def test_midpoint_violation_is_nonmonotone() -> None:
    def cost(strength: float) -> float:
        lookup = {0.0: 0.0, 1.0: 0.10, 0.5: 0.20}
        return lookup[strength]

    result = calibrate_additive_branch(
        cost,
        (0.0, 1.0),
        tolerance=0.0025,
    )

    assert not result.converged
    assert result.note == "nonmonotone_calibration"
    assert result.n_iterations == 1


def test_unreachable_target_never_extrapolates() -> None:
    observed: list[float] = []

    def cost(strength: float) -> float:
        observed.append(strength)
        return 0.01 * strength

    result = calibrate_additive_branch(cost, (0.0, 1.0, 2.0))

    assert not result.converged
    assert result.note == "unreachable_matched_strength"
    assert result.selected_strength is None
    assert observed == [0.0, 1.0, 2.0]


def test_iteration_exhaustion_is_registered_numerical_failure() -> None:
    result = calibrate_additive_branch(
        lambda strength: 0.10 * strength,
        (0.0, 1.0),
        target=0.053,
        tolerance=0.0001,
        max_iterations=1,
    )

    assert not result.converged
    assert result.note == "calibration_numerical_failure"
    assert result.n_iterations == 1
    assert result.selected_strength is None


def test_paired_bootstrap_is_deterministic_and_chunk_invariant() -> None:
    baseline = np.linspace(0.001, 0.01, 40)
    perturbed = baseline + np.linspace(0.03, 0.07, 40)

    first = paired_bootstrap_additive_cost(
        baseline,
        perturbed,
        draws=101,
        bootstrap_seed=71,
        chunk_size=1,
    )
    second = paired_bootstrap_additive_cost(
        baseline,
        perturbed,
        draws=101,
        bootstrap_seed=71,
        chunk_size=19,
    )

    assert first.point_cost == pytest.approx(0.05)
    np.testing.assert_array_equal(first.estimates, second.estimates)
    assert first.ci_lower == second.ci_lower
    assert first.ci_upper == second.ci_upper


def test_heldout_cost_check_applies_band_precision_and_p5_gates() -> None:
    baseline = np.full(1024, 0.01)
    valid_perturbed = baseline + 0.05
    valid = validate_heldout_additive_cost(
        baseline,
        valid_perturbed,
        bootstrap_seed=89,
        p5_point_cost=0.052,
        p5_reference_valid=True,
        draws=101,
    )
    assert valid.cost_match_valid
    assert valid.invalid_reasons == ()

    outside = validate_heldout_additive_cost(
        baseline,
        baseline + 0.061,
        bootstrap_seed=90,
        draws=101,
    )
    assert outside.invalid_reasons == ("cost_band_failure",)

    imprecise_baseline = np.full(1024, 1.0)
    variable_differences = np.tile([-0.15, 0.25], 512)
    imprecise = validate_heldout_additive_cost(
        imprecise_baseline,
        imprecise_baseline + variable_differences,
        bootstrap_seed=91,
        draws=1000,
    )
    assert not imprecise.cost_precision_valid
    assert "cost_precision_failure" in imprecise.invalid_reasons

    mismatch = validate_heldout_additive_cost(
        baseline,
        valid_perturbed,
        bootstrap_seed=92,
        p5_point_cost=0.04,
        p5_reference_valid=False,
        draws=101,
    )
    assert mismatch.p5_cost_gap == pytest.approx(0.01)
    assert mismatch.invalid_reasons == (
        "p5_reference_invalid",
        "p5_cost_mismatch",
    )
    with pytest.raises(ValueError, match="exactly 1024"):
        validate_heldout_additive_cost(
            baseline[:-1],
            valid_perturbed[:-1],
            bootstrap_seed=93,
            draws=10,
        )
    with pytest.raises(ValueError, match="supplied together"):
        validate_heldout_additive_cost(
            baseline,
            valid_perturbed,
            bootstrap_seed=94,
            p5_point_cost=0.05,
            draws=10,
        )


def test_module_does_not_import_or_call_proportional_calibration() -> None:
    source = Path(calibration_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = []
    called_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.append(node.func.attr)

    assert "wm_rnn.perturbation_calibration" not in imported_modules
    assert not any("proportional" in name for name in called_names)
