"""Tests for pre-registered psilocybin-signature outcome metrics."""

from __future__ import annotations

import numpy as np
import pytest

from wm_rnn.perturbation_metrics import (
    activation_slope_and_saturation,
    assess_settling_validity,
    delay_decoding_error,
    distractor_drift_and_recovery,
    marginal_state_entropy,
    response_geometry_measures,
    signed_circular_error,
    time_to_threshold,
)


def _settling_metrics(
    response_errors: np.ndarray,
    *,
    vector_length: np.ndarray | None = None,
    dwell_steps: int = 1,
) -> dict[str, object]:
    response_errors = np.asarray(response_errors, dtype=float)
    if response_errors.ndim == 1:
        response_errors = response_errors[:, None]
    if vector_length is None:
        vector_length = np.ones_like(response_errors)
    return time_to_threshold(
        response_errors,
        np.asarray(vector_length, dtype=float),
        slice(0, response_errors.shape[0]),
        baseline_vector_length=1.0,
        threshold_degrees=15.0,
        dwell_steps=dwell_steps,
        amplitude_fraction=0.5,
    )


def test_settling_returns_first_sustained_crossing_step() -> None:
    metrics = _settling_metrics(
        np.array([30.0, 14.0, 10.0, 9.0, 8.0]),
        dwell_steps=3,
    )

    np.testing.assert_allclose(metrics["settling_steps"], np.array([1.0]))
    assert metrics["median_settling_steps"] == pytest.approx(1.0)
    assert metrics["restricted_mean_settling_steps"] == pytest.approx(1.0)
    assert metrics["fraction_settled"] == pytest.approx(1.0)
    assert metrics["failure_rate"] == pytest.approx(0.0)


def test_settling_never_crosses_returns_nan_and_full_failure() -> None:
    metrics = _settling_metrics(np.array([30.0, 25.0, 20.0, 16.0]))

    assert np.isnan(np.asarray(metrics["settling_steps"])[0])
    assert np.isnan(float(metrics["median_settling_steps"]))
    assert metrics["restricted_mean_settling_steps"] == pytest.approx(4.0)
    assert metrics["fraction_settled"] == pytest.approx(0.0)
    assert metrics["failure_rate"] == pytest.approx(1.0)


def test_transient_dip_shorter_than_dwell_does_not_settle() -> None:
    metrics = _settling_metrics(
        np.array([30.0, 10.0, 9.0, 20.0, 8.0, 18.0]),
        dwell_steps=3,
    )

    assert np.isnan(np.asarray(metrics["settling_steps"])[0])
    assert metrics["fraction_settled"] == pytest.approx(0.0)


def test_silent_output_below_error_threshold_does_not_settle() -> None:
    errors = np.array([10.0, 8.0, 5.0])
    silent_vector_length = np.full((3, 1), 0.49)

    metrics = _settling_metrics(
        errors,
        vector_length=silent_vector_length,
        dwell_steps=1,
    )

    assert np.isnan(np.asarray(metrics["settling_steps"])[0])
    assert metrics["fraction_settled"] == pytest.approx(0.0)


def test_restricted_mean_exposes_failures_that_bias_conditional_median() -> None:
    # Columns are trials. All trials settle in the first cell, at steps 1, 2,
    # and 5. In the second cell the slowest trial becomes a failure and is
    # capped at the six-step response-window length.
    all_settle = np.array(
        [
            [30.0, 30.0, 30.0],
            [10.0, 30.0, 30.0],
            [10.0, 10.0, 30.0],
            [10.0, 10.0, 30.0],
            [10.0, 10.0, 30.0],
            [10.0, 10.0, 10.0],
        ]
    )
    slowest_fails = all_settle.copy()
    slowest_fails[-1, -1] = 30.0

    before = _settling_metrics(all_settle)
    after = _settling_metrics(slowest_fails)

    assert after["median_settling_steps"] < before["median_settling_steps"]
    assert (
        after["restricted_mean_settling_steps"]
        > before["restricted_mean_settling_steps"]
    )
    assert after["failure_rate"] == pytest.approx(
        1.0 - float(after["fraction_settled"])
    )


def test_low_fraction_settled_is_scored_as_failure_rate_not_latency() -> None:
    validity = assess_settling_validity(
        fixation_accuracy=0.98,
        baseline_fraction_settled=0.90,
        perturbed_fraction_settled=0.40,
    )

    assert validity["latency_valid"] is False
    assert validity["settling_score"] == "NA"
    assert validity["latency_score"] == "NA"
    assert validity["response_failure_score"] == "eligible"
    assert validity["settling_validity_reason"] == "low_fraction_settled"
    assert validity["primary_response_outcome"] == "failure_rate"


def test_fixation_failure_marks_settling_na_without_removing_raw_metrics() -> None:
    metrics = _settling_metrics(np.array([30.0, 10.0, 8.0]))
    metrics.update(
        assess_settling_validity(
            fixation_accuracy=0.89,
            baseline_fraction_settled=1.0,
            perturbed_fraction_settled=float(metrics["fraction_settled"]),
        )
    )

    assert metrics["latency_valid"] is False
    assert metrics["settling_score"] == "NA"
    assert metrics["latency_score"] == "NA"
    assert metrics["response_failure_score"] == "NA"
    assert metrics["settling_validity_reason"] == "fixation_failure"
    assert metrics["primary_response_outcome"] == "non_settling_only"
    np.testing.assert_allclose(metrics["settling_steps"], np.array([1.0]))
    assert metrics["fraction_settled"] == pytest.approx(1.0)


def test_valid_settling_cell_uses_latency_as_primary_outcome() -> None:
    validity = assess_settling_validity(
        fixation_accuracy=0.90,
        baseline_fraction_settled=0.50,
        perturbed_fraction_settled=0.50,
    )

    assert validity["latency_valid"] is True
    assert validity["settling_score"] == "eligible"
    assert validity["latency_score"] == "eligible"
    assert validity["response_failure_score"] == "eligible"
    assert validity["settling_validity_reason"] is None
    assert validity["primary_response_outcome"] == "latency"


def test_delay_decoding_error_recovers_known_circular_code() -> None:
    angles = np.array([0.0, np.pi / 2.0, 2.0 * np.pi - 0.1])
    circular_code = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    hidden_states = np.repeat(circular_code[None, :, :], repeats=3, axis=0)

    metrics = delay_decoding_error(
        hidden_states,
        angles,
        np.eye(2),
        slice(1, 3),
    )

    assert metrics["mean_error_degrees"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["median_error_degrees"] == pytest.approx(0.0, abs=1e-10)


def test_distractor_attraction_has_target_distractor_and_away_anchors() -> None:
    target = np.array([0.0])
    distractor = np.array([np.pi / 2.0])

    flat = distractor_drift_and_recovery(
        np.zeros((4, 1)),
        target,
        distractor,
        slice(0, 2),
        slice(2, 4),
    )
    assert flat["peak_attraction"] == pytest.approx(0.0)
    assert np.isnan(flat["recovered_fraction"])

    reaches_distractor = distractor_drift_and_recovery(
        np.array([[0.0], [np.pi / 2.0], [np.pi / 4.0], [np.pi / 8.0]]),
        target,
        distractor,
        slice(0, 2),
        slice(2, 4),
    )
    assert reaches_distractor["peak_target_error_degrees"] == pytest.approx(90.0)
    assert reaches_distractor["peak_attraction"] == pytest.approx(1.0)
    assert reaches_distractor["end_attraction"] == pytest.approx(0.25)
    assert reaches_distractor["recovered_fraction"] == pytest.approx(0.75)

    moves_away = distractor_drift_and_recovery(
        np.full((2, 1), -np.pi / 4.0),
        target,
        distractor,
        slice(0, 1),
        slice(1, 2),
    )
    assert moves_away["peak_attraction"] == pytest.approx(-0.5)


def test_end_of_delay_distractor_marks_recovery_unavailable() -> None:
    metrics = distractor_drift_and_recovery(
        np.array([[0.0], [np.pi / 4.0], [np.pi / 2.0]]),
        np.array([0.0]),
        np.array([np.pi / 2.0]),
        slice(1, 3),
        slice(3, 3),
    )

    assert metrics["peak_attraction"] == pytest.approx(1.0)
    assert metrics["end_attraction"] == pytest.approx(1.0)
    assert np.isnan(metrics["recovered_fraction"])


def test_signed_circular_error_retains_sign_and_wraps() -> None:
    decoded = np.deg2rad(np.array([10.0, 350.0, 1.0, 359.0, 180.0]))
    target = np.deg2rad(np.array([0.0, 0.0, 359.0, 1.0, 0.0]))

    errors = signed_circular_error(decoded, target)

    np.testing.assert_allclose(errors, np.array([10.0, -10.0, 2.0, -2.0, 180.0]))


def test_signed_circular_error_does_not_flip_near_negative_branch() -> None:
    errors = signed_circular_error(
        np.deg2rad(np.array([-180.0 + 1e-8])),
        np.array([0.0]),
    )

    np.testing.assert_allclose(errors, np.array([-180.0 + 1e-8]), atol=1e-10)


def test_response_geometry_bias_is_zero_for_symmetric_errors_and_positive_for_offset() -> None:
    symmetric = np.array(
        [
            [[-10.0, 10.0], [-10.0, 10.0]],
            [[-20.0, 20.0], [-20.0, 20.0]],
        ]
    ).reshape(2, 4)
    positive_offset = np.full((2, 4), 25.0)
    predictions = np.zeros((2, 4, 8))
    predictions[:, :, 0] = 1.0

    unbiased = response_geometry_measures(symmetric, predictions, slice(0, 2))
    biased = response_geometry_measures(
        positive_offset,
        predictions,
        slice(0, 2),
    )

    assert unbiased["mean_signed_error_degrees"] == pytest.approx(0.0)
    assert unbiased["circular_bias_degrees"] == pytest.approx(0.0, abs=1e-12)
    assert biased["mean_signed_error_degrees"] == pytest.approx(25.0)
    assert biased["circular_bias_degrees"] == pytest.approx(25.0)


def test_response_geometry_marks_undefined_circular_bias_nan() -> None:
    errors = np.array([[0.0, 180.0]])
    predictions = np.zeros((1, 2, 8))
    predictions[:, :, 0] = 1.0

    metrics = response_geometry_measures(errors, predictions, slice(0, 1))

    assert np.isnan(metrics["circular_bias_degrees"])


def test_activation_saturation_is_one_for_all_point_99_states() -> None:
    metrics = activation_slope_and_saturation(np.full((3, 4, 5), 0.99))

    assert metrics["saturation_fraction"] == pytest.approx(1.0)
    assert metrics["mean_tanh_slope"] == pytest.approx(1.0 - 0.99**2)


def test_marginal_entropy_is_higher_for_broad_than_concentrated_states() -> None:
    rng = np.random.default_rng(0)
    concentrated = rng.normal(0.0, 0.01, size=(20, 100, 3))
    broad = rng.uniform(-0.95, 0.95, size=(20, 100, 3))

    narrow_metrics = marginal_state_entropy(concentrated, bins=32)
    broad_metrics = marginal_state_entropy(broad, bins=32)

    assert np.asarray(narrow_metrics["per_unit_entropy"]).shape == (3,)
    assert np.asarray(broad_metrics["per_unit_entropy"]).shape == (3,)
    assert broad_metrics["mean_entropy"] > narrow_metrics["mean_entropy"]


def test_marginal_entropy_rejects_values_outside_tanh_support() -> None:
    with pytest.raises(ValueError, match="tanh support"):
        marginal_state_entropy(np.array([[-1.01, 0.0, 1.01]]))


def test_settling_rejects_non_finite_values() -> None:
    errors = np.array([[10.0], [np.nan]])
    vector_length = np.ones_like(errors)

    with pytest.raises(ValueError, match="finite"):
        time_to_threshold(errors, vector_length, slice(0, 2), 1.0)


def test_time_windows_must_use_consecutive_steps() -> None:
    with pytest.raises(ValueError, match="consecutive"):
        time_to_threshold(
            np.zeros((4, 1)),
            np.ones((4, 1)),
            slice(0, 4, 2),
            1.0,
            dwell_steps=1,
        )


@pytest.mark.parametrize(
    ("function", "args", "match"),
    [
        (
            time_to_threshold,
            (
                np.zeros((4, 2)),
                np.zeros((4, 3)),
                slice(0, 4),
                1.0,
            ),
            "shape",
        ),
        (
            time_to_threshold,
            (
                np.zeros((4, 2)),
                np.ones((4, 2)),
                slice(0, 4),
                0.0,
            ),
            "baseline_vector_length",
        ),
        (
            delay_decoding_error,
            (
                np.zeros((4, 2, 3)),
                np.zeros(2),
                np.zeros((4, 2)),
                slice(0, 4),
            ),
            "decoder_weights",
        ),
        (
            distractor_drift_and_recovery,
            (
                np.zeros((4, 2)),
                np.zeros(3),
                np.ones(2),
                slice(0, 2),
                slice(2, 4),
            ),
            "trial",
        ),
        (
            response_geometry_measures,
            (
                np.zeros((4, 2)),
                np.zeros((4, 3, 8)),
                slice(0, 4),
            ),
            "shape",
        ),
        (
            marginal_state_entropy,
            (np.zeros((4, 2, 3)), 1),
            "bins",
        ),
    ],
)
def test_interpretation_critical_inputs_are_validated(
    function: object,
    args: tuple[object, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        function(*args)  # type: ignore[operator]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"fixation_accuracy": np.nan}, "finite"),
        ({"fixation_accuracy": 1.01}, "fixation_accuracy"),
        ({"baseline_fraction_settled": -0.01}, "baseline_fraction_settled"),
        ({"perturbed_fraction_settled": 1.01}, "perturbed_fraction_settled"),
    ],
)
def test_settling_validity_rejects_invalid_proportions(
    kwargs: dict[str, float],
    match: str,
) -> None:
    valid = {
        "fixation_accuracy": 0.95,
        "baseline_fraction_settled": 0.80,
        "perturbed_fraction_settled": 0.80,
    }
    valid.update(kwargs)

    with pytest.raises(ValueError, match=match):
        assess_settling_validity(**valid)
