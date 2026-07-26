"""Pure NumPy outcome metrics for the perturbation-signature experiment.

The functions in this module deliberately do not load models or read files.
Callers are responsible for supplying the pre-registered time windows and for
keeping decoder fitting, metric-reference, calibration, cost-check, and final
evaluation trials disjoint.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_TWO_PI = 2.0 * np.pi
_TANH_BOUND_TOLERANCE = 1e-7


def _as_nonempty_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _window(array: np.ndarray, window: slice, name: str) -> np.ndarray:
    if not isinstance(window, slice):
        raise TypeError(f"{name} must be a slice")
    if window.step not in (None, 1):
        raise ValueError(f"{name} must select consecutive time steps")
    selected = array[window]
    if selected.shape[0] == 0:
        raise ValueError(f"{name} selects no time steps")
    return selected


def _bounded_tanh_states(hidden_states: np.ndarray) -> np.ndarray:
    hidden = _as_nonempty_array(hidden_states, "hidden_states")
    if np.any(hidden < -1.0 - _TANH_BOUND_TOLERANCE) or np.any(
        hidden > 1.0 + _TANH_BOUND_TOLERANCE
    ):
        raise ValueError("hidden_states must lie within the tanh support [-1, 1]")
    return np.clip(hidden, -1.0, 1.0)


def _signed_wrapped_radians(angle_difference: np.ndarray) -> np.ndarray:
    """Wrap radians to (-pi, pi], retaining +pi at the branch cut."""
    wrapped = (angle_difference + np.pi) % _TWO_PI - np.pi
    at_negative_branch = wrapped == -np.pi
    return np.where(at_negative_branch, np.pi, wrapped)


def time_to_threshold(
    decoded_error_degrees: np.ndarray,
    vector_length: np.ndarray,
    response_slice: slice,
    baseline_vector_length: float,
    threshold_degrees: float = 15.0,
    dwell_steps: int = 3,
    amplitude_fraction: float = 0.5,
) -> dict[str, Any]:
    """Return per-trial settling steps and the pre-registered D5 summaries.

    Settling is the first zero-based response step at which both the angular
    error and response-amplitude criteria hold for ``dwell_steps`` consecutive
    samples. Unsettled trials remain NaN in ``settling_steps`` and are assigned
    the response-window length in the restricted mean.
    """
    errors = _as_nonempty_array(decoded_error_degrees, "decoded_error_degrees")
    amplitudes = _as_nonempty_array(vector_length, "vector_length")
    if errors.ndim != 2:
        raise ValueError("decoded_error_degrees must have shape [time, trials]")
    if amplitudes.shape != errors.shape:
        raise ValueError(
            "vector_length shape must match decoded_error_degrees shape"
        )
    if not np.isfinite(baseline_vector_length) or baseline_vector_length <= 0.0:
        raise ValueError("baseline_vector_length must be finite and positive")
    if not np.isfinite(threshold_degrees) or threshold_degrees <= 0.0:
        raise ValueError("threshold_degrees must be finite and positive")
    if not isinstance(dwell_steps, (int, np.integer)) or dwell_steps <= 0:
        raise ValueError("dwell_steps must be a positive integer")
    if not np.isfinite(amplitude_fraction) or amplitude_fraction < 0.0:
        raise ValueError("amplitude_fraction must be finite and non-negative")

    response_errors = _window(errors, response_slice, "response_slice")
    response_amplitudes = _window(amplitudes, response_slice, "response_slice")
    response_steps, n_trials = response_errors.shape
    amplitude_threshold = amplitude_fraction * baseline_vector_length
    eligible = (
        np.isfinite(response_errors)
        & np.isfinite(response_amplitudes)
        & (np.abs(response_errors) < threshold_degrees)
        & (response_amplitudes >= amplitude_threshold)
    )

    settling_steps = np.full(n_trials, np.nan, dtype=np.float64)
    if dwell_steps <= response_steps:
        for start in range(response_steps - dwell_steps + 1):
            sustained = np.all(eligible[start : start + dwell_steps], axis=0)
            newly_settled = sustained & np.isnan(settling_steps)
            settling_steps[newly_settled] = float(start)

    settled = np.isfinite(settling_steps)
    fraction_settled = float(np.mean(settled))
    conditional_median = (
        float(np.median(settling_steps[settled])) if np.any(settled) else float("nan")
    )
    capped_steps = np.where(settled, settling_steps, float(response_steps))

    return {
        "settling_steps": settling_steps,
        "median_settling_steps": conditional_median,
        "restricted_mean_settling_steps": float(np.mean(capped_steps)),
        "fraction_settled": fraction_settled,
        "failure_rate": float(1.0 - fraction_settled),
    }


def assess_settling_validity(
    fixation_accuracy: float,
    baseline_fraction_settled: float,
    perturbed_fraction_settled: float,
    fixation_floor: float = 0.90,
    fraction_settled_floor: float = 0.50,
) -> dict[str, Any]:
    """Apply the D9 fixation and response-failure interpretation gates."""
    values = (
        fixation_accuracy,
        baseline_fraction_settled,
        perturbed_fraction_settled,
        fixation_floor,
        fraction_settled_floor,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("settling-validity inputs must all be finite")
    if not 0.0 <= fixation_accuracy <= 1.0:
        raise ValueError("fixation_accuracy must lie in [0, 1]")
    if not 0.0 <= baseline_fraction_settled <= 1.0:
        raise ValueError("baseline_fraction_settled must lie in [0, 1]")
    if not 0.0 <= perturbed_fraction_settled <= 1.0:
        raise ValueError("perturbed_fraction_settled must lie in [0, 1]")
    if not 0.0 <= fixation_floor <= 1.0:
        raise ValueError("fixation_floor must lie in [0, 1]")
    if not 0.0 <= fraction_settled_floor <= 1.0:
        raise ValueError("fraction_settled_floor must lie in [0, 1]")

    if fixation_accuracy < fixation_floor:
        return {
            "latency_valid": False,
            "settling_score": "NA",
            "latency_score": "NA",
            "response_failure_score": "NA",
            "settling_validity_reason": "fixation_failure",
            "primary_response_outcome": "non_settling_only",
        }
    if (
        baseline_fraction_settled < fraction_settled_floor
        or perturbed_fraction_settled < fraction_settled_floor
    ):
        return {
            "latency_valid": False,
            "settling_score": "NA",
            "latency_score": "NA",
            "response_failure_score": "eligible",
            "settling_validity_reason": "low_fraction_settled",
            "primary_response_outcome": "failure_rate",
        }
    return {
        "latency_valid": True,
        "settling_score": "eligible",
        "latency_score": "eligible",
        "response_failure_score": "eligible",
        "settling_validity_reason": None,
        "primary_response_outcome": "latency",
    }


def delay_decoding_error(
    hidden_states: np.ndarray,
    angles: np.ndarray,
    decoder_weights: np.ndarray,
    window: slice,
) -> dict[str, float]:
    """Return mean and median absolute circular decode error in degrees."""
    hidden = _as_nonempty_array(hidden_states, "hidden_states")
    targets = _as_nonempty_array(angles, "angles")
    weights = _as_nonempty_array(decoder_weights, "decoder_weights")
    if hidden.ndim != 3:
        raise ValueError("hidden_states must have shape [time, trials, units]")
    if targets.ndim != 1 or targets.shape[0] != hidden.shape[1]:
        raise ValueError("angles must have shape [trials]")
    if weights.shape != (hidden.shape[2], 2):
        raise ValueError("decoder_weights must have shape [units, 2]")

    selected = _window(hidden, window, "window")
    decoded_components = selected @ weights
    decoded_angles = np.arctan2(
        decoded_components[..., 1], decoded_components[..., 0]
    )
    signed_error = _signed_wrapped_radians(decoded_angles - targets[None, :])
    absolute_error_degrees = np.degrees(np.abs(signed_error))
    return {
        "mean_error_degrees": float(np.mean(absolute_error_degrees)),
        "median_error_degrees": float(np.median(absolute_error_degrees)),
    }


def distractor_drift_and_recovery(
    decoded_angles: np.ndarray,
    target_angles: np.ndarray,
    distractor_angles: np.ndarray,
    distractor_slice: slice,
    post_distractor_slice: slice,
) -> dict[str, float]:
    """Quantify signed distractor attraction and subsequent recovery.

    Attraction is the signed target-to-decoded displacement divided by the
    signed shortest target-to-distractor arc. The reported peak is the largest
    absolute group-mean excursion, with its sign retained, so movement directly
    away from the distractor remains negative rather than being clipped to zero.
    """
    decoded = _as_nonempty_array(decoded_angles, "decoded_angles")
    targets = _as_nonempty_array(target_angles, "target_angles")
    distractors = _as_nonempty_array(distractor_angles, "distractor_angles")
    if decoded.ndim != 2:
        raise ValueError("decoded_angles must have shape [time, trials]")
    expected = (decoded.shape[1],)
    if targets.shape != expected or distractors.shape != expected:
        raise ValueError("target_angles and distractor_angles must have shape [trials]")

    during = _window(decoded, distractor_slice, "distractor_slice")
    after = _window(decoded, post_distractor_slice, "post_distractor_slice")
    trajectory = np.concatenate((during, after), axis=0)
    distractor_arc = _signed_wrapped_radians(distractors - targets)
    if np.any(np.isclose(distractor_arc, 0.0, atol=1e-12)):
        raise ValueError("target and distractor angles must differ on every trial")

    displacement = _signed_wrapped_radians(trajectory - targets[None, :])
    attraction = displacement / distractor_arc[None, :]
    mean_attraction_trajectory = np.mean(attraction, axis=1)
    peak_index = int(np.argmax(np.abs(mean_attraction_trajectory)))
    peak_attraction = float(mean_attraction_trajectory[peak_index])
    end_attraction = float(np.mean(attraction[-1]))

    mean_target_error_degrees = np.mean(np.degrees(np.abs(displacement)), axis=1)
    peak_target_error = float(np.max(mean_target_error_degrees))
    if np.isclose(peak_attraction, 0.0, atol=1e-12):
        recovered_fraction = float("nan")
    else:
        recovered_fraction = float((peak_attraction - end_attraction) / peak_attraction)

    return {
        "peak_target_error_degrees": peak_target_error,
        "peak_attraction": peak_attraction,
        "end_attraction": end_attraction,
        "recovered_fraction": recovered_fraction,
        "distractor_peak_drift_degrees": peak_target_error,
        "distractor_peak_attraction_fraction": peak_attraction,
        "distractor_end_attraction_fraction": end_attraction,
        "distractor_recovery_fraction": recovered_fraction,
    }


def response_geometry_measures(
    signed_error_degrees: np.ndarray,
    population_predictions: np.ndarray,
    response_slice: slice,
) -> dict[str, float]:
    """Return exploratory directional-error and output-commitment measures."""
    signed_errors = _as_nonempty_array(signed_error_degrees, "signed_error_degrees")
    populations = _as_nonempty_array(
        population_predictions, "population_predictions"
    )
    if signed_errors.ndim != 2:
        raise ValueError("signed_error_degrees must have shape [time, trials]")
    if populations.ndim != 3 or populations.shape[:2] != signed_errors.shape:
        raise ValueError(
            "population_predictions must have shape [time, trials, n_tuned_units]"
        )
    if populations.shape[2] < 3:
        raise ValueError("population_predictions must contain at least 3 tuned units")

    selected_errors = _window(signed_errors, response_slice, "response_slice")
    selected_populations = _window(
        populations, response_slice, "response_slice"
    )
    preferred_angles = np.linspace(
        0.0, _TWO_PI, selected_populations.shape[2], endpoint=False
    )
    x_component = np.sum(
        selected_populations * np.cos(preferred_angles), axis=-1
    )
    y_component = np.sum(
        selected_populations * np.sin(preferred_angles), axis=-1
    )
    vector_lengths = np.hypot(x_component, y_component)
    trial_vector_lengths = np.mean(vector_lengths, axis=0)
    mean_vector_length = float(np.mean(trial_vector_lengths))
    vector_length_cv = (
        float(np.std(trial_vector_lengths, ddof=0) / mean_vector_length)
        if not np.isclose(mean_vector_length, 0.0)
        else float("nan")
    )

    error_radians = np.radians(selected_errors)
    circular_resultant = np.mean(np.exp(1j * error_radians))
    circular_bias = (
        float(np.degrees(np.angle(circular_resultant)))
        if np.abs(circular_resultant) > 1e-12
        else float("nan")
    )

    return {
        "mean_signed_error_degrees": float(np.mean(selected_errors)),
        "circular_bias_degrees": circular_bias,
        "mean_vector_length": mean_vector_length,
        "vector_length_cv": vector_length_cv,
    }


def signed_circular_error(
    decoded_angles: np.ndarray,
    target_angles: np.ndarray,
) -> np.ndarray:
    """Return signed wrapped error in degrees in the interval (-180, 180]."""
    decoded = np.asarray(decoded_angles, dtype=np.float64)
    targets = np.asarray(target_angles, dtype=np.float64)
    if not np.all(np.isfinite(decoded)) or not np.all(np.isfinite(targets)):
        raise ValueError("decoded_angles and target_angles must contain only finite values")
    try:
        difference = decoded - targets
    except ValueError as error:
        raise ValueError("decoded_angles and target_angles must be broadcastable") from error
    return np.degrees(_signed_wrapped_radians(difference))


def activation_slope_and_saturation(hidden_states: np.ndarray) -> dict[str, float]:
    """Return mean tanh-slope proxy and the fixed |h| >= 0.95 saturation rate."""
    hidden = _bounded_tanh_states(hidden_states)
    mean_tanh_slope = float(np.mean(1.0 - np.square(hidden)))
    return {
        "mean_tanh_slope": mean_tanh_slope,
        "mean_activation_slope": mean_tanh_slope,
        "saturation_fraction": float(np.mean(np.abs(hidden) >= 0.95)),
    }


def marginal_state_entropy(
    hidden_states: np.ndarray,
    bins: int = 64,
) -> dict[str, Any]:
    """Estimate per-unit fixed-bin differential entropy on [-1, 1].

    All axes except the final unit axis are pooled. The histogram support and
    bin widths are fixed, making estimates comparable across matched windows.
    """
    hidden = _bounded_tanh_states(hidden_states)
    if hidden.ndim < 2:
        raise ValueError("hidden_states must have a final unit axis")
    if not isinstance(bins, (int, np.integer)) or bins <= 1:
        raise ValueError("bins must be an integer greater than 1")
    samples_by_unit = hidden.reshape(-1, hidden.shape[-1])
    bin_width = 2.0 / bins
    per_unit_entropy = np.empty(hidden.shape[-1], dtype=np.float64)
    for unit in range(hidden.shape[-1]):
        counts, _ = np.histogram(
            samples_by_unit[:, unit], bins=bins, range=(-1.0, 1.0)
        )
        probabilities = counts.astype(np.float64) / counts.sum()
        nonzero = probabilities > 0.0
        per_unit_entropy[unit] = -np.sum(
            probabilities[nonzero]
            * np.log(probabilities[nonzero] / bin_width)
        )

    return {
        "per_unit_entropy": per_unit_entropy,
        "mean_entropy": float(np.mean(per_unit_entropy)),
    }
