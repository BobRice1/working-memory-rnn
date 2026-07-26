"""Matched proportional-cost calibration and held-out precision utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    """One branch-restricted matched-cost calibration result."""

    strength: float
    achieved_proportional_cost: float
    converged: bool
    n_iterations: int
    note: str
    bracket_lower: float | None
    bracket_upper: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-ready record."""
        return asdict(self)


@dataclass(frozen=True)
class CostMatchCheck:
    """Held-out D7 cost, precision, and optional P5 pairwise checks."""

    proportional_clean_cost: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    bootstrap_ci_half_width: float
    cost_precision_valid: bool
    cost_band_valid: bool
    cost_match_valid: bool
    p5_cost_gap: float | None
    p5_cost_gap_valid: bool | None
    invalid_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-ready record."""
        return asdict(self)


def proportional_cost(
    baseline_mean_error: float,
    perturbed_mean_error: float,
) -> float:
    """Return checkpoint-normalized proportional clean-task cost."""
    baseline = float(baseline_mean_error)
    perturbed = float(perturbed_mean_error)
    if not np.isfinite(baseline) or baseline <= 0.0:
        raise ValueError("baseline_mean_error must be finite and positive")
    if not np.isfinite(perturbed) or perturbed < 0.0:
        raise ValueError(
            "perturbed_mean_error must be finite and non-negative"
        )
    return (perturbed - baseline) / baseline


def required_cost_check_trials(
    baseline_mean_error: float,
    trial_error_sd: float,
    *,
    proportional_half_width: float = 0.10,
    z_value: float = 1.96,
) -> float:
    """Return the conservative D7 independent-means sample requirement."""
    mean = float(baseline_mean_error)
    sd = float(trial_error_sd)
    half_width = float(proportional_half_width)
    if not np.isfinite(mean) or mean <= 0.0:
        raise ValueError("baseline_mean_error must be finite and positive")
    if not np.isfinite(sd) or sd < 0.0:
        raise ValueError("trial_error_sd must be finite and non-negative")
    if not np.isfinite(half_width) or half_width <= 0.0:
        raise ValueError("proportional_half_width must be finite and positive")
    if not np.isfinite(z_value) or z_value <= 0.0:
        raise ValueError("z_value must be finite and positive")
    return 2.0 * (z_value * sd / (half_width * mean)) ** 2


def round_up_to_batch(value: float, batch_size: int = 64) -> int:
    """Round a positive sample requirement up to complete batches."""
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("value must be finite and non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return int(np.ceil(value / batch_size) * batch_size)


def _grid_values(
    cost_function: Callable[[float], float],
    strength_grid: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    strengths = np.asarray(strength_grid, dtype=np.float64)
    if strengths.ndim != 1 or strengths.size == 0:
        raise ValueError("strength_grid must be a non-empty one-dimensional grid")
    if not np.all(np.isfinite(strengths)):
        raise ValueError("strength_grid must contain finite values")
    if len(np.unique(strengths)) != strengths.size:
        raise ValueError("strength_grid values must be unique")
    strengths = np.sort(strengths)
    costs = np.asarray(
        [float(cost_function(float(value))) for value in strengths],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(costs)):
        raise ValueError("cost_function must return finite values")
    return strengths, costs


def calibrate_strength(
    cost_function: Callable[[float], float],
    strength_grid: Sequence[float],
    *,
    target_proportional_cost: float = 0.30,
    tolerance: float = 0.01,
    max_iterations: int = 12,
) -> CalibrationResult:
    """Calibrate within a supplied monotone branch without extrapolation."""
    target = float(target_proportional_cost)
    if not np.isfinite(target):
        raise ValueError("target_proportional_cost must be finite")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    strengths, costs = _grid_values(cost_function, strength_grid)
    differences = costs - target
    exact = np.flatnonzero(np.abs(differences) <= tolerance)
    if exact.size:
        index = int(exact[np.argmin(np.abs(differences[exact]))])
        return CalibrationResult(
            strength=float(strengths[index]),
            achieved_proportional_cost=float(costs[index]),
            converged=True,
            n_iterations=0,
            note="grid_point_within_tolerance",
            bracket_lower=float(strengths[index]),
            bracket_upper=float(strengths[index]),
        )

    bracket: tuple[int, int] | None = None
    for index in range(strengths.size - 1):
        if differences[index] * differences[index + 1] < 0.0:
            bracket = (index, index + 1)
            break
    if bracket is None:
        closest = int(np.argmin(np.abs(differences)))
        return CalibrationResult(
            strength=float(strengths[closest]),
            achieved_proportional_cost=float(costs[closest]),
            converged=False,
            n_iterations=0,
            note="target_unreachable_no_extrapolation",
            bracket_lower=None,
            bracket_upper=None,
        )

    lower_index, upper_index = bracket
    lower_strength = float(strengths[lower_index])
    upper_strength = float(strengths[upper_index])
    lower_difference = float(differences[lower_index])
    best_strength = (
        lower_strength
        if abs(lower_difference) <= abs(float(differences[upper_index]))
        else upper_strength
    )
    best_cost = float(cost_function(best_strength))

    for iteration in range(1, max_iterations + 1):
        midpoint = 0.5 * (lower_strength + upper_strength)
        midpoint_cost = float(cost_function(midpoint))
        if not np.isfinite(midpoint_cost):
            raise ValueError("cost_function must return finite values")
        midpoint_difference = midpoint_cost - target
        if abs(midpoint_difference) < abs(best_cost - target):
            best_strength, best_cost = midpoint, midpoint_cost
        if abs(midpoint_difference) <= tolerance:
            return CalibrationResult(
                strength=midpoint,
                achieved_proportional_cost=midpoint_cost,
                converged=True,
                n_iterations=iteration,
                note="bisection_converged",
                bracket_lower=float(strengths[lower_index]),
                bracket_upper=float(strengths[upper_index]),
            )
        if lower_difference * midpoint_difference <= 0.0:
            upper_strength = midpoint
        else:
            lower_strength = midpoint
            lower_difference = midpoint_difference

    return CalibrationResult(
        strength=best_strength,
        achieved_proportional_cost=best_cost,
        converged=abs(best_cost - target) <= tolerance,
        n_iterations=max_iterations,
        note=(
            "bisection_converged"
            if abs(best_cost - target) <= tolerance
            else "maximum_iterations_reached"
        ),
        bracket_lower=float(strengths[lower_index]),
        bracket_upper=float(strengths[upper_index]),
    )


def calibrate_bidirectional(
    cost_function: Callable[[float], float],
    strength_grid: Sequence[float],
    *,
    neutral_strength: float = 1.0,
    target_proportional_cost: float = 0.30,
    tolerance: float = 0.01,
    max_iterations: int = 12,
) -> dict[str, CalibrationResult]:
    """Calibrate below- and above-neutral branches independently."""
    strengths = np.asarray(strength_grid, dtype=np.float64)
    if neutral_strength not in strengths:
        raise ValueError("neutral_strength must be present in strength_grid")
    return {
        "below_neutral": calibrate_strength(
            cost_function,
            strengths[strengths <= neutral_strength],
            target_proportional_cost=target_proportional_cost,
            tolerance=tolerance,
            max_iterations=max_iterations,
        ),
        "above_neutral": calibrate_strength(
            cost_function,
            strengths[strengths >= neutral_strength],
            target_proportional_cost=target_proportional_cost,
            tolerance=tolerance,
            max_iterations=max_iterations,
        ),
    }


def paired_bootstrap_proportional_cost(
    baseline_trial_errors: np.ndarray,
    perturbed_trial_errors: np.ndarray,
    *,
    draws: int = 10_000,
    bootstrap_seed: int = 202607250,
) -> tuple[float, float, float, float]:
    """Return point cost, paired-bootstrap bounds, and interval half-width."""
    baseline = np.asarray(baseline_trial_errors, dtype=np.float64)
    perturbed = np.asarray(perturbed_trial_errors, dtype=np.float64)
    if baseline.ndim != 1 or perturbed.shape != baseline.shape:
        raise ValueError("paired trial errors must be one-dimensional and shape matched")
    if baseline.size == 0:
        raise ValueError("paired trial errors must not be empty")
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(perturbed)):
        raise ValueError("paired trial errors must contain only finite values")
    if np.mean(baseline) <= 0.0:
        raise ValueError("baseline trial-error mean must be positive")
    if draws <= 0:
        raise ValueError("draws must be positive")

    point = proportional_cost(float(np.mean(baseline)), float(np.mean(perturbed)))
    rng = np.random.default_rng(int(bootstrap_seed))
    estimates = np.empty(draws, dtype=np.float64)
    chunk_size = min(256, draws)
    for start in range(0, draws, chunk_size):
        stop = min(start + chunk_size, draws)
        indices = rng.integers(
            0, baseline.size, size=(stop - start, baseline.size)
        )
        baseline_means = np.mean(baseline[indices], axis=1)
        perturbed_means = np.mean(perturbed[indices], axis=1)
        estimates[start:stop] = (
            perturbed_means - baseline_means
        ) / baseline_means
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    half_width = 0.5 * float(upper - lower)
    return point, float(lower), float(upper), half_width


def validate_cost_match(
    baseline_trial_errors: np.ndarray,
    perturbed_trial_errors: np.ndarray,
    *,
    p5_proportional_cost: float | None = None,
    band: tuple[float, float] = (0.20, 0.40),
    maximum_half_width: float = 0.10,
    maximum_p5_gap: float = 0.05,
    draws: int = 10_000,
    bootstrap_seed: int = 202607250,
) -> CostMatchCheck:
    """Apply the held-out D7 band, precision, and pairwise P5 gates."""
    lower_band, upper_band = map(float, band)
    if not lower_band <= upper_band:
        raise ValueError("band lower bound must not exceed upper bound")
    point, lower, upper, half_width = paired_bootstrap_proportional_cost(
        baseline_trial_errors,
        perturbed_trial_errors,
        draws=draws,
        bootstrap_seed=bootstrap_seed,
    )
    precision_valid = half_width <= maximum_half_width
    band_valid = lower_band <= point <= upper_band
    p5_gap: float | None = None
    p5_gap_valid: bool | None = None
    if p5_proportional_cost is not None:
        if not np.isfinite(p5_proportional_cost):
            raise ValueError("p5_proportional_cost must be finite")
        p5_gap = point - float(p5_proportional_cost)
        p5_gap_valid = abs(p5_gap) <= maximum_p5_gap

    if not band_valid:
        reason = "cost_band_failure"
    elif not precision_valid:
        reason = "cost_precision_failure"
    elif p5_gap_valid is False:
        reason = "p5_cost_mismatch"
    else:
        reason = None
    return CostMatchCheck(
        proportional_clean_cost=point,
        bootstrap_ci_lower=lower,
        bootstrap_ci_upper=upper,
        bootstrap_ci_half_width=half_width,
        cost_precision_valid=precision_valid,
        cost_band_valid=band_valid,
        cost_match_valid=reason is None,
        p5_cost_gap=p5_gap,
        p5_cost_gap_valid=p5_gap_valid,
        invalid_reason=reason,
    )
