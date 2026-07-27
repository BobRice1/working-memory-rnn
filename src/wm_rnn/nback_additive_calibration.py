"""Frozen additive-cost calibration utilities for the N-back branch.

This module is deliberately outcome-runner agnostic.  It defines the
pre-registered profiles and random-seed addressing, performs branch-restricted
additive calibration against synthetic or externally collected costs, and
validates held-out paired sequence-CE costs.  It does not load checkpoints or
construct perturbation operators.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


CALIBRATION_TARGET = 0.050
CALIBRATION_TOLERANCE = 0.0025
MAXIMUM_BISECTION_ITERATIONS = 12
COST_CHECK_SEQUENCES = 1024
COST_CHECK_DRAWS = 10_000
COST_CHECK_BAND = (0.040, 0.060)
MAXIMUM_COST_CHECK_HALF_WIDTH = 0.005
MAXIMUM_P5_COST_GAP = 0.005

P2_VECTOR_SEEDS = (3101, 3102, 3103)
P5_REPLICATE_LABELS = (4101, 4102, 4103)
CONFIRMATORY_PROFILE_IDS = (1, 4, 7, 9, 10, 12)


@dataclass(frozen=True)
class OperatorProfile:
    """One immutable row of the registered operator-profile manifest."""

    profile_id: int
    operator: str
    variant: str
    branch: str
    ordered_grid: tuple[float, ...]
    profile_class: str


MULTIPLICATIVE_BELOW = (1.0, 0.975, 0.95, 0.90)
MULTIPLICATIVE_ABOVE = (1.0, 1.025, 1.05, 1.10, 1.15, 1.20)
P2_ABOVE = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15)
P7_BELOW = (1.0, 0.95, 0.90, 0.80)
P7_ABOVE = (1.0, 1.05, 1.10, 1.25)
P5_ABOVE = (0.0, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10)

OPERATOR_PROFILES = (
    OperatorProfile(
        0,
        "synaptic_drive_gain",
        "bias_outside",
        "below",
        MULTIPLICATIVE_BELOW,
        "descriptive",
    ),
    OperatorProfile(
        1,
        "synaptic_drive_gain",
        "bias_outside",
        "above",
        MULTIPLICATIVE_ABOVE,
        "confirmatory",
    ),
    OperatorProfile(
        2,
        "synaptic_drive_gain",
        "bias_inside",
        "below",
        MULTIPLICATIVE_BELOW,
        "descriptive",
    ),
    OperatorProfile(
        3,
        "synaptic_drive_gain",
        "bias_inside",
        "above",
        MULTIPLICATIVE_ABOVE,
        "descriptive",
    ),
    OperatorProfile(
        4,
        "heterogeneous_drive_gain",
        "bias_outside",
        "above",
        P2_ABOVE,
        "confirmatory",
    ),
    OperatorProfile(
        5,
        "heterogeneous_drive_gain",
        "bias_inside",
        "above",
        P2_ABOVE,
        "descriptive",
    ),
    OperatorProfile(
        6,
        "sensory_input_gain",
        "six_sensory_channels",
        "below",
        MULTIPLICATIVE_BELOW,
        "descriptive",
    ),
    OperatorProfile(
        7,
        "sensory_input_gain",
        "six_sensory_channels",
        "above",
        MULTIPLICATIVE_ABOVE,
        "confirmatory",
    ),
    OperatorProfile(
        8,
        "recurrent_gain",
        "weights_only",
        "below",
        MULTIPLICATIVE_BELOW,
        "descriptive",
    ),
    OperatorProfile(
        9,
        "recurrent_gain",
        "weights_only",
        "above",
        MULTIPLICATIVE_ABOVE,
        "confirmatory",
    ),
    OperatorProfile(
        10,
        "state_persistence",
        "carried_state_only",
        "below",
        MULTIPLICATIVE_BELOW,
        "confirmatory",
    ),
    OperatorProfile(
        11,
        "state_persistence",
        "carried_state_only",
        "above",
        MULTIPLICATIVE_ABOVE,
        "descriptive",
    ),
    OperatorProfile(
        12,
        "time_constant",
        "conserved_integrator",
        "below",
        P7_BELOW,
        "confirmatory",
    ),
    OperatorProfile(
        13,
        "time_constant",
        "conserved_integrator",
        "above",
        P7_ABOVE,
        "descriptive",
    ),
    OperatorProfile(
        14,
        "gaussian_state_noise",
        "generic_control",
        "above",
        P5_ABOVE,
        "comparator",
    ),
)

PROFILE_BY_ID = {
    profile.profile_id: profile for profile in OPERATOR_PROFILES
}


@dataclass(frozen=True)
class TaskBank:
    """One frozen task bank and its allowed axes."""

    base: int
    n_batches: int
    condition_codes: tuple[int, ...]


TASK_BANKS = {
    "calibration": TaskBank(132000000, 4, (0,)),
    "cost_check": TaskBank(133000000, 8, (0,)),
    "confirmatory": TaskBank(134000000, 8, (0, 1)),
}

P5_PHASE_CODES = {
    "calibration": 0,
    "cost_check": 1,
    "confirmatory": 2,
}


@dataclass(frozen=True)
class AdditiveCostSummary:
    """Paired sequence-level additive CE summary."""

    n_sequences: int
    baseline_mean_ce: float
    perturbed_mean_ce: float
    additive_cost: float
    paired_difference_sd: float


@dataclass(frozen=True)
class CalibrationEvaluation:
    """One cached strength evaluation in execution order."""

    strength: float
    additive_cost: float


@dataclass(frozen=True)
class AdditiveCalibrationResult:
    """Result of one registered branch-grid plus bisection calibration."""

    selected_strength: float | None
    achieved_additive_cost: float | None
    converged: bool
    n_iterations: int
    note: str
    bracket_lower: float | None
    bracket_upper: float | None
    evaluations: tuple[CalibrationEvaluation, ...]


@dataclass(frozen=True)
class AdditiveBootstrapResult:
    """Paired-bootstrap point estimate, interval, and full audit draws."""

    point_cost: float
    ci_lower: float
    ci_upper: float
    ci_half_width: float
    draws: int
    bootstrap_seed: int
    estimates: np.ndarray


@dataclass(frozen=True)
class AdditiveCostCheck:
    """All registered held-out additive-cost validity gates."""

    point_cost: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    bootstrap_ci_half_width: float
    cost_band_valid: bool
    cost_precision_valid: bool
    p5_point_cost: float | None
    p5_cost_gap: float | None
    p5_cost_gap_valid: bool | None
    p5_reference_valid: bool | None
    cost_match_valid: bool
    invalid_reasons: tuple[str, ...]


def _validate_checkpoint_ordinal(checkpoint_ordinal: int) -> int:
    ordinal = int(checkpoint_ordinal)
    if not 0 <= ordinal < 10:
        raise ValueError("checkpoint_ordinal must lie in [0, 9]")
    return ordinal


def task_seed(
    phase: str,
    checkpoint_ordinal: int,
    condition_code: int,
    batch_index: int,
) -> int:
    """Return one frozen calibration, cost-check, or outcome task seed."""
    if phase not in TASK_BANKS:
        raise ValueError(f"unknown task phase: {phase}")
    bank = TASK_BANKS[phase]
    ordinal = _validate_checkpoint_ordinal(checkpoint_ordinal)
    condition = int(condition_code)
    batch = int(batch_index)
    if condition not in bank.condition_codes:
        raise ValueError("condition_code is not allowed for this phase")
    if not 0 <= batch < bank.n_batches:
        raise ValueError("batch_index is outside the registered bank")
    return bank.base + 10_000 * ordinal + 1_000 * condition + batch


def p2_vector_seed(replicate_ordinal: int) -> int:
    """Return the frozen literal P2 seed, invariant over all other axes."""
    ordinal = int(replicate_ordinal)
    if not 0 <= ordinal < len(P2_VECTOR_SEEDS):
        raise ValueError("P2 replicate_ordinal must lie in [0, 2]")
    return P2_VECTOR_SEEDS[ordinal]


def p5_generator_seed(
    phase: str,
    checkpoint_ordinal: int,
    condition_code: int,
    replicate_ordinal: int,
    batch_index: int,
) -> int:
    """Return the frozen collision-free P5 generator seed."""
    if phase not in P5_PHASE_CODES:
        raise ValueError(f"unknown P5 phase: {phase}")
    bank = TASK_BANKS[phase]
    ordinal = _validate_checkpoint_ordinal(checkpoint_ordinal)
    condition = int(condition_code)
    replicate = int(replicate_ordinal)
    batch = int(batch_index)
    if condition not in bank.condition_codes:
        raise ValueError("condition_code is not allowed for this phase")
    if not 0 <= replicate < len(P5_REPLICATE_LABELS):
        raise ValueError("P5 replicate_ordinal must lie in [0, 2]")
    if not 0 <= batch < bank.n_batches:
        raise ValueError("batch_index is outside the registered bank")
    return (
        138000000
        + 1_000_000 * ordinal
        + 100_000 * P5_PHASE_CODES[phase]
        + 10_000 * condition
        + 1_000 * replicate
        + batch
    )


def cost_check_bootstrap_seed(
    checkpoint_ordinal: int,
    profile_id: int,
) -> int:
    """Return the frozen held-out paired-bootstrap seed."""
    ordinal = _validate_checkpoint_ordinal(checkpoint_ordinal)
    resolved_profile = int(profile_id)
    if resolved_profile not in PROFILE_BY_ID:
        raise ValueError("profile_id is outside the registered manifest")
    return 136000000 + 1_000 * ordinal + resolved_profile


def _validated_ce_vector(
    values: np.ndarray | Sequence[float],
    *,
    name: str,
) -> np.ndarray:
    units = np.asarray(values, dtype=np.float64)
    if units.ndim != 1 or units.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(units)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(units < 0.0):
        raise ValueError(f"{name} must contain only non-negative CE values")
    return units


def average_replicate_sequence_units(
    replicate_units: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Average exactly three P2/P5 replicates without increasing sequence n."""
    units = np.asarray(replicate_units, dtype=np.float64)
    if units.ndim != 2 or units.shape[0] != 3 or units.shape[1] == 0:
        raise ValueError(
            "replicate_units must have shape [3, n_sequences] with n > 0"
        )
    if not np.all(np.isfinite(units)):
        raise ValueError("all three replicate vectors must be finite")
    if np.any(units < 0.0):
        raise ValueError("replicate CE values must be non-negative")
    return np.mean(units, axis=0)


def summarize_additive_cost(
    baseline_units: np.ndarray | Sequence[float],
    perturbed_units: np.ndarray | Sequence[float],
) -> AdditiveCostSummary:
    """Summarize paired additive sequence CE, never a proportional ratio."""
    baseline = _validated_ce_vector(baseline_units, name="baseline_units")
    perturbed = _validated_ce_vector(
        perturbed_units, name="perturbed_units"
    )
    if perturbed.shape != baseline.shape:
        raise ValueError("baseline and perturbed CE vectors must be paired")
    differences = perturbed - baseline
    return AdditiveCostSummary(
        n_sequences=int(baseline.size),
        baseline_mean_ce=float(np.mean(baseline)),
        perturbed_mean_ce=float(np.mean(perturbed)),
        additive_cost=float(np.mean(differences)),
        paired_difference_sd=(
            float(np.std(differences, ddof=1))
            if differences.size > 1
            else 0.0
        ),
    )


def _validated_ordered_grid(
    ordered_grid: Sequence[float],
) -> tuple[float, ...]:
    grid = tuple(float(value) for value in ordered_grid)
    if len(grid) < 2:
        raise ValueError("ordered_grid must contain at least two strengths")
    if not np.all(np.isfinite(grid)):
        raise ValueError("ordered_grid strengths must be finite")
    if len(set(grid)) != len(grid):
        raise ValueError("ordered_grid strengths must be unique")
    return grid


def calibrate_additive_branch(
    cost_function: Callable[[float], float],
    ordered_grid: Sequence[float],
    *,
    target: float = CALIBRATION_TARGET,
    tolerance: float = CALIBRATION_TOLERANCE,
    max_iterations: int = MAXIMUM_BISECTION_ITERATIONS,
) -> AdditiveCalibrationResult:
    """Run the cached, monotonic, no-extrapolation additive calibration."""
    grid = _validated_ordered_grid(ordered_grid)
    resolved_target = float(target)
    resolved_tolerance = float(tolerance)
    if not np.isfinite(resolved_target):
        raise ValueError("target must be finite")
    if not np.isfinite(resolved_tolerance) or resolved_tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if int(max_iterations) <= 0:
        raise ValueError("max_iterations must be positive")

    cache: dict[float, float] = {}
    trace: list[CalibrationEvaluation] = []

    def evaluate(strength: float) -> float:
        resolved_strength = float(strength)
        if resolved_strength not in cache:
            cost = float(cost_function(resolved_strength))
            if not np.isfinite(cost):
                raise ValueError("cost_function must return finite costs")
            cache[resolved_strength] = cost
            trace.append(
                CalibrationEvaluation(
                    strength=resolved_strength,
                    additive_cost=cost,
                )
            )
        return cache[resolved_strength]

    grid_costs = tuple(evaluate(strength) for strength in grid)

    def result(
        *,
        strength: float | None,
        cost: float | None,
        converged: bool,
        iterations: int,
        note: str,
        bracket: tuple[float, float] | None = None,
    ) -> AdditiveCalibrationResult:
        return AdditiveCalibrationResult(
            selected_strength=strength,
            achieved_additive_cost=cost,
            converged=converged,
            n_iterations=iterations,
            note=note,
            bracket_lower=(
                min(bracket) if bracket is not None else None
            ),
            bracket_upper=(
                max(bracket) if bracket is not None else None
            ),
            evaluations=tuple(trace),
        )

    if any(
        later < earlier - resolved_tolerance
        for earlier, later in zip(grid_costs, grid_costs[1:])
    ):
        return result(
            strength=None,
            cost=None,
            converged=False,
            iterations=0,
            note="nonmonotone_calibration",
        )

    within = [
        index
        for index, cost in enumerate(grid_costs)
        if abs(cost - resolved_target) <= resolved_tolerance
    ]
    if within:
        selected = min(
            within,
            key=lambda index: (
                abs(grid_costs[index] - resolved_target),
                index,
            ),
        )
        strength = grid[selected]
        return result(
            strength=strength,
            cost=grid_costs[selected],
            converged=True,
            iterations=0,
            note="grid_point_within_tolerance",
            bracket=(strength, strength),
        )

    bracket_index: int | None = None
    for index in range(len(grid) - 1):
        first = grid_costs[index] - resolved_target
        second = grid_costs[index + 1] - resolved_target
        if first * second < 0.0:
            bracket_index = index
            break
    if bracket_index is None:
        return result(
            strength=None,
            cost=None,
            converged=False,
            iterations=0,
            note="unreachable_matched_strength",
        )

    first_strength = grid[bracket_index]
    second_strength = grid[bracket_index + 1]
    first_cost = grid_costs[bracket_index]
    second_cost = grid_costs[bracket_index + 1]
    original_bracket = (first_strength, second_strength)

    for iteration in range(1, int(max_iterations) + 1):
        midpoint = 0.5 * (first_strength + second_strength)
        midpoint_cost = evaluate(midpoint)
        endpoint_min = min(first_cost, second_cost)
        endpoint_max = max(first_cost, second_cost)
        if (
            midpoint_cost < endpoint_min - resolved_tolerance
            or midpoint_cost > endpoint_max + resolved_tolerance
        ):
            return result(
                strength=None,
                cost=None,
                converged=False,
                iterations=iteration,
                note="nonmonotone_calibration",
                bracket=original_bracket,
            )
        if abs(midpoint_cost - resolved_target) <= resolved_tolerance:
            return result(
                strength=midpoint,
                cost=midpoint_cost,
                converged=True,
                iterations=iteration,
                note="bisection_converged",
                bracket=original_bracket,
            )
        first_difference = first_cost - resolved_target
        midpoint_difference = midpoint_cost - resolved_target
        if first_difference * midpoint_difference <= 0.0:
            second_strength = midpoint
            second_cost = midpoint_cost
        else:
            first_strength = midpoint
            first_cost = midpoint_cost

    return result(
        strength=None,
        cost=None,
        converged=False,
        iterations=int(max_iterations),
        note="calibration_numerical_failure",
        bracket=original_bracket,
    )


def calibrate_profile(
    profile: OperatorProfile,
    cost_function: Callable[[float], float],
) -> AdditiveCalibrationResult:
    """Calibrate one immutable registered profile."""
    if PROFILE_BY_ID.get(profile.profile_id) != profile:
        raise ValueError("profile is not an exact registered manifest row")
    return calibrate_additive_branch(cost_function, profile.ordered_grid)


def paired_bootstrap_additive_cost(
    baseline_units: np.ndarray | Sequence[float],
    perturbed_units: np.ndarray | Sequence[float],
    *,
    draws: int = COST_CHECK_DRAWS,
    bootstrap_seed: int,
    chunk_size: int = 256,
) -> AdditiveBootstrapResult:
    """Bootstrap complete paired sequence differences in bounded memory."""
    summary = summarize_additive_cost(baseline_units, perturbed_units)
    baseline = _validated_ce_vector(baseline_units, name="baseline_units")
    perturbed = _validated_ce_vector(
        perturbed_units, name="perturbed_units"
    )
    resolved_draws = int(draws)
    resolved_chunk = int(chunk_size)
    if resolved_draws <= 0 or resolved_chunk <= 0:
        raise ValueError("draws and chunk_size must be positive")
    differences = perturbed - baseline
    rng = np.random.default_rng(int(bootstrap_seed))
    estimates = np.empty(resolved_draws, dtype=np.float64)
    for start in range(0, resolved_draws, resolved_chunk):
        stop = min(start + resolved_chunk, resolved_draws)
        indices = rng.integers(
            0,
            summary.n_sequences,
            size=(stop - start, summary.n_sequences),
            dtype=np.int32,
        )
        estimates[start:stop] = np.mean(differences[indices], axis=1)
    lower, upper = np.quantile(
        estimates, [0.025, 0.975], method="linear"
    )
    return AdditiveBootstrapResult(
        point_cost=summary.additive_cost,
        ci_lower=float(lower),
        ci_upper=float(upper),
        ci_half_width=0.5 * float(upper - lower),
        draws=resolved_draws,
        bootstrap_seed=int(bootstrap_seed),
        estimates=estimates,
    )


def validate_heldout_additive_cost(
    baseline_units: np.ndarray | Sequence[float],
    perturbed_units: np.ndarray | Sequence[float],
    *,
    bootstrap_seed: int,
    p5_point_cost: float | None = None,
    p5_reference_valid: bool | None = None,
    required_sequences: int = COST_CHECK_SEQUENCES,
    draws: int = COST_CHECK_DRAWS,
    chunk_size: int = 256,
) -> AdditiveCostCheck:
    """Apply the registered band, precision, P5-reference, and gap gates."""
    baseline = _validated_ce_vector(baseline_units, name="baseline_units")
    perturbed = _validated_ce_vector(
        perturbed_units, name="perturbed_units"
    )
    if baseline.shape != perturbed.shape:
        raise ValueError("held-out CE vectors must be paired")
    if baseline.size != int(required_sequences):
        raise ValueError(
            f"held-out cost check requires exactly {required_sequences} "
            "sequences"
        )
    bootstrap = paired_bootstrap_additive_cost(
        baseline,
        perturbed,
        draws=draws,
        bootstrap_seed=bootstrap_seed,
        chunk_size=chunk_size,
    )
    lower_band, upper_band = COST_CHECK_BAND
    band_valid = lower_band <= bootstrap.point_cost <= upper_band
    precision_valid = (
        bootstrap.ci_half_width <= MAXIMUM_COST_CHECK_HALF_WIDTH
    )

    p5_cost: float | None = None
    p5_gap: float | None = None
    p5_gap_valid: bool | None = None
    if (p5_point_cost is None) != (p5_reference_valid is None):
        raise ValueError(
            "p5_point_cost and p5_reference_valid must be supplied together"
        )
    if p5_point_cost is not None:
        p5_cost = float(p5_point_cost)
        if not np.isfinite(p5_cost):
            raise ValueError("p5_point_cost must be finite")
        p5_gap = bootstrap.point_cost - p5_cost
        p5_gap_valid = abs(p5_gap) <= MAXIMUM_P5_COST_GAP
    if p5_reference_valid is not None and not isinstance(
        p5_reference_valid, (bool, np.bool_)
    ):
        raise ValueError("p5_reference_valid must be boolean or None")

    reasons: list[str] = []
    if not band_valid:
        reasons.append("cost_band_failure")
    if not precision_valid:
        reasons.append("cost_precision_failure")
    if p5_reference_valid is False:
        reasons.append("p5_reference_invalid")
    if p5_gap_valid is False:
        reasons.append("p5_cost_mismatch")
    return AdditiveCostCheck(
        point_cost=bootstrap.point_cost,
        bootstrap_ci_lower=bootstrap.ci_lower,
        bootstrap_ci_upper=bootstrap.ci_upper,
        bootstrap_ci_half_width=bootstrap.ci_half_width,
        cost_band_valid=band_valid,
        cost_precision_valid=precision_valid,
        p5_point_cost=p5_cost,
        p5_cost_gap=p5_gap,
        p5_cost_gap_valid=p5_gap_valid,
        p5_reference_valid=p5_reference_valid,
        cost_match_valid=not reasons,
        invalid_reasons=tuple(reasons),
    )
