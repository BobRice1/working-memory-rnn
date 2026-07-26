"""Result-contingent P2 gain-assignment sensitivity utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


ASSIGNMENT_COLUMNS = [
    "family",
    "operator",
    "variant",
    "strength",
    "seed",
    "gain_vector_seed",
    "assignment_seed",
    "gain_in_strength_correlation",
    "gain_out_strength_correlation",
    "gain_total_strength_correlation",
    "delta_angular_error_degrees",
    "delta_restricted_mean_settling_steps",
    "distractor_difference_in_differences",
    "load_difference_in_differences",
    "mean_late_delay_entropy_change",
    "within_checkpoint_slope",
    "note",
]


def permute_gain_multiset(
    gain_vector: np.ndarray, assignment_seed: int
) -> np.ndarray:
    """Return a deterministic permutation preserving the multiset exactly."""
    gains = np.asarray(gain_vector, dtype=np.float64)
    if gains.ndim != 1 or gains.size == 0:
        raise ValueError("gain_vector must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(gains)) or np.any(gains <= 0.0):
        raise ValueError("gain_vector must contain finite positive values")
    rng = np.random.default_rng(int(assignment_seed))
    return gains[rng.permutation(gains.size)]


def recurrent_strengths(
    recurrent_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return absolute incoming, outgoing, and total strength per recurrent unit."""
    weights = np.asarray(recurrent_weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("recurrent_weights must be a square matrix [out, in]")
    if not np.all(np.isfinite(weights)):
        raise ValueError("recurrent_weights must contain only finite values")
    incoming = np.sum(np.abs(weights), axis=1)
    outgoing = np.sum(np.abs(weights), axis=0)
    return {
        "in_strength": incoming,
        "out_strength": outgoing,
        "total_strength": incoming + outgoing,
    }


def safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Return Pearson correlation, or NaN when either vector is constant."""
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("correlation inputs must be matched one-dimensional arrays")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("correlation inputs must contain only finite values")
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def gain_strength_correlations(
    gain_vector: np.ndarray, recurrent_weights: np.ndarray
) -> dict[str, float]:
    """Correlate one gain assignment with the three frozen strength statistics."""
    gains = np.asarray(gain_vector, dtype=np.float64)
    strengths = recurrent_strengths(recurrent_weights)
    if gains.shape != strengths["in_strength"].shape:
        raise ValueError("gain_vector length must match recurrent unit count")
    return {
        "gain_in_strength_correlation": safe_correlation(
            gains, strengths["in_strength"]
        ),
        "gain_out_strength_correlation": safe_correlation(
            gains, strengths["out_strength"]
        ),
        "gain_total_strength_correlation": safe_correlation(
            gains, strengths["total_strength"]
        ),
    }


def within_checkpoint_slope(
    alignment: np.ndarray,
    outcome: np.ndarray,
) -> float:
    """Return the OLS outcome-on-alignment slope within one checkpoint."""
    x = np.asarray(alignment, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError("alignment and outcome must be matched vectors of length >=2")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("alignment and outcome must contain only finite values")
    if np.std(x) == 0.0:
        raise ValueError("alignment must vary within checkpoint")
    return float(np.polyfit(x, y, 1)[0])


def unit_entropy_regression(
    entropy_change: np.ndarray,
    gain_vector: np.ndarray,
    recurrent_weights: np.ndarray,
) -> dict[str, float]:
    """Fit the descriptive unit-level entropy model with frozen predictors."""
    entropy = np.asarray(entropy_change, dtype=np.float64)
    gains = np.asarray(gain_vector, dtype=np.float64)
    strengths = recurrent_strengths(recurrent_weights)
    if entropy.shape != gains.shape or entropy.shape != strengths["in_strength"].shape:
        raise ValueError("unit-level arrays must have the same shape")
    design = np.column_stack(
        (
            np.ones(entropy.size),
            gains,
            strengths["in_strength"],
            strengths["out_strength"],
            strengths["total_strength"],
        )
    )
    coefficients, *_ = np.linalg.lstsq(design, entropy, rcond=None)
    return {
        "intercept": float(coefficients[0]),
        "gain_coefficient": float(coefficients[1]),
        "in_strength_coefficient": float(coefficients[2]),
        "out_strength_coefficient": float(coefficients[3]),
        "total_strength_coefficient": float(coefficients[4]),
    }


def write_assignment_csv(
    path: str | Path, rows: list[dict[str, Any]]
) -> Path:
    """Write assignment-ensemble rows using the frozen schema."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ASSIGNMENT_COLUMNS)
        writer.writeheader()
        writer.writerows(
            [{column: row.get(column, "") for column in ASSIGNMENT_COLUMNS} for row in rows]
        )
    return target
