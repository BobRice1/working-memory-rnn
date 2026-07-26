"""Frozen descriptive and confirmatory scoring for perturbation signatures."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import optimize, stats

from wm_rnn.perturbation_experiment import (
    compute_excess_constraints,
    condition_normalized_change,
)


PRIMARY_PROFILES = (
    ("synaptic_drive_gain", "bias_outside", "above_neutral"),
    ("heterogeneous_drive_gain", "bias_outside", "above_neutral"),
    ("sensory_input_gain", "tuned_only", "above_neutral"),
    ("recurrent_gain", "weights_only", "above_neutral"),
    ("state_persistence", "carried_state_only", "below_neutral"),
    ("time_constant", "conserved_integrator", "below_neutral"),
)

DESCRIPTIVE_PROFILES = (
    ("synaptic_drive_gain", "bias_outside", "below_neutral"),
    ("synaptic_drive_gain", "bias_inside", "above_neutral"),
    ("synaptic_drive_gain", "bias_inside", "below_neutral"),
    ("heterogeneous_drive_gain", "bias_inside", "above_neutral"),
    ("sensory_input_gain", "tuned_only", "below_neutral"),
    ("recurrent_gain", "weights_only", "below_neutral"),
    ("state_persistence", "carried_state_only", "above_neutral"),
    ("time_constant", "conserved_integrator", "above_neutral"),
    ("distractor_input_gain", "distractor_only", "matched_distractor"),
)

PROFILE_COLUMNS = [
    "family",
    "operator",
    "variant",
    "branch",
    "profile_class",
    "n_checkpoints",
    "mean_x1",
    "dz_x1",
    "sign_fraction_x1",
    "p_c1",
    "mean_x2",
    "dz_x2",
    "sign_fraction_x2",
    "p_c2",
    "mean_x3",
    "dz_x3",
    "sign_fraction_x3",
    "p_c3",
    "mean_x2_gap_adjusted",
    "mean_x3_gap_adjusted",
    "p_iut",
    "p_iut_holm",
    "all_cost_checks_valid",
    "all_metric_gates_valid",
    "max_abs_p5_cost_gap",
    "all_p5_gaps_valid",
    "strictest_holm_alpha",
    "component_mde_dz_80",
    "outcome_label",
    "invalid_reason",
]

SCORE_COLUMNS = [
    "operator",
    "variant",
    "branch",
    "settling_slowing",
    "response_failure",
    "retention_dependent",
    "distractor_selective",
    "load_dependent",
    "dose_ordered",
    "dynamics_differ_from_p5",
    "assignment_sensitive",
]


@dataclass(frozen=True)
class ComponentTest:
    """Checkpoint-level summary for one one-sided component test."""

    mean: float
    dz: float
    sign_fraction: float
    p_value: float
    n: int


def one_sided_component_test(values: np.ndarray) -> ComponentTest:
    """Test H0 mean <= 0 against H1 mean > 0 across checkpoints."""
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or sample.size < 2:
        raise ValueError("values must contain at least two checkpoints")
    if not np.all(np.isfinite(sample)):
        raise ValueError("values must contain only finite values")
    mean = float(np.mean(sample))
    sd = float(np.std(sample, ddof=1))
    if sd == 0.0:
        if mean > 0.0:
            statistic, p_value, dz = np.inf, 0.0, np.inf
        elif mean < 0.0:
            statistic, p_value, dz = -np.inf, 1.0, -np.inf
        else:
            statistic, p_value, dz = 0.0, 0.5, 0.0
    else:
        statistic = mean / (sd / np.sqrt(sample.size))
        p_value = float(stats.t.sf(statistic, df=sample.size - 1))
        dz = mean / sd
    return ComponentTest(
        mean=mean,
        dz=float(dz),
        sign_fraction=float(np.mean(sample > 0.0)),
        p_value=float(p_value),
        n=int(sample.size),
    )


def intersection_union_pvalue(component_pvalues: list[float]) -> float:
    """Return the IUT p-value, exactly the maximum component p-value."""
    values = np.asarray(component_pvalues, dtype=np.float64)
    if values.shape != (3,) or not np.all((0.0 <= values) & (values <= 1.0)):
        raise ValueError("exactly three component p-values in [0, 1] are required")
    return float(np.max(values))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm familywise-adjusted p-values in original order."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a non-empty one-dimensional list")
    if not np.all((0.0 <= values) & (values <= 1.0)):
        raise ValueError("p_values must lie in [0, 1]")
    order = np.argsort(values)
    adjusted_sorted = np.empty(values.size, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (values.size - rank) * values[index]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty(values.size, dtype=np.float64)
    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]
    return adjusted.astype(float).tolist()


def minimum_detectable_dz(
    n: int,
    alpha: float,
    power: float,
) -> float:
    """Reproduce the frozen exact noncentral-t power boundary."""
    if n < 2:
        raise ValueError("n must be at least 2")
    if not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("alpha and power must lie in (0, 1)")
    df = n - 1
    critical = stats.t.ppf(1.0 - alpha, df)
    return float(
        optimize.brentq(
            lambda dz: stats.nct.sf(
                critical, df, dz * np.sqrt(n)
            )
            - power,
            0.01,
            10.0,
        )
    )


def cost_gap_valid(p5_cost_gap: float, tolerance: float = 0.05) -> bool:
    """Apply the pairwise candidate-to-P5 D7 cost tolerance."""
    if not np.isfinite(p5_cost_gap):
        return False
    return bool(abs(float(p5_cost_gap)) <= tolerance)


def classify_profile(
    *,
    profile_class: str,
    component_means: list[float],
    sign_fractions: list[float],
    adjusted_iut_pvalue: float | None,
    all_cost_checks_valid: bool,
    all_metric_gates_valid: bool,
    invalid_reason: str | None = None,
) -> tuple[str, str | None]:
    """Assign exactly one frozen D10 outcome label."""
    if profile_class == "descriptive_only":
        return "descriptive_only", None
    if profile_class != "primary":
        raise ValueError("profile_class must be primary or descriptive_only")
    if not all_cost_checks_valid or not all_metric_gates_valid:
        return "not_testable_validity", invalid_reason or "validity_gate_failure"
    if adjusted_iut_pvalue is None:
        raise ValueError("a testable primary profile requires adjusted IUT p-value")
    substantive_match = (
        all(float(value) > 0.0 for value in component_means)
        and all(float(value) >= 0.80 for value in sign_fractions)
        and float(adjusted_iut_pvalue) < 0.05
    )
    return (
        ("confirmatory_match", None)
        if substantive_match
        else ("tested_null", None)
    )


def descriptive_cell(
    values: np.ndarray,
    *,
    predicted_positive: bool = True,
    valid_fraction: float = 1.0,
) -> str:
    """Apply the frozen yes/partial/no/NA descriptive screening rule."""
    sample = np.asarray(values, dtype=np.float64)
    if valid_fraction < 0.80 or sample.size < 2 or not np.all(np.isfinite(sample)):
        return "NA"
    directed = sample if predicted_positive else -sample
    sign_fraction = float(np.mean(directed > 0.0))
    if sign_fraction < 0.80:
        return "no"
    interval = stats.t.interval(
        0.95,
        df=sample.size - 1,
        loc=float(np.mean(directed)),
        scale=float(stats.sem(directed)),
    )
    return "yes" if interval[0] > 0.0 else "partial"


def gap_adjusted_intercept(values: np.ndarray, gaps: np.ndarray) -> float:
    """Return the pre-registered linear sensitivity intercept at zero cost gap."""
    outcomes = np.asarray(values, dtype=np.float64)
    covariate = np.asarray(gaps, dtype=np.float64)
    if outcomes.shape != covariate.shape or outcomes.ndim != 1:
        raise ValueError("values and gaps must be matched one-dimensional arrays")
    design = np.column_stack((np.ones(outcomes.size), covariate))
    coefficients, *_ = np.linalg.lstsq(design, outcomes, rcond=None)
    return float(coefficients[0])


def _coerce(value: str) -> Any:
    if value == "":
        return ""
    if value in {"True", "False"}:
        return value == "True"
    try:
        return float(value)
    except ValueError:
        return value


def read_csv_records(path: str | Path) -> list[dict[str, Any]]:
    """Read experiment CSV rows with basic numeric/bool coercion."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [
            {key: _coerce(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _branch_from_strength_kind(strength_kind: str) -> str | None:
    return {
        "matched_above": "above_neutral",
        "matched_below": "below_neutral",
        "matched_distractor": "matched_distractor",
    }.get(strength_kind)


def _aggregate_nested(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str, int, str], dict[str, float]]:
    """Average vector/noise replicates within checkpoint and condition."""
    grouped: dict[
        tuple[str, str, str, int, str], list[dict[str, Any]]
    ] = {}
    for row in rows:
        branch = _branch_from_strength_kind(str(row["strength_kind"]))
        if (
            branch is None
            or row["item_position"] != "pooled"
            or int(row["delay_steps"]) != 20
        ):
            continue
        key = (
            str(row["operator"]),
            str(row["variant"]),
            branch,
            int(row["seed"]),
            str(row["condition"]),
        )
        grouped.setdefault(key, []).append(row)
    output: dict[
        tuple[str, str, str, int, str], dict[str, float]
    ] = {}
    metric_names = (
        "baseline_mean_angular_error_degrees",
        "mean_angular_error_degrees",
        "baseline_restricted_mean_settling_steps",
        "restricted_mean_settling_steps",
        "latency_valid",
    )
    for key, members in grouped.items():
        output[key] = {
            metric: float(np.mean([float(row[metric]) for row in members]))
            for metric in metric_names
        }
    return output


def score_profile_rows(
    grid_rows: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build all 15 Family B profile records from matched rows."""
    aggregated = _aggregate_nested(grid_rows)
    seeds = sorted(
        {
            int(row["seed"])
            for row in grid_rows
            if row["family"] == "B"
        }
    )
    cost_lookup = {
        (
            str(row["operator"]),
            str(row["variant"]),
            str(row["branch"]),
            int(row["seed"]),
        ): row
        for row in cost_rows
        if row["family"] == "B"
    }
    p5_by_seed = {}
    for seed in seeds:
        key_base = (
            "gaussian_state_noise",
            "generic_control",
            "above_neutral",
            seed,
        )
        p5_by_seed[seed] = {
            condition: aggregated[key_base + (condition,)]
            for condition in (
                "load1_clean",
                "load2_clean",
                "load1_distractor",
            )
        }

    profiles = [
        (*profile, "primary") for profile in PRIMARY_PROFILES
    ] + [
        (*profile, "descriptive_only") for profile in DESCRIPTIVE_PROFILES
    ]
    records: list[dict[str, Any]] = []
    for operator, variant, branch, profile_class in profiles:
        required_keys = [
            (name, profile_variant, profile_branch, seed, condition)
            for name, profile_variant, profile_branch in (
                (operator, variant, branch),
                (
                    "gaussian_state_noise",
                    "generic_control",
                    "above_neutral",
                ),
            )
            for seed in seeds
            for condition in (
                "load1_clean",
                "load2_clean",
                "load1_distractor",
            )
        ]
        if any(key not in aggregated for key in required_keys):
            records.append(
                {
                    "family": "B",
                    "operator": operator,
                    "variant": variant,
                    "branch": branch,
                    "profile_class": profile_class,
                    "n_checkpoints": len(seeds),
                    "mean_x1": np.nan,
                    "dz_x1": np.nan,
                    "sign_fraction_x1": np.nan,
                    "p_c1": 1.0,
                    "mean_x2": np.nan,
                    "dz_x2": np.nan,
                    "sign_fraction_x2": np.nan,
                    "p_c2": 1.0,
                    "mean_x3": np.nan,
                    "dz_x3": np.nan,
                    "sign_fraction_x3": np.nan,
                    "p_c3": 1.0,
                    "mean_x2_gap_adjusted": np.nan,
                    "mean_x3_gap_adjusted": np.nan,
                    "p_iut": 1.0,
                    "p_iut_holm": "",
                    "all_cost_checks_valid": False,
                    "all_metric_gates_valid": False,
                    "max_abs_p5_cost_gap": np.nan,
                    "all_p5_gaps_valid": False,
                    "strictest_holm_alpha": 0.05 / 6,
                    "component_mde_dz_80": minimum_detectable_dz(
                        10, 0.05 / 6, 0.80
                    ),
                    "_component_means": [np.nan, np.nan, np.nan],
                    "_sign_fractions": [np.nan, np.nan, np.nan],
                    "_invalid_reason": "unreachable_matched_strength",
                }
            )
            continue
        x1_values: list[float] = []
        x2_values: list[float] = []
        x3_values: list[float] = []
        gaps: list[float] = []
        metric_valid = True
        cost_valid = True
        invalid_reasons: list[str] = []
        for seed in seeds:
            candidate = {
                condition: aggregated[
                    (operator, variant, branch, seed, condition)
                ]
                for condition in (
                    "load1_clean",
                    "load2_clean",
                    "load1_distractor",
                )
            }
            baseline_errors = {
                condition: candidate[condition][
                    "baseline_mean_angular_error_degrees"
                ]
                for condition in candidate
            }
            candidate_errors = {
                condition: candidate[condition][
                    "mean_angular_error_degrees"
                ]
                for condition in candidate
            }
            p5_errors = {
                condition: p5_by_seed[seed][condition][
                    "mean_angular_error_degrees"
                ]
                for condition in candidate
            }
            contrast = compute_excess_constraints(
                baseline_errors,
                candidate_errors,
                p5_errors,
                baseline_rmst=candidate["load1_clean"][
                    "baseline_restricted_mean_settling_steps"
                ],
                candidate_rmst=candidate["load1_clean"][
                    "restricted_mean_settling_steps"
                ],
                p5_rmst=p5_by_seed[seed]["load1_clean"][
                    "restricted_mean_settling_steps"
                ],
            )
            x1_values.append(contrast["x1"])
            x2_values.append(contrast["x2"])
            x3_values.append(contrast["x3"])
            metric_valid &= bool(
                candidate["load1_clean"]["latency_valid"]
                and p5_by_seed[seed]["load1_clean"]["latency_valid"]
            )
            cost = cost_lookup.get((operator, variant, branch, seed))
            p5_cost = cost_lookup.get(
                (
                    "gaussian_state_noise",
                    "generic_control",
                    "above_neutral",
                    seed,
                )
            )
            if cost is None or p5_cost is None:
                cost_valid = False
                invalid_reasons.append("unreachable_matched_strength")
                gaps.append(np.nan)
            else:
                cost_valid &= bool(
                    cost["cost_match_valid"]
                    and p5_cost["cost_match_valid"]
                    and cost["p5_cost_gap_valid"]
                )
                gaps.append(float(cost["p5_cost_gap"]))
                if not bool(cost["cost_match_valid"]):
                    invalid_reasons.append("cost_band_failure")
                if not bool(cost["p5_cost_gap_valid"]):
                    invalid_reasons.append("p5_cost_mismatch")

        components = [
            one_sided_component_test(np.asarray(values))
            for values in (x1_values, x2_values, x3_values)
        ]
        p_iut = intersection_union_pvalue(
            [component.p_value for component in components]
        )
        finite_gaps = np.asarray(gaps, dtype=float)
        gap_valid = np.all(np.isfinite(finite_gaps)) and np.all(
            np.abs(finite_gaps) <= 0.05
        )
        record = {
            "family": "B",
            "operator": operator,
            "variant": variant,
            "branch": branch,
            "profile_class": profile_class,
            "n_checkpoints": len(seeds),
            "mean_x1": components[0].mean,
            "dz_x1": components[0].dz,
            "sign_fraction_x1": components[0].sign_fraction,
            "p_c1": components[0].p_value,
            "mean_x2": components[1].mean,
            "dz_x2": components[1].dz,
            "sign_fraction_x2": components[1].sign_fraction,
            "p_c2": components[1].p_value,
            "mean_x3": components[2].mean,
            "dz_x3": components[2].dz,
            "sign_fraction_x3": components[2].sign_fraction,
            "p_c3": components[2].p_value,
            "mean_x2_gap_adjusted": (
                gap_adjusted_intercept(
                    np.asarray(x2_values), finite_gaps
                )
                if np.all(np.isfinite(finite_gaps))
                else np.nan
            ),
            "mean_x3_gap_adjusted": (
                gap_adjusted_intercept(
                    np.asarray(x3_values), finite_gaps
                )
                if np.all(np.isfinite(finite_gaps))
                else np.nan
            ),
            "p_iut": p_iut,
            "p_iut_holm": "",
            "all_cost_checks_valid": bool(cost_valid),
            "all_metric_gates_valid": bool(metric_valid),
            "max_abs_p5_cost_gap": (
                float(np.max(np.abs(finite_gaps)))
                if np.all(np.isfinite(finite_gaps))
                else np.nan
            ),
            "all_p5_gaps_valid": bool(gap_valid),
            "strictest_holm_alpha": 0.05 / 6,
            "component_mde_dz_80": minimum_detectable_dz(
                10, 0.05 / 6, 0.80
            ),
            "_component_means": [component.mean for component in components],
            "_sign_fractions": [
                component.sign_fraction for component in components
            ],
            "_invalid_reason": (
                invalid_reasons[0]
                if invalid_reasons
                else (
                    "low_fraction_settled"
                    if not metric_valid
                    else None
                )
            ),
        }
        records.append(record)

    primary_indices = [
        index
        for index, row in enumerate(records)
        if row["profile_class"] == "primary"
    ]
    adjusted = holm_adjust(
        [float(records[index]["p_iut"]) for index in primary_indices]
    )
    for index, adjusted_p in zip(primary_indices, adjusted):
        records[index]["p_iut_holm"] = adjusted_p
    for record in records:
        label, reason = classify_profile(
            profile_class=record["profile_class"],
            component_means=record.pop("_component_means"),
            sign_fractions=record.pop("_sign_fractions"),
            adjusted_iut_pvalue=(
                float(record["p_iut_holm"])
                if record["p_iut_holm"] != ""
                else None
            ),
            all_cost_checks_valid=record["all_cost_checks_valid"],
            all_metric_gates_valid=record["all_metric_gates_valid"],
            invalid_reason=record.pop("_invalid_reason"),
        )
        record["outcome_label"] = label
        record["invalid_reason"] = reason or ""
    return records


def write_profile_csv(
    path: str | Path, rows: list[dict[str, Any]]
) -> Path:
    """Write profile rows with the exact pre-registration schema."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROFILE_COLUMNS)
        writer.writeheader()
        writer.writerows(
            [{column: row.get(column, "") for column in PROFILE_COLUMNS} for row in rows]
        )
    return target
