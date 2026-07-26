"""Programmatic scoring and dissertation figures for the signature experiment."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from wm_rnn.perturbation_signature_scoring import (
    DESCRIPTIVE_PROFILES,
    PRIMARY_PROFILES,
    PROFILE_COLUMNS,
    SCORE_COLUMNS,
    read_csv_records,
    score_profile_rows,
    write_profile_csv,
)


def _write_scores(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _cell_from_profile(
    sign_fraction: float, one_sided_p: float
) -> str:
    if sign_fraction < 0.80:
        return "no"
    return "yes" if one_sided_p < 0.025 else "partial"


def build_signature_scores(
    profile_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fill the broad score table deterministically from profile statistics."""
    rows = []
    for profile in profile_rows:
        rows.append(
            {
                "operator": profile["operator"],
                "variant": profile["variant"],
                "branch": profile["branch"],
                "settling_slowing": _cell_from_profile(
                    float(profile["sign_fraction_x1"]),
                    float(profile["p_c1"]),
                ),
                "response_failure": "NA",
                "retention_dependent": "NA",
                "distractor_selective": _cell_from_profile(
                    float(profile["sign_fraction_x3"]),
                    float(profile["p_c3"]),
                ),
                "load_dependent": _cell_from_profile(
                    float(profile["sign_fraction_x2"]),
                    float(profile["p_c2"]),
                ),
                "dose_ordered": "NA",
                "dynamics_differ_from_p5": "NA",
                "assignment_sensitive": (
                    "NA"
                    if profile["operator"] != "heterogeneous_drive_gain"
                    else "NA"
                ),
            }
        )
    return rows


def _matched_rows(grid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in grid_rows
        if str(row["strength_kind"]).startswith("matched")
        and row["item_position"] in {"", "pooled"}
    ]


def _t_interval(values: list[float]) -> tuple[float, float, float]:
    sample = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(sample))
    if sample.size < 2:
        return mean, mean, mean
    low, high = stats.t.interval(
        0.95,
        df=sample.size - 1,
        loc=mean,
        scale=float(stats.sem(sample)),
    )
    return mean, float(low), float(high)


def plot_settling_vs_strength(
    grid_rows: list[dict[str, Any]], path: Path
) -> None:
    clean = [
        row
        for row in grid_rows
        if row["condition"] in {"clean", "load1_clean"}
        and int(row["delay_steps"]) == 20
        and row["item_position"] in {"", "pooled"}
        and bool(row["latency_valid"])
    ]
    nested: dict[tuple[str, float, int], list[float]] = defaultdict(list)
    for row in clean:
        nested[
            (
                str(row["operator"]),
                float(row["strength"]),
                int(row["seed"]),
            )
        ].append(
            float(row["delta_restricted_mean_settling_steps"])
        )
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for (operator, strength, _seed), values in nested.items():
        grouped[(operator, strength)].append(float(np.mean(values)))
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for operator in sorted({key[0] for key in grouped}):
        strengths = sorted(
            strength for name, strength in grouped if name == operator
        )
        summaries = [
            _t_interval(grouped[(operator, strength)])
            for strength in strengths
        ]
        means = [summary[0] for summary in summaries]
        lower = [mean - summary[1] for mean, summary in zip(means, summaries)]
        upper = [summary[2] - mean for mean, summary in zip(means, summaries)]
        ax.errorbar(
            strengths,
            means,
            yerr=np.asarray([lower, upper]),
            marker="o",
            linewidth=1.2,
            label=operator,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Perturbation strength")
    ax.set_ylabel("Delta restricted-mean settling steps")
    ax.set_title("Settling analogue across perturbation strength")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_delay_condition_heatmaps(
    grid_rows: list[dict[str, Any]], path: Path
) -> None:
    matched = _matched_rows(grid_rows)
    operators = sorted({str(row["operator"]) for row in matched})
    cells = sorted(
        {
            (str(row["condition"]), int(row["delay_steps"]))
            for row in matched
        }
    )
    matrix = np.full((len(operators), len(cells)), np.nan)
    for row_index, operator in enumerate(operators):
        for column_index, (condition, delay) in enumerate(cells):
            values = [
                float(row["delta_angular_error_degrees"])
                for row in matched
                if row["operator"] == operator
                and row["condition"] == condition
                and int(row["delay_steps"]) == delay
            ]
            if values:
                matrix[row_index, column_index] = float(np.mean(values))
    fig, ax = plt.subplots(figsize=(max(8.0, len(cells) * 0.45), 5.5))
    image = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(operators)), operators)
    ax.set_xticks(
        range(len(cells)),
        [f"{condition}\n{delay}" for condition, delay in cells],
        rotation=90,
    )
    fig.colorbar(image, ax=ax, label="Delta angular error (degrees)")
    ax.set_title("Matched-cost delay-by-condition effects")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_profile_components(
    profile_rows: list[dict[str, Any]], path: Path
) -> None:
    labels = [
        f"{row['operator']}\n{row['variant']}\n{row['branch']}"
        for row in profile_rows
    ]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10.0, len(labels) * 0.75), 5.5))
    for offset, component in zip(
        (-width, 0.0, width), ("mean_x1", "mean_x2", "mean_x3")
    ):
        ax.bar(
            x + offset,
            [float(row[component]) for row in profile_rows],
            width,
            label=component.replace("mean_", "").upper(),
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=75, ha="right")
    ax.set_ylabel("Excess-over-P5 contrast")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_distractor_drift_recovery(
    grid_rows: list[dict[str, Any]], path: Path
) -> None:
    rows = [
        row
        for row in _matched_rows(grid_rows)
        if "distractor" in str(row["condition"])
        and int(row["delay_steps"]) == 20
    ]
    operators = sorted({str(row["operator"]) for row in rows})
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for operator in operators:
        peak = np.mean(
            [
                float(row["distractor_peak_attraction_fraction"])
                for row in rows
                if row["operator"] == operator
            ]
        )
        end = np.mean(
            [
                float(row["distractor_end_attraction_fraction"])
                for row in rows
                if row["operator"] == operator
            ]
        )
        ax.plot([0, 1], [peak, end], marker="o", label=operator)
    ax.set_xticks([0, 1], ["Peak distraction", "End of delay"])
    ax.set_ylabel("Target-to-distractor attraction fraction")
    ax.set_title("Reference distraction and recovery summary")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ordinal_constraints(
    profile_rows: list[dict[str, Any]], path: Path
) -> None:
    labels = [
        f"{row['operator']}\n{row['branch']}" for row in profile_rows
    ]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(12.0, max(5.0, len(labels) * 0.42)))
    for axis, component in zip(axes, ("x1", "x2", "x3")):
        means = np.asarray(
            [float(row[f"mean_{component}"]) for row in profile_rows]
        )
        axis.scatter(means, y)
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_title(component.upper())
        axis.set_yticks(y, labels if component == "x1" else [])
    fig.suptitle("Noise-referenced ordinal constraints (checkpoint means)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def operator_distinguishability(
    grid_rows: list[dict[str, Any]],
) -> tuple[list[str], np.ndarray]:
    """Return the Section 9d core-profile centroid-distance ratio matrix."""
    matched = _matched_rows(grid_rows)
    profiles = sorted(
        {
            (
                str(row["operator"]),
                str(row["variant"]),
                str(row["branch"]),
                str(row["strength_kind"]),
            )
            for row in matched
        }
    )
    seeds = sorted({int(row["seed"]) for row in matched})
    vectors: dict[tuple[str, str, str, str], list[np.ndarray]] = {}
    for profile in profiles:
        profile_vectors = []
        for seed in seeds:
            subset = [
                row
                for row in matched
                if (
                    row["operator"],
                    row["variant"],
                    row["branch"],
                    row["strength_kind"],
                )
                == profile
                and int(row["seed"]) == seed
            ]
            if not subset:
                continue
            clean = [
                row
                for row in subset
                if row["condition"] in {"clean", "load1_clean"}
            ]
            delay20 = [
                row for row in clean if int(row["delay_steps"]) == 20
            ]
            delays = sorted(
                {
                    int(row["delay_steps"])
                    for row in clean
                    if int(row["delay_steps"]) <= 80
                }
            )
            delay_means = [
                np.mean(
                    [
                        float(row["delta_angular_error_degrees"])
                        for row in clean
                        if int(row["delay_steps"]) == delay
                    ]
                )
                for delay in delays
            ]
            retention_slope = (
                float(
                    np.polyfit(
                        np.log2(np.asarray(delays) / 10.0),
                        delay_means,
                        1,
                    )[0]
                )
                if len(delays) >= 2
                else 0.0
            )
            conditions = {
                condition: np.mean(
                    [
                        float(row["delta_angular_error_degrees"])
                        for row in subset
                        if row["condition"] == condition
                        and int(row["delay_steps"]) == 20
                    ]
                )
                for condition in {
                    str(row["condition"]) for row in subset
                }
            }
            distractor_did = (
                conditions.get("load1_distractor", conditions.get("distractor", 0.0))
                - conditions.get("load1_clean", conditions.get("clean", 0.0))
            )
            load_did = conditions.get("load2_clean", 0.0) - conditions.get(
                "load1_clean", 0.0
            )
            profile_vectors.append(
                np.asarray(
                    [
                        np.mean(
                            [
                                float(row["delta_failure_rate"])
                                for row in delay20
                            ]
                        ),
                        retention_slope,
                        distractor_did,
                        load_did,
                        np.mean(
                            [
                                float(
                                    row[
                                        "delta_mean_late_delay_state_entropy"
                                    ]
                                )
                                for row in delay20
                            ]
                        ),
                    ]
                )
            )
        vectors[profile] = profile_vectors
    all_vectors = np.vstack(
        [np.vstack(values) for values in vectors.values() if values]
    )
    scale = np.std(all_vectors, axis=0, ddof=1)
    scale[scale == 0.0] = 1.0
    standardized = {
        profile: np.vstack(values) / scale
        for profile, values in vectors.items()
        if values
    }
    labels = ["|".join(profile) for profile in standardized]
    centroids = {
        profile: np.mean(values, axis=0)
        for profile, values in standardized.items()
    }
    spreads = {
        profile: float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (values - centroids[profile]) ** 2, axis=1
                    )
                )
            )
        )
        for profile, values in standardized.items()
    }
    matrix = np.zeros((len(labels), len(labels)))
    keys = list(standardized)
    for i, first in enumerate(keys):
        for j, second in enumerate(keys):
            denominator = np.sqrt(
                (spreads[first] ** 2 + spreads[second] ** 2) / 2.0
            )
            distance = np.linalg.norm(centroids[first] - centroids[second])
            matrix[i, j] = (
                distance / denominator if denominator > 0.0 else np.nan
            )
    return labels, matrix


def plot_distinguishability(
    grid_rows: list[dict[str, Any]], path: Path
) -> None:
    labels, matrix = operator_distinguishability(grid_rows)
    fig, ax = plt.subplots(figsize=(max(7.0, len(labels) * 0.55), 6.5))
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    fig.colorbar(image, ax=ax, label="R_ij")
    ax.set_title("Core-profile operator distinguishability")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_outcome_taxonomy(
    profile_rows: list[dict[str, Any]], path: Path
) -> None:
    labels = [
        "confirmatory_match",
        "tested_null",
        "not_testable_validity",
        "descriptive_only",
    ]
    counts = [
        sum(row["outcome_label"] == label for row in profile_rows)
        for label in labels
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(labels, counts, color=["#009E73", "#D55E00", "#CC79A7", "#999999"])
    ax.set_ylabel("Number of profiles")
    ax.set_title("Pre-registered outcome taxonomy")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_signature_artifacts(
    grid_path: str | Path,
    cost_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Generate all frozen Phase 7 score tables and figure files."""
    grid_rows = read_csv_records(grid_path)
    cost_rows = read_csv_records(cost_path)
    profile_rows = score_profile_rows(grid_rows, cost_rows)
    output = Path(output_dir)
    metrics = output / "metrics"
    figures = output / "figures"
    metrics.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    profile_path = write_profile_csv(
        metrics / "profile_match.csv", profile_rows
    )
    score_path = _write_scores(
        metrics / "signature_scores.csv",
        build_signature_scores(profile_rows),
    )
    paths = {
        "profile_match": profile_path,
        "signature_scores": score_path,
        "settling_vs_strength": figures / "settling_vs_strength.png",
        "delay_condition_heatmaps": figures / "delay_condition_heatmaps.png",
        "dissociation_summary": figures / "dissociation_summary.png",
        "distractor_drift_recovery": figures / "distractor_drift_recovery.png",
        "ordinal_constraints": figures / "ordinal_constraints.png",
        "operator_distinguishability": figures
        / "operator_distinguishability.png",
        "outcome_taxonomy": figures / "outcome_taxonomy.png",
    }
    plot_settling_vs_strength(grid_rows, paths["settling_vs_strength"])
    plot_delay_condition_heatmaps(
        grid_rows, paths["delay_condition_heatmaps"]
    )
    plot_profile_components(profile_rows, paths["dissociation_summary"])
    plot_distractor_drift_recovery(
        grid_rows, paths["distractor_drift_recovery"]
    )
    plot_ordinal_constraints(profile_rows, paths["ordinal_constraints"])
    plot_distinguishability(
        grid_rows, paths["operator_distinguishability"]
    )
    plot_outcome_taxonomy(profile_rows, paths["outcome_taxonomy"])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score and plot psilocybin-signature perturbation results."
    )
    parser.add_argument("--grid", required=True)
    parser.add_argument("--cost-check", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    paths = generate_signature_artifacts(
        args.grid, args.cost_check, args.output_dir
    )
    for name, path in paths.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
