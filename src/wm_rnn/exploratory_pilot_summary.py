"""Summarize the frozen exploratory two-task signature pilot."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def circular_signatures(path: Path) -> list[dict[str, Any]]:
    """Collapse technical replicates and compute checkpoint-level contrasts."""
    grouped: dict[tuple[Any, ...], list[dict[str, str]]] = defaultdict(list)
    for row in _read(path):
        key = (
            int(row["checkpoint_seed"]),
            row["operator"],
            row["variant"],
            float(row["strength"]),
            row["condition"],
            int(row["delay_steps"]),
        )
        grouped[key].append(row)
    cells: dict[tuple[Any, ...], dict[str, float]] = {}
    fields = (
        "mean_angular_error_degrees",
        "baseline_mean_angular_error_degrees",
        "delta_restricted_mean_settling_steps",
        "fixation_accuracy",
        "fraction_settled",
    )
    for key, rows in grouped.items():
        cells[key] = {
            field: _mean([float(row[field]) for row in rows])
            for field in fields
        }

    def impairment(cell: dict[str, float]) -> float:
        baseline = cell["baseline_mean_angular_error_degrees"]
        return (cell["mean_angular_error_degrees"] - baseline) / baseline

    records: list[dict[str, Any]] = []
    settings = sorted({key[1:4] for key in cells})
    seeds = sorted({int(key[0]) for key in cells})
    for operator, variant, strength in settings:
        for seed in seeds:
            clean20 = cells.get(
                (seed, operator, variant, strength, "clean", 20)
            )
            if clean20 is None:
                # P3b exists only in the distractor window and is summarized
                # separately rather than being assigned nonexistent clean data.
                continue
            clean10 = cells[(seed, operator, variant, strength, "clean", 10)]
            clean80 = cells[(seed, operator, variant, strength, "clean", 80)]
            distractor = cells.get(
                (seed, operator, variant, strength, "distractor", 20)
            )
            clean_cost = impairment(clean20)
            latency_valid = (
                clean20["fixation_accuracy"] >= 0.90
                and clean20["fraction_settled"] >= 0.80
            )
            records.append(
                {
                    "checkpoint_seed": seed,
                    "operator": operator,
                    "variant": variant,
                    "strength": strength,
                    "clean20_proportional_error_impairment": clean_cost,
                    "clean20_settling_delta": clean20[
                        "delta_restricted_mean_settling_steps"
                    ],
                    "slowing_with_preservation": bool(
                        latency_valid
                        and clean20["delta_restricted_mean_settling_steps"] > 0
                        and clean_cost <= 0.20
                    ),
                    "delay_selectivity": impairment(clean80)
                    - impairment(clean10),
                    "distractor_selectivity": (
                        impairment(distractor) - clean_cost
                        if distractor is not None
                        else np.nan
                    ),
                    "latency_valid": latency_valid,
                }
            )
    return records


def nback_signatures(path: Path) -> list[dict[str, Any]]:
    """Read the already checkpoint-level N-back signature table."""
    output: list[dict[str, Any]] = []
    for row in _read(path):
        output.append(
            {
                "checkpoint_seed": int(row["checkpoint_seed"]),
                "operator": row["operator"],
                "variant": row["variant"],
                "strength": float(row["strength"]),
                "load_selectivity": float(row["load_selectivity"]),
            }
        )
    return output


def cross_task_summary(
    circular: list[dict[str, Any]],
    nback: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score exact operator-strength settings present in both task families."""
    circular_groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(
        list
    )
    nback_groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in circular:
        circular_groups[(row["operator"], row["strength"])].append(row)
    for row in nback:
        nback_groups[(row["operator"], row["strength"])].append(row)
    rows: list[dict[str, Any]] = []
    for operator, strength in sorted(circular_groups.keys() & nback_groups.keys()):
        if operator == "gaussian_state_noise":
            continue
        crows = circular_groups[(operator, strength)]
        nrows = nback_groups[(operator, strength)]
        slowing_count = sum(row["slowing_with_preservation"] for row in crows)
        delay_values = [row["delay_selectivity"] for row in crows]
        distractor_values = [row["distractor_selectivity"] for row in crows]
        load_values = [row["load_selectivity"] for row in nrows]
        circular_required = len(crows) // 2 + 1
        nback_required = len(nrows) // 2 + 1
        delay_count = sum(value > 0 for value in delay_values)
        distractor_count = sum(value > 0 for value in distractor_values)
        load_count = sum(value > 0 for value in load_values)
        mean_settling = _mean(
            [row["clean20_settling_delta"] for row in crows]
        )
        slowing_match = (
            mean_settling > 0 and slowing_count >= circular_required
        )
        delay_match = (
            _mean(delay_values) > 0 and delay_count >= circular_required
        )
        distractor_match = (
            _mean(distractor_values) > 0
            and distractor_count >= circular_required
        )
        load_match = (
            _mean(load_values) > 0 and load_count >= nback_required
        )
        rows.append(
            {
                "operator": operator,
                "strength": strength,
                "mean_clean20_proportional_error_impairment": _mean(
                    [
                        row["clean20_proportional_error_impairment"]
                        for row in crows
                    ]
                ),
                "mean_clean20_settling_delta": mean_settling,
                "slowing_with_preservation_count": slowing_count,
                "slowing_with_preservation": slowing_match,
                "mean_delay_selectivity": _mean(delay_values),
                "delay_selectivity_count": delay_count,
                "delay_selective": delay_match,
                "mean_distractor_selectivity": _mean(distractor_values),
                "distractor_selectivity_count": distractor_count,
                "distractor_selective": distractor_match,
                "mean_nback_load_selectivity": _mean(load_values),
                "nback_load_selectivity_count": load_count,
                "load_selective": load_match,
                "complete_primary_pattern": bool(
                    slowing_match and distractor_match and load_match
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_leader(
    path: Path,
    leader: dict[str, Any],
    circular: list[dict[str, Any]],
    nback: list[dict[str, Any]],
) -> None:
    key = (leader["operator"], leader["strength"])
    crows = [
        row
        for row in circular
        if (row["operator"], row["strength"]) == key
    ]
    nrows = [
        row for row in nback if (row["operator"], row["strength"]) == key
    ]
    measures = [
        [row["clean20_settling_delta"] for row in crows],
        [row["delay_selectivity"] for row in crows],
        [row["distractor_selectivity"] for row in crows],
        [row["load_selectivity"] for row in nrows],
    ]
    labels = [
        "Settling delta\n(steps)",
        "Long-short delay\nselectivity",
        "Distractor-clean\nselectivity",
        "2-back-0-back\nselectivity",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    for axis, values, label in zip(axes, measures, labels):
        axis.axhline(0, color="#777777", linewidth=1)
        axis.scatter(np.ones(len(values)), values, color="#345995", zorder=2)
        axis.scatter([1], [_mean(values)], marker="_", s=500, color="#d1495b")
        axis.set_xlim(0.7, 1.3)
        axis.set_xticks([])
        axis.set_title(label, fontsize=9)
    fig.suptitle(
        f"Leading exploratory profile: {leader['operator']} "
        f"at {leader['strength']:g} (points are trained seeds)",
        fontsize=11,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_summary(
    root: Path,
    *,
    circular_grid: Path | None = None,
    nback_signature_table: Path | None = None,
) -> dict[str, Any]:
    """Summarise circular and N-back results from explicit or legacy paths."""
    circular = circular_signatures(
        circular_grid
        or root / "circular_family_a/metrics/circular_family_a_grid.csv"
    )
    nback = nback_signatures(
        nback_signature_table or root / "nback/pilot_signatures.csv"
    )
    summary = cross_task_summary(circular, nback)
    summary.sort(
        key=lambda row: (
            row["complete_primary_pattern"],
            row["slowing_with_preservation_count"]
            + row["distractor_selectivity_count"]
            + row["nback_load_selectivity_count"],
        ),
        reverse=True,
    )
    metrics = root / "summary"
    _write_csv(metrics / "cross_task_signature_summary.csv", summary)
    with (metrics / "cross_task_signature_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "status": "exploratory_descriptive_only",
                "n_circular_checkpoint_profiles": len(circular),
                "n_nback_checkpoint_profiles": len(nback),
                "profiles": summary,
            },
            handle,
            indent=2,
        )
    leader = summary[0]
    _plot_leader(metrics / "leading_profile_seed_points.png", leader, circular, nback)
    return leader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="outputs/exploratory_psilocybin_signature_pilot",
        type=Path,
    )
    parser.add_argument("--circular-grid", type=Path)
    parser.add_argument("--nback-signatures", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run_summary(
                args.root,
                circular_grid=args.circular_grid,
                nback_signature_table=args.nback_signatures,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
