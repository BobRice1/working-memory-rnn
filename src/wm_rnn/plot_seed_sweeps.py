"""Plot delay-sweep curves from a multi-seed summary file."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class SeedSweepPlotResult:
    """Output produced by combined seed-sweep plotting.

    Attributes:
        figure_path: PNG figure containing per-seed and aggregate curves.
        seed_count: Number of seed runs included in the plot.
        delays: Delay lengths included in the aggregate curve.
    """

    figure_path: Path
    seed_count: int
    delays: list[int]


def plot_seed_sweeps(
    summary_path: str | Path,
    output_path: str | Path | None = None,
    trained_delay_steps: int | None = None,
    chance_accuracy: float | None = None,
) -> SeedSweepPlotResult:
    """Plot all per-seed delay sweeps from a seed-sweep summary JSON file.

    Args:
        summary_path: Path to ``*_seed_sweep_summary.json``.
        output_path: Optional PNG path. If omitted, a default path under the
            base run's ``figures`` directory is used.
        trained_delay_steps: Optional vertical reference line. Defaults to the
            value in the summary if present, otherwise 20.
        chance_accuracy: Optional chance-level reference line. Defaults to the
            value in the summary if present, otherwise 0.25.

    Returns:
        ``SeedSweepPlotResult`` with the saved figure path and plotted delays.

    Raises:
        ValueError: If the summary does not contain any usable delay sweeps.
    """
    summary_file = Path(summary_path)
    with summary_file.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    base_output_dir = Path(summary.get("base_output_dir", summary_file.parents[1]))
    base_run_name = summary.get("base_run_name", "baseline_delay")
    trained_delay = int(trained_delay_steps or summary.get("trained_delay_steps", 20))
    chance = float(chance_accuracy if chance_accuracy is not None else summary.get("chance_accuracy", 0.25))
    target = Path(output_path) if output_path else base_output_dir / "figures" / f"{base_run_name}_seed_sweep_delay_curves.png"

    seed_curves = _load_seed_curves(summary)
    if not seed_curves:
        raise ValueError("summary does not contain any delay_sweep_csv entries with data")

    delays = sorted({delay for curve in seed_curves for delay in curve["accuracies"].keys()})
    matrix = np.full((len(seed_curves), len(delays)), np.nan, dtype=np.float64)
    for seed_idx, curve in enumerate(seed_curves):
        for delay_idx, delay in enumerate(delays):
            if delay in curve["accuracies"]:
                matrix[seed_idx, delay_idx] = curve["accuracies"][delay]

    mean_accuracy = np.nanmean(matrix, axis=0)
    std_accuracy = np.nanstd(matrix, axis=0)

    target.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 4.8))
    for curve in seed_curves:
        curve_delays = sorted(curve["accuracies"].keys())
        curve_accuracy = [curve["accuracies"][delay] for delay in curve_delays]
        plt.plot(curve_delays, curve_accuracy, marker="o", linewidth=1, alpha=0.35, label=f"seed {curve['seed']}")

    plt.plot(delays, mean_accuracy, marker="o", linewidth=2.5, color="#111111", label="mean")
    if len(seed_curves) > 1:
        lower = np.clip(mean_accuracy - std_accuracy, 0.0, 1.0)
        upper = np.clip(mean_accuracy + std_accuracy, 0.0, 1.0)
        plt.fill_between(delays, lower, upper, color="#111111", alpha=0.12, label="+/- 1 SD")

    plt.axhline(chance, linestyle="--", linewidth=1.25, color="#666666", label="chance")
    plt.axvline(trained_delay, linestyle=":", linewidth=1.5, color="#444444", label="trained delay")
    plt.ylim(0.0, 1.05)
    plt.xlabel("Delay length (time steps)")
    plt.ylabel("Response accuracy")
    plt.title("Delay-length sweeps across training seeds")
    plt.legend(fontsize=8, ncols=2)
    plt.tight_layout()
    plt.savefig(target, dpi=160)
    plt.close()

    return SeedSweepPlotResult(figure_path=target, seed_count=len(seed_curves), delays=delays)


def _load_seed_curves(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Load delay-accuracy mappings from each seed's delay-sweep CSV."""
    curves = []
    for row in summary.get("results", []):
        csv_path = row.get("delay_sweep_csv")
        if not csv_path:
            continue
        path = Path(csv_path)
        if not path.exists():
            continue

        accuracies = {}
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for csv_row in reader:
                accuracies[int(csv_row["delay_steps"])] = float(csv_row["accuracy"])

        if accuracies:
            curves.append({"seed": row.get("seed", "?"), "accuracies": accuracies})
    return curves


def main() -> None:
    """Parse command-line arguments and plot combined seed-sweep curves."""
    parser = argparse.ArgumentParser(description="Plot delay-sweep curves across multiple training seeds.")
    parser.add_argument("--summary", required=True, help="Path to *_seed_sweep_summary.json.")
    parser.add_argument("--output", help="Optional output PNG path.")
    parser.add_argument("--trained-delay", type=int, help="Delay length used during training.")
    parser.add_argument("--chance", type=float, help="Chance accuracy reference line.")
    args = parser.parse_args()

    result = plot_seed_sweeps(
        args.summary,
        output_path=args.output,
        trained_delay_steps=args.trained_delay,
        chance_accuracy=args.chance,
    )
    print(f"figure={result.figure_path}")
    print(f"seeds={result.seed_count}")
    print(f"delays={' '.join(str(delay) for delay in result.delays)}")


if __name__ == "__main__":
    main()
