"""Generate publication figures for the full candidate perturbation write-up."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

from wm_rnn.exploratory_pilot_summary import (
    circular_signatures,
    nback_signatures,
)


COLOURS = {
    "blue": "#315A8C",
    "orange": "#D9822B",
    "green": "#3C8D6B",
    "red": "#C94C4C",
    "grey": "#6B7280",
    "light": "#EEF2F6",
}


def _setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def task_schematic(path: Path) -> None:
    """Draw the circular delayed-response and N-back task structures."""
    _setup()
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.6))
    ax = axes[0]
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.1, 2.3)
    ax.axis("off")
    phases = [
        (2, 20, "Pre-cue\nfixation", COLOURS["light"]),
        (22, 18, "Circular cue", "#BDD7EE"),
        (40, 37, "Memory delay", "#DCEAD7"),
        (77, 21, "Response", "#F5D2B8"),
    ]
    for start, width, label, colour in phases:
        ax.add_patch(
            patches.FancyBboxPatch(
                (start, 1.25),
                width,
                0.65,
                boxstyle="round,pad=0.02",
                facecolor=colour,
                edgecolor="#4B5563",
                linewidth=0.8,
            )
        )
        ax.text(start + width / 2, 1.575, label, ha="center", va="center")
    ax.annotate(
        "trained irrelevant cue",
        xy=(61, 1.25),
        xytext=(61, 0.55),
        ha="center",
        arrowprops={"arrowstyle": "-|>", "color": COLOURS["red"]},
        color=COLOURS["red"],
    )
    ax.text(
        1,
        2.12,
        "A   Fixation-gated circular delayed-response task",
        weight="bold",
        fontsize=11,
    )
    ax.text(
        2,
        0.08,
        "Balanced clean/distractor training; clean delays span 10--80 steps "
        "and the angle is reported only after the go cue.",
        color="#374151",
    )

    ax = axes[1]
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.2, 3.2)
    ax.axis("off")
    centres = np.linspace(8, 92, 10)
    identities = ["A", "B", "C", "B", "D", "B", "A", "C", "D", "A"]
    for centre, identity in zip(centres, identities):
        item = patches.FancyBboxPatch(
            (centre - 2.4, 1.20),
            4.8,
            0.70,
            boxstyle="round,pad=0.03",
            facecolor="#BDD7EE",
            edgecolor="#4B5563",
            linewidth=0.8,
        )
        ax.add_patch(item)
        ax.text(centre, 1.55, identity, ha="center", va="center", weight="bold")
    ax.annotate(
        "0-back: respond when item = target A",
        xy=(92, 1.55),
        xytext=(62, 2.65),
        arrowprops={"arrowstyle": "-|>", "color": COLOURS["blue"]},
        color=COLOURS["blue"],
    )
    ax.annotate(
        "2-back: respond when item = item two positions earlier",
        xy=(36, 1.55),
        xytext=(45, 0.25),
        ha="center",
        arrowprops={"arrowstyle": "-|>", "color": COLOURS["orange"]},
        color=COLOURS["orange"],
    )
    ax.text(
        1,
        3.0,
        "B   Context-cued 0-back and 2-back task",
        weight="bold",
        fontsize=11,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _group_records(records: list[dict]) -> dict[tuple[str, float], list[dict]]:
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in records:
        groups[(row["operator"], float(row["strength"]))].append(row)
    return groups


def signature_screen(
    path: Path, circular: list[dict], nback: list[dict]
) -> None:
    """Plot sign-consistency fractions for every shared candidate setting."""
    _setup()
    cg = _group_records(circular)
    ng = _group_records(nback)
    keys = sorted(
        cg.keys() & ng.keys(),
        key=lambda key: (key[0], key[1]),
    )
    matrix = []
    labels = []
    for key in keys:
        crows, nrows = cg[key], ng[key]
        matrix.append(
            [
                np.mean([row["slowing_with_preservation"] for row in crows]),
                np.mean([row["delay_selectivity"] > 1e-9 for row in crows]),
                np.mean(
                    [row["distractor_selectivity"] > 1e-9 for row in crows]
                ),
                np.mean([row["load_selectivity"] > 1e-9 for row in nrows]),
            ]
        )
        labels.append(f"{key[0].replace('_', ' ')}  {key[1]:g}")
    values = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(8.8, 10.0))
    image = ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels(
        [
            "Slowing with\npreservation",
            "Long-delay\nselectivity",
            "Distractor\nselectivity",
            "N-back load\nselectivity",
        ]
    )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.tick_params(length=0)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "#111827",
                fontsize=7,
            )
    ax.set_title(
        "Consistency of predicted signature direction across trained networks",
        pad=12,
        weight="bold",
    )
    colourbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.03)
    colourbar.set_label("Fraction of checkpoint seeds")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def persistence_dose_response(
    path: Path, circular: list[dict], nback: list[dict]
) -> None:
    """Plot seed-level persistence-strength trajectories for four signatures."""
    _setup()
    crows = [row for row in circular if row["operator"] == "state_persistence"]
    nrows = [row for row in nback if row["operator"] == "state_persistence"]
    strengths = sorted({float(row["strength"]) for row in crows})
    cseeds = sorted({int(row["checkpoint_seed"]) for row in crows})
    nseeds = sorted({int(row["checkpoint_seed"]) for row in nrows})
    metrics = [
        ("clean20_settling_delta", "Settling change (steps)", crows, cseeds),
        ("delay_selectivity", "Long-short delay selectivity", crows, cseeds),
        (
            "distractor_selectivity",
            "Distractor-clean selectivity",
            crows,
            cseeds,
        ),
        ("load_selectivity", "2-back-0-back selectivity", nrows, nseeds),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.5), sharex=True)
    for axis, (field, title, rows, seeds) in zip(axes.flat, metrics):
        lookup = {
            (int(row["checkpoint_seed"]), float(row["strength"])): float(
                row[field]
            )
            for row in rows
        }
        trajectories = np.asarray(
            [[lookup[(seed, strength)] for strength in strengths] for seed in seeds]
        )
        for trajectory in trajectories:
            axis.plot(
                strengths,
                trajectory,
                color=COLOURS["grey"],
                linewidth=0.8,
                alpha=0.45,
            )
            axis.scatter(
                strengths,
                trajectory,
                color=COLOURS["grey"],
                s=9,
                alpha=0.45,
            )
        means = trajectories.mean(axis=0)
        standard_deviation = trajectories.std(axis=0, ddof=1)
        axis.errorbar(
            strengths,
            means,
            yerr=standard_deviation,
            color=COLOURS["blue"],
            marker="o",
            linewidth=2,
            capsize=3,
            label="mean ± SD",
        )
        axis.axhline(0, color="#111827", linewidth=0.8)
        axis.axvline(0.95, color=COLOURS["red"], linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("State-persistence scale")
    axes[0, 0].legend(frameon=False, loc="best")
    fig.suptitle(
        "Response across carried-state persistence scales",
        weight="bold",
        fontsize=12,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("outputs/full_candidate_perturbation_1024"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/reports/figures/full_candidate_perturbation"),
    )
    parser.add_argument("--circular-grid", type=Path)
    parser.add_argument("--nback-signatures", type=Path)
    args = parser.parse_args()
    circular = circular_signatures(
        args.circular_grid
        or args.results_root
        / "circular_family_a/metrics/circular_family_a_grid.csv"
    )
    nback = nback_signatures(
        args.nback_signatures
        or args.results_root / "nback/pilot_signatures.csv"
    )
    task_schematic(args.output_dir / "task_schematic.png")
    signature_screen(args.output_dir / "signature_screen.png", circular, nback)
    persistence_dose_response(
        args.output_dir / "persistence_dose_response.png", circular, nback
    )


if __name__ == "__main__":
    main()
