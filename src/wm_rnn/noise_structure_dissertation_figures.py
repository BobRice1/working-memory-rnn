"""Dissertation-oriented figures for the structured-noise experiment.

Marker-facing six-figure set: noise definitions, validation, hero behaviour,
delay timecourse, ring geometry, and epoch sensitivity.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyBboxPatch, Patch

from wm_rnn.config import load_config
from wm_rnn.noise_structure_experiment import (
    COLORS,
    CONDITIONS,
    generate_perturbations,
    phase_mask,
    require_cuda,
)
from wm_rnn.training_utils import fresh_model, generate_batch_for_task, task_config_from_dict

LABELS = {
    "unperturbed": "Clean (no noise)",
    "independent_gaussian": "Independent",
    "temporally_correlated": "Temporal AR(1)",
    "context_topology_correlated": "Topology AR(1)",
}
SHORT_LABELS = {
    "unperturbed": "Clean",
    "independent_gaussian": "Independent",
    "temporally_correlated": "Temporal",
    "context_topology_correlated": "Topology",
}
NOISE_CONDITIONS = CONDITIONS[1:]
SEEDS = (20260714, 20260715, 20260716, 20260717, 20260718)
PRIMARY_RMS = 0.05
PRIMARY_DELAY = 80
RMS_AXIS_LABEL = "Perturbation strength (matched RMS of added noise)"
RMS_NOTE = (
    "RMS = root-mean-square size of the added noise, matched across conditions. "
    "It is a model intensity, not a drug dose."
)
PHASE_COLORS = {"cue": "#009E73", "delay": "#D55E00", "response": "#0072B2"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, root: Path, name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(root / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def seed_summaries(rows: list[dict[str, str]], metric: str) -> dict[tuple[int, str, float, int], float]:
    grouped: dict[tuple[int, str, float, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (int(row["delay"]), row["condition"], float(row["strength"]), int(row["seed"]))
        grouped[key].append(float(row[metric]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def bootstrap_ci(values: np.ndarray, seed: int = 20260714, draws: int = 5000) -> tuple[float, float]:
    if len(values) < 2:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def mean_ci(
    summary: dict[tuple[int, str, float, int], float],
    delay: int,
    condition: str,
    strength: float,
) -> tuple[float, float, float, np.ndarray]:
    values = np.array([summary[(delay, condition, strength, seed)] for seed in SEEDS])
    low, high = bootstrap_ci(values, seed=delay + int(strength * 10000))
    return float(values.mean()), low, high, values


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.6)


def plot_schematic(figure_dir: Path) -> None:
    """Figure 1: what the three noises mean, with equations and example traces."""
    fig = plt.figure(figsize=(10.0, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(4, 3, height_ratios=[0.55, 1.15, 1.0, 0.85])

    ax_timeline = fig.add_subplot(gs[0, :])
    ax_timeline.set_xlim(0, 10)
    ax_timeline.set_ylim(0, 1)
    ax_timeline.axis("off")
    stages = [
        (0.3, 2.8, "Cue\nencode angle", "#C8E6C9"),
        (3.4, 3.4, "Delay\nhold memory\n(+ noise here)", "#FFE082"),
        (7.1, 2.6, "Response\nreport angle", "#C5CAE9"),
    ]
    for x, width, text, color in stages:
        ax_timeline.add_patch(
            FancyBboxPatch((x, 0.15), width, 0.7, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=color, edgecolor="0.3", linewidth=1.0)
        )
        ax_timeline.text(x + width / 2, 0.5, text, ha="center", va="center", fontsize=9)
    ax_timeline.annotate("", xy=(3.4, 0.5), xytext=(3.1, 0.5), arrowprops={"arrowstyle": "->", "color": "0.3"})
    ax_timeline.annotate("", xy=(7.1, 0.5), xytext=(6.8, 0.5), arrowprops={"arrowstyle": "->", "color": "0.3"})
    ax_timeline.set_title("A. Task timeline (noise added to hidden-state updates mainly in the delay)", loc="left", fontsize=11)

    definitions = [
        (
            "independent_gaussian",
            "Independent Gaussian",
            r"$\varepsilon_t \sim \mathcal{N}(0, I)$",
            "Fresh random kick every step.\nNo memory across time.\nUnits kicked independently.",
        ),
        (
            "temporally_correlated",
            "Temporal AR(1)",
            r"$z_t = 0.9\,z_{t-1} + \sqrt{1-0.9^2}\,\eta_t$",
            "Sticky kick: each step resembles\nthe previous one.\nUnits still independent.",
        ),
        (
            "context_topology_correlated",
            "Topology-correlated AR(1)",
            r"$\varepsilon_t = L z_t,\quad LL^\top=\Sigma(W_{\mathrm{rec}})$",
            "Sticky kick, then shared across\nunits with similar recurrent wiring.\n$\Sigma = 0.25I + 0.75\,W_{\mathrm{rec}}W_{\mathrm{rec}}^\top$ (normalized).",
        ),
    ]

    for col, (condition, title, equation, plain) in enumerate(definitions):
        ax_eq = fig.add_subplot(gs[1, col])
        ax_eq.axis("off")
        ax_eq.set_xlim(0, 1)
        ax_eq.set_ylim(0, 1)
        ax_eq.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.9, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor="#FAFAFA", edgecolor=COLORS[condition], linewidth=1.8))
        ax_eq.text(0.5, 0.88, title, ha="center", va="top", fontsize=10, fontweight="bold", color=COLORS[condition])
        ax_eq.text(0.5, 0.62, equation, ha="center", va="center", fontsize=10)
        ax_eq.text(0.5, 0.28, plain, ha="center", va="center", fontsize=8.2, color="0.2")

        ax_tr = fig.add_subplot(gs[2, col])
        rng = np.random.default_rng(10 + col)
        t = np.arange(60)
        if condition == "independent_gaussian":
            y = rng.normal(size=len(t))
        else:
            y = np.zeros(len(t))
            innov = np.sqrt(1 - 0.9**2)
            y[0] = rng.normal()
            for i in range(1, len(t)):
                y[i] = 0.9 * y[i - 1] + innov * rng.normal()
        y = 0.05 * y / np.sqrt(np.mean(y**2))
        ax_tr.plot(t, y, color=COLORS[condition], linewidth=1.3)
        ax_tr.axhline(0, color="0.75", linewidth=0.6)
        ax_tr.set_ylim(-0.18, 0.18)
        ax_tr.set_xlabel("Delay step")
        ax_tr.set_ylabel("Example noise\non one unit" if col == 0 else "")
        ax_tr.set_title("Example trace (RMS matched to 0.05)", fontsize=8.5)
        style_axis(ax_tr)

    ax_note = fig.add_subplot(gs[3, :])
    ax_note.axis("off")
    ax_note.text(
        0.5,
        0.55,
        "All three noises are rescaled to the same realized RMS before use.\n"
        "So differences in outcome are due to structure (time stickiness / unit coupling), not total noise energy.\n"
        "These are psilocybin-informed modelling assumptions, not a pharmacological simulation.",
        ha="center",
        va="center",
        fontsize=9,
        color="0.2",
    )
    fig.suptitle("What the three equal-RMS noise interventions mean", fontsize=13, y=1.01)
    save_figure(fig, figure_dir, "figure_01_task_perturbation_schematic")


def plot_manipulation_validation(config: dict[str, Any], figure_dir: Path, data_dir: Path) -> None:
    """Figure 2: check that RMS matched and structure differed as intended."""
    device = require_cuda()
    local = {**config, "task": dict(config["task"])}
    local["task"]["seed"] = SEEDS[0]
    model = fresh_model(local, device)
    checkpoint = (
        Path(config["paths"]["output_dir"])
        / "seed_sweep"
        / f"seed_{SEEDS[0]}"
        / "checkpoints"
        / f"yang_fixation_circular_working_memory_seed_{SEEDS[0]}.pt"
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model_state"])
    model.eval()
    task = replace(task_config_from_dict(local, batch_size=256), delay_steps=80, seed=99117)
    batch = generate_batch_for_task(task)
    mask = phase_mask(task.seq_len, batch.phase_index, "delay", device)

    fig = plt.figure(figsize=(10.0, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.85])
    plotted: list[dict[str, Any]] = []
    offdiags = []

    for col, condition in enumerate(NOISE_CONDITIONS):
        generator = torch.Generator(device=device).manual_seed(113 + col)
        noise = generate_perturbations(
            condition,
            (task.seq_len, 256, model.config.hidden_size),
            PRIMARY_RMS,
            mask,
            generator,
            model.rnn.h2h.weight,
        )
        active = noise[mask].detach().cpu().numpy()
        trace = active[:, 0, 0]
        denom = np.mean(active**2)
        max_lag = min(20, len(trace) - 1)
        acf = np.array([1.0] + [np.mean(active[:-lag] * active[lag:]) / denom for lag in range(1, max_lag + 1)])
        expected = np.array(
            [1.0]
            + ([0.0] * max_lag if condition == "independent_gaussian" else [0.9**lag for lag in range(1, max_lag + 1)])
        )
        corr = np.corrcoef(active.reshape(-1, active.shape[-1]), rowvar=False)
        offdiag = float(np.mean(np.abs(corr[~np.eye(corr.shape[0], dtype=bool)])))
        offdiags.append(offdiag)
        realized = float(np.sqrt(np.mean(active**2)))

        ax0 = fig.add_subplot(gs[0, col])
        ax0.plot(np.arange(len(trace)), trace, color=COLORS[condition], linewidth=1.0)
        ax0.axhline(0, color="0.7", linewidth=0.6)
        ax0.set_title(LABELS[condition], fontsize=10, color=COLORS[condition])
        ax0.set_xlabel("Delay time step")
        if col == 0:
            ax0.set_ylabel("Added noise value\n(one hidden unit)")
        ax0.text(
            0.03,
            0.95,
            f"Matched RMS = {realized:.3f}",
            transform=ax0.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2},
        )
        ax0.set_ylim(-0.22, 0.22)
        style_axis(ax0)

        ax1 = fig.add_subplot(gs[1, col])
        ax1.plot(np.arange(len(acf)), acf, color=COLORS[condition], linewidth=2.0, label="Measured")
        ax1.plot(np.arange(len(expected)), expected, color="0.25", linewidth=1.2, linestyle=":", label="Expected from formula")
        ax1.axhline(0, color="0.6", linewidth=0.5)
        ax1.axhline(1, color="0.85", linewidth=0.5, linestyle="--")
        ax1.set_ylim(-0.15, 1.08)
        ax1.set_xlabel("Time lag k (steps)")
        if col == 0:
            ax1.set_ylabel("Similarity of noise to itself\nk steps later\n(1 = identical, 0 = unrelated)")
        if col == 2:
            ax1.legend(frameon=False, fontsize=7.5, loc="upper right")
        style_axis(ax1)

        plotted.append({"condition": condition, "realized_rms": realized, "lag1": float(acf[1]), "mean_abs_offdiag_correlation": offdiag})

    ax2 = fig.add_subplot(gs[2, :])
    xpos = np.arange(len(NOISE_CONDITIONS))
    bars = ax2.bar(xpos, offdiags, color=[COLORS[c] for c in NOISE_CONDITIONS], alpha=0.8, edgecolor="white")
    ax2.set_xticks(xpos, [LABELS[c] for c in NOISE_CONDITIONS])
    ax2.set_ylabel("Average |correlation|\nbetween different units")
    ax2.set_xlabel("Noise condition")
    ax2.set_title(
        "C. Do different hidden units get pushed together?  (near 0 = independent units; higher = coordinated population push)",
        loc="left",
        fontsize=10,
    )
    for bar, value in zip(bars, offdiags):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 0.005, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    style_axis(ax2)
    ax2.set_ylim(0, max(offdiags) * 1.25)

    fig.suptitle("Manipulation check: same noise size, different structure", fontsize=13)
    fig.text(
        0.5,
        -0.01,
        "Row A: example noise traces at matched RMS.  Row B: temporal stickiness (autocorrelation).  "
        "Panel C: replaces correlation heatmaps with a direct summary of unit–unit coupling.",
        ha="center",
        va="top",
        fontsize=8,
        color="0.3",
    )
    save_figure(fig, figure_dir, "figure_02_manipulation_validation")
    write_csv(data_dir / "figure_02_manipulation_validation.csv", plotted)


def plot_behavioural_hero(rows: list[dict[str, str]], figure_dir: Path, data_dir: Path) -> None:
    response = seed_summaries(rows, "response_error_degrees")
    decoder = seed_summaries(rows, "delay_decoder_error_degrees")
    fixation = seed_summaries(rows, "fixation_accuracy")
    delays = (20, 80, 160)
    strengths = (0.0, 0.01, 0.025, 0.05, 0.1)
    plotted: list[dict[str, Any]] = []

    fig = plt.figure(figsize=(10.8, 6.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.05], width_ratios=[1.35, 1.05, 0.9])
    ax_bars = fig.add_subplot(gs[0, :])
    ax_dose = fig.add_subplot(gs[1, 0:2])
    ax_fix = fig.add_subplot(gs[1, 2])

    x = np.arange(len(delays))
    width = 0.18
    for idx, condition in enumerate(CONDITIONS):
        means, lows, highs = [], [], []
        offsets = x + (idx - 1.5) * width
        for d_i, delay in enumerate(delays):
            mean, low, high, seed_values = mean_ci(response, delay, condition, PRIMARY_RMS)
            means.append(mean)
            lows.append(low)
            highs.append(high)
            jitter = np.linspace(-0.04, 0.04, len(SEEDS))
            ax_bars.scatter(offsets[d_i] + jitter, seed_values, color=COLORS[condition], s=18, alpha=0.7, zorder=3, edgecolors="white", linewidths=0.3)
            for seed, value in zip(SEEDS, seed_values):
                plotted.append(
                    {
                        "panel": "primary_bars",
                        "metric": "response_error_degrees",
                        "delay": delay,
                        "condition": condition,
                        "strength": PRIMARY_RMS,
                        "seed": seed,
                        "seed_mean": value,
                        "grand_mean": mean,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
        ax_bars.bar(offsets, means, width=width, color=COLORS[condition], alpha=0.78, label=SHORT_LABELS[condition], edgecolor="white", linewidth=0.6)
        ax_bars.errorbar(
            offsets,
            means,
            yerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)],
            fmt="none",
            ecolor="0.15",
            capsize=2.5,
            linewidth=1.1,
        )

    ax_bars.set_xticks(x, [f"Delay {d} steps" for d in delays])
    ax_bars.set_ylabel("Absolute response error (degrees)")
    ax_bars.set_title(f"A. Main result at matched RMS = {PRIMARY_RMS}: structured noise hurts more, especially at long delays", loc="left", fontsize=10.5)
    ax_bars.legend(
        handles=[
            Patch(facecolor=COLORS[c], edgecolor="white", label=SHORT_LABELS[c]) for c in CONDITIONS
        ]
        + [
            Line2D([0], [0], marker="o", color="0.35", linestyle="None", markersize=5, label="One trained model (seed)"),
            Line2D([0], [0], color="0.15", linewidth=1.2, label="Error bar: 95% CI across 5 models"),
        ],
        frameon=False,
        ncol=3,
        fontsize=7.5,
        loc="upper left",
    )
    style_axis(ax_bars)

    # Dose panel: ONLY four mean curves + visible CI (no seed spaghetti).
    for condition in CONDITIONS:
        means, lows, highs = [], [], []
        for strength in strengths:
            mean, low, high, seed_values = mean_ci(response, PRIMARY_DELAY, condition, strength)
            means.append(mean)
            lows.append(low)
            highs.append(high)
            for seed, value in zip(SEEDS, seed_values):
                plotted.append(
                    {
                        "panel": "dose_delay80",
                        "metric": "response_error_degrees",
                        "delay": PRIMARY_DELAY,
                        "condition": condition,
                        "strength": strength,
                        "seed": seed,
                        "seed_mean": value,
                        "grand_mean": mean,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
        means_a = np.array(means)
        lows_a = np.array(lows)
        highs_a = np.array(highs)
        ax_dose.fill_between(strengths, lows_a, highs_a, color=COLORS[condition], alpha=0.38, linewidth=0, zorder=1)
        ax_dose.plot(strengths, lows_a, color=COLORS[condition], linewidth=1.5, linestyle="--", alpha=1.0, zorder=2)
        ax_dose.plot(strengths, highs_a, color=COLORS[condition], linewidth=1.5, linestyle="--", alpha=1.0, zorder=2)
        ax_dose.plot(strengths, means_a, color=COLORS[condition], linewidth=2.8, solid_capstyle="round", label=SHORT_LABELS[condition], zorder=3)

    ax_dose.set_xlabel(RMS_AXIS_LABEL)
    ax_dose.set_ylabel("Absolute response error (degrees)")
    ax_dose.set_title(f"B. Dose–response at delay {PRIMARY_DELAY} (solid = mean of 5 models; dashed = 95% CI bounds)", loc="left", fontsize=10)
    ax_dose.legend(frameon=False, fontsize=8, loc="upper left")
    style_axis(ax_dose)

    # Fixation: categorical bars, not ambiguous black dots.
    for idx, condition in enumerate(CONDITIONS):
        mean, low, high, seed_values = mean_ci(fixation, PRIMARY_DELAY, condition, PRIMARY_RMS)
        ax_fix.bar(idx, mean, color=COLORS[condition], alpha=0.75, edgecolor="white")
        ax_fix.errorbar(idx, mean, yerr=[[mean - low], [high - mean]], fmt="none", ecolor="0.15", capsize=3, linewidth=1.1)
        ax_fix.scatter(np.full(5, idx) + np.linspace(-0.12, 0.12, 5), seed_values, color="0.15", s=16, zorder=4)
        for seed, value in zip(SEEDS, seed_values):
            plotted.append(
                {
                    "panel": "fixation",
                    "metric": "fixation_accuracy",
                    "delay": PRIMARY_DELAY,
                    "condition": condition,
                    "strength": PRIMARY_RMS,
                    "seed": seed,
                    "seed_mean": value,
                    "grand_mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    ax_fix.set_xticks(range(len(CONDITIONS)), [SHORT_LABELS[c] for c in CONDITIONS], rotation=20)
    ax_fix.set_ylabel("Fraction of timesteps\nwith correct wait/go gating")
    ax_fix.set_ylim(0.94, 1.005)
    ax_fix.set_title("C. Fixation rule still works", loc="left", fontsize=10)
    ax_fix.legend(
        handles=[
            Line2D([0], [0], marker="o", color="0.15", linestyle="None", markersize=5, label="One model"),
            Line2D([0], [0], color="0.15", linewidth=1.2, label="95% CI across 5 models"),
        ],
        frameon=False,
        fontsize=7,
        loc="lower left",
    )
    style_axis(ax_fix)

    fig.suptitle("Structured noise impairs remembered angle more than matched random noise", fontsize=13)
    fig.text(0.5, -0.02, RMS_NOTE + "  CI = bootstrap interval over the five independently trained models.", ha="center", va="top", fontsize=8, color="0.3")
    save_figure(fig, figure_dir, "figure_03_behavioural_hero")
    write_csv(data_dir / "figure_03_behavioural_hero.csv", plotted)

    table_rows = []
    for delay in delays:
        row: dict[str, Any] = {"delay": delay}
        for condition in CONDITIONS:
            mean, _, _, seed_values = mean_ci(response, delay, condition, PRIMARY_RMS)
            row[condition] = mean
            row[f"{condition}_sd"] = float(seed_values.std(ddof=1))
            dec_mean, _, _, dec_values = mean_ci(decoder, delay, condition, PRIMARY_RMS)
            row[f"{condition}_decoder"] = dec_mean
            row[f"{condition}_decoder_sd"] = float(dec_values.std(ddof=1))
        table_rows.append(row)
    write_csv(data_dir / "table_01_response_error_rms05.csv", table_rows)


def plot_delay_timecourse(time_rows: list[dict[str, Any]], figure_dir: Path, data_dir: Path) -> None:
    grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in time_rows:
        grouped[(row["condition"], int(row["seed"]), int(row["time"]))].append(float(row["decoder_error_degrees"]))
    times = sorted({int(r["time"]) for r in time_rows})
    times_a = np.array(times)
    plotted: list[dict[str, Any]] = []

    fig, ax = plt.subplots(figsize=(9.4, 4.4), constrained_layout=True)
    # Neutral-but-visible epoch bands; avoid yellow fill that clashes with Temporal.
    ax.axvspan(25, 44, color="#E8F5E9", alpha=0.85, zorder=-2)
    ax.axvspan(45, 124, color="#FFF3E0", alpha=0.7, zorder=-2)
    ax.axvspan(125, 149, color="#E8EAF6", alpha=0.8, zorder=-2)
    ax.axvline(45, color="0.25", linewidth=1.4, linestyle="--")
    ax.axvline(125, color="0.25", linewidth=1.4, linestyle="--")

    for condition in CONDITIONS:
        matrix = np.array([[np.nanmean(grouped[(condition, seed, t)]) for t in times] for seed in SEEDS])
        mean = np.nanmean(matrix, axis=0)
        sem = np.nanstd(matrix, axis=0, ddof=1) / np.sqrt(len(SEEDS))
        low = mean - 1.96 * sem
        high = mean + 1.96 * sem
        # Light fill + strong dashed bounds (dashes carry the CI when colours overlap).
        ax.fill_between(times_a, low, high, color=COLORS[condition], alpha=0.18, linewidth=0, zorder=1)
        ax.plot(times_a, low, color=COLORS[condition], linestyle="--", linewidth=1.7, alpha=1.0, zorder=2)
        ax.plot(times_a, high, color=COLORS[condition], linestyle="--", linewidth=1.7, alpha=1.0, zorder=2)
        ax.plot(times_a, mean, color=COLORS[condition], linestyle="-", linewidth=2.7, label=LABELS[condition], zorder=3)
        for t_idx, t in enumerate(times):
            plotted.append({"condition": condition, "time": t, "grand_mean": float(mean[t_idx]), "ci_low": float(low[t_idx]), "ci_high": float(high[t_idx])})

    ax.set_xlim(40, 149)
    ax.set_ylim(0, 46)
    ax.text(34.5, 44.5, "Cue", ha="center", va="top", fontsize=9, color="0.2")
    ax.text(84, 44.5, "Delay (noise active)", ha="center", va="top", fontsize=9, color="0.1", fontweight="bold")
    ax.text(137, 44.5, "Response", ha="center", va="top", fontsize=9, color="0.2")
    ax.set_xlabel("Trial time step")
    ax.set_ylabel("Error of a clean-trained decoder (°)\nreading the remembered angle")
    ax.legend(
        handles=[Line2D([0], [0], color=COLORS[c], linewidth=2.4, label=LABELS[c]) for c in CONDITIONS]
        + [
            Line2D([0], [0], color="0.35", linewidth=1.2, linestyle="--", label="Dashed: 95% CI across 5 models"),
            Patch(facecolor="0.5", alpha=0.2, label="Shaded band between CI bounds"),
        ],
        frameon=False,
        ncol=2,
        fontsize=7.5,
        loc="upper left",
    )
    style_axis(ax)
    fig.suptitle(f"Memory code degrades during the hold under structured noise (delay {PRIMARY_DELAY}, RMS {PRIMARY_RMS})", fontsize=12)
    save_figure(fig, figure_dir, "figure_04_delay_timecourse")
    write_csv(data_dir / "figure_04_delay_timecourse.csv", plotted)


def _confidence_ellipse(points: np.ndarray, ax: plt.Axes, color: str) -> None:
    if len(points) < 2:
        return
    cov = np.cov(points.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * 1.96 * np.sqrt(np.maximum(vals, 0) / len(points))
    ax.add_patch(Ellipse(points.mean(0), width, height, angle=angle, facecolor=color, edgecolor=color, alpha=0.15, linewidth=1))


def plot_ring_geometry(
    centroid_rows: list[dict[str, Any]],
    geometry_rows: list[dict[str, Any]],
    figure_dir: Path,
    data_dir: Path,
) -> None:
    fig = plt.figure(figsize=(10.4, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1.05])
    top = [fig.add_subplot(gs[0, i]) for i in range(4)]
    bottom = [fig.add_subplot(gs[1, :2]), fig.add_subplot(gs[1, 2:])]
    plotted: list[dict[str, Any]] = []

    for ax, condition in zip(top, CONDITIONS):
        condition_means = []
        clean_means = []
        for b in range(16):
            pts = np.array(
                [[r["condition_pc1"], r["condition_pc2"]] for r in centroid_rows if r["condition"] == condition and int(r["angle_bin"]) == b]
            )
            clean = np.array(
                [[r["clean_pc1"], r["clean_pc2"]] for r in centroid_rows if r["condition"] == condition and int(r["angle_bin"]) == b]
            )
            p = pts.mean(0)
            c = clean.mean(0)
            _confidence_ellipse(pts, ax, COLORS[condition])
            ax.annotate("", xy=p, xytext=c, arrowprops={"arrowstyle": "->", "color": COLORS[condition], "lw": 0.9, "alpha": 0.75})
            ax.scatter(*p, color=COLORS[condition], s=18, zorder=3)
            ax.scatter(*c, color="0.35", s=10, alpha=0.55, zorder=2)
            condition_means.append(p)
            clean_means.append(c)
        condition_means = np.array(condition_means)
        clean_means = np.array(clean_means)
        ax.plot(*np.vstack((condition_means, condition_means[0])).T, color=COLORS[condition], linewidth=1.0, alpha=0.7)
        ax.plot(*np.vstack((clean_means, clean_means[0])).T, color="0.4", linewidth=0.7, alpha=0.4)
        ax.set_title(SHORT_LABELS[condition], fontsize=10, color=COLORS[condition])
        ax.set_aspect("equal")
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.7, 1.7)
        ax.set_xlabel("PC1 of clean memory plane")
        ax.set_ylabel("PC2 of clean memory plane" if condition == "unperturbed" else "")
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        style_axis(ax)

    metrics = (
        (
            "within_angle_dispersion",
            "How spread out are trials\nthat should be the same angle?\n(larger = blurrier memory cloud)",
            False,
        ),
        (
            "separation_normalized_to_clean",
            "How far apart are different angles?\n(1 = same as clean; lower =\nangles harder to tell apart)",
            True,
        ),
    )
    for ax, (metric, ylabel, draw_ref) in zip(bottom, metrics):
        means = []
        for idx, condition in enumerate(CONDITIONS):
            seed_values = []
            for seed in SEEDS:
                values = [float(r[metric]) for r in geometry_rows if r["condition"] == condition and int(r["seed"]) == seed]
                seed_values.append(float(np.mean(values)))
            seed_values_arr = np.array(seed_values)
            mean = float(seed_values_arr.mean())
            low, high = bootstrap_ci(seed_values_arr, seed=idx + 72)
            means.append(mean)
            ax.bar(idx, mean, color=COLORS[condition], alpha=0.78, edgecolor="white", width=0.72)
            ax.errorbar(idx, mean, yerr=[[mean - low], [high - mean]], fmt="none", ecolor="0.15", capsize=3, linewidth=1.1)
            ax.scatter(np.full(5, idx) + np.linspace(-0.15, 0.15, 5), seed_values_arr, color="0.15", s=16, zorder=4)
            plotted.append(
                {
                    "metric": metric,
                    "condition": condition,
                    "grand_mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    **{f"seed_{seed}": value for seed, value in zip(SEEDS, seed_values_arr)},
                }
            )
        if draw_ref:
            ax.axhline(1.0, color="0.45", linestyle=":", linewidth=1.2)
            ax.text(3.35, 1.02, "clean reference", fontsize=7.5, color="0.4", ha="right")
        ax.set_xticks(range(len(CONDITIONS)), [SHORT_LABELS[c] for c in CONDITIONS])
        ax.set_xlabel("Noise condition (four separate experiments, not a continuous axis)")
        ax.set_ylabel(ylabel)
        style_axis(ax)

    bottom[0].legend(
        handles=[
            Line2D([0], [0], marker="o", color="0.15", linestyle="None", markersize=5, label="One trained model"),
            Line2D([0], [0], color="0.15", linewidth=1.2, label="95% CI across 5 models"),
        ],
        frameon=False,
        fontsize=7.5,
        loc="upper left",
    )
    fig.suptitle(
        f"The circular memory map gets blurrier and more collapsed under structured noise (delay {PRIMARY_DELAY}, RMS {PRIMARY_RMS})",
        fontsize=12,
    )
    fig.text(
        0.5,
        -0.01,
        "Top: each coloured point is the average hidden-state location for one remembered angle. Grey = clean reference. "
        "Bottom: summary scores for the same four conditions.",
        ha="center",
        va="top",
        fontsize=8,
        color="0.3",
    )
    save_figure(fig, figure_dir, "figure_05_ring_geometry")
    write_csv(data_dir / "figure_05_ring_geometry.csv", plotted)


def plot_epoch_sensitivity(rows: list[dict[str, str]], figure_dir: Path, data_dir: Path) -> None:
    selected = [r for r in rows if r["condition"] == "context_topology_correlated"]
    metrics = (
        ("response_error_degrees", "Absolute response error (degrees)"),
        ("delay_decoder_error_degrees", "Clean-decoder error during delay (°)"),
    )
    phases = ("cue", "delay", "response")
    plotted: list[dict[str, Any]] = []
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)

    for ax, (metric, ylabel) in zip(axes, metrics):
        grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
        for row in selected:
            if abs(float(row["strength"]) - PRIMARY_RMS) > 1e-12:
                continue
            grouped[(row["phase"], int(row["seed"]))].append(float(row[metric]))
        for idx, phase in enumerate(phases):
            seed_values = np.array([np.mean(grouped[(phase, seed)]) for seed in SEEDS])
            mean = float(seed_values.mean())
            low, high = bootstrap_ci(seed_values, seed=100 + idx)
            ax.bar(idx, mean, color=PHASE_COLORS[phase], alpha=0.75, edgecolor="white", width=0.7)
            ax.errorbar(idx, mean, yerr=[[mean - low], [high - mean]], fmt="none", ecolor="0.15", capsize=3, linewidth=1.1)
            ax.scatter(np.full(5, idx) + np.linspace(-0.15, 0.15, 5), seed_values, color="0.15", s=18, zorder=4)
            for seed, value in zip(SEEDS, seed_values):
                plotted.append(
                    {
                        "metric": metric,
                        "phase": phase,
                        "strength": PRIMARY_RMS,
                        "seed": seed,
                        "seed_mean": float(value),
                        "grand_mean": mean,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
        ax.set_xticks(range(3), ["Cue only", "Delay only", "Response only"])
        ax.set_xlabel("When topology noise was applied (one phase per condition)")
        ax.set_ylabel(ylabel)
        style_axis(ax)

    axes[0].set_title("A. Final behavioural error", loc="left", fontsize=11)
    axes[1].set_title("B. Quality of the maintained memory code", loc="left", fontsize=11)
    axes[0].legend(
        handles=[
            Line2D([0], [0], marker="o", color="0.15", linestyle="None", markersize=5, label="One trained model"),
            Line2D([0], [0], color="0.15", linewidth=1.2, label="95% CI across 5 models"),
        ],
        frameon=False,
        fontsize=7.5,
        loc="upper left",
    )
    fig.suptitle(f"Topology noise is most damaging during memory maintenance (delay {PRIMARY_DELAY}, RMS {PRIMARY_RMS})", fontsize=12)
    fig.text(
        0.5,
        -0.04,
        "This is task-phase context in the model, not human experiential context under psilocybin.",
        ha="center",
        va="top",
        fontsize=8,
        color="0.3",
    )
    save_figure(fig, figure_dir, "figure_06_epoch_sensitivity")
    write_csv(data_dir / "figure_06_epoch_sensitivity.csv", plotted)


def _load_hidden_rows(data_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    time_rows = [
        {
            **r,
            "seed": int(r["seed"]),
            "replicate": int(r["replicate"]),
            "time": int(r["time"]),
            "decoder_error_degrees": float(r["decoder_error_degrees"]),
            "hidden_speed": float(r["hidden_speed"]),
        }
        for r in read_csv(data_dir / "hidden_timecourse.csv")
    ]
    centroid_rows = [
        {
            **r,
            "seed": int(r["seed"]),
            "angle_bin": int(r["angle_bin"]),
            "condition_pc1": float(r["condition_pc1"]),
            "condition_pc2": float(r["condition_pc2"]),
            "clean_pc1": float(r["clean_pc1"]),
            "clean_pc2": float(r["clean_pc2"]),
        }
        for r in read_csv(data_dir / "hidden_centroids.csv")
    ]
    geometry_rows = [
        {
            **r,
            "seed": int(r["seed"]),
            "replicate": int(r["replicate"]),
            "within_angle_dispersion": float(r["within_angle_dispersion"]),
            "separation_normalized_to_clean": float(r["separation_normalized_to_clean"]),
        }
        for r in read_csv(data_dir / "hidden_geometry.csv")
    ]
    return time_rows, centroid_rows, geometry_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/yang_fixation_circular_working_memory.yaml")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/yang_fixation_circular_working_memory/perturbation_experiments/noise_structure"),
    )
    parser.add_argument(
        "--archive-hidden",
        type=Path,
        default=Path("outputs/archive/noise_structure_initial_figure_suite_2026-07-14/figure_data"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    figure_dir = args.root / "figures"
    data_dir = args.root / "figure_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    for name in ("hidden_timecourse.csv", "hidden_centroids.csv", "hidden_geometry.csv"):
        src = args.archive_hidden / name
        dst = data_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    main_rows = read_csv(args.root / "full" / "summary_metrics.csv")
    timing_rows = read_csv(args.root / "epoch_timing_full" / "summary_metrics.csv")
    time_rows, centroid_rows, geometry_rows = _load_hidden_rows(data_dir)

    plot_schematic(figure_dir)
    plot_manipulation_validation(config, figure_dir, data_dir)
    plot_behavioural_hero(main_rows, figure_dir, data_dir)
    plot_delay_timecourse(time_rows, figure_dir, data_dir)
    plot_ring_geometry(centroid_rows, geometry_rows, figure_dir, data_dir)
    plot_epoch_sensitivity(timing_rows, figure_dir, data_dir)
    print(f"Wrote dissertation figures to {figure_dir}")


if __name__ == "__main__":
    main()
