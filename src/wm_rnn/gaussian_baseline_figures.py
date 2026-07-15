"""Gaussian-noise vs clean baseline figure pack for supervisor progress.

Uses the completed structured-noise CSVs (clean + independent Gaussian only).
Optionally runs a focused CUDA settling-time analysis on the five frozen seeds.
"""

from __future__ import annotations

import argparse
import csv
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
from matplotlib.patches import Ellipse, Patch

from wm_rnn.config import load_config
from wm_rnn.noise_structure_experiment import (
    COLORS,
    generate_perturbations,
    phase_mask,
    require_cuda,
)
from wm_rnn.training_utils import batch_to_tensors, fresh_model, generate_batch_for_task, task_config_from_dict
from wm_rnn.tuned_task import circular_angular_error, decode_population_angle

PAIR = ("unperturbed", "independent_gaussian")
LABELS = {
    "unperturbed": "Clean baseline",
    "independent_gaussian": "Gaussian noise",
}
SEEDS = (20260714, 20260715, 20260716, 20260717, 20260718)
PRIMARY_DELAY = 80
STRENGTHS = (0.0, 0.01, 0.025, 0.05, 0.1)
DELAYS = (20, 80, 160)
RMS_NOTE = (
    "RMS = matched root-mean-square size of additive hidden-state noise. "
    "Not a drug dose. Seed is the replication unit (n = 5 models)."
)

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 9,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9})


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


def bootstrap_ci(values: np.ndarray, seed: int = 0, draws: int = 5000) -> tuple[float, float]:
    if len(values) < 2:
        v = float(values[0])
        return v, v
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def seed_means(rows: list[dict[str, str]], metric: str) -> dict[tuple[int, str, float, int], float]:
    grouped: dict[tuple[int, str, float, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["condition"] not in PAIR:
            continue
        key = (int(row["delay"]), row["condition"], float(row["strength"]), int(row["seed"]))
        grouped[key].append(float(row[metric]))
    return {key: float(np.mean(vals)) for key, vals in grouped.items()}


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.6)


def mean_ci_curve(
    summary: dict[tuple[int, str, float, int], float],
    delay: int,
    condition: str,
    strengths: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means, lows, highs = [], [], []
    for strength in strengths:
        values = np.array([summary[(delay, condition, strength, seed)] for seed in SEEDS])
        mean = float(values.mean())
        low, high = bootstrap_ci(values, seed=delay + int(strength * 10000))
        means.append(mean)
        lows.append(low)
        highs.append(high)
    return np.array(means), np.array(lows), np.array(highs)


def plot_dose_delay_behaviour(rows: list[dict[str, str]], figure_dir: Path, data_dir: Path) -> None:
    summary = seed_means(rows, "response_error_degrees")
    plotted: list[dict[str, Any]] = []
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 5.65), sharex=True, constrained_layout=False)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.96, bottom=0.08, hspace=0.30)
    all_lows, all_highs = [], []
    for ax, delay in zip(axes, DELAYS):
        for condition in PAIR:
            means, lows, highs = mean_ci_curve(summary, delay, condition, STRENGTHS)
            all_lows.extend(lows); all_highs.extend(highs)
            ax.fill_between(STRENGTHS, lows, highs, color=COLORS[condition], alpha=0.28, linewidth=0)
            ax.plot(STRENGTHS, lows, color=COLORS[condition], linestyle="--", linewidth=1.3)
            ax.plot(STRENGTHS, highs, color=COLORS[condition], linestyle="--", linewidth=1.3)
            ax.plot(STRENGTHS, means, color=COLORS[condition], linewidth=2.5, label=LABELS[condition])
            for strength, mean, low, high in zip(STRENGTHS, means, lows, highs):
                for seed in SEEDS:
                    value = summary[(delay, condition, strength, seed)]
                    plotted.append(
                        {
                            "delay": delay,
                            "condition": condition,
                            "strength": strength,
                            "seed": seed,
                            "seed_mean": value,
                            "grand_mean": mean,
                            "ci_low": low,
                            "ci_high": high,
                        }
                    )
        ax.set_title(f"Delay {delay}", fontsize=11, loc="left")
        ax.set_xlabel("Perturbation strength (RMS)")
        style_axis(ax)
    y_min, y_max = min(all_lows), max(all_highs)
    pad = 0.06 * (y_max - y_min)
    for ax in axes:
        ax.set_ylim(y_min - pad, y_max + pad)
    axes[0].set_ylabel("Absolute response error (°)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    save_figure(fig, figure_dir, "figure_g1_dose_delay_behaviour")
    write_csv(data_dir / "figure_g1_dose_delay_behaviour.csv", plotted)


def plot_maintenance_metrics(rows: list[dict[str, str]], figure_dir: Path, data_dir: Path) -> None:
    metrics = (
        ("delay_decoder_error_degrees", "Clean-decoder error during delay (°)"),
        ("memory_drift_degrees", "Decoded angle drift over delay (°)"),
        ("fixation_accuracy", "Fixation / wait–go accuracy"),
    )
    plotted: list[dict[str, Any]] = []
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 7.2), sharex=True, constrained_layout=False)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.96, bottom=0.06, hspace=0.42)
    for ax, (metric, ylabel) in zip(axes, metrics):
        summary = seed_means(rows, metric)
        for condition in PAIR:
            means, lows, highs = mean_ci_curve(summary, PRIMARY_DELAY, condition, STRENGTHS)
            ax.fill_between(STRENGTHS, lows, highs, color=COLORS[condition], alpha=0.28, linewidth=0)
            ax.plot(STRENGTHS, lows, color=COLORS[condition], linestyle="--", linewidth=1.3)
            ax.plot(STRENGTHS, highs, color=COLORS[condition], linestyle="--", linewidth=1.3)
            ax.plot(STRENGTHS, means, color=COLORS[condition], linewidth=2.5, label=LABELS[condition])
            for strength, mean, low, high in zip(STRENGTHS, means, lows, highs):
                for seed in SEEDS:
                    plotted.append(
                        {
                            "metric": metric,
                            "delay": PRIMARY_DELAY,
                            "condition": condition,
                            "strength": strength,
                            "seed": seed,
                            "seed_mean": summary[(PRIMARY_DELAY, condition, strength, seed)],
                            "grand_mean": mean,
                            "ci_low": low,
                            "ci_high": high,
                        }
                    )
        ax.set_xlabel("Perturbation strength (RMS)")
        ax.set_ylabel(ylabel)
        style_axis(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    if axes[2].get_ylim()[0] > 0.9:
        axes[2].set_ylim(0.94, 1.005)
    save_figure(fig, figure_dir, "figure_g2_maintenance_metrics")
    write_csv(data_dir / "figure_g2_maintenance_metrics.csv", plotted)


def _confidence_ellipse(points: np.ndarray, ax: plt.Axes, color: str) -> None:
    if len(points) < 2:
        return
    cov = np.cov(points.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * 1.96 * np.sqrt(np.maximum(vals, 0) / len(points))
    ax.add_patch(Ellipse(points.mean(0), width, height, angle=angle, facecolor=color, edgecolor=color, alpha=0.15, linewidth=1))


def plot_geometry(centroid_rows: list[dict[str, Any]], geometry_rows: list[dict[str, Any]], figure_dir: Path, data_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 3.35), constrained_layout=False)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.98, bottom=0.16)
    plotted = []
    clean_means = []
    for b in range(16):
        clean = np.array([[r["clean_pc1"], r["clean_pc2"]] for r in centroid_rows if int(r["angle_bin"]) == b])
        c = clean.mean(0); clean_means.append(c); ax.scatter(*c, color="0.25", s=18, zorder=3)
    clean_means = np.array(clean_means)
    ax.plot(*np.vstack((clean_means, clean_means[0])).T, color="0.25", linestyle="--", linewidth=1.2, label="Clean baseline")
    condition_means = []
    for b in range(16):
        pts = np.array([[r["condition_pc1"], r["condition_pc2"]] for r in centroid_rows if r["condition"] == "independent_gaussian" and int(r["angle_bin"]) == b])
        p = pts.mean(0); condition_means.append(p); _confidence_ellipse(pts, ax, COLORS["independent_gaussian"])
        ax.scatter(*p, color=COLORS["independent_gaussian"], s=20, zorder=4)
        plotted.append({"condition": "independent_gaussian", "angle_bin": b, "condition_pc1_mean": float(p[0]), "condition_pc2_mean": float(p[1])})
    condition_means = np.array(condition_means)
    ax.plot(*np.vstack((condition_means, condition_means[0])).T, color=COLORS["independent_gaussian"], linewidth=1.8, alpha=0.62, label="Gaussian noise")
    ax.set_aspect("equal"); ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.legend(frameon=False, fontsize=9, loc="upper right"); style_axis(ax)
    save_figure(fig, figure_dir, "figure_g3_geometry"); write_csv(data_dir / "figure_g3_geometry.csv", plotted)


def _old_plot_geometry(centroid_rows: list[dict[str, Any]], geometry_rows: list[dict[str, Any]], figure_dir: Path, data_dir: Path) -> None:
    fig = plt.figure(figsize=(9.6, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
    top = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    bottom = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    plotted: list[dict[str, Any]] = []

    for ax, condition in zip(top, PAIR):
        condition_means, clean_means = [], []
        for b in range(16):
            pts = np.array(
                [
                    [r["condition_pc1"], r["condition_pc2"]]
                    for r in centroid_rows
                    if r["condition"] == condition and int(r["angle_bin"]) == b
                ]
            )
            clean = np.array(
                [
                    [r["clean_pc1"], r["clean_pc2"]]
                    for r in centroid_rows
                    if r["condition"] == condition and int(r["angle_bin"]) == b
                ]
            )
            p, c = pts.mean(0), clean.mean(0)
            _confidence_ellipse(pts, ax, COLORS[condition])
            ax.annotate("", xy=p, xytext=c, arrowprops={"arrowstyle": "->", "color": COLORS[condition], "lw": 0.9, "alpha": 0.75})
            ax.scatter(*p, color=COLORS[condition], s=20, zorder=3)
            ax.scatter(*c, color="0.35", s=10, alpha=0.55, zorder=2)
            condition_means.append(p)
            clean_means.append(c)
        condition_means = np.array(condition_means)
        clean_means = np.array(clean_means)
        ax.plot(*np.vstack((condition_means, condition_means[0])).T, color=COLORS[condition], linewidth=1.0, alpha=0.7)
        ax.plot(*np.vstack((clean_means, clean_means[0])).T, color="0.4", linewidth=0.7, alpha=0.4)
        ax.set_title(LABELS[condition], color=COLORS[condition], fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.7, 1.7)
        ax.set_xlabel("PC1 of clean memory plane")
        ax.set_ylabel("PC2 of clean memory plane")
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        style_axis(ax)

    metric_specs = (
        ("within_angle_dispersion", "Same-angle spread\n(larger = blurrier)"),
        ("separation_normalized_to_clean", "Between-angle spacing\n(1 = same as clean)"),
    )
    for ax, (metric, ylabel) in zip(bottom, metric_specs):
        for idx, condition in enumerate(PAIR):
            seed_values = []
            for seed in SEEDS:
                vals = [
                    float(r[metric])
                    for r in geometry_rows
                    if r["condition"] == condition and int(r["seed"]) == seed
                ]
                seed_values.append(float(np.mean(vals)))
            arr = np.array(seed_values)
            mean = float(arr.mean())
            low, high = bootstrap_ci(arr, seed=40 + idx)
            ax.bar(idx, mean, color=COLORS[condition], alpha=0.8, edgecolor="white", width=0.65)
            ax.errorbar(idx, mean, yerr=[[mean - low], [high - mean]], fmt="none", ecolor="0.15", capsize=3)
            ax.scatter(np.full(5, idx) + np.linspace(-0.14, 0.14, 5), arr, color="0.15", s=18, zorder=4)
            plotted.append({"metric": metric, "condition": condition, "grand_mean": mean, "ci_low": low, "ci_high": high})
        if metric.endswith("clean"):
            ax.axhline(1.0, color="0.45", linestyle=":", linewidth=1.2)
        ax.set_xticks([0, 1], [LABELS[c] for c in PAIR])
        ax.set_xlabel("Condition (categorical)")
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
    fig.suptitle(f"Gaussian noise mildly blurs the ring (delay {PRIMARY_DELAY}, RMS 0.05)", fontsize=12)
    fig.text(
        0.5,
        -0.02,
        "Top: angle centroids in clean PCA space (grey = clean reference). Bottom: summary scores.",
        ha="center",
        fontsize=8,
        color="0.3",
    )
    save_figure(fig, figure_dir, "figure_g3_geometry")
    write_csv(data_dir / "figure_g3_geometry.csv", plotted)


def write_summary_table(rows: list[dict[str, str]], data_dir: Path) -> None:
    metrics = (
        "response_error_degrees",
        "delay_decoder_error_degrees",
        "memory_drift_degrees",
        "fixation_accuracy",
    )
    out_rows = []
    for delay in DELAYS:
        for strength in (0.0, 0.05, 0.1):
            row: dict[str, Any] = {"delay": delay, "strength": strength}
            for metric in metrics:
                summary = seed_means(rows, metric)
                for condition in PAIR:
                    values = np.array([summary[(delay, condition, strength, seed)] for seed in SEEDS])
                    row[f"{condition}_{metric}_mean"] = float(values.mean())
                    row[f"{condition}_{metric}_sd"] = float(values.std(ddof=1))
            out_rows.append(row)
    write_csv(data_dir / "table_gaussian_vs_clean_summary.csv", out_rows)


def compute_settling_times(
    config: dict[str, Any],
    delay: int = PRIMARY_DELAY,
    strengths: tuple[float, ...] = (0.0, 0.05, 0.1),
    trials: int = 128,
    threshold_deg: float = 20.0,
    skip_transition: int = 5,
) -> list[dict[str, Any]]:
    """First response step (after transition) with absolute angular error < threshold."""
    device = require_cuda()
    rows: list[dict[str, Any]] = []
    checkpoint_root = Path(config["paths"]["output_dir"]) / "seed_sweep"
    for seed in SEEDS:
        local = {**config, "task": dict(config["task"])}
        local["task"]["seed"] = seed
        model = fresh_model(local, device)
        checkpoint = (
            checkpoint_root
            / f"seed_{seed}"
            / "checkpoints"
            / f"yang_fixation_circular_working_memory_seed_{seed}.pt"
        )
        model.load_state_dict(torch.load(checkpoint, map_location=device)["model_state"])
        model.eval()
        task = replace(task_config_from_dict(local, batch_size=trials), delay_steps=delay, seed=seed + 880000)
        batch = generate_batch_for_task(task)
        inputs, _, _ = batch_to_tensors(batch, device)
        mask = phase_mask(inputs.size(0), batch.phase_index, "delay", device)
        response = batch.phase_index["response"]
        n_resp = response.stop - response.start
        for c_idx, condition in enumerate(PAIR):
            for strength in strengths:
                effective_strength = 0.0 if condition == "unperturbed" else float(strength)
                gen = torch.Generator(device=device).manual_seed(seed + 17 * c_idx + int(strength * 1000))
                noise = generate_perturbations(
                    condition if effective_strength > 0 else "unperturbed",
                    (inputs.size(0), trials, model.config.hidden_size),
                    effective_strength,
                    mask,
                    gen,
                    model.rnn.h2h.weight,
                )
                with torch.no_grad():
                    logits, _ = model(inputs, perturbations=noise)
                pop = logits[response, :, : len(batch.preferred_angles)].detach().cpu().numpy()
                decoded = decode_population_angle(pop, batch.preferred_angles)
                errors = np.degrees(circular_angular_error(decoded, batch.angles.reshape(1, -1)))
                settle = np.full(trials, n_resp, dtype=np.float64)
                settled = np.zeros(trials, dtype=bool)
                start = min(skip_transition, n_resp - 1)
                for trial in range(trials):
                    for t in range(start, n_resp):
                        if errors[t, trial] < threshold_deg:
                            settle[trial] = t
                            settled[trial] = True
                            break
                rows.append(
                    {
                        "seed": seed,
                        "delay": delay,
                        "condition": condition,
                        "strength": effective_strength if condition == "independent_gaussian" else 0.0,
                        "threshold_degrees": threshold_deg,
                        "skip_transition_steps": skip_transition,
                        "mean_settling_steps": float(settle.mean()),
                        "median_settling_steps": float(np.median(settle)),
                        "fraction_settled": float(settled.mean()),
                        "mean_final_error_degrees": float(errors[-1].mean()),
                        "n_trials": trials,
                    }
                )
    return rows


def plot_settling(settling_rows: list[dict[str, Any]], figure_dir: Path, data_dir: Path) -> None:
    write_csv(data_dir / "figure_g4_settling.csv", settling_rows)
    strengths = sorted({float(r["strength"]) for r in settling_rows if r["condition"] == "independent_gaussian"} | {0.0})
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 5.4), sharex=True, constrained_layout=False)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.96, bottom=0.07, hspace=0.42)
    for ax, metric, ylabel in ((axes[0], "mean_settling_steps", "Mean settling time (steps)"), (axes[1], "mean_final_error_degrees", "Final response error (°)")):
        for condition in PAIR:
            xs=[]; means=[]; lows=[]; highs=[]
            for strength in strengths:
                if condition == "unperturbed" and strength != 0: continue
                x = 0.0 if condition == "unperturbed" else strength
                vals = np.array([float(r[metric]) for r in settling_rows if r["condition"] == condition and abs(float(r["strength"])-x)<1e-12])
                if len(vals)==0: continue
                xs.append(x); means.append(vals.mean()); lo,hi=bootstrap_ci(vals, seed=90+int(x*100)); lows.append(lo); highs.append(hi)
            ax.fill_between(xs,lows,highs,color=COLORS[condition],alpha=.25)
            if condition == "unperturbed":
                clean_level = float(means[0])
                ax.axhline(clean_level, color=COLORS[condition], linewidth=2.2, linestyle="-", zorder=3, label="Clean baseline")
                ax.plot([0.0], [clean_level], marker="o", color=COLORS[condition], markersize=5, zorder=4)
            else:
                ax.plot(xs,means,color=COLORS[condition],marker="o",linewidth=1.8,label=LABELS[condition])
            ax.plot(xs,lows,color=COLORS[condition],linestyle="--",linewidth=1); ax.plot(xs,highs,color=COLORS[condition],linestyle="--",linewidth=1)
        ax.set_ylabel(ylabel); style_axis(ax)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left"); axes[1].set_xlabel("Perturbation strength (RMS)")
    save_figure(fig, figure_dir, "figure_g4_settling")


def _old_plot_settling(settling_rows: list[dict[str, Any]], figure_dir: Path, data_dir: Path) -> None:
    write_csv(data_dir / "figure_g4_settling.csv", settling_rows)
    strengths = sorted({float(r["strength"]) for r in settling_rows if r["condition"] == "independent_gaussian"} | {0.0})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)

    for ax, metric, ylabel in (
        (axes[0], "mean_settling_steps", "Mean settling time\n(steps after response cue)"),
        (axes[1], "mean_final_error_degrees", "Final response error (°)"),
    ):
        for condition in PAIR:
            means, lows, highs = [], [], []
            xs = []
            for strength in strengths:
                if condition == "unperturbed" and strength != 0.0:
                    continue
                plot_strength = 0.0 if condition == "unperturbed" else strength
                values = np.array(
                    [
                        float(r[metric])
                        for r in settling_rows
                        if r["condition"] == condition and abs(float(r["strength"]) - plot_strength) < 1e-12
                    ]
                )
                if len(values) == 0:
                    continue
                mean = float(values.mean())
                low, high = bootstrap_ci(values, seed=90 + int(plot_strength * 100))
                xs.append(plot_strength)
                means.append(mean)
                lows.append(low)
                highs.append(high)
                ax.scatter(
                    np.full(len(values), plot_strength) + np.linspace(-0.004, 0.004, len(values)),
                    values,
                    color=COLORS[condition],
                    s=22,
                    alpha=0.75,
                    zorder=3,
                )
            if not xs:
                continue
            ax.fill_between(xs, lows, highs, color=COLORS[condition], alpha=0.25, linewidth=0)
            ax.plot(xs, means, color=COLORS[condition], linewidth=2.4, marker="o", label=LABELS[condition])
            ax.plot(xs, lows, color=COLORS[condition], linestyle="--", linewidth=1.2)
            ax.plot(xs, highs, color=COLORS[condition], linestyle="--", linewidth=1.2)
        ax.set_xlabel("Perturbation strength (RMS)")
        ax.set_ylabel(ylabel)
        style_axis(ax)

    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_title("A. Response settling (RT analogue)", loc="left", fontsize=10)
    axes[1].set_title("B. Final accuracy at end of response", loc="left", fontsize=10)
    fig.suptitle(
        f"Gaussian noise vs clean: settling after response cue (delay {PRIMARY_DELAY}; threshold 20°)",
        fontsize=12,
    )
    fig.text(
        0.5,
        -0.05,
        "Settling = first post-transition response step with absolute circular error < 20°. "
        "Noise was applied during delay only. Points = model seeds.",
        ha="center",
        fontsize=8,
        color="0.3",
    )
    save_figure(fig, figure_dir, "figure_g4_settling")


def write_markdown_summary(data_dir: Path, settling_rows: list[dict[str, Any]] | None) -> None:
    table = read_csv(data_dir / "table_gaussian_vs_clean_summary.csv")

    def grab(delay: int, strength: float, condition: str, metric: str) -> str:
        for row in table:
            if int(float(row["delay"])) == delay and abs(float(row["strength"]) - strength) < 1e-12:
                mean = float(row[f"{condition}_{metric}_mean"])
                sd = float(row[f"{condition}_{metric}_sd"])
                return f"{mean:.2f} ± {sd:.2f}"
        return "n/a"

    lines = [
        "# Gaussian Noise vs Clean Baseline",
        "",
        "Supervisor-facing control pack extracted from the completed five-seed",
        "structured-noise experiment. Temporal/topology conditions are omitted.",
        "",
        "## Headline numbers (delay 80, RMS 0.05)",
        "",
        f"- Response error (°): clean {grab(80, 0.05, 'unperturbed', 'response_error_degrees')}; "
        f"Gaussian {grab(80, 0.05, 'independent_gaussian', 'response_error_degrees')}",
        f"- Delay decoder error (°): clean {grab(80, 0.05, 'unperturbed', 'delay_decoder_error_degrees')}; "
        f"Gaussian {grab(80, 0.05, 'independent_gaussian', 'delay_decoder_error_degrees')}",
        f"- Decoded drift (°): clean {grab(80, 0.05, 'unperturbed', 'memory_drift_degrees')}; "
        f"Gaussian {grab(80, 0.05, 'independent_gaussian', 'memory_drift_degrees')}",
        f"- Fixation accuracy: clean {grab(80, 0.05, 'unperturbed', 'fixation_accuracy')}; "
        f"Gaussian {grab(80, 0.05, 'independent_gaussian', 'fixation_accuracy')}",
        "",
        "## Interpretation for the meeting",
        "",
        "- Gaussian noise is a **weak, nonspecific degradation control** at RMS 0.05.",
        "- Effects grow with delay and strength, but remain far milder than the benched",
        "  structured-noise conditions.",
        "- Use this pack to define ordinary stochastic disruption before testing",
        "  distractor / gain candidates against the literature signature.",
        "",
        "## Figures",
        "",
        "- `figure_g1_dose_delay_behaviour`",
        "- `figure_g2_maintenance_metrics`",
        "- `figure_g3_geometry`",
        "- `figure_g4_settling` (if generated)",
        "",
    ]
    if settling_rows:
        clean = [r for r in settling_rows if r["condition"] == "unperturbed"]
        gauss = [r for r in settling_rows if r["condition"] == "independent_gaussian" and abs(float(r["strength"]) - 0.05) < 1e-12]
        if clean and gauss:
            lines.extend(
                [
                    "## Settling (delay 80, threshold 20°)",
                    "",
                    f"- Clean mean settling steps: {np.mean([r['mean_settling_steps'] for r in clean]):.2f}",
                    f"- Gaussian RMS 0.05 mean settling steps: {np.mean([r['mean_settling_steps'] for r in gauss]):.2f}",
                    f"- Clean final error (°): {np.mean([r['mean_final_error_degrees'] for r in clean]):.2f}",
                    f"- Gaussian RMS 0.05 final error (°): {np.mean([r['mean_final_error_degrees'] for r in gauss]):.2f}",
                    "",
                ]
            )
    (data_dir / "gaussian_vs_baseline_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/yang_fixation_circular_working_memory.yaml")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/yang_fixation_circular_working_memory/perturbation_experiments/noise_structure"),
    )
    parser.add_argument("--skip-settling", action="store_true")
    args = parser.parse_args()

    figure_dir = args.root / "gaussian_control" / "figures"
    data_dir = args.root / "gaussian_control" / "figure_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    main_rows = read_csv(args.root / "full" / "summary_metrics.csv")
    # Prefer live figure_data; fall back to archive if needed.
    hidden_root = args.root / "figure_data"
    archive_hidden = Path("outputs/archive/noise_structure_initial_figure_suite_2026-07-14/figure_data")
    for name in ("hidden_centroids.csv", "hidden_geometry.csv"):
        if not (hidden_root / name).exists() and (archive_hidden / name).exists():
            hidden_root = archive_hidden
            break

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
        for r in read_csv(hidden_root / "hidden_centroids.csv")
        if r["condition"] in PAIR
    ]
    geometry_rows = [
        {
            **r,
            "seed": int(r["seed"]),
            "within_angle_dispersion": float(r["within_angle_dispersion"]),
            "separation_normalized_to_clean": float(r["separation_normalized_to_clean"]),
        }
        for r in read_csv(hidden_root / "hidden_geometry.csv")
        if r["condition"] in PAIR
    ]

    plot_dose_delay_behaviour(main_rows, figure_dir, data_dir)
    plot_maintenance_metrics(main_rows, figure_dir, data_dir)
    plot_geometry(centroid_rows, geometry_rows, figure_dir, data_dir)
    write_summary_table(main_rows, data_dir)

    settling_rows = None
    if not args.skip_settling:
        config = load_config(args.config)
        try:
            settling_rows = compute_settling_times(config)
        except RuntimeError as exc:
            cached = data_dir / "figure_g4_settling.csv"
            if "CUDA is mandatory" not in str(exc) or not cached.exists():
                raise
            settling_rows = read_csv(cached)
            for row in settling_rows:
                for key in ("strength", "mean_settling_steps", "mean_final_error_degrees"):
                    row[key] = float(row[key])
        plot_settling(settling_rows, figure_dir, data_dir)

    write_markdown_summary(data_dir, settling_rows)
    print(f"Wrote Gaussian-control figures to {figure_dir}")
    print(f"Summary: {data_dir / 'gaussian_vs_baseline_summary.md'}")


if __name__ == "__main__":
    main()
