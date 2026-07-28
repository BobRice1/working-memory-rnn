"""Baseline competence and hidden-state figures for circular and N-back pools.

Generates a parallel figure set for the current distractor-trained circular
family and the competence-screened N-back family: pool behaviour, delay or
load summary, PCA trajectories, hidden-state stability, and a task-specific
memory/readout timecourse.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from wm_rnn.circular_distractor_pool import (
    _load_trained_model,
    generate_paired_batch,
)
from wm_rnn.tuned_task import (
    TunedDelayTaskConfig,
    circular_angular_error,
    decode_population_angle,
)
from wm_rnn.config import load_config
from wm_rnn.device import select_device
from wm_rnn.hidden_angle_decoder import (
    angle_error_degrees,
    decode_angles_from_hidden,
    fit_hidden_angle_decoder,
)
from wm_rnn.io import ensure_run_dirs, write_json
from wm_rnn.nback_evaluation import evaluate_nback_conditions, resolve_nback_bank_seed
from wm_rnn.nback_task import generate_nback_batch
from wm_rnn.training_utils import batch_to_tensors, fresh_model, task_config_from_dict


COLOURS = {
    "blue": "#315A8C",
    "orange": "#D9822B",
    "green": "#3C8D6B",
    "red": "#C94C4C",
    "grey": "#6B7280",
}


@dataclass(frozen=True)
class BaselineFigureBundle:
    """Paths produced for one task family's baseline figure suite."""

    output_dir: Path
    report_dir: Path
    figure_paths: dict[str, Path]
    summary_path: Path


def _setup_style() -> None:
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


def _save_fig(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _copy_to_report(figures: dict[str, Path], report_dir: Path) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    for name, source in figures.items():
        destination = report_dir / source.name
        shutil.copy2(source, destination)
        copied[name] = destination
    return copied


def _phase_bounds(batch) -> dict[str, tuple[int, int]]:
    return {name: (span.start, span.stop) for name, span in batch.phase_index.items()}


def _shade_phases(ax, phase_bounds: dict[str, tuple[int, int]], ymax: float) -> None:
    colours = {
        "pre_cue": "#EEF2F6",
        "cue": "#BDD7EE",
        "delay": "#DCEAD7",
        "response": "#F5D2B8",
        "distractor": "#F8D7DA",
    }
    for name, (start, stop) in phase_bounds.items():
        ax.axvspan(start, stop, color=colours.get(name, "#F3F4F6"), alpha=0.45, lw=0)


def _hidden_speed(hidden: np.ndarray) -> np.ndarray:
    deltas = hidden[1:] - hidden[:-1]
    return np.linalg.norm(deltas, axis=-1)


def _load_checkpoint_model(
    checkpoint: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    return _load_trained_model(checkpoint, device)


def plot_circular_pool_competence(
    pool_summary: dict[str, Any],
    figure_path: Path,
) -> Path:
    """Bar chart of clean vs distractor angular error across pool seeds."""
    _setup_style()
    rows = sorted(pool_summary["results"], key=lambda row: int(row["seed"]))
    seeds = [int(row["seed"]) for row in rows]
    clean = [float(row["clean_mean_angular_error_degrees"]) for row in rows]
    distractor = [
        float(row["distractor_mean_angular_error_degrees"]) for row in rows
    ]
    retained = {
        int(seed) for seed in pool_summary.get("retained_checkpoint_seeds", [])
    }
    x = np.arange(len(seeds))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x - width / 2, clean, width, label="Clean", color=COLOURS["blue"])
    ax.bar(
        x + width / 2,
        distractor,
        width,
        label="Distractor",
        color=COLOURS["orange"],
    )
    for idx, seed in enumerate(seeds):
        if seed not in retained:
            ax.axvspan(idx - 0.48, idx + 0.48, color="#F8D7DA", alpha=0.55, zorder=0)
    ax.axhline(10.0, color=COLOURS["blue"], ls="--", lw=0.9, label="Clean gate 10°")
    ax.axhline(
        15.0, color=COLOURS["orange"], ls="--", lw=0.9, label="Distractor gate 15°"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(seed) for seed in seeds], rotation=30, ha="right")
    ax.set_ylabel("Mean angular error (degrees)")
    ax.set_xlabel("Checkpoint seed")
    ax.set_title("Distractor-trained circular pool competence")
    ax.legend(frameon=False, fontsize=8)
    return _save_fig(figure_path)


def plot_nback_pool_competence(
    pool_summary: dict[str, Any],
    figure_path: Path,
) -> Path:
    """Bar chart of 0-back vs 2-back accuracy and discriminability."""
    _setup_style()
    rows = [
        row
        for row in pool_summary["results"]
        if bool(row.get("retained", False))
    ]
    rows = sorted(rows, key=lambda row: int(row["seed"]))
    seeds = [int(row["seed"]) for row in rows]
    x = np.arange(len(seeds))
    width = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharex=True)
    metrics = [
        (
            axes[0],
            "Accuracy",
            [
                float(row["zero_back_accuracy"]) for row in rows
            ],
            [float(row["two_back_accuracy"]) for row in rows],
            0.95,
        ),
        (
            axes[1],
            "Discriminability (HR − FAR)",
            [
                float(row["zero_back_discriminability"]) for row in rows
            ],
            [float(row["two_back_discriminability"]) for row in rows],
            0.90,
        ),
    ]
    for ax, title, zero, two, gate in metrics:
        ax.bar(x - width, zero, width, label="0-back", color=COLOURS["blue"])
        ax.bar(x, two, width, label="2-back", color=COLOURS["orange"])
        lure = [float(row["two_back_lure_accuracy"]) for row in rows]
        if title.startswith("Accuracy"):
            ax.bar(
                x + width,
                lure,
                width,
                label="2-back lure",
                color=COLOURS["green"],
            )
        ax.axhline(gate, color=COLOURS["grey"], ls="--", lw=0.9)
        ax.set_ylim(0.85, 1.01)
        ax.set_title(title)
        ax.set_ylabel(title.split()[0])
        ax.set_xticks(x)
        ax.set_xticklabels([str(seed) for seed in seeds], rotation=35, ha="right")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_xlabel("Checkpoint seed")
    axes[1].set_xlabel("Checkpoint seed")
    fig.suptitle("Competence-screened N-back pool", y=1.02, fontsize=11)
    return _save_fig(figure_path)


def evaluate_circular_delay_sweep(
    model: torch.nn.Module,
    config: dict[str, Any],
    delays: list[int],
    *,
    trials_per_cell: int = 256,
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    """Evaluate clean delays plus a distractor delay-20 cell."""
    device = next(model.parameters()).device
    base = task_config_from_dict(config, batch_size=batch_size)
    if not isinstance(base, TunedDelayTaskConfig):
        raise TypeError("circular delay sweep requires a tuned task")
    response_transition = int(
        config["training"].get("response_transition_steps", 5)
    )
    rows: list[dict[str, Any]] = []
    conditions: list[tuple[str, int, int]] = [
        ("clean", delay, 0) for delay in delays
    ]
    if 20 in delays:
        conditions.append(("distractor", 20, int(base.distractor_steps)))

    with torch.inference_mode():
        for condition, delay, distractor_steps in conditions:
            task = replace(
                base,
                pre_cue_steps=int(config["task"]["pre_cue_steps"]),
                cue_steps=int(config["task"]["cue_steps"]),
                delay_steps=int(delay),
                distractor_steps=int(distractor_steps),
            )
            errors: list[float] = []
            n_batches = max(1, trials_per_cell // batch_size)
            for batch_index in range(n_batches):
                batch = generate_paired_batch(
                    task,
                    int(config["evaluation"]["seed_base"])
                    + 17_000
                    + delay * 100
                    + batch_index
                    + (1_000 if condition == "distractor" else 0),
                )
                inputs, _, _ = batch_to_tensors(batch, device)
                outputs, _ = model(inputs)
                response = batch.phase_index["response"]
                scored = slice(response.start + response_transition, response.stop)
                decoded = decode_population_angle(
                    outputs[scored, :, : task.n_tuned_units].cpu().numpy(),
                    batch.preferred_angles,
                )
                target = np.broadcast_to(
                    batch.angles[np.newaxis, :],
                    decoded.shape,
                )
                trial_errors = np.degrees(
                    circular_angular_error(decoded, target)
                ).mean(axis=0)
                errors.extend(float(value) for value in trial_errors)
            arr = np.asarray(errors, dtype=np.float64)
            rows.append(
                {
                    "condition": condition,
                    "delay_steps": int(delay),
                    "mean_angular_error_degrees": float(arr.mean()),
                    "median_angular_error_degrees": float(np.median(arr)),
                    "p95_angular_error_degrees": float(np.percentile(arr, 95)),
                    "trials": int(arr.size),
                }
            )
    return rows


def plot_circular_delay_sweep(
    rows: list[dict[str, Any]],
    figure_path: Path,
) -> Path:
    """Plot clean delay generalization and distractor delay-20 marker."""
    _setup_style()
    clean = sorted(
        [row for row in rows if row["condition"] == "clean"],
        key=lambda row: int(row["delay_steps"]),
    )
    distractor = [
        row for row in rows if row["condition"] == "distractor"
    ]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    delays = [int(row["delay_steps"]) for row in clean]
    means = [float(row["mean_angular_error_degrees"]) for row in clean]
    ax.plot(delays, means, "o-", color=COLOURS["blue"], label="Clean")
    if distractor:
        ax.scatter(
            [int(distractor[0]["delay_steps"])],
            [float(distractor[0]["mean_angular_error_degrees"])],
            s=55,
            color=COLOURS["orange"],
            zorder=3,
            label="Distractor (delay 20)",
        )
    ax.set_xlabel("Delay (model steps)")
    ax.set_ylabel("Mean angular error (degrees)")
    ax.set_title("Circular delay generalization")
    ax.legend(frameon=False)
    return _save_fig(figure_path)


def plot_circular_pca(
    projected: np.ndarray,
    angles: np.ndarray,
    phase_index: dict[str, slice],
    figure_path: Path,
    explained: list[float],
) -> Path:
    """PCA trajectories coloured by remembered cue angle."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    cmap = plt.get_cmap("hsv")
    delay = phase_index["delay"]
    n_show = min(projected.shape[1], 48)
    for trial_idx in range(n_show):
        colour = cmap((angles[trial_idx] % (2 * np.pi)) / (2 * np.pi))
        traj = projected[:, trial_idx, :2]
        ax.plot(traj[:, 0], traj[:, 1], color=colour, alpha=0.35, lw=0.9)
        ax.scatter(
            traj[delay.stop - 1, 0],
            traj[delay.stop - 1, 1],
            color=colour,
            s=18,
            zorder=3,
        )
    ax.set_xlabel(f"PC1 ({100 * explained[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({100 * explained[1]:.1f}% var)")
    ax.set_title("Circular hidden-state PCA (colour = cue angle)")
    ax.set_aspect("equal", adjustable="datalim")
    return _save_fig(figure_path)


def plot_stability_timecourse(
    norm_mean: np.ndarray,
    speed_mean: np.ndarray,
    phase_bounds: dict[str, tuple[int, int]],
    figure_path: Path,
    title: str,
) -> Path:
    """Hidden-state norm and step speed across trial time."""
    _setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.8), sharex=True)
    t_norm = np.arange(norm_mean.size)
    t_speed = np.arange(1, speed_mean.size + 1)
    for ax, series, times, ylabel in (
        (axes[0], norm_mean, t_norm, "Hidden ||h||"),
        (axes[1], speed_mean, t_speed, "Step speed ||Δh||"),
    ):
        _shade_phases(ax, phase_bounds, float(np.nanmax(series)))
        ax.plot(times, series, color=COLOURS["blue"], lw=1.4)
        ax.set_ylabel(ylabel)
    axes[1].set_xlabel("Time (model steps)")
    axes[0].set_title(title)
    return _save_fig(figure_path)


def plot_decoded_angle_over_time(
    decoded: np.ndarray,
    targets: np.ndarray,
    phase_bounds: dict[str, tuple[int, int]],
    figure_path: Path,
    n_examples: int = 12,
) -> Path:
    """Hidden-state decoded angle trajectories for example clean trials."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    _shade_phases(ax, phase_bounds, 2 * np.pi)
    n_show = min(n_examples, decoded.shape[1])
    cmap = plt.get_cmap("hsv")
    for trial_idx in range(n_show):
        colour = cmap((targets[trial_idx] % (2 * np.pi)) / (2 * np.pi))
        ax.plot(decoded[:, trial_idx], color=colour, alpha=0.75, lw=1.0)
        ax.axhline(targets[trial_idx], color=colour, ls=":", lw=0.7, alpha=0.5)
    ax.set_ylim(-0.1, 2 * np.pi + 0.1)
    ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_yticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax.set_xlabel("Time (model steps)")
    ax.set_ylabel("Decoded angle (rad)")
    ax.set_title("Hidden-state decoded angle over time (clean trials)")
    return _save_fig(figure_path)


def plot_nback_pca(
    projected_by_rule: dict[str, np.ndarray],
    stimuli_by_rule: dict[str, np.ndarray],
    event_onsets: np.ndarray,
    stimulus_steps: int,
    figure_path: Path,
    explained_by_rule: dict[str, list[float]],
) -> Path:
    """PCA snapshots at stimulus onsets coloured by item identity."""
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    cmap = plt.get_cmap("tab10")
    for ax, rule in zip(axes, ("0-back", "2-back")):
        projected = projected_by_rule[rule]
        stimuli = stimuli_by_rule[rule]
        explained = explained_by_rule[rule]
        # Sample mid-stimulus frames for each item.
        sample_times = event_onsets + max(stimulus_steps // 2, 0)
        for trial_idx in range(min(projected.shape[1], 40)):
            for item_idx, time_idx in enumerate(sample_times):
                if time_idx >= projected.shape[0]:
                    continue
                identity = int(stimuli[item_idx, trial_idx])
                ax.scatter(
                    projected[time_idx, trial_idx, 0],
                    projected[time_idx, trial_idx, 1],
                    color=cmap(identity % 10),
                    s=10,
                    alpha=0.55,
                )
        ax.set_title(f"{rule} PCA (colour = stimulus id)")
        ax.set_xlabel(f"PC1 ({100 * explained[0]:.1f}% var)")
        ax.set_ylabel(f"PC2 ({100 * explained[1]:.1f}% var)")
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("N-back hidden-state geometry at stimulus samples", y=1.02)
    return _save_fig(figure_path)


def plot_nback_match_probability(
    match_prob: np.ndarray,
    item_labels: np.ndarray,
    item_scored: np.ndarray,
    event_onsets: np.ndarray,
    event_steps: int,
    figure_path: Path,
    title: str,
) -> Path:
    """Mean match probability for match vs non-match scored items."""
    _setup_style()
    n_items = item_labels.shape[0]
    match_curves: list[np.ndarray] = []
    nonmatch_curves: list[np.ndarray] = []
    for item_idx in range(n_items):
        onset = int(event_onsets[item_idx])
        stop = onset + int(event_steps)
        segment = match_prob[onset:stop]
        scored = item_scored[item_idx]
        is_match = item_labels[item_idx] == 1
        for trial_idx in range(segment.shape[1]):
            if not bool(scored[trial_idx]):
                continue
            curve = segment[:, trial_idx]
            if bool(is_match[trial_idx]):
                match_curves.append(curve)
            else:
                nonmatch_curves.append(curve)
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    t = np.arange(event_steps)
    if match_curves:
        stacked = np.stack(match_curves, axis=1)
        ax.plot(
            t,
            stacked.mean(axis=1),
            color=COLOURS["green"],
            lw=1.6,
            label="Match items",
        )
        ax.fill_between(
            t,
            stacked.mean(axis=1) - stacked.std(axis=1),
            stacked.mean(axis=1) + stacked.std(axis=1),
            color=COLOURS["green"],
            alpha=0.2,
        )
    if nonmatch_curves:
        stacked = np.stack(nonmatch_curves, axis=1)
        ax.plot(
            t,
            stacked.mean(axis=1),
            color=COLOURS["red"],
            lw=1.6,
            label="Non-match items",
        )
        ax.fill_between(
            t,
            stacked.mean(axis=1) - stacked.std(axis=1),
            stacked.mean(axis=1) + stacked.std(axis=1),
            color=COLOURS["red"],
            alpha=0.2,
        )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time within item event (steps)")
    ax.set_ylabel("P(match)")
    ax.set_title(title)
    ax.legend(frameon=False)
    return _save_fig(figure_path)


def _circular_dynamics_for_seed(
    model: torch.nn.Module,
    config: dict[str, Any],
    *,
    n_trials: int = 64,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    task = task_config_from_dict(config, seed_offset=40_000, batch_size=n_trials)
    task = replace(
        task,
        pre_cue_steps=int(config["task"]["pre_cue_steps"]),
        cue_steps=int(config["task"]["cue_steps"]),
        delay_steps=int(config["task"]["delay_steps"]),
        distractor_steps=0,
    )
    batch = generate_paired_batch(task, int(config["task"]["seed"]) + 40_000)
    with torch.inference_mode():
        inputs, _, _ = batch_to_tensors(batch, device)
        _, hidden = model(inputs)
    hidden_np = hidden.detach().cpu().numpy()
    delay = batch.phase_index["delay"]
    weights = fit_hidden_angle_decoder(
        hidden_np[delay],
        batch.angles,
        ridge_alpha=1.0,
    )
    decoded = decode_angles_from_hidden(hidden_np, weights)
    pca = PCA(n_components=2)
    projected = pca.fit_transform(
        hidden_np.reshape(-1, hidden_np.shape[-1])
    ).reshape(hidden_np.shape[0], hidden_np.shape[1], 2)
    norm = np.linalg.norm(hidden_np, axis=-1).mean(axis=1)
    speed = _hidden_speed(hidden_np).mean(axis=1)
    delay_error = angle_error_degrees(
        decoded[delay],
        np.broadcast_to(batch.angles[np.newaxis, :], decoded[delay].shape),
    ).mean()
    return {
        "projected": projected,
        "angles": batch.angles,
        "phase_index": batch.phase_index,
        "phase_bounds": _phase_bounds(batch),
        "explained": [float(x) for x in pca.explained_variance_ratio_],
        "decoded": decoded,
        "norm_mean": norm,
        "speed_mean": speed,
        "mean_delay_decode_error_degrees": float(delay_error),
        "hidden": hidden_np,
    }


def _nback_dynamics_for_seed(
    model: torch.nn.Module,
    config: dict[str, Any],
    *,
    n_sequences: int = 48,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    base = task_config_from_dict(config, batch_size=n_sequences)
    projected_by_rule: dict[str, np.ndarray] = {}
    stimuli_by_rule: dict[str, np.ndarray] = {}
    explained_by_rule: dict[str, list[float]] = {}
    stability: dict[str, dict[str, np.ndarray]] = {}
    match_payload: dict[str, Any] = {}
    with torch.inference_mode():
        for offset, n_back in enumerate((0, 2)):
            task = replace(
                base,
                n_back=n_back,
                seed=int(config["task"]["seed"]) + 50_000 + offset,
            )
            batch = generate_nback_batch(task)
            inputs, _, _ = batch_to_tensors(batch, device)
            logits, hidden = model(inputs)
            hidden_np = hidden.detach().cpu().numpy()
            probs = torch.softmax(logits, dim=-1)[..., 1].detach().cpu().numpy()
            pca = PCA(n_components=2)
            projected = pca.fit_transform(
                hidden_np.reshape(-1, hidden_np.shape[-1])
            ).reshape(hidden_np.shape[0], hidden_np.shape[1], 2)
            rule = f"{n_back}-back"
            projected_by_rule[rule] = projected
            stimuli_by_rule[rule] = batch.stimuli
            explained_by_rule[rule] = [
                float(x) for x in pca.explained_variance_ratio_
            ]
            stability[rule] = {
                "norm_mean": np.linalg.norm(hidden_np, axis=-1).mean(axis=1),
                "speed_mean": _hidden_speed(hidden_np).mean(axis=1),
            }
            if n_back == 2:
                match_payload = {
                    "match_prob": probs,
                    "item_labels": batch.item_labels,
                    "item_scored": batch.item_scored,
                    "event_onsets": batch.event_onsets,
                    "event_steps": batch.event_steps,
                }
    return {
        "projected_by_rule": projected_by_rule,
        "stimuli_by_rule": stimuli_by_rule,
        "explained_by_rule": explained_by_rule,
        "stability": stability,
        "match_payload": match_payload,
        "event_onsets": match_payload["event_onsets"],
        "stimulus_steps": int(config["task"]["stimulus_steps"]),
    }


def generate_circular_baseline_figures(
    config_path: str | Path,
    pool_summary_path: str | Path,
    output_dir: str | Path,
    report_dir: str | Path,
    *,
    representative_seed: int = 20260735,
    delays: list[int] | None = None,
    trials_per_delay: int = 256,
    n_trials_dynamics: int = 64,
) -> BaselineFigureBundle:
    """Generate the distractor-trained circular baseline figure suite."""
    config = load_config(config_path)
    pool = _read_json(Path(pool_summary_path))
    retained = [int(seed) for seed in pool["retained_checkpoint_seeds"]]
    if representative_seed not in retained:
        representative_seed = retained[0]
    delays = delays or [10, 20, 40, 80]
    dirs = ensure_run_dirs(output_dir)
    report = Path(report_dir)
    selected = select_device(config["training"].get("device", "auto"))

    figures: dict[str, Path] = {}
    figures["pool_competence"] = plot_circular_pool_competence(
        pool,
        dirs["figures"] / "circular_trained_distractor_pool_competence.png",
    )

    # Delay sweep on the representative retained seed.
    seed_row = next(
        row for row in pool["results"] if int(row["seed"]) == representative_seed
    )
    checkpoint = Path(seed_row["checkpoint"])
    model, embedded = _load_checkpoint_model(checkpoint, selected.device)
    delay_rows = evaluate_circular_delay_sweep(
        model,
        embedded,
        delays,
        trials_per_cell=trials_per_delay,
    )
    for row in delay_rows:
        row["seed"] = representative_seed
    _write_csv(dirs["metrics"] / "circular_delay_sweep.csv", delay_rows)
    figures["delay_sweep"] = plot_circular_delay_sweep(
        delay_rows,
        dirs["figures"] / "circular_trained_distractor_delay_sweep.png",
    )

    dynamics = _circular_dynamics_for_seed(
        model, embedded, n_trials=n_trials_dynamics
    )
    figures["pca"] = plot_circular_pca(
        dynamics["projected"],
        dynamics["angles"],
        dynamics["phase_index"],
        dirs["figures"] / "circular_trained_distractor_pca_trajectories.png",
        dynamics["explained"],
    )
    figures["stability"] = plot_stability_timecourse(
        dynamics["norm_mean"],
        dynamics["speed_mean"],
        dynamics["phase_bounds"],
        dirs["figures"] / "circular_trained_distractor_stability.png",
        title=f"Circular hidden-state stability (seed {representative_seed})",
    )
    figures["decoded_angle"] = plot_decoded_angle_over_time(
        dynamics["decoded"],
        dynamics["angles"],
        dynamics["phase_bounds"],
        dirs["figures"] / "circular_trained_distractor_decoded_angle_over_time.png",
    )
    np.savez_compressed(
        dirs["arrays"] / "circular_trained_distractor_baseline_dynamics.npz",
        projected=dynamics["projected"],
        angles=dynamics["angles"],
        decoded=dynamics["decoded"],
        norm_mean=dynamics["norm_mean"],
        speed_mean=dynamics["speed_mean"],
        explained=np.asarray(dynamics["explained"]),
    )

    # Reconfirm retained-seed competence for the summary JSON.
    competence_rows = []
    for seed in retained:
        row = next(item for item in pool["results"] if int(item["seed"]) == seed)
        competence_rows.append(
            {
                "seed": seed,
                "clean_mean_angular_error_degrees": float(
                    row["clean_mean_angular_error_degrees"]
                ),
                "distractor_mean_angular_error_degrees": float(
                    row["distractor_mean_angular_error_degrees"]
                ),
                "competence_passed": bool(row["competence_passed"]),
            }
        )

    copied = _copy_to_report(figures, report)
    summary = {
        "family": "circular_trained_distractor",
        "device": selected.description,
        "representative_seed": representative_seed,
        "retained_seeds": retained,
        "delay_sweep": delay_rows,
        "mean_delay_decode_error_degrees": dynamics[
            "mean_delay_decode_error_degrees"
        ],
        "pca_explained_variance_ratio": dynamics["explained"],
        "competence_rows": competence_rows,
        "figures": {name: str(path) for name, path in figures.items()},
        "report_figures": {name: str(path) for name, path in copied.items()},
    }
    summary_path = write_json(
        dirs["metrics"] / "circular_trained_distractor_baseline_figures.json",
        summary,
    )
    return BaselineFigureBundle(
        output_dir=Path(output_dir),
        report_dir=report,
        figure_paths=copied,
        summary_path=summary_path,
    )


def generate_nback_baseline_figures(
    config_path: str | Path,
    pool_summary_path: str | Path,
    output_dir: str | Path,
    report_dir: str | Path,
    *,
    representative_seed: int = 20260916,
    n_sequences_dynamics: int = 48,
) -> BaselineFigureBundle:
    """Generate the screened N-back baseline figure suite."""
    config = load_config(config_path)
    pool = _read_json(Path(pool_summary_path))
    retained = [int(seed) for seed in pool["retained_seeds"]]
    if representative_seed not in retained:
        representative_seed = retained[0]
    dirs = ensure_run_dirs(output_dir)
    report = Path(report_dir)
    selected = select_device(config["training"].get("device", "auto"))

    figures: dict[str, Path] = {}
    figures["pool_competence"] = plot_nback_pool_competence(
        pool,
        dirs["figures"] / "nback_screened_pool_competence.png",
    )

    seed_row = next(
        row for row in pool["results"] if int(row["seed"]) == representative_seed
    )
    checkpoint = Path(seed_row["checkpoint"])
    payload = torch.load(checkpoint, map_location=selected.device, weights_only=False)
    embedded = payload["config"]
    model = fresh_model(embedded, selected.device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    # Load comparison using the frozen evaluation bank size, but a smaller
    # figure-oriented bank is enough for the representative dynamics panels.
    eval_seed = resolve_nback_bank_seed(embedded, "evaluation")
    metrics = evaluate_nback_conditions(
        model,
        embedded,
        seed=eval_seed,
        sequences_per_condition=int(
            embedded["evaluation"]["sequences_per_condition"]
        ),
    )
    load_rows = []
    for rule, values in metrics.items():
        lure = values.get("one_back_lure_accuracy")
        load_rows.append(
            {
                "rule": rule,
                "accuracy": float(values["accuracy"]),
                "discriminability": float(values["discriminability"]),
                "one_back_lure_accuracy": (
                    float(lure) if lure is not None else float("nan")
                ),
            }
        )
    _write_csv(dirs["metrics"] / "nback_representative_load_metrics.csv", load_rows)

    _setup_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    labels = [row["rule"] for row in load_rows]
    acc = [row["accuracy"] for row in load_rows]
    disc = [row["discriminability"] for row in load_rows]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, acc, 0.36, label="Accuracy", color=COLOURS["blue"])
    ax.bar(x + 0.18, disc, 0.36, label="HR − FAR", color=COLOURS["orange"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.85, 1.01)
    ax.set_ylabel("Metric")
    ax.set_title(f"N-back load comparison (seed {representative_seed})")
    ax.legend(frameon=False)
    figures["load_comparison"] = _save_fig(
        dirs["figures"] / "nback_screened_load_comparison.png"
    )

    dynamics = _nback_dynamics_for_seed(
        model, embedded, n_sequences=n_sequences_dynamics
    )
    figures["pca"] = plot_nback_pca(
        dynamics["projected_by_rule"],
        dynamics["stimuli_by_rule"],
        dynamics["event_onsets"],
        dynamics["stimulus_steps"],
        dirs["figures"] / "nback_screened_pca_trajectories.png",
        dynamics["explained_by_rule"],
    )
    # Stability for 2-back as the harder updating condition.
    stab = dynamics["stability"]["2-back"]
    # Synthetic phase shading by item events.
    phase_bounds = {
        f"item_{idx}": (int(onset), int(onset) + int(dynamics["match_payload"]["event_steps"]))
        for idx, onset in enumerate(dynamics["event_onsets"][:4])
    }
    figures["stability"] = plot_stability_timecourse(
        stab["norm_mean"],
        stab["speed_mean"],
        phase_bounds,
        dirs["figures"] / "nback_screened_stability.png",
        title=f"N-back hidden-state stability, 2-back (seed {representative_seed})",
    )
    figures["match_probability"] = plot_nback_match_probability(
        dynamics["match_payload"]["match_prob"],
        dynamics["match_payload"]["item_labels"],
        dynamics["match_payload"]["item_scored"],
        dynamics["match_payload"]["event_onsets"],
        dynamics["match_payload"]["event_steps"],
        dirs["figures"] / "nback_screened_match_probability.png",
        title=f"2-back match probability (seed {representative_seed})",
    )
    np.savez_compressed(
        dirs["arrays"] / "nback_screened_baseline_dynamics.npz",
        projected_0=dynamics["projected_by_rule"]["0-back"],
        projected_2=dynamics["projected_by_rule"]["2-back"],
        stimuli_0=dynamics["stimuli_by_rule"]["0-back"],
        stimuli_2=dynamics["stimuli_by_rule"]["2-back"],
        norm_2=stab["norm_mean"],
        speed_2=stab["speed_mean"],
    )

    copied = _copy_to_report(figures, report)
    summary = {
        "family": "nback_screened_final",
        "device": selected.description,
        "representative_seed": representative_seed,
        "retained_seeds": retained,
        "representative_load_metrics": load_rows,
        "pca_explained_variance_ratio": dynamics["explained_by_rule"],
        "figures": {name: str(path) for name, path in figures.items()},
        "report_figures": {name: str(path) for name, path in copied.items()},
    }
    summary_path = write_json(
        dirs["metrics"] / "nback_screened_baseline_figures.json",
        summary,
    )
    return BaselineFigureBundle(
        output_dir=Path(output_dir),
        report_dir=report,
        figure_paths=copied,
        summary_path=summary_path,
    )


def generate_all_baseline_figures(
    repo_root: str | Path | None = None,
) -> dict[str, BaselineFigureBundle]:
    """Generate circular and N-back baseline figure suites from defaults."""
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    report_dir = root / "docs" / "reports" / "figures" / "baseline_competence"
    circular = generate_circular_baseline_figures(
        config_path=root / "configs" / "fixation_circular_distractor_working_memory.yaml",
        pool_summary_path=(
            root
            / "outputs"
            / "fixation_circular_distractor_working_memory"
            / "metrics"
            / "fixation_circular_distractor_working_memory_pool_summary.json"
        ),
        output_dir=root / "outputs" / "baseline_competence_figures" / "circular_trained_distractor",
        report_dir=report_dir,
    )
    nback = generate_nback_baseline_figures(
        config_path=root / "configs" / "nback_working_memory_screened_final.yaml",
        pool_summary_path=(
            root
            / "outputs"
            / "nback_working_memory_screened_final"
            / "metrics"
            / "nback_working_memory_screened_final_screened_pool_summary.json"
        ),
        output_dir=root / "outputs" / "baseline_competence_figures" / "nback_screened_final",
        report_dir=report_dir,
    )
    return {"circular": circular, "nback": nback}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate baseline competence and hidden-state figures."
    )
    parser.add_argument(
        "--family",
        choices=("all", "circular", "nback"),
        default="all",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.repo_root
    report_dir = root / "docs" / "reports" / "figures" / "baseline_competence"
    if args.family in {"all", "circular"}:
        circular = generate_circular_baseline_figures(
            config_path=root
            / "configs"
            / "fixation_circular_distractor_working_memory.yaml",
            pool_summary_path=(
                root
                / "outputs"
                / "fixation_circular_distractor_working_memory"
                / "metrics"
                / "fixation_circular_distractor_working_memory_pool_summary.json"
            ),
            output_dir=(
                root
                / "outputs"
                / "baseline_competence_figures"
                / "circular_trained_distractor"
            ),
            report_dir=report_dir,
        )
        print(f"circular figures -> {circular.report_dir}")
        print(f"circular summary -> {circular.summary_path}")
    if args.family in {"all", "nback"}:
        nback = generate_nback_baseline_figures(
            config_path=root / "configs" / "nback_working_memory_screened_final.yaml",
            pool_summary_path=(
                root
                / "outputs"
                / "nback_working_memory_screened_final"
                / "metrics"
                / "nback_working_memory_screened_final_screened_pool_summary.json"
            ),
            output_dir=(
                root / "outputs" / "baseline_competence_figures" / "nback_screened_final"
            ),
            report_dir=report_dir,
        )
        print(f"nback figures -> {nback.report_dir}")
        print(f"nback summary -> {nback.summary_path}")


if __name__ == "__main__":
    main()
