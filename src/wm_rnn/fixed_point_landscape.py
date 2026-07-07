"""PCA landscape visualization for tuned fixed points.

This analysis follows the visualization logic used in the reference RNN
dynamical-systems notebook: fit PCA on task-evoked hidden trajectories, search
for approximate fixed points from multiple starting-state families, then plot
the found fixed points in the same low-dimensional state space.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from wm_rnn.config import load_config
from wm_rnn.device import select_device
from wm_rnn.fixed_point_analysis import _optimize_fixed_points
from wm_rnn.io import ensure_run_dirs, write_json
from wm_rnn.tuned_task import circular_angular_error, decode_population_angle
from wm_rnn.training_utils import batch_to_tensors, fresh_model, generate_batch_for_task, task_config_from_dict


@dataclass(frozen=True)
class FixedPointLandscapeResult:
    """Output paths and summary metrics for the fixed-point landscape."""

    figure_path: Path
    summary_path: Path
    arrays_path: Path
    csv_path: Path
    metrics: dict[str, Any]


def run_fixed_point_landscape(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    n_trajectory_trials: int = 64,
    n_random_starts: int = 128,
    perturbations_per_trial: int = 2,
    perturbation_scale: float = 0.15,
    max_steps: int = 3000,
    lbfgs_steps: int = 100,
    learning_rate: float = 0.03,
    tolerance: float = 1e-7,
    residual_threshold: float = 1e-3,
    anchor_weight: float = 0.0,
) -> FixedPointLandscapeResult:
    """Search fixed points from several start families and project into PCA space."""
    if str(config["task"].get("task_type", "categorical")) != "tuned":
        raise ValueError("fixed-point landscape currently requires task.task_type: tuned")
    if n_trajectory_trials <= 2:
        raise ValueError("n_trajectory_trials must be greater than 2")
    if n_random_starts < 0:
        raise ValueError("n_random_starts must be non-negative")
    if perturbations_per_trial < 0:
        raise ValueError("perturbations_per_trial must be non-negative")
    if perturbation_scale < 0:
        raise ValueError("perturbation_scale must be non-negative")
    if n_random_starts == 0 and perturbations_per_trial == 0:
        raise ValueError("at least one random or perturbed start is required")

    device_info = select_device(config["training"].get("device", "auto"))
    dirs = ensure_run_dirs(config["paths"]["output_dir"])
    model = fresh_model(config, device_info.device)
    checkpoint = torch.load(checkpoint_path, map_location=device_info.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    task_config = task_config_from_dict(config, seed_offset=70000, batch_size=n_trajectory_trials)
    batch = generate_batch_for_task(task_config)
    inputs, _, _ = batch_to_tensors(batch, device_info.device)
    blank_input = torch.zeros(1, task_config.input_size, device=device_info.device)
    blank_input[:, -1] = 1.0

    with torch.no_grad():
        _, hidden_states = model(inputs)
        delay_end_index = batch.phase_index["delay"].stop - 1
        late_delay_hidden = hidden_states[delay_end_index].detach()

    start_hidden, start_source, source_angles = _make_start_states(
        late_delay_hidden=late_delay_hidden,
        target_angles=batch.angles,
        n_random_starts=n_random_starts,
        perturbations_per_trial=perturbations_per_trial,
        perturbation_scale=perturbation_scale,
        seed=int(config["task"].get("seed", 0)) + 70000,
    )
    start_hidden = start_hidden.to(device_info.device)
    blank_inputs = blank_input.repeat(start_hidden.shape[0], 1)

    fixed_points, residual_history = _optimize_fixed_points(
        model=model,
        blank_input=blank_inputs,
        initial_hidden=start_hidden,
        max_steps=max_steps,
        lbfgs_steps=lbfgs_steps,
        learning_rate=learning_rate,
        tolerance=tolerance,
        anchor_weight=anchor_weight,
    )

    with torch.no_grad():
        next_hidden = model.rnn.recurrence(blank_inputs, fixed_points)
        residuals = torch.linalg.norm(next_hidden - fixed_points, dim=-1).detach().cpu().numpy()
        fixed_outputs = model.readout(fixed_points).detach().cpu().numpy()

    hidden_np = hidden_states.detach().cpu().numpy()
    flattened_trajectories = hidden_np.reshape(-1, hidden_np.shape[-1])
    pca = PCA(n_components=2)
    pca.fit(flattened_trajectories)
    trajectory_pc = pca.transform(flattened_trajectories).reshape(hidden_np.shape[0], hidden_np.shape[1], 2)
    start_pc = pca.transform(start_hidden.detach().cpu().numpy())
    fixed_pc = pca.transform(fixed_points.detach().cpu().numpy())

    decoded_angles = decode_population_angle(fixed_outputs, batch.preferred_angles)
    source_angles_array = np.asarray(source_angles, dtype=np.float32)
    angle_error = np.full_like(decoded_angles, fill_value=np.nan, dtype=np.float32)
    known_angle_mask = np.isfinite(source_angles_array)
    angle_error[known_angle_mask] = np.degrees(
        circular_angular_error(decoded_angles[known_angle_mask], source_angles_array[known_angle_mask])
    )
    displacement_pc = np.linalg.norm(fixed_pc - start_pc, axis=-1)
    converged = residuals < residual_threshold

    rows = _build_rows(
        start_source=start_source,
        source_angles=source_angles_array,
        decoded_angles=decoded_angles,
        residuals=residuals,
        angle_error=angle_error,
        start_pc=start_pc,
        fixed_pc=fixed_pc,
        displacement_pc=displacement_pc,
        converged=converged,
    )
    source_counts = {source: int(np.sum(start_source == source)) for source in np.unique(start_source)}
    source_converged = {
        source: float(converged[start_source == source].mean()) for source in np.unique(start_source)
    }
    source_mean_residual = {
        source: float(residuals[start_source == source].mean()) for source in np.unique(start_source)
    }
    source_known_angle_error = {
        source: float(np.nanmean(angle_error[start_source == source]))
        for source in np.unique(start_source)
        if np.isfinite(angle_error[start_source == source]).any()
    }
    metrics = {
        "device": device_info.description,
        "checkpoint": str(checkpoint_path),
        "task_type": "tuned",
        "n_trajectory_trials": n_trajectory_trials,
        "n_random_starts": n_random_starts,
        "perturbations_per_trial": perturbations_per_trial,
        "perturbation_scale": perturbation_scale,
        "n_total_starts": int(len(start_source)),
        "source_counts": source_counts,
        "max_steps": max_steps,
        "lbfgs_steps": lbfgs_steps,
        "learning_rate": learning_rate,
        "tolerance": tolerance,
        "residual_threshold": residual_threshold,
        "anchor_weight": anchor_weight,
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "mean_fixed_point_residual": float(residuals.mean()),
        "median_fixed_point_residual": float(np.median(residuals)),
        "max_fixed_point_residual": float(residuals.max()),
        "converged_fraction": float(converged.mean()),
        "converged_fraction_by_source": source_converged,
        "mean_residual_by_source": source_mean_residual,
        "mean_known_angle_error_degrees": float(np.nanmean(angle_error)),
        "p95_known_angle_error_degrees": float(np.nanpercentile(angle_error, 95)),
        "mean_known_angle_error_degrees_by_source": source_known_angle_error,
        "mean_pc_displacement": float(displacement_pc.mean()),
        "interpretation_note": (
            "This analysis does not map the full 64D hidden-state cube. It samples task-reached, "
            "locally perturbed, and random bounded starting states, then visualizes where fixed-point "
            "search lands in the same PCA space as task trajectories."
        ),
    }

    run_name = config["paths"].get("run_name", "tuned_delay")
    arrays_path = dirs["arrays"] / f"{run_name}_fixed_point_landscape.npz"
    np.savez_compressed(
        arrays_path,
        hidden_states=hidden_np,
        trajectory_pc=trajectory_pc,
        late_delay_hidden=late_delay_hidden.detach().cpu().numpy(),
        start_hidden=start_hidden.detach().cpu().numpy(),
        fixed_points=fixed_points.detach().cpu().numpy(),
        start_pc=start_pc,
        fixed_pc=fixed_pc,
        start_source=start_source,
        source_angles=source_angles_array,
        decoded_angles=decoded_angles,
        angle_error_degrees=angle_error,
        residuals=residuals,
        residual_history=np.asarray(residual_history, dtype=np.float64),
        displacement_pc=displacement_pc,
        target_angles=batch.angles,
        preferred_angles=batch.preferred_angles,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )
    csv_path = _write_csv(dirs["metrics"] / f"{run_name}_fixed_point_landscape_points.csv", rows)
    figure_path = _plot_landscape(
        dirs["figures"] / f"{run_name}_fixed_point_landscape.png",
        trajectory_pc=trajectory_pc,
        target_angles=batch.angles,
        fixed_pc=fixed_pc,
        start_source=start_source,
        decoded_angles=decoded_angles,
        residuals=residuals,
        residual_threshold=residual_threshold,
        angle_error=angle_error,
    )
    summary_path = write_json(
        dirs["metrics"] / f"{run_name}_fixed_point_landscape_summary.json",
        {
            **metrics,
            "figure": str(figure_path),
            "arrays": str(arrays_path),
            "points_csv": str(csv_path),
        },
    )
    return FixedPointLandscapeResult(
        figure_path=figure_path,
        summary_path=summary_path,
        arrays_path=arrays_path,
        csv_path=csv_path,
        metrics=metrics,
    )


def _make_start_states(
    late_delay_hidden: torch.Tensor,
    target_angles: np.ndarray,
    n_random_starts: int,
    perturbations_per_trial: int,
    perturbation_scale: float,
    seed: int,
) -> tuple[torch.Tensor, np.ndarray, list[float]]:
    """Create late-delay, perturbed, and random bounded hidden starts."""
    rng = np.random.default_rng(seed)
    late_cpu = late_delay_hidden.detach().cpu()
    starts = [late_cpu]
    sources = [np.full(late_cpu.shape[0], "late_delay", dtype=object)]
    angles: list[float] = [float(x) for x in target_angles]

    if perturbations_per_trial > 0:
        repeated = late_cpu.repeat_interleave(perturbations_per_trial, dim=0)
        noise = torch.from_numpy(
            rng.normal(0.0, perturbation_scale, size=tuple(repeated.shape)).astype(np.float32)
        )
        perturbed = torch.clamp(repeated + noise, -0.999, 0.999)
        starts.append(perturbed)
        sources.append(np.full(perturbed.shape[0], "perturbed_late_delay", dtype=object))
        angles.extend(float(x) for x in np.repeat(target_angles, perturbations_per_trial))

    if n_random_starts > 0:
        random_values = rng.uniform(-0.95, 0.95, size=(n_random_starts, late_cpu.shape[1])).astype(np.float32)
        random_starts = torch.from_numpy(random_values)
        starts.append(random_starts)
        sources.append(np.full(n_random_starts, "random", dtype=object))
        angles.extend([float("nan")] * n_random_starts)

    return torch.cat(starts, dim=0), np.concatenate(sources), angles


def _build_rows(
    start_source: np.ndarray,
    source_angles: np.ndarray,
    decoded_angles: np.ndarray,
    residuals: np.ndarray,
    angle_error: np.ndarray,
    start_pc: np.ndarray,
    fixed_pc: np.ndarray,
    displacement_pc: np.ndarray,
    converged: np.ndarray,
) -> list[dict[str, float | int | str]]:
    """Build per-fixed-point CSV rows."""
    rows: list[dict[str, float | int | str]] = []
    for index in range(len(start_source)):
        source_angle = source_angles[index]
        rows.append(
            {
                "point": index,
                "start_source": str(start_source[index]),
                "source_angle_degrees": float(np.degrees(source_angle)) if np.isfinite(source_angle) else "",
                "decoded_fixed_point_angle_degrees": float(np.degrees(decoded_angles[index])),
                "angle_error_degrees": float(angle_error[index]) if np.isfinite(angle_error[index]) else "",
                "fixed_point_residual": float(residuals[index]),
                "converged": int(converged[index]),
                "start_pc1": float(start_pc[index, 0]),
                "start_pc2": float(start_pc[index, 1]),
                "fixed_pc1": float(fixed_pc[index, 0]),
                "fixed_pc2": float(fixed_pc[index, 1]),
                "pc_displacement": float(displacement_pc[index]),
            }
        )
    return rows


def _write_csv(path: str | Path, rows: list[dict[str, float | int | str]]) -> Path:
    """Write fixed-point landscape rows to CSV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return target


def _plot_landscape(
    path: str | Path,
    trajectory_pc: np.ndarray,
    target_angles: np.ndarray,
    fixed_pc: np.ndarray,
    start_source: np.ndarray,
    decoded_angles: np.ndarray,
    residuals: np.ndarray,
    residual_threshold: float,
    angle_error: np.ndarray,
) -> Path:
    """Plot task trajectories and fixed-point search endpoints in PCA space."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("hsv")
    source_styles = {
        "late_delay": {"marker": "x", "label": "actual late-delay starts", "s": 42},
        "perturbed_late_delay": {"marker": "+", "label": "perturbed late-delay starts", "s": 32},
        "random": {"marker": "o", "label": "random hidden-state starts", "s": 18},
    }
    converged = residuals < residual_threshold

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    ax = axes[0, 0]
    for trial_idx in range(min(trajectory_pc.shape[1], 48)):
        color = cmap(float(target_angles[trial_idx] % (2.0 * np.pi)) / (2.0 * np.pi))
        xy = trajectory_pc[:, trial_idx, :]
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=0.8, alpha=0.35)
    for source, style in source_styles.items():
        mask = start_source == source
        if mask.any():
            ax.scatter(
                fixed_pc[mask, 0],
                fixed_pc[mask, 1],
                marker=style["marker"],
                s=style["s"],
                color="#111111",
                alpha=0.7 if source != "random" else 0.35,
                label=style["label"],
            )
    ax.set_title("Task trajectories plus fixed points")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(title="Fixed-point search starts", frameon=False, fontsize=7, title_fontsize=8, loc="lower right")

    ax = axes[0, 1]
    decoded_degrees = np.degrees(decoded_angles % (2.0 * np.pi))
    scatter_angle = ax.scatter(fixed_pc[:, 0], fixed_pc[:, 1], c=decoded_degrees, s=22, alpha=0.75, cmap="hsv", vmin=0, vmax=360)
    ax.set_title("Fixed points colored by decoded angle")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    cbar_angle = fig.colorbar(scatter_angle, ax=ax)
    cbar_angle.set_label("Decoded angle (deg)")

    ax = axes[1, 0]
    residual_plot = np.maximum(residuals, 1e-8)
    scatter = ax.scatter(fixed_pc[:, 0], fixed_pc[:, 1], c=np.log10(residual_plot), s=24, cmap="viridis", alpha=0.75)
    ax.set_title("Fixed-point residual in PCA space")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("log10 fixed-point residual")

    ax = axes[1, 1]
    for source, style in source_styles.items():
        mask = (start_source == source) & np.isfinite(angle_error)
        if mask.any():
            ax.hist(angle_error[mask], bins=20, alpha=0.55, label=style["label"])
    ax.set_title("Fixed-point angle error for starts with known target")
    ax.set_xlabel("Angular error (degrees)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(f"Converged below {residual_threshold:g}: {converged.mean():.2f}", y=1.01)
    plt.tight_layout()
    plt.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return target


def main() -> None:
    """Parse arguments and run fixed-point landscape analysis."""
    parser = argparse.ArgumentParser(description="Visualize tuned fixed-point search endpoints in PCA state space.")
    parser.add_argument("--config", default="configs/tuned_delay_stable.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint produced by training.")
    parser.add_argument("--n-trajectory-trials", type=int, default=64, help="Task trajectories used to fit PCA.")
    parser.add_argument("--n-random-starts", type=int, default=128, help="Number of random hidden-state starts.")
    parser.add_argument("--perturbations-per-trial", type=int, default=2, help="Perturbed starts per late-delay state.")
    parser.add_argument("--perturbation-scale", type=float, default=0.15, help="Gaussian perturbation SD for late-delay starts.")
    parser.add_argument("--max-steps", type=int, default=3000, help="Maximum Adam fixed-point optimization steps.")
    parser.add_argument("--lbfgs-steps", type=int, default=100, help="LBFGS polishing iterations after Adam.")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="Adam fixed-point optimizer learning rate.")
    parser.add_argument("--tolerance", type=float, default=1e-7, help="Optimizer early-stop tolerance.")
    parser.add_argument("--residual-threshold", type=float, default=1e-3, help="Residual threshold for convergence summary.")
    parser.add_argument("--anchor-weight", type=float, default=0.0, help="Penalty for moving away from start states.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Override analysis device.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device
    result = run_fixed_point_landscape(
        config,
        args.checkpoint,
        n_trajectory_trials=args.n_trajectory_trials,
        n_random_starts=args.n_random_starts,
        perturbations_per_trial=args.perturbations_per_trial,
        perturbation_scale=args.perturbation_scale,
        max_steps=args.max_steps,
        lbfgs_steps=args.lbfgs_steps,
        learning_rate=args.learning_rate,
        tolerance=args.tolerance,
        residual_threshold=args.residual_threshold,
        anchor_weight=args.anchor_weight,
    )
    print(f"figure={result.figure_path}")
    print(f"summary={result.summary_path}")
    print(f"arrays={result.arrays_path}")
    print(f"csv={result.csv_path}")
    print(f"converged_fraction={result.metrics['converged_fraction']:.3f}")
    print(f"mean_fixed_point_residual={result.metrics['mean_fixed_point_residual']:.8f}")
    print(f"mean_known_angle_error_degrees={result.metrics['mean_known_angle_error_degrees']:.3f}")


if __name__ == "__main__":
    main()
