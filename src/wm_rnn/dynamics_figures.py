"""Additional dynamics figures for the tuned working-memory model.

The figures in this module are intended to make the model mechanism easier to
inspect: decoded memory over time, the fixed-point ring, local Jacobian
stability, and deterministic recovery after off-manifold perturbations.
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
from wm_rnn.io import ensure_run_dirs, write_json
from wm_rnn.tuned_task import circular_angular_error, decode_population_angle
from wm_rnn.training_utils import batch_to_tensors, fresh_model, generate_batch_for_task, task_config_from_dict


FIGURE_RATIONALE = {
    "ring_manifold": {
        "purpose": "Show task trajectories and fixed-point endpoints occupying a circular memory manifold.",
        "references": [
            "Kilpatrick, Ermentrout & Doiron 2013: persistent bump activity, drift, and attractor stability.",
            "Ghazizadeh & Ching 2021: trained RNN attractor/manifold analysis.",
        ],
    },
    "decoded_angle_over_time": {
        "purpose": "Show whether the model decodes and maintains the remembered angle across cue, delay, and response.",
        "references": [
            "Murray et al. 2017: stable population coding over working-memory epochs.",
            "Ghazizadeh & Ching 2021: delay-period decoding and forgetting dynamics.",
        ],
    },
    "jacobian_spectrum": {
        "purpose": "Visualize local recurrent stability around sampled fixed points.",
        "references": [
            "Ghazizadeh & Ching 2021: fixed-point and attractor-landscape interpretation.",
            "Yang/GY RNN dynamical-system notebook: Jacobian eigenspectrum around fixed points.",
        ],
    },
    "perturbation_recovery": {
        "purpose": "Test deterministic recovery toward the ring after off-manifold hidden-state perturbations.",
        "references": [
            "Kilpatrick, Ermentrout & Doiron 2013: perturbations, drift, and attractor robustness.",
            "Ghazizadeh & Ching 2021: robustness of working-memory manifolds.",
        ],
    },
}


@dataclass(frozen=True)
class DynamicsFiguresResult:
    """Output paths and metrics for additional dynamics figures."""

    summary_path: Path
    arrays_path: Path
    recovery_csv_path: Path
    figure_paths: dict[str, Path]
    metrics: dict[str, Any]


def run_dynamics_figures(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    n_trials: int = 64,
    example_trials: int = 12,
    perturbation_scales: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8),
    recovery_steps: int = 100,
) -> DynamicsFiguresResult:
    """Generate mechanism-inspection figures for a tuned checkpoint."""
    if str(config["task"].get("task_type", "categorical")) != "tuned":
        raise ValueError("dynamics figures currently require task.task_type: tuned")
    if n_trials <= 2:
        raise ValueError("n_trials must be greater than 2")
    if example_trials <= 0:
        raise ValueError("example_trials must be positive")
    if recovery_steps <= 0:
        raise ValueError("recovery_steps must be positive")
    if any(scale < 0 for scale in perturbation_scales):
        raise ValueError("perturbation scales must be non-negative")

    device_info = select_device(config["training"].get("device", "auto"))
    dirs = ensure_run_dirs(config["paths"]["output_dir"])
    model = fresh_model(config, device_info.device)
    checkpoint = torch.load(checkpoint_path, map_location=device_info.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    task_config = task_config_from_dict(config, seed_offset=80000, batch_size=n_trials)
    batch = generate_batch_for_task(task_config)
    inputs, _, _ = batch_to_tensors(batch, device_info.device)

    with torch.no_grad():
        outputs, hidden_states = model(inputs)
        output_np = outputs.detach().cpu().numpy()
        hidden_np = hidden_states.detach().cpu().numpy()

    decoded_over_time = decode_population_angle(output_np, batch.preferred_angles)
    target_angles = batch.angles
    angular_error_over_time = np.degrees(
        circular_angular_error(decoded_over_time, target_angles.reshape(1, -1))
    )
    flattened_hidden = hidden_np.reshape(-1, hidden_np.shape[-1])
    pca = PCA(n_components=2)
    trajectory_pc = pca.fit_transform(flattened_hidden).reshape(hidden_np.shape[0], hidden_np.shape[1], 2)

    run_name = config["paths"].get("run_name", "tuned_delay")
    landscape_arrays = _load_npz(dirs["arrays"] / f"{run_name}_fixed_point_landscape.npz")
    fixed_point_arrays = _load_npz(dirs["arrays"] / f"{run_name}_fixed_point_analysis.npz")
    fixed_points = np.asarray(landscape_arrays["fixed_points"])
    fixed_point_pc = pca.transform(fixed_points)
    fixed_point_decoded = np.asarray(landscape_arrays["decoded_angles"])
    eigenvalues = np.asarray(fixed_point_arrays["eigenvalues"])

    recovery = _run_perturbation_recovery(
        model=model,
        task_config=task_config,
        late_delay_hidden=hidden_states[batch.phase_index["delay"].stop - 1].detach(),
        target_angles=target_angles,
        preferred_angles=batch.preferred_angles,
        fixed_points=fixed_points,
        pca=pca,
        perturbation_scales=perturbation_scales,
        recovery_steps=recovery_steps,
        seed=int(config["task"].get("seed", 0)) + 80000,
        device=device_info.device,
    )

    figure_paths = {
        "ring_manifold": _plot_ring_manifold(
            dirs["figures"] / f"{run_name}_ring_manifold.png",
            trajectory_pc=trajectory_pc,
            target_angles=target_angles,
        fixed_point_pc=fixed_point_pc,
        fixed_point_decoded=fixed_point_decoded,
        phase_index=batch.phase_index,
    ),
        "decoded_angle_over_time": _plot_decoded_angle_over_time(
            dirs["figures"] / f"{run_name}_decoded_angle_over_time.png",
            decoded_over_time=decoded_over_time,
            target_angles=target_angles,
            phase_index=batch.phase_index,
            example_trials=min(example_trials, n_trials),
        ),
        "jacobian_spectrum": _plot_jacobian_spectrum(
            dirs["figures"] / f"{run_name}_jacobian_spectrum.png",
            eigenvalues=eigenvalues,
        ),
        "perturbation_recovery": _plot_perturbation_recovery(
            dirs["figures"] / f"{run_name}_perturbation_recovery.png",
            recovery=recovery,
        ),
    }

    recovery_csv_path = _write_recovery_csv(dirs["metrics"] / f"{run_name}_perturbation_recovery.csv", recovery)
    arrays_path = dirs["arrays"] / f"{run_name}_dynamics_figures.npz"
    np.savez_compressed(
        arrays_path,
        decoded_over_time=decoded_over_time,
        angular_error_over_time=angular_error_over_time,
        target_angles=target_angles,
        trajectory_pc=trajectory_pc,
        fixed_point_pc=fixed_point_pc,
        fixed_point_decoded=fixed_point_decoded,
        eigenvalues=eigenvalues,
        perturbation_scales=np.asarray(perturbation_scales, dtype=np.float32),
        recovery_final_error_degrees=recovery["final_error_degrees"],
        recovery_initial_distance_to_ring=recovery["initial_distance_to_ring"],
        recovery_final_distance_to_ring=recovery["final_distance_to_ring"],
    )

    response_slice = batch.phase_index["response"]
    metrics = {
        "device": device_info.description,
        "checkpoint": str(checkpoint_path),
        "task_type": "tuned",
        "n_trials": n_trials,
        "example_trials": min(example_trials, n_trials),
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "mean_delay_angular_error_degrees": float(
            angular_error_over_time[batch.phase_index["delay"], :].mean()
        ),
        "mean_response_angular_error_degrees": float(angular_error_over_time[response_slice, :].mean()),
        "jacobian_mean_abs_eigenvalue": float(np.abs(eigenvalues).mean()),
        "jacobian_max_abs_eigenvalue": float(np.abs(eigenvalues).max()),
        "recovery_steps": recovery_steps,
        "perturbation_scales": [float(x) for x in perturbation_scales],
        "recovery_mean_final_error_by_scale_degrees": {
            str(scale): float(recovery["final_error_degrees"][index].mean())
            for index, scale in enumerate(perturbation_scales)
        },
        "recovery_mean_final_distance_to_ring_by_scale": {
            str(scale): float(recovery["final_distance_to_ring"][index].mean())
            for index, scale in enumerate(perturbation_scales)
        },
        "figure_rationale": FIGURE_RATIONALE,
    }
    summary_path = write_json(
        dirs["metrics"] / f"{run_name}_dynamics_figures_summary.json",
        {
            **metrics,
            "figures": {name: str(path) for name, path in figure_paths.items()},
            "arrays": str(arrays_path),
            "recovery_csv": str(recovery_csv_path),
        },
    )
    return DynamicsFiguresResult(
        summary_path=summary_path,
        arrays_path=arrays_path,
        recovery_csv_path=recovery_csv_path,
        figure_paths=figure_paths,
        metrics=metrics,
    )


def _load_npz(path: Path) -> np.lib.npyio.NpzFile:
    """Load a required NumPy archive with a clear error message."""
    if not path.exists():
        raise FileNotFoundError(f"required analysis array file not found: {path}")
    return np.load(path, allow_pickle=True)


def _run_perturbation_recovery(
    model: torch.nn.Module,
    task_config: Any,
    late_delay_hidden: torch.Tensor,
    target_angles: np.ndarray,
    preferred_angles: np.ndarray,
    fixed_points: np.ndarray,
    pca: PCA,
    perturbation_scales: tuple[float, ...],
    recovery_steps: int,
    seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Apply deterministic off-manifold perturbations and run blank dynamics."""
    rng = np.random.default_rng(seed)
    n_trials = late_delay_hidden.shape[0]
    fixed_points_torch = torch.from_numpy(fixed_points.astype(np.float32)).to(device)
    blank_input = torch.zeros(n_trials, task_config.input_size, device=device)
    blank_input[:, -1] = 1.0
    final_error = []
    initial_distance = []
    final_distance = []
    initial_pc = []
    final_pc = []

    with torch.no_grad():
        for scale in perturbation_scales:
            noise = torch.from_numpy(
                rng.normal(0.0, scale, size=tuple(late_delay_hidden.shape)).astype(np.float32)
            ).to(device)
            hidden = torch.clamp(late_delay_hidden + noise, -0.999, 0.999)
            initial_distance.append(_nearest_distance(hidden, fixed_points_torch).detach().cpu().numpy())
            initial_pc.append(pca.transform(hidden.detach().cpu().numpy()))
            for _ in range(recovery_steps):
                hidden = model.rnn.recurrence(blank_input, hidden)
            final_distance.append(_nearest_distance(hidden, fixed_points_torch).detach().cpu().numpy())
            final_pc.append(pca.transform(hidden.detach().cpu().numpy()))
            final_output = model.readout(hidden).detach().cpu().numpy()
            final_angles = decode_population_angle(final_output, preferred_angles)
            final_error.append(np.degrees(circular_angular_error(final_angles, target_angles)))

    return {
        "scales": np.asarray(perturbation_scales, dtype=np.float32),
        "final_error_degrees": np.stack(final_error, axis=0),
        "initial_distance_to_ring": np.stack(initial_distance, axis=0),
        "final_distance_to_ring": np.stack(final_distance, axis=0),
        "initial_pc": np.stack(initial_pc, axis=0),
        "final_pc": np.stack(final_pc, axis=0),
    }


def _nearest_distance(states: torch.Tensor, fixed_points: torch.Tensor) -> torch.Tensor:
    """Return nearest Euclidean distance from each state to sampled ring points."""
    distances = torch.cdist(states, fixed_points)
    return distances.min(dim=1).values


def _plot_ring_manifold(
    path: Path,
    trajectory_pc: np.ndarray,
    target_angles: np.ndarray,
    fixed_point_pc: np.ndarray,
    fixed_point_decoded: np.ndarray,
    phase_index: dict[str, slice],
) -> Path:
    """Plot task trajectories and fixed-point ring in shared PCA space."""
    cmap = plt.get_cmap("hsv")
    fig, ax = plt.subplots(figsize=(6.3, 5.5), facecolor="black")
    ax.set_facecolor("black")
    n_plotted = min(trajectory_pc.shape[1], 48)
    late_delay_index = phase_index["delay"].stop - 1
    for trial_idx in range(n_plotted):
        color = cmap(float(target_angles[trial_idx] % (2.0 * np.pi)) / (2.0 * np.pi))
        xy = trajectory_pc[:, trial_idx, :]
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.15, alpha=0.72)
        if trial_idx < 20:
            arrow_start = xy[phase_index["cue"].stop - 1]
            arrow_end = xy[min(phase_index["cue"].stop + 3, xy.shape[0] - 1)]
            ax.annotate(
                "",
                xy=arrow_end,
                xytext=arrow_start,
                arrowprops={"arrowstyle": "->", "color": color, "lw": 0.9, "alpha": 0.8},
            )
        ax.scatter(xy[0, 0], xy[0, 1], marker="s", s=14, color="white", alpha=0.42, edgecolors="none")
        ax.scatter(
            xy[late_delay_index, 0],
            xy[late_delay_index, 1],
            marker="^",
            s=20,
            color=color,
            alpha=0.95,
            edgecolors="white",
            linewidths=0.25,
        )
    decoded_degrees = np.degrees(fixed_point_decoded % (2.0 * np.pi))
    scatter = ax.scatter(
        fixed_point_pc[:, 0],
        fixed_point_pc[:, 1],
        c=decoded_degrees,
        cmap="hsv",
        s=28,
        alpha=0.95,
        edgecolors="none",
        vmin=0.0,
        vmax=360.0,
    )
    ax.scatter([], [], marker="s", s=28, color="white", alpha=0.6, label="trajectory start")
    ax.scatter([], [], marker="^", s=34, color="white", alpha=0.9, label="late-delay state")
    ax.plot([], [], color="white", alpha=0.75, lw=1.0, label="task trajectory path")
    ax.scatter([], [], marker="o", s=34, color="white", alpha=0.9, label="fixed point")
    legend = ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    for text in legend.get_texts():
        text.set_color("white")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Decoded fixed-point angle (deg)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    cbar.outline.set_edgecolor("white")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Fixed-point ring in task PCA space")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    plt.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    plt.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _plot_decoded_angle_over_time(
    path: Path,
    decoded_over_time: np.ndarray,
    target_angles: np.ndarray,
    phase_index: dict[str, slice],
    example_trials: int,
) -> Path:
    """Plot decoded angle over time against the target angle."""
    time = np.arange(decoded_over_time.shape[0])
    chosen = np.linspace(0, decoded_over_time.shape[1] - 1, example_trials, dtype=int)
    fig, axes = plt.subplots(3, 4, figsize=(12, 7), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for axis, trial in zip(axes_flat, chosen):
        decoded_deg = np.degrees(np.unwrap(decoded_over_time[:, trial]))
        target_deg = float(np.degrees(target_angles[trial]))
        center = decoded_deg[phase_index["cue"].stop - 1]
        decoded_deg = target_deg + ((decoded_deg - target_deg + 180.0) % 360.0 - 180.0)
        if abs(decoded_deg[0] - center) > 270:
            decoded_deg = target_deg + ((decoded_deg - target_deg + 180.0) % 360.0 - 180.0)
        axis.plot(time, decoded_deg, color="#1f77b4", linewidth=1.8)
        axis.axhline(target_deg, color="#d62728", linestyle=":", linewidth=1.2)
        _shade_phases(axis, phase_index)
        axis.set_title(f"target {target_deg:.0f} deg", fontsize=9)
    for axis in axes[-1, :]:
        axis.set_xlabel("Time step")
    for axis in axes[:, 0]:
        axis.set_ylabel("Decoded angle (deg)")
    phase_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#bbdefb", alpha=0.35, label="fixation"),
        plt.Rectangle((0, 0), 1, 1, color="#dddddd", alpha=0.35, label="cue"),
        plt.Rectangle((0, 0), 1, 1, color="#c8e6c9", alpha=0.35, label="delay"),
        plt.Rectangle((0, 0), 1, 1, color="#ffe0b2", alpha=0.35, label="response"),
        plt.Line2D([0], [0], color="#1f77b4", linewidth=1.8, label="decoded angle"),
        plt.Line2D([0], [0], color="#d62728", linestyle=":", linewidth=1.2, label="target angle"),
    ]
    fig.legend(handles=phase_handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Decoded remembered angle over trial time", y=1.01)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_jacobian_spectrum(path: Path, eigenvalues: np.ndarray) -> Path:
    """Plot sampled Jacobian eigenvalues in the complex plane."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    eig = eigenvalues.reshape(-1)
    theta = np.linspace(0, 2.0 * np.pi, 360)
    ax.plot(np.cos(theta), np.sin(theta), color="#444444", linestyle=":", linewidth=1.2, label="unit circle")
    ax.scatter(np.real(eig), np.imag(eig), s=10, alpha=0.25, color="#1f77b4", edgecolors="none")
    leading = eigenvalues[np.arange(eigenvalues.shape[0]), np.argmax(np.abs(eigenvalues), axis=1)]
    ax.scatter(np.real(leading), np.imag(leading), s=24, alpha=0.8, color="#d62728", label="largest per point")
    ax.axhline(0.0, color="#cccccc", linewidth=0.8)
    ax.axvline(0.0, color="#cccccc", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Real eigenvalue")
    ax.set_ylabel("Imaginary eigenvalue")
    ax.set_title("Jacobian spectrum around sampled fixed points")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_perturbation_recovery(path: Path, recovery: dict[str, np.ndarray]) -> Path:
    """Plot recovery after deterministic hidden-state perturbations."""
    scales = recovery["scales"]
    final_error = recovery["final_error_degrees"]
    initial_distance = recovery["initial_distance_to_ring"]
    final_distance = recovery["final_distance_to_ring"]
    initial_pc = recovery["initial_pc"]
    final_pc = recovery["final_pc"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    error_mean = final_error.mean(axis=1)
    error_p95 = np.percentile(final_error, 95, axis=1)
    axes[0].plot(scales, error_mean, marker="o", linewidth=2, label="mean final angular error")
    axes[0].plot(scales, error_p95, marker="o", linewidth=2, label="95th-percentile final angular error")
    axes[0].set_xlabel("Hidden perturbation SD")
    axes[0].set_ylabel("Final angular error (deg)")
    axes[0].set_title("Angle after recovery")
    axes[0].legend(frameon=False)

    axes[1].plot(scales, initial_distance.mean(axis=1), marker="o", linewidth=2, label="distance immediately after perturbation")
    axes[1].plot(scales, final_distance.mean(axis=1), marker="o", linewidth=2, label="distance after recovery")
    axes[1].set_xlabel("Hidden perturbation SD")
    axes[1].set_ylabel("Distance to sampled ring")
    axes[1].set_title("Return toward ring")
    axes[1].legend(frameon=False)

    max_scale_idx = len(scales) - 1
    axes[2].scatter(
        initial_pc[max_scale_idx, :, 0],
        initial_pc[max_scale_idx, :, 1],
        s=16,
        alpha=0.35,
        label="states immediately after perturbation",
    )
    axes[2].scatter(
        final_pc[max_scale_idx, :, 0],
        final_pc[max_scale_idx, :, 1],
        s=16,
        alpha=0.55,
        label="states after blank-delay recovery",
    )
    axes[2].set_xlabel("PC 1")
    axes[2].set_ylabel("PC 2")
    axes[2].set_title(f"Largest perturbation SD={scales[max_scale_idx]:.2f}")
    axes[2].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)

    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _shade_phases(axis: plt.Axes, phase_index: dict[str, slice]) -> None:
    """Shade cue, delay, and response phases."""
    colors = {
        "fixation": "#bbdefb",
        "cue": "#dddddd",
        "delay": "#c8e6c9",
        "response": "#ffe0b2",
    }
    for name, span in phase_index.items():
        axis.axvspan(span.start, span.stop, color=colors.get(name, "#eeeeee"), alpha=0.35)


def _write_recovery_csv(path: Path, recovery: dict[str, np.ndarray]) -> Path:
    """Write perturbation recovery summary by scale."""
    rows = []
    for index, scale in enumerate(recovery["scales"]):
        rows.append(
            {
                "perturbation_scale": float(scale),
                "mean_final_error_degrees": float(recovery["final_error_degrees"][index].mean()),
                "p95_final_error_degrees": float(np.percentile(recovery["final_error_degrees"][index], 95)),
                "mean_initial_distance_to_ring": float(recovery["initial_distance_to_ring"][index].mean()),
                "mean_final_distance_to_ring": float(recovery["final_distance_to_ring"][index].mean()),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    """Parse command-line arguments and generate dynamics figures."""
    parser = argparse.ArgumentParser(description="Generate tuned RNN dynamics figures.")
    parser.add_argument("--config", default="configs/tuned_delay_stable.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint produced by training.")
    parser.add_argument("--n-trials", type=int, default=64, help="Number of trials to analyze.")
    parser.add_argument("--example-trials", type=int, default=12, help="Number of decoded-angle examples.")
    parser.add_argument("--recovery-steps", type=int, default=100, help="Blank dynamics steps after perturbation.")
    parser.add_argument(
        "--perturbation-scales",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2, 0.4, 0.8],
        help="Hidden-state perturbation standard deviations.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Override analysis device.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device
    result = run_dynamics_figures(
        config,
        args.checkpoint,
        n_trials=args.n_trials,
        example_trials=args.example_trials,
        perturbation_scales=tuple(args.perturbation_scales),
        recovery_steps=args.recovery_steps,
    )
    print(f"summary={result.summary_path}")
    print(f"arrays={result.arrays_path}")
    print(f"recovery_csv={result.recovery_csv_path}")
    for name, path in result.figure_paths.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
