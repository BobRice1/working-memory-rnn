"""Fixed-point and Jacobian analysis for tuned working-memory RNNs.

This module searches for low-speed hidden states under blank-delay input, then
linearizes the recurrent dynamics around those states. For a continuous
ring-like memory, useful evidence is: low fixed-point residuals, decoded angles
that preserve the remembered location, and a Jacobian spectrum with one
near-neutral direction plus contraction in the remaining directions.
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
import torch.nn.functional as F

from wm_rnn.config import load_config
from wm_rnn.device import select_device
from wm_rnn.hidden_angle_decoder import (
    angle_error_degrees,
    decode_angles_from_hidden,
    fit_hidden_angle_decoder,
    resolve_angle_decode_method,
)
from wm_rnn.io import ensure_run_dirs, write_json
from wm_rnn.tuned_task import decode_population_angle
from wm_rnn.training_utils import batch_to_tensors, fresh_model, generate_batch_for_task, task_config_from_dict


@dataclass(frozen=True)
class FixedPointAnalysisResult:
    """Output paths and summary metrics from fixed-point analysis."""

    figure_path: Path
    summary_path: Path
    arrays_path: Path
    csv_path: Path
    metrics: dict[str, Any]


def run_fixed_point_analysis(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    n_trials: int = 64,
    max_steps: int = 2000,
    lbfgs_steps: int = 100,
    learning_rate: float = 0.05,
    tolerance: float = 1e-7,
    residual_threshold: float = 1e-3,
    anchor_weight: float = 1e-4,
) -> FixedPointAnalysisResult:
    """Find nearby delay-input fixed points and analyze local Jacobians."""
    if str(config["task"].get("task_type", "categorical")) != "tuned":
        raise ValueError("fixed-point analysis currently requires task.task_type: tuned")
    if n_trials <= 2:
        raise ValueError("n_trials must be greater than 2")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if lbfgs_steps < 0:
        raise ValueError("lbfgs_steps must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if residual_threshold <= 0:
        raise ValueError("residual_threshold must be positive")
    if anchor_weight < 0:
        raise ValueError("anchor_weight must be non-negative")

    device_info = select_device(config["training"].get("device", "auto"))
    dirs = ensure_run_dirs(config["paths"]["output_dir"])
    model = fresh_model(config, device_info.device)
    checkpoint = torch.load(checkpoint_path, map_location=device_info.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    task_config = task_config_from_dict(config, seed_offset=60000, batch_size=n_trials)
    batch = generate_batch_for_task(task_config)
    inputs, _, _ = batch_to_tensors(batch, device_info.device)
    blank_input = torch.zeros(n_trials, task_config.input_size, device=device_info.device)
    blank_input[:, -1] = 1.0

    with torch.no_grad():
        _, hidden_states = model(inputs)
        delay_end_index = batch.phase_index["delay"].stop - 1
        initial_hidden = hidden_states[delay_end_index].detach()

    fixed_points, residual_history = _optimize_fixed_points(
        model=model,
        blank_input=blank_input,
        initial_hidden=initial_hidden,
        max_steps=max_steps,
        lbfgs_steps=lbfgs_steps,
        learning_rate=learning_rate,
        tolerance=tolerance,
        anchor_weight=anchor_weight,
    )

    with torch.no_grad():
        next_hidden = model.rnn.recurrence(blank_input, fixed_points)
        residuals = torch.linalg.norm(next_hidden - fixed_points, dim=-1).detach().cpu().numpy()
        initial_distances = torch.linalg.norm(fixed_points - initial_hidden, dim=-1).detach().cpu().numpy()
        fixed_outputs = model.readout(fixed_points).detach().cpu().numpy()

    target_angles = batch.angles
    decode_method = resolve_angle_decode_method(config)
    ridge_alpha = float(config.get("decoder", {}).get("ridge_alpha", 1.0))
    if decode_method == "hidden_ridge":
        # Fit on a held-out delay batch so fixed-point angle is not readout-silent.
        decoder_trials = int(config.get("decoder", {}).get("n_trials", 512))
        decoder_task = task_config_from_dict(config, seed_offset=61000, batch_size=decoder_trials)
        decoder_batch = generate_batch_for_task(decoder_task)
        decoder_inputs, _, _ = batch_to_tensors(decoder_batch, device_info.device)
        with torch.no_grad():
            _, decoder_hidden = model(decoder_inputs)
            delay_slice = decoder_batch.phase_index["delay"]
            train_hidden = decoder_hidden[delay_slice].detach()
        decoder_weights = fit_hidden_angle_decoder(
            train_hidden, decoder_batch.angles, ridge_alpha=ridge_alpha
        )
        decoded_angles = decode_angles_from_hidden(fixed_points, decoder_weights)
        initial_angles = decode_angles_from_hidden(initial_hidden, decoder_weights)
    else:
        decoder_weights = None
        decoded_angles = decode_population_angle(fixed_outputs, batch.preferred_angles)
        initial_outputs = model.readout(initial_hidden).detach().cpu().numpy()
        initial_angles = decode_population_angle(initial_outputs, batch.preferred_angles)

    fixed_point_error_degrees = angle_error_degrees(decoded_angles, target_angles)
    drift_from_late_delay_degrees = angle_error_degrees(decoded_angles, initial_angles)

    jacobian_metrics, eigenvalues, leading_vectors, tangent_alignment = _analyze_jacobians(
        model=model,
        blank_input=blank_input,
        fixed_points=fixed_points,
        target_angles=target_angles,
    )

    per_trial_rows = _build_trial_rows(
        target_angles=target_angles,
        decoded_angles=decoded_angles,
        residuals=residuals,
        initial_distances=initial_distances,
        fixed_point_error_degrees=fixed_point_error_degrees,
        drift_from_late_delay_degrees=drift_from_late_delay_degrees,
        jacobian_metrics=jacobian_metrics,
        tangent_alignment=tangent_alignment,
    )

    converged = residuals < residual_threshold
    spectral_radius = np.array([row["spectral_radius"] for row in jacobian_metrics], dtype=np.float64)
    second_abs = np.array([row["second_largest_abs_eigenvalue"] for row in jacobian_metrics], dtype=np.float64)
    leading_abs = np.array([row["largest_abs_eigenvalue"] for row in jacobian_metrics], dtype=np.float64)
    metrics = {
        "device": device_info.description,
        "checkpoint": str(checkpoint_path),
        "task_type": "tuned",
        "n_trials": n_trials,
        "max_steps": max_steps,
        "lbfgs_steps": lbfgs_steps,
        "learning_rate": learning_rate,
        "tolerance": tolerance,
        "residual_threshold": residual_threshold,
        "anchor_weight": anchor_weight,
        "final_mean_fixed_point_residual": float(residuals.mean()),
        "final_median_fixed_point_residual": float(np.median(residuals)),
        "final_max_fixed_point_residual": float(residuals.max()),
        "converged_fraction": float(converged.mean()),
        "mean_distance_from_late_delay_state": float(initial_distances.mean()),
        "mean_fixed_point_error_degrees": float(fixed_point_error_degrees.mean()),
        "median_fixed_point_error_degrees": float(np.median(fixed_point_error_degrees)),
        "p95_fixed_point_error_degrees": float(np.percentile(fixed_point_error_degrees, 95)),
        "mean_drift_from_late_delay_degrees": float(drift_from_late_delay_degrees.mean()),
        "p95_drift_from_late_delay_degrees": float(np.percentile(drift_from_late_delay_degrees, 95)),
        "mean_spectral_radius": float(spectral_radius.mean()),
        "median_spectral_radius": float(np.median(spectral_radius)),
        "max_spectral_radius": float(spectral_radius.max()),
        "mean_largest_abs_eigenvalue": float(leading_abs.mean()),
        "mean_second_largest_abs_eigenvalue": float(second_abs.mean()),
        "mean_near_neutral_eigenvalue_count_abs_gt_0_99": float(
            np.mean([row["num_abs_eigenvalues_gt_0_99"] for row in jacobian_metrics])
        ),
        "mean_unstable_eigenvalue_count_abs_gt_1": float(
            np.mean([row["num_abs_eigenvalues_gt_1"] for row in jacobian_metrics])
        ),
        "mean_tangent_alignment_with_leading_eigenvector": float(np.nanmean(tangent_alignment)),
        "angle_decode_method": decode_method,
        "interpretation_note": (
            "Low residuals mean the optimizer found approximate fixed points under blank-delay input. "
            "For a continuous memory attractor, expect preserved decoded angle, a leading eigenvalue "
            "near 1 along the circular manifold, and remaining eigenvalues below 1. "
            "For fixation-gated models, angle is decoded from hidden states with a ridge decoder "
            "because the circular readout is silent during delay."
            if decode_method == "hidden_ridge"
            else
            "Low residuals mean the optimizer found approximate fixed points under blank-delay input. "
            "For a continuous memory attractor, expect preserved decoded angle, a leading eigenvalue "
            "near 1 along the circular manifold, and remaining eigenvalues below 1. This is stronger "
            "than trajectory drift analysis but still depends on sampled starting states."
        ),
    }

    run_name = config["paths"].get("run_name", "circular_working_memory")
    arrays_path = dirs["arrays"] / f"{run_name}_fixed_point_analysis.npz"
    array_payload = {
        "initial_hidden": initial_hidden.detach().cpu().numpy(),
        "fixed_points": fixed_points.detach().cpu().numpy(),
        "residual_history": np.asarray(residual_history, dtype=np.float64),
        "residuals": residuals,
        "target_angles": target_angles,
        "initial_angles": initial_angles,
        "decoded_angles": decoded_angles,
        "fixed_point_error_degrees": fixed_point_error_degrees,
        "drift_from_late_delay_degrees": drift_from_late_delay_degrees,
        "eigenvalues": eigenvalues,
        "leading_eigenvectors": leading_vectors,
        "tangent_alignment": tangent_alignment,
        "angle_decode_method": np.asarray(decode_method),
    }
    if decoder_weights is not None:
        array_payload["hidden_decoder_weights"] = decoder_weights
    np.savez_compressed(arrays_path, **array_payload)
    csv_path = _write_csv(dirs["metrics"] / f"{run_name}_fixed_point_analysis_trials.csv", per_trial_rows)
    figure_path = _plot_fixed_point_analysis(
        dirs["figures"] / f"{run_name}_fixed_point_analysis.png",
        residual_history=np.asarray(residual_history, dtype=np.float64),
        target_angles=target_angles,
        fixed_point_error_degrees=fixed_point_error_degrees,
        drift_from_late_delay_degrees=drift_from_late_delay_degrees,
        leading_abs=leading_abs,
        second_abs=second_abs,
        tangent_alignment=tangent_alignment,
    )
    summary_path = write_json(
        dirs["metrics"] / f"{run_name}_fixed_point_analysis_summary.json",
        {
            **metrics,
            "figure": str(figure_path),
            "arrays": str(arrays_path),
            "per_trial_csv": str(csv_path),
        },
    )
    return FixedPointAnalysisResult(
        figure_path=figure_path,
        summary_path=summary_path,
        arrays_path=arrays_path,
        csv_path=csv_path,
        metrics=metrics,
    )


def _optimize_fixed_points(
    model: torch.nn.Module,
    blank_input: torch.Tensor,
    initial_hidden: torch.Tensor,
    max_steps: int,
    lbfgs_steps: int,
    learning_rate: float,
    tolerance: float,
    anchor_weight: float,
) -> tuple[torch.Tensor, list[float]]:
    """Optimize bounded hidden states to minimize one-step recurrent speed."""
    clipped = initial_hidden.clamp(-0.999, 0.999)
    raw_hidden = torch.atanh(clipped).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([raw_hidden], lr=learning_rate)
    residual_history: list[float] = []

    for _ in range(max_steps):
        optimizer.zero_grad()
        fixed_candidate = torch.tanh(raw_hidden)
        next_hidden = model.rnn.recurrence(blank_input, fixed_candidate)
        residual_loss = F.mse_loss(next_hidden, fixed_candidate)
        anchor_loss = F.mse_loss(fixed_candidate, initial_hidden)
        loss = residual_loss + anchor_weight * anchor_loss
        loss.backward()
        optimizer.step()

        residual_speed = torch.linalg.norm(next_hidden - fixed_candidate, dim=-1).mean().detach().item()
        residual_history.append(float(residual_speed))
        if residual_speed < tolerance:
            break

    if residual_history[-1] >= tolerance and lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            [raw_hidden],
            lr=1.0,
            max_iter=lbfgs_steps,
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            fixed_candidate = torch.tanh(raw_hidden)
            next_hidden = model.rnn.recurrence(blank_input, fixed_candidate)
            residual_loss = F.mse_loss(next_hidden, fixed_candidate)
            anchor_loss = F.mse_loss(fixed_candidate, initial_hidden)
            loss = residual_loss + anchor_weight * anchor_loss
            loss.backward()
            return loss

        lbfgs.step(closure)
        with torch.no_grad():
            fixed_candidate = torch.tanh(raw_hidden)
            next_hidden = model.rnn.recurrence(blank_input, fixed_candidate)
            residual_speed = torch.linalg.norm(next_hidden - fixed_candidate, dim=-1).mean().item()
            residual_history.append(float(residual_speed))

    return torch.tanh(raw_hidden).detach(), residual_history


def _analyze_jacobians(
    model: torch.nn.Module,
    blank_input: torch.Tensor,
    fixed_points: torch.Tensor,
    target_angles: np.ndarray,
) -> tuple[list[dict[str, float | int]], np.ndarray, np.ndarray, np.ndarray]:
    """Compute local Jacobian eigenvalue summaries at each fixed point."""
    fixed_points_cpu = fixed_points.detach().cpu()
    blank_input_cpu = blank_input.detach().cpu()
    model_cpu = model.to("cpu").eval()

    target_order = np.argsort(target_angles)
    tangent_vectors = _ring_tangent_vectors(fixed_points_cpu.numpy(), target_order)

    rows: list[dict[str, float | int]] = []
    eigenvalues: list[np.ndarray] = []
    leading_vectors: list[np.ndarray] = []
    tangent_alignment: list[float] = []

    for trial in range(fixed_points_cpu.shape[0]):
        input_t = blank_input_cpu[trial : trial + 1]
        hidden = fixed_points_cpu[trial].detach().clone().requires_grad_(True)

        def one_step(hidden_vector: torch.Tensor) -> torch.Tensor:
            return model_cpu.rnn.recurrence(input_t, hidden_vector.unsqueeze(0)).squeeze(0)

        jacobian = torch.autograd.functional.jacobian(one_step, hidden, vectorize=True)
        jacobian_np = jacobian.detach().numpy()
        eigvals, eigvecs = np.linalg.eig(jacobian_np)
        abs_eigvals = np.abs(eigvals)
        order = np.argsort(abs_eigvals)[::-1]
        sorted_abs = abs_eigvals[order]
        leading_index = int(order[0])
        leading_vector = np.real(eigvecs[:, leading_index])
        leading_norm = np.linalg.norm(leading_vector)
        if leading_norm > 0:
            leading_vector = leading_vector / leading_norm
        tangent = tangent_vectors[trial]
        tangent_norm = np.linalg.norm(tangent)
        alignment = float(abs(np.dot(leading_vector, tangent / tangent_norm))) if tangent_norm > 0 else float("nan")

        rows.append(
            {
                "spectral_radius": float(sorted_abs[0]),
                "largest_abs_eigenvalue": float(sorted_abs[0]),
                "second_largest_abs_eigenvalue": float(sorted_abs[1]) if len(sorted_abs) > 1 else float("nan"),
                "largest_real_eigenvalue": float(np.max(np.real(eigvals))),
                "num_abs_eigenvalues_gt_0_99": int(np.sum(abs_eigvals > 0.99)),
                "num_abs_eigenvalues_gt_1": int(np.sum(abs_eigvals > 1.0)),
            }
        )
        eigenvalues.append(eigvals.astype(np.complex64))
        leading_vectors.append(leading_vector.astype(np.float32))
        tangent_alignment.append(alignment)

    return rows, np.stack(eigenvalues, axis=0), np.stack(leading_vectors, axis=0), np.asarray(tangent_alignment)


def _ring_tangent_vectors(fixed_points: np.ndarray, sorted_indices: np.ndarray) -> np.ndarray:
    """Approximate the local tangent around the sampled fixed-point ring."""
    tangents = np.zeros_like(fixed_points)
    for position, trial_index in enumerate(sorted_indices):
        prev_index = sorted_indices[(position - 1) % len(sorted_indices)]
        next_index = sorted_indices[(position + 1) % len(sorted_indices)]
        tangents[trial_index] = fixed_points[next_index] - fixed_points[prev_index]
    return tangents


def _build_trial_rows(
    target_angles: np.ndarray,
    decoded_angles: np.ndarray,
    residuals: np.ndarray,
    initial_distances: np.ndarray,
    fixed_point_error_degrees: np.ndarray,
    drift_from_late_delay_degrees: np.ndarray,
    jacobian_metrics: list[dict[str, float | int]],
    tangent_alignment: np.ndarray,
) -> list[dict[str, float | int]]:
    """Build per-trial CSV rows."""
    rows: list[dict[str, float | int]] = []
    for trial in range(len(target_angles)):
        rows.append(
            {
                "trial": trial,
                "target_angle_degrees": float(np.degrees(target_angles[trial])),
                "decoded_fixed_point_angle_degrees": float(np.degrees(decoded_angles[trial])),
                "fixed_point_residual": float(residuals[trial]),
                "distance_from_late_delay_state": float(initial_distances[trial]),
                "fixed_point_error_degrees": float(fixed_point_error_degrees[trial]),
                "drift_from_late_delay_degrees": float(drift_from_late_delay_degrees[trial]),
                "spectral_radius": float(jacobian_metrics[trial]["spectral_radius"]),
                "largest_abs_eigenvalue": float(jacobian_metrics[trial]["largest_abs_eigenvalue"]),
                "second_largest_abs_eigenvalue": float(jacobian_metrics[trial]["second_largest_abs_eigenvalue"]),
                "largest_real_eigenvalue": float(jacobian_metrics[trial]["largest_real_eigenvalue"]),
                "num_abs_eigenvalues_gt_0_99": int(jacobian_metrics[trial]["num_abs_eigenvalues_gt_0_99"]),
                "num_abs_eigenvalues_gt_1": int(jacobian_metrics[trial]["num_abs_eigenvalues_gt_1"]),
                "tangent_alignment_with_leading_eigenvector": float(tangent_alignment[trial]),
            }
        )
    return rows


def _write_csv(path: str | Path, rows: list[dict[str, float | int]]) -> Path:
    """Write per-trial fixed-point metrics to CSV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return target


def _plot_fixed_point_analysis(
    path: str | Path,
    residual_history: np.ndarray,
    target_angles: np.ndarray,
    fixed_point_error_degrees: np.ndarray,
    drift_from_late_delay_degrees: np.ndarray,
    leading_abs: np.ndarray,
    second_abs: np.ndarray,
    tangent_alignment: np.ndarray,
) -> Path:
    """Plot optimizer residuals, decoded error, and Jacobian summaries."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    angle_degrees = np.degrees(target_angles)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(
        np.arange(1, len(residual_history) + 1),
        residual_history,
        color="#1f77b4",
        linewidth=2,
        label="mean one-step residual",
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Optimization step")
    axes[0, 0].set_ylabel("Mean fixed-point residual")
    axes[0, 0].set_title("Fixed-point search")
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].scatter(
        angle_degrees,
        fixed_point_error_degrees,
        s=18,
        color="#d62728",
        alpha=0.75,
        label="fixed point",
    )
    axes[0, 1].axhline(0.0, color="#444444", linestyle=":", linewidth=1, label="perfect decoding")
    axes[0, 1].set_xlabel("Remembered angle (degrees)")
    axes[0, 1].set_ylabel("Decoded error (degrees)")
    axes[0, 1].set_title("Fixed-point decoded error")
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].scatter(
        angle_degrees,
        drift_from_late_delay_degrees,
        s=18,
        color="#2ca02c",
        alpha=0.75,
        label="fixed point vs late-delay state",
    )
    axes[1, 0].axhline(0.0, color="#444444", linestyle=":", linewidth=1, label="no drift")
    axes[1, 0].set_xlabel("Remembered angle (degrees)")
    axes[1, 0].set_ylabel("Drift from late-delay state (degrees)")
    axes[1, 0].set_title("Fixed point versus trajectory state")
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].scatter(angle_degrees, leading_abs, s=18, color="#9467bd", alpha=0.75, label="largest")
    axes[1, 1].scatter(angle_degrees, second_abs, s=18, color="#ff7f0e", alpha=0.75, label="second")
    axes[1, 1].axhline(1.0, color="#444444", linestyle=":", linewidth=1, label="neutral stability")
    axes[1, 1].set_xlabel("Remembered angle (degrees)")
    axes[1, 1].set_ylabel("Abs eigenvalue")
    axes[1, 1].set_title("Jacobian eigenvalue magnitudes")
    axes[1, 1].legend(frameon=False, fontsize=8)

    finite_alignment = tangent_alignment[np.isfinite(tangent_alignment)]
    if finite_alignment.size:
        fig.suptitle(f"Mean tangent alignment: {finite_alignment.mean():.3f}", y=1.01)

    plt.tight_layout()
    plt.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return target


def main() -> None:
    """Parse command-line arguments and run fixed-point analysis."""
    parser = argparse.ArgumentParser(description="Run tuned RNN fixed-point and Jacobian analysis.")
    parser.add_argument("--config", default="configs/yang_fixation_circular_working_memory.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint produced by training.")
    parser.add_argument("--n-trials", type=int, default=64, help="Number of late-delay states to analyze.")
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum fixed-point optimization steps.")
    parser.add_argument("--lbfgs-steps", type=int, default=100, help="LBFGS polishing iterations after Adam.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Fixed-point optimizer learning rate.")
    parser.add_argument("--tolerance", type=float, default=1e-7, help="Optimizer early-stop tolerance.")
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=1e-3,
        help="Practical residual threshold used when reporting converged_fraction.",
    )
    parser.add_argument("--anchor-weight", type=float, default=1e-4, help="Weak penalty for moving away from trajectory states.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Override analysis device.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device
    result = run_fixed_point_analysis(
        config,
        args.checkpoint,
        n_trials=args.n_trials,
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
    print(f"mean_fixed_point_residual={result.metrics['final_mean_fixed_point_residual']:.8f}")
    print(f"mean_fixed_point_error_degrees={result.metrics['mean_fixed_point_error_degrees']:.3f}")
    print(f"mean_spectral_radius={result.metrics['mean_spectral_radius']:.6f}")
    print(f"mean_second_largest_abs_eigenvalue={result.metrics['mean_second_largest_abs_eigenvalue']:.6f}")


if __name__ == "__main__":
    main()
