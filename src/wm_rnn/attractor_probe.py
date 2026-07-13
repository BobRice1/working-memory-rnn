"""Attractor-style probe for tuned working-memory RNN checkpoints.

This analysis starts from hidden states near the end of the delay period and
continues the recurrent dynamics with the cue removed. It is stronger than a
trajectory-speed summary, but still descriptive: it probes whether the trained
network has low-drift, angle-preserving, low-speed end states under the same
blank-delay input, rather than proving all fixed points analytically.
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

from wm_rnn.config import load_config
from wm_rnn.device import select_device
from wm_rnn.io import ensure_run_dirs, write_json
from wm_rnn.tuned_task import circular_angular_error, decode_population_angle
from wm_rnn.training_utils import batch_to_tensors, fresh_model, generate_batch_for_task, task_config_from_dict


@dataclass(frozen=True)
class AttractorProbeResult:
    """Output paths and summary metrics from an attractor probe."""

    figure_path: Path
    summary_path: Path
    arrays_path: Path
    csv_path: Path
    metrics: dict[str, Any]


def run_attractor_probe(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    n_trials: int = 128,
    probe_steps: int = 100,
) -> AttractorProbeResult:
    """Probe whether late-delay tuned hidden states remain stable without cue input."""
    if str(config["task"].get("task_type", "categorical")) != "tuned":
        raise ValueError("attractor probe currently requires task.task_type: tuned")
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if probe_steps <= 0:
        raise ValueError("probe_steps must be positive")

    device_info = select_device(config["training"].get("device", "auto"))
    dirs = ensure_run_dirs(config["paths"]["output_dir"])
    model = fresh_model(config, device_info.device)
    checkpoint = torch.load(checkpoint_path, map_location=device_info.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    task_config = task_config_from_dict(config, seed_offset=50000, batch_size=n_trials)
    batch = generate_batch_for_task(task_config)
    inputs, targets, _ = batch_to_tensors(batch, device_info.device)

    with torch.no_grad():
        _, hidden_states = model(inputs)
        delay_end_index = batch.phase_index["delay"].stop - 1
        initial_hidden = hidden_states[delay_end_index]

        blank_input = torch.zeros(n_trials, task_config.input_size, device=device_info.device)
        blank_input[:, -1] = 1.0
        hidden = initial_hidden
        autonomous_states = [hidden.detach().cpu().numpy()]
        for _ in range(probe_steps):
            hidden = model.rnn.recurrence(blank_input, hidden)
            autonomous_states.append(hidden.detach().cpu().numpy())

        initial_output = model.readout(initial_hidden).detach().cpu().numpy()
        final_output = model.readout(hidden).detach().cpu().numpy()

    states = np.stack(autonomous_states, axis=0)
    speeds = np.linalg.norm(states[1:] - states[:-1], axis=-1)
    displacement = np.linalg.norm(states[-1] - states[0], axis=-1)
    norm_by_step = np.linalg.norm(states, axis=-1)

    initial_angles = decode_population_angle(initial_output, batch.preferred_angles)
    final_angles = decode_population_angle(final_output, batch.preferred_angles)
    target_angles = batch.angles
    initial_error_degrees = np.degrees(circular_angular_error(initial_angles, target_angles))
    final_error_degrees = np.degrees(circular_angular_error(final_angles, target_angles))
    drift_degrees = np.degrees(circular_angular_error(final_angles, initial_angles))

    first_window = speeds[: min(10, len(speeds))]
    last_window = speeds[-min(10, len(speeds)) :]
    mean_initial_probe_speed = float(first_window.mean())
    mean_final_probe_speed = float(last_window.mean())
    speed_settling_ratio = (
        float(mean_final_probe_speed / mean_initial_probe_speed)
        if mean_initial_probe_speed > 0
        else float("nan")
    )

    per_trial_rows = [
        {
            "trial": trial,
            "target_angle_degrees": float(np.degrees(target_angles[trial])),
            "initial_error_degrees": float(initial_error_degrees[trial]),
            "final_error_degrees": float(final_error_degrees[trial]),
            "drift_degrees": float(drift_degrees[trial]),
            "state_displacement": float(displacement[trial]),
            "final_step_speed": float(speeds[-1, trial]),
        }
        for trial in range(n_trials)
    ]

    metrics = {
        "device": device_info.description,
        "checkpoint": str(checkpoint_path),
        "task_type": "tuned",
        "n_trials": n_trials,
        "probe_steps": probe_steps,
        "mean_initial_probe_speed": mean_initial_probe_speed,
        "mean_final_probe_speed": mean_final_probe_speed,
        "speed_settling_ratio": speed_settling_ratio,
        "mean_state_displacement": float(displacement.mean()),
        "median_state_displacement": float(np.median(displacement)),
        "mean_initial_error_degrees": float(initial_error_degrees.mean()),
        "mean_final_error_degrees": float(final_error_degrees.mean()),
        "median_final_error_degrees": float(np.median(final_error_degrees)),
        "p95_final_error_degrees": float(np.percentile(final_error_degrees, 95)),
        "mean_autonomous_drift_degrees": float(drift_degrees.mean()),
        "median_autonomous_drift_degrees": float(np.median(drift_degrees)),
        "p95_autonomous_drift_degrees": float(np.percentile(drift_degrees, 95)),
        "interpretation_note": (
            "This probe continues late-delay hidden states under blank-delay input. "
            "Low final speed plus low angular drift supports an attractor-like or "
            "slow-manifold interpretation, but does not mathematically prove all fixed points."
        ),
    }

    run_name = config["paths"].get("run_name", "circular_working_memory")
    arrays_path = dirs["arrays"] / f"{run_name}_attractor_probe.npz"
    np.savez_compressed(
        arrays_path,
        states=states,
        speeds=speeds,
        norm_by_step=norm_by_step,
        target_angles=target_angles,
        initial_angles=initial_angles,
        final_angles=final_angles,
        initial_error_degrees=initial_error_degrees,
        final_error_degrees=final_error_degrees,
        drift_degrees=drift_degrees,
        displacement=displacement,
    )
    csv_path = _write_probe_csv(dirs["metrics"] / f"{run_name}_attractor_probe_trials.csv", per_trial_rows)
    figure_path = _plot_probe(
        dirs["figures"] / f"{run_name}_attractor_probe.png",
        speeds=speeds,
        target_angles=target_angles,
        drift_degrees=drift_degrees,
        displacement=displacement,
    )
    summary_path = write_json(
        dirs["metrics"] / f"{run_name}_attractor_probe_summary.json",
        {
            **metrics,
            "figure": str(figure_path),
            "arrays": str(arrays_path),
            "per_trial_csv": str(csv_path),
        },
    )
    return AttractorProbeResult(
        figure_path=figure_path,
        summary_path=summary_path,
        arrays_path=arrays_path,
        csv_path=csv_path,
        metrics=metrics,
    )


def _write_probe_csv(path: str | Path, rows: list[dict[str, float | int]]) -> Path:
    """Write per-trial attractor-probe metrics to CSV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return target


def _plot_probe(
    path: str | Path,
    speeds: np.ndarray,
    target_angles: np.ndarray,
    drift_degrees: np.ndarray,
    displacement: np.ndarray,
) -> Path:
    """Plot aggregate speed decay plus per-angle drift/displacement."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    speed_mean = speeds.mean(axis=1)
    speed_std = speeds.std(axis=1)
    steps = np.arange(1, len(speed_mean) + 1)
    angle_degrees = np.degrees(target_angles)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(steps, speed_mean, color="#1f77b4", linewidth=2, label="mean recurrent step speed")
    axes[0].fill_between(
        steps,
        speed_mean - speed_std,
        speed_mean + speed_std,
        color="#1f77b4",
        alpha=0.2,
        label="+/- 1 SD across trials",
    )
    axes[0].set_xlabel("Autonomous probe step")
    axes[0].set_ylabel("Step-to-step speed")
    axes[0].set_title("Speed after cue removal")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].scatter(
        angle_degrees,
        drift_degrees,
        s=18,
        alpha=0.75,
        c=angle_degrees,
        cmap="hsv",
        vmin=0,
        vmax=360,
        label="trial",
    )
    axes[1].set_xlabel("Remembered angle (degrees)")
    axes[1].set_ylabel("Angular drift (degrees)")
    axes[1].set_title("Decoded drift")
    axes[1].axhline(0.0, color="#444444", linestyle=":", linewidth=1)

    scatter_disp = axes[2].scatter(
        angle_degrees,
        displacement,
        s=18,
        alpha=0.75,
        c=angle_degrees,
        cmap="hsv",
        vmin=0,
        vmax=360,
    )
    axes[2].set_xlabel("Remembered angle (degrees)")
    axes[2].set_ylabel("Hidden-state displacement")
    axes[2].set_title("State displacement")
    cbar = fig.colorbar(scatter_disp, ax=axes[2])
    cbar.set_label("Remembered angle (deg)")

    plt.tight_layout()
    plt.savefig(target, dpi=160)
    plt.close(fig)
    return target


def main() -> None:
    """Parse command-line arguments and run the attractor probe."""
    parser = argparse.ArgumentParser(description="Probe tuned RNN late-delay states for attractor-like stability.")
    parser.add_argument("--config", default="configs/circular_working_memory.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint produced by training.")
    parser.add_argument("--n-trials", type=int, default=128, help="Number of trials to probe.")
    parser.add_argument("--probe-steps", type=int, default=100, help="Autonomous recurrence steps after delay.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Override analysis device.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device
    result = run_attractor_probe(config, args.checkpoint, n_trials=args.n_trials, probe_steps=args.probe_steps)
    print(f"figure={result.figure_path}")
    print(f"summary={result.summary_path}")
    print(f"arrays={result.arrays_path}")
    print(f"csv={result.csv_path}")
    print(f"mean_final_probe_speed={result.metrics['mean_final_probe_speed']:.6f}")
    print(f"mean_autonomous_drift_degrees={result.metrics['mean_autonomous_drift_degrees']:.3f}")


if __name__ == "__main__":
    main()
