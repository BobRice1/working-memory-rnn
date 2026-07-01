"""Hidden-state stability analysis for trained working-memory RNN checkpoints.

This module tests whether a trained model's hidden state settles into a
steady value during the delay period (a tonic, attractor-like signature) or
keeps changing throughout the delay (a phasic or ramping signature). It does
this by tracking, at every time step, how much the hidden state moves
relative to the previous step and how large the hidden state has grown.

This analysis is descriptive only. It does not search for fixed points or
compute Jacobians; it summarizes trajectory speed and magnitude as a fast,
low-cost first diagnostic before any heavier dynamical-systems analysis.
"""

from __future__ import annotations

import argparse
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
from wm_rnn.task import generate_delay_batch
from wm_rnn.training_utils import batch_to_tensors, fresh_model, task_config_from_dict


@dataclass(frozen=True)
class StabilityResult:
    """Output paths and headline numbers produced by stability analysis.

    Attributes:
        figure_path: PNG figure showing hidden-state norm and step-to-step
            speed across trial time.
        summary_path: JSON file containing phase-averaged norm/speed values,
            the delay settling ratio, and output paths.
        arrays_path: Compressed NumPy archive with per-time-step norm and
            speed statistics.
    """

    figure_path: Path
    summary_path: Path
    arrays_path: Path


def run_stability_analysis(
    config: dict[str, Any], checkpoint_path: str | Path, n_trials: int = 64
) -> StabilityResult:
    """Measure hidden-state growth and step-to-step speed for a checkpoint.

    Args:
        config: Experiment configuration dictionary.
        checkpoint_path: Path to a checkpoint produced by ``train_model``.
        n_trials: Number of fresh trials to generate for the analysis batch.

    Returns:
        ``StabilityResult`` containing the saved figure, summary, and array
        paths.
    """
    device_info = select_device(config["training"].get("device", "auto"))
    dirs = ensure_run_dirs(config["paths"]["output_dir"])
    model = fresh_model(config, device_info.device)
    checkpoint = torch.load(checkpoint_path, map_location=device_info.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    task_config = task_config_from_dict(config, seed_offset=40000, batch_size=n_trials)
    batch = generate_delay_batch(task_config)

    with torch.no_grad():
        inputs, _, _ = batch_to_tensors(batch, device_info.device)
        _, hidden_states = model(inputs)

    hidden_np = hidden_states.detach().cpu().numpy()  # (time, trials, hidden)
    norm_per_trial = np.linalg.norm(hidden_np, axis=-1)  # (time, trials)
    step_delta = hidden_np[1:] - hidden_np[:-1]
    speed_per_trial = np.linalg.norm(step_delta, axis=-1)  # (time - 1, trials)

    norm_mean = norm_per_trial.mean(axis=1)
    norm_std = norm_per_trial.std(axis=1)
    speed_mean = speed_per_trial.mean(axis=1)
    speed_std = speed_per_trial.std(axis=1)

    cue_steps = int(config["task"]["cue_steps"])
    delay_steps = int(config["task"]["delay_steps"])
    response_steps = int(config["task"]["response_steps"])
    phase_bounds = {
        "cue": (0, cue_steps),
        "delay": (cue_steps, cue_steps + delay_steps),
        "response": (cue_steps + delay_steps, cue_steps + delay_steps + response_steps),
    }

    def _phase_mean(series: np.ndarray, start: int, end: int) -> float:
        """Average a per-time-step series over ``[start, end)``, skipping empty ranges."""
        start = min(start, len(series))
        end = min(end, len(series))
        if end <= start:
            return float("nan")
        return float(series[start:end].mean())

    phase_speed = {name: _phase_mean(speed_mean, start, end) for name, (start, end) in phase_bounds.items()}
    phase_norm = {name: _phase_mean(norm_mean, start, end) for name, (start, end) in phase_bounds.items()}

    delay_start, delay_end = phase_bounds["delay"]
    delay_len = delay_end - delay_start
    third = max(delay_len // 3, 1)
    early_delay_speed = _phase_mean(speed_mean, delay_start, delay_start + third)
    late_delay_speed = _phase_mean(speed_mean, delay_end - third, delay_end)
    if early_delay_speed and early_delay_speed > 0 and not np.isnan(early_delay_speed):
        delay_settling_ratio = float(late_delay_speed / early_delay_speed)
    else:
        delay_settling_ratio = float("nan")

    run_name = config["paths"].get("run_name", "baseline_delay")
    figure_path = _plot_stability(
        dirs["figures"] / f"{run_name}_stability.png",
        norm_mean=norm_mean,
        norm_std=norm_std,
        speed_mean=speed_mean,
        speed_std=speed_std,
        phase_bounds=phase_bounds,
    )
    arrays_path = dirs["arrays"] / f"{run_name}_stability.npz"
    np.savez_compressed(
        arrays_path,
        norm_mean=norm_mean,
        norm_std=norm_std,
        speed_mean=speed_mean,
        speed_std=speed_std,
    )
    summary_path = write_json(
        dirs["metrics"] / f"{run_name}_stability_summary.json",
        {
            "device": device_info.description,
            "checkpoint": str(checkpoint_path),
            "n_trials": n_trials,
            "phase_bounds_steps": {name: list(bounds) for name, bounds in phase_bounds.items()},
            "mean_hidden_norm_by_phase": phase_norm,
            "mean_step_speed_by_phase": phase_speed,
            "early_delay_speed": early_delay_speed,
            "late_delay_speed": late_delay_speed,
            "delay_settling_ratio": delay_settling_ratio,
            "interpretation_note": (
                "delay_settling_ratio is late-delay speed divided by early-delay speed. "
                "A ratio well below 1 means the hidden state is slowing down and settling "
                "during the delay (attractor-like). A ratio near or above 1 means the hidden "
                "state keeps changing at a similar or increasing rate throughout the delay "
                "(ramping/phasic, not settled)."
            ),
            "figure": str(figure_path),
            "arrays": str(arrays_path),
        },
    )
    return StabilityResult(figure_path=figure_path, summary_path=summary_path, arrays_path=arrays_path)


def _plot_stability(
    path: str | Path,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    speed_mean: np.ndarray,
    speed_std: np.ndarray,
    phase_bounds: dict[str, tuple[int, int]],
) -> Path:
    """Plot hidden-state norm and step-to-step speed across trial time."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    time_norm = np.arange(len(norm_mean))
    time_speed = np.arange(len(speed_mean)) + 0.5

    fig, (ax_norm, ax_speed) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax_norm.plot(time_norm, norm_mean, color="#1f77b4", linewidth=2)
    ax_norm.fill_between(time_norm, norm_mean - norm_std, norm_mean + norm_std, color="#1f77b4", alpha=0.2)
    ax_norm.set_ylabel("Hidden-state norm")
    ax_norm.set_title("Hidden-state magnitude across trial time")

    ax_speed.plot(time_speed, speed_mean, color="#d62728", linewidth=2)
    ax_speed.fill_between(time_speed, speed_mean - speed_std, speed_mean + speed_std, color="#d62728", alpha=0.2)
    ax_speed.set_ylabel("Step-to-step speed")
    ax_speed.set_xlabel("Time step")
    ax_speed.set_title("Hidden-state step-to-step speed across trial time")

    colors = {"cue": "#999999", "delay": "#66bb6a", "response": "#ffa726"}
    for axis in (ax_norm, ax_speed):
        for name, (start, end) in phase_bounds.items():
            axis.axvspan(start, end, color=colors.get(name, "#cccccc"), alpha=0.12)
        for start, _ in phase_bounds.values():
            axis.axvline(start, color="#444444", linestyle=":", linewidth=1)

    ax_norm.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.25) for color in colors.values()],
        labels=list(colors.keys()),
        loc="upper left",
    )

    plt.tight_layout()
    plt.savefig(target, dpi=160)
    plt.close(fig)
    return target


def main() -> None:
    """Parse command-line arguments and run hidden-state stability analysis."""
    parser = argparse.ArgumentParser(
        description="Diagnose whether a trained working-memory RNN settles (attractor-like) or keeps "
        "changing (ramping/phasic) during the delay period."
    )
    parser.add_argument("--config", default="configs/baseline_delay.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint produced by training.")
    parser.add_argument("--n-trials", type=int, default=64, help="Number of trials to analyze.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Override analysis device.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device
    result = run_stability_analysis(config, args.checkpoint, n_trials=args.n_trials)
    print(f"figure={result.figure_path}")
    print(f"summary={result.summary_path}")
    print(f"arrays={result.arrays_path}")


if __name__ == "__main__":
    main()
