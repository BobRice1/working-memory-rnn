"""Animated hidden-state visualization for the tuned working-memory model.

The movie shows how cue-driven hidden states move through PCA space toward the
sampled fixed-point ring while the task input and decoded output evolve.
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
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA

from wm_rnn.config import load_config
from wm_rnn.device import select_device
from wm_rnn.io import ensure_run_dirs, write_json
from wm_rnn.tuned_task import circular_angular_error, decode_population_angle
from wm_rnn.training_utils import (
    batch_to_tensors,
    fresh_model,
    generate_batch_for_task,
    task_config_from_dict,
    with_delay_steps,
)


@dataclass(frozen=True)
class HiddenStateMovieResult:
    """Output paths and summary metrics for the hidden-state movie."""

    movie_path: Path
    arrays_path: Path
    summary_path: Path
    metrics: dict[str, Any]


def run_hidden_state_movie(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    n_trials: int = 64,
    example_trials: int = 16,
    fps: int = 6,
    output_format: str = "gif",
    delay_steps: int | None = None,
    perturbation_scale: float = 0.4,
    recovery_steps: int | None = None,
    frames_per_step: int = 1,
    recovery_source: str = "landscape-perturbed",
    recovery_perturbations_per_trial: int = 4,
) -> HiddenStateMovieResult:
    """Generate a time-resolved hidden-state movie for a tuned checkpoint."""
    if str(config["task"].get("task_type", "categorical")) != "tuned":
        raise ValueError("hidden-state movie currently requires task.task_type: tuned")
    if n_trials <= 2:
        raise ValueError("n_trials must be greater than 2")
    if example_trials <= 0:
        raise ValueError("example_trials must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if output_format not in {"gif", "mp4"}:
        raise ValueError("output_format must be 'gif' or 'mp4'")
    if delay_steps is not None and delay_steps <= 0:
        raise ValueError("delay_steps must be positive when provided")
    if perturbation_scale < 0:
        raise ValueError("perturbation_scale must be non-negative")
    if recovery_steps is not None and recovery_steps <= 0:
        raise ValueError("recovery_steps must be positive when provided")
    if frames_per_step <= 0:
        raise ValueError("frames_per_step must be positive")
    if recovery_source not in {"landscape-perturbed", "wide-perturbed"}:
        raise ValueError("recovery_source must be 'landscape-perturbed' or 'wide-perturbed'")
    if recovery_perturbations_per_trial <= 0:
        raise ValueError("recovery_perturbations_per_trial must be positive")

    device_info = select_device(config["training"].get("device", "auto"))
    dirs = ensure_run_dirs(config["paths"]["output_dir"])
    model = fresh_model(config, device_info.device)
    checkpoint = torch.load(checkpoint_path, map_location=device_info.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    task_config = task_config_from_dict(config, seed_offset=90000, batch_size=n_trials)
    if delay_steps is not None:
        task_config = with_delay_steps(task_config, delay_steps)
    batch = generate_batch_for_task(task_config)
    inputs, _, _ = batch_to_tensors(batch, device_info.device)

    with torch.no_grad():
        outputs, hidden_states = model(inputs)

    input_np = inputs.detach().cpu().numpy()
    output_np = outputs.detach().cpu().numpy()
    hidden_np = hidden_states.detach().cpu().numpy()
    decoded_over_time = decode_population_angle(output_np, batch.preferred_angles)
    target_angles = batch.angles
    angular_error_degrees = np.degrees(
        circular_angular_error(decoded_over_time, target_angles.reshape(1, -1))
    )

    flattened_hidden = hidden_np.reshape(-1, hidden_np.shape[-1])
    pca = PCA(n_components=2)
    trajectory_pc = pca.fit_transform(flattened_hidden).reshape(hidden_np.shape[0], hidden_np.shape[1], 2)

    run_name = config["paths"].get("run_name", "circular_working_memory")
    landscape_path = dirs["arrays"] / f"{run_name}_fixed_point_landscape.npz"
    fixed_point_pc = np.empty((0, 2), dtype=np.float32)
    fixed_point_decoded = np.empty((0,), dtype=np.float32)
    landscape_recovery_hidden = None
    landscape_recovery_angles = None
    if landscape_path.exists():
        with np.load(landscape_path, allow_pickle=True) as landscape:
            fixed_points = np.asarray(landscape["fixed_points"])
            fixed_point_pc = pca.transform(fixed_points)
            fixed_point_decoded = np.asarray(landscape["decoded_angles"])
            if {"start_hidden", "start_source", "source_angles"}.issubset(set(landscape.files)):
                start_source = np.asarray(landscape["start_source"]).astype(str)
                perturbed_mask = start_source == "perturbed_late_delay"
                if perturbed_mask.any():
                    landscape_recovery_hidden = np.asarray(landscape["start_hidden"])[perturbed_mask]
                    landscape_recovery_angles = np.asarray(landscape["source_angles"])[perturbed_mask]

    chosen_trials = _choose_trials(target_angles, min(example_trials, n_trials))
    resolved_recovery_steps = int(recovery_steps if recovery_steps is not None else hidden_np.shape[0])
    if recovery_source == "landscape-perturbed" and landscape_recovery_hidden is not None:
        recovery_initial_hidden = torch.from_numpy(landscape_recovery_hidden.astype(np.float32)).to(device_info.device)
        recovery_angles = landscape_recovery_angles.astype(np.float32, copy=False)
        resolved_recovery_source = "fixed_point_landscape_perturbed_late_delay"
    else:
        late_delay_for_recovery = hidden_states[batch.phase_index["delay"].stop - 1].detach()
        repeated_late_delay = late_delay_for_recovery.repeat_interleave(recovery_perturbations_per_trial, dim=0)
        recovery_initial_hidden = _make_perturbed_starts(
            late_delay_hidden=repeated_late_delay,
            perturbation_scale=perturbation_scale,
            seed=int(config["task"].get("seed", 0)) + 91000,
            device=device_info.device,
        )
        recovery_angles = np.repeat(target_angles, recovery_perturbations_per_trial)
        resolved_recovery_source = "wide_perturbed_late_delay"
    recovery_pc = _run_recovery_paths(
        model=model,
        task_config=task_config,
        initial_hidden=recovery_initial_hidden,
        pca=pca,
        steps=resolved_recovery_steps,
        device=device_info.device,
    )
    movie_path = dirs["figures"] / f"{run_name}_hidden_state_movie.{output_format}"
    arrays_path = dirs["arrays"] / f"{run_name}_hidden_state_movie.npz"

    np.savez_compressed(
        arrays_path,
        input_population=input_np[:, :, :-1],
        output_population=output_np,
        hidden_states=hidden_np,
        trajectory_pc=trajectory_pc,
        fixed_point_pc=fixed_point_pc,
        fixed_point_decoded=fixed_point_decoded,
        decoded_over_time=decoded_over_time,
        angular_error_degrees=angular_error_degrees,
        target_angles=target_angles,
        preferred_angles=batch.preferred_angles,
        phase_names=np.asarray(list(batch.phase_index.keys())),
        phase_bounds=np.asarray([(span.start, span.stop) for span in batch.phase_index.values()], dtype=np.int32),
        chosen_trials=chosen_trials,
        recovery_pc=recovery_pc,
        recovery_angles=recovery_angles,
        recovery_source=np.asarray(resolved_recovery_source),
        recovery_perturbation_scale=np.asarray(perturbation_scale, dtype=np.float32),
        pca_explained_variance_ratio=pca.explained_variance_ratio_,
    )

    _render_movie(
        path=movie_path,
        input_population=input_np[:, :, :-1],
        output_population=output_np,
        trajectory_pc=trajectory_pc,
        fixed_point_pc=fixed_point_pc,
        fixed_point_decoded=fixed_point_decoded,
        decoded_over_time=decoded_over_time,
        angular_error_degrees=angular_error_degrees,
        target_angles=target_angles,
        preferred_angles=batch.preferred_angles,
        phase_index=batch.phase_index,
        chosen_trials=chosen_trials,
        recovery_pc=recovery_pc,
        recovery_angles=recovery_angles,
        recovery_source=resolved_recovery_source,
        recovery_perturbation_scale=perturbation_scale,
        fps=fps,
        frames_per_step=frames_per_step,
        output_format=output_format,
    )

    metrics = {
        "device": device_info.description,
        "checkpoint": str(checkpoint_path),
        "task_type": "tuned",
        "n_trials": n_trials,
        "example_trials": int(len(chosen_trials)),
        "fps": fps,
        "frames_per_step": frames_per_step,
        "format": output_format,
        "seq_len": int(hidden_np.shape[0]),
        "cue_steps": int(task_config.cue_steps),
        "delay_steps": int(task_config.delay_steps),
        "response_steps": int(task_config.response_steps),
        "recovery_steps": resolved_recovery_steps,
        "recovery_points": int(recovery_pc.shape[1]),
        "recovery_source": resolved_recovery_source,
        "requested_recovery_source": recovery_source,
        "recovery_perturbations_per_trial": recovery_perturbations_per_trial,
        "recovery_perturbation_scale": float(perturbation_scale),
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "mean_angular_error_degrees": float(angular_error_degrees.mean()),
        "mean_delay_angular_error_degrees": float(angular_error_degrees[batch.phase_index["delay"], :].mean()),
        "mean_response_angular_error_degrees": float(angular_error_degrees[batch.phase_index["response"], :].mean()),
        "fixed_point_overlay_source": str(landscape_path) if landscape_path.exists() else "",
        "manim_note": (
            "The NPZ archive stores PCA trajectories, fixed-point ring samples, phase bounds, "
            "input/output population activity, and decoded angles for a future ManimGL scene."
        ),
    }
    summary_path = write_json(
        dirs["metrics"] / f"{run_name}_hidden_state_movie_summary.json",
        {
            **metrics,
            "movie": str(movie_path),
            "arrays": str(arrays_path),
        },
    )
    return HiddenStateMovieResult(movie_path=movie_path, arrays_path=arrays_path, summary_path=summary_path, metrics=metrics)


def _choose_trials(target_angles: np.ndarray, n_examples: int) -> np.ndarray:
    """Choose example trials spread around the circular target space."""
    order = np.argsort(target_angles)
    positions = np.linspace(0, len(order) - 1, n_examples, dtype=int)
    return order[positions]


def _make_perturbed_starts(
    late_delay_hidden: torch.Tensor,
    perturbation_scale: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Return perturbed copies of late-delay hidden states."""
    rng = np.random.default_rng(seed)
    noise = torch.from_numpy(
        rng.normal(0.0, perturbation_scale, size=tuple(late_delay_hidden.shape)).astype(np.float32)
    ).to(device)
    return torch.clamp(late_delay_hidden + noise, -0.999, 0.999)


def _run_recovery_paths(
    model: torch.nn.Module,
    task_config: Any,
    initial_hidden: torch.Tensor,
    pca: PCA,
    steps: int,
    device: torch.device,
) -> np.ndarray:
    """Return PCA paths for initial states under blank dynamics."""
    hidden = initial_hidden
    blank_input = torch.zeros(hidden.shape[0], task_config.input_size, device=device)
    blank_input[:, -1] = 1.0
    states = []

    with torch.no_grad():
        for _ in range(steps):
            states.append(hidden.detach().cpu().numpy())
            hidden = model.rnn.recurrence(blank_input, hidden)

    stacked = np.stack(states, axis=0)
    return pca.transform(stacked.reshape(-1, stacked.shape[-1])).reshape(stacked.shape[0], stacked.shape[1], 2)


def _current_phase(frame: int, phase_index: dict[str, slice]) -> str:
    """Return the task phase active at a frame."""
    for name, span in phase_index.items():
        if span.start <= frame < span.stop:
            return name
    return "trial"


def _phase_color(name: str) -> str:
    """Return a display color for a task phase."""
    return {"cue": "#d9d9d9", "delay": "#62c370", "response": "#ffb454"}.get(name, "#ffffff")


def _axis_limits(points: np.ndarray, fixed_points: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return padded axis limits covering trajectories and fixed points."""
    all_points = points.reshape(-1, 2)
    if fixed_points.size:
        all_points = np.vstack([all_points, fixed_points])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-3)
    padding = 0.12 * span
    return (float(mins[0] - padding[0]), float(maxs[0] + padding[0])), (float(mins[1] - padding[1]), float(maxs[1] + padding[1]))


def _draw_fixed_points(axis: plt.Axes, fixed_point_pc: np.ndarray, fixed_point_decoded: np.ndarray) -> Any | None:
    """Draw sampled fixed points on an axis and return the scatter handle."""
    if not fixed_point_pc.size:
        return None
    fixed_degrees = np.degrees(fixed_point_decoded % (2.0 * np.pi))
    return axis.scatter(
        fixed_point_pc[:, 0],
        fixed_point_pc[:, 1],
        c=fixed_degrees,
        cmap="hsv",
        vmin=0.0,
        vmax=360.0,
        s=18,
        alpha=0.55,
        edgecolors="none",
    )


def _make_segments(xy: np.ndarray) -> np.ndarray:
    """Convert a 2D trajectory into line segments."""
    return np.stack([xy[:-1], xy[1:]], axis=1)


def _interp_time(values: np.ndarray, position: float) -> np.ndarray:
    """Linearly interpolate a time-major array at a fractional position."""
    if values.shape[0] == 1:
        return values[0]
    lower = int(np.floor(position))
    upper = min(lower + 1, values.shape[0] - 1)
    lower = min(lower, values.shape[0] - 1)
    weight = float(position - lower)
    return (1.0 - weight) * values[lower] + weight * values[upper]


def _trajectory_until(values: np.ndarray, position: float) -> np.ndarray:
    """Return a trajectory up to a fractional time position."""
    lower = int(np.floor(position))
    lower = min(lower, values.shape[0] - 1)
    xy = values[: lower + 1]
    if lower < values.shape[0] - 1 and position > lower:
        xy = np.vstack([xy, _interp_time(values, position)])
    return xy


def _render_movie(
    path: Path,
    input_population: np.ndarray,
    output_population: np.ndarray,
    trajectory_pc: np.ndarray,
    fixed_point_pc: np.ndarray,
    fixed_point_decoded: np.ndarray,
    decoded_over_time: np.ndarray,
    angular_error_degrees: np.ndarray,
    target_angles: np.ndarray,
    preferred_angles: np.ndarray,
    phase_index: dict[str, slice],
    chosen_trials: np.ndarray,
    recovery_pc: np.ndarray,
    recovery_angles: np.ndarray,
    recovery_source: str,
    recovery_perturbation_scale: float,
    fps: int,
    frames_per_step: int,
    output_format: str,
) -> None:
    """Render the hidden-state movie to GIF or MP4."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("hsv")
    target_degrees = np.degrees(target_angles)
    preferred_degrees = np.degrees(preferred_angles)
    displayed_points = np.vstack(
        [
            trajectory_pc[:, chosen_trials, :].reshape(-1, 2),
            recovery_pc.reshape(-1, 2),
        ]
    ).reshape(-1, 1, 2)
    xlim, ylim = _axis_limits(displayed_points, fixed_point_pc)
    representative = int(chosen_trials[len(chosen_trials) // 3])
    time = np.arange(trajectory_pc.shape[0])
    visual_frames = max(trajectory_pc.shape[0], recovery_pc.shape[0]) * frames_per_step

    fig = plt.figure(figsize=(18, 8), facecolor="black")
    grid = fig.add_gridspec(2, 4, width_ratios=[1.65, 1.65, 1.0, 1.18], height_ratios=[1.0, 1.0])
    ax_state = fig.add_subplot(grid[:, 0], facecolor="black")
    ax_recovery = fig.add_subplot(grid[:, 1], facecolor="black")
    ax_input = fig.add_subplot(grid[0, 2], facecolor="black")
    ax_output = fig.add_subplot(grid[1, 2], facecolor="black")
    ax_decode = fig.add_subplot(grid[0, 3], facecolor="black")
    ax_text = fig.add_subplot(grid[1, 3], facecolor="black")
    ax_text.axis("off")

    for axis in (ax_state, ax_recovery, ax_input, ax_output, ax_decode):
        axis.tick_params(colors="white")
        axis.xaxis.label.set_color("white")
        axis.yaxis.label.set_color("white")
        axis.title.set_color("white")
        for spine in axis.spines.values():
            spine.set_color("#777777")

    fixed_scatter = _draw_fixed_points(ax_state, fixed_point_pc, fixed_point_decoded)
    _draw_fixed_points(ax_recovery, fixed_point_pc, fixed_point_decoded)
    if fixed_scatter is not None:
        cbar = fig.colorbar(fixed_scatter, ax=ax_state, orientation="horizontal", fraction=0.055, pad=0.09)
        cbar.set_label("Fixed-point angle (deg)", color="white")
        cbar.ax.xaxis.set_tick_params(color="white", labelcolor="white")
        cbar.outline.set_edgecolor("white")

    lines: list[LineCollection] = []
    heads = []
    for trial in chosen_trials:
        color = cmap(float(target_angles[trial] % (2.0 * np.pi)) / (2.0 * np.pi))
        collection = LineCollection([], colors=[color], linewidths=1.5, alpha=0.75)
        ax_state.add_collection(collection)
        lines.append(collection)
        head = ax_state.scatter([], [], marker="o", s=36, color=[color], edgecolors="white", linewidths=0.35)
        heads.append(head)

    ax_state.set_xlim(*xlim)
    ax_state.set_ylim(*ylim)
    ax_state.set_xlabel("PC 1")
    ax_state.set_ylabel("PC 2")
    ax_state.set_title("Hidden states moving toward the ring attractor")
    ax_state.scatter([], [], marker="o", s=28, color="white", label="current hidden state")
    ax_state.scatter([], [], marker="o", s=28, color="#888888", label="sampled fixed points")
    state_legend = ax_state.legend(frameon=False, loc="lower left", fontsize=8)
    for text in state_legend.get_texts():
        text.set_color("white")

    recovery_colors = np.degrees(recovery_angles % (2.0 * np.pi))
    recovery_scatter = ax_recovery.scatter(
        recovery_pc[0, :, 0],
        recovery_pc[0, :, 1],
        c=recovery_colors,
        cmap="hsv",
        vmin=0.0,
        vmax=360.0,
        s=8 if recovery_pc.shape[1] > 256 else 24,
        alpha=0.55 if recovery_pc.shape[1] > 256 else 0.85,
        edgecolors="none",
    )
    ax_recovery.set_xlim(*xlim)
    ax_recovery.set_ylim(*ylim)
    ax_recovery.set_xlabel("PC 1")
    ax_recovery.set_ylabel("PC 2")
    ax_recovery.set_title(f"Perturbed states returning to the attractor (n={recovery_pc.shape[1]})")
    ax_recovery.scatter([], [], marker="o", s=28, color="white", label="perturbed state")
    ax_recovery.scatter([], [], marker="o", s=28, color="#888888", label="sampled fixed points")
    recovery_legend = ax_recovery.legend(frameon=False, loc="lower left", fontsize=8)
    for text in recovery_legend.get_texts():
        text.set_color("white")

    input_line, = ax_input.plot([], [], color="#62c370", linewidth=2)
    output_line, = ax_output.plot([], [], color="#4aa3ff", linewidth=2)
    for axis, title in ((ax_input, "Cue input population"), (ax_output, "Model output population")):
        axis.set_xlim(0.0, 360.0)
        axis.set_ylim(-0.05, 1.05)
        axis.set_xlabel("Preferred angle (deg)")
        axis.set_ylabel("Activity")
        axis.set_title(title)

    decoded_line, = ax_decode.plot([], [], color="#4aa3ff", linewidth=2, label="decoded")
    target_line = ax_decode.axhline(target_degrees[representative], color="#ff5c5c", linestyle=":", linewidth=1.5, label="target")
    frame_line = ax_decode.axvline(0.0, color="#ffffff", linewidth=1.0, alpha=0.8)
    ax_decode.set_xlim(0, trajectory_pc.shape[0] - 1)
    ax_decode.set_ylim(target_degrees[representative] - 35.0, target_degrees[representative] + 35.0)
    ax_decode.set_xlabel("Time step")
    ax_decode.set_ylabel("Angle (deg)")
    ax_decode.set_title("Decoded memory angle")
    decode_legend = ax_decode.legend(frameon=False, loc="upper right", fontsize=8)
    for text in decode_legend.get_texts():
        text.set_color("white")

    for name, span in phase_index.items():
        ax_decode.axvspan(span.start, span.stop, color=_phase_color(name), alpha=0.16)

    status = ax_text.text(0.02, 0.92, "", color="white", fontsize=11, va="top", family="monospace")
    explanation = ax_text.text(
        0.02,
        0.28,
        "Cue: input bump moves the state.\n"
        "Delay: recurrent dynamics hold the ring.\n"
        "Middle: perturbed delay states\n"
        "return under blank dynamics.\n"
        "Smooth frames are interpolated\n"
        "between real model steps.",
        color="#d8d8d8",
        fontsize=8.0,
        va="top",
        linespacing=1.45,
    )
    explanation.set_wrap(True)
    if recovery_source.startswith("fixed_point_landscape"):
        recovery_source_label = "landscape perturbed starts"
    elif recovery_source.startswith("wide_perturbed"):
        recovery_source_label = "wide perturbation starts"
    else:
        recovery_source_label = "movie selected starts"

    def update(frame: int) -> list[Any]:
        model_position = min(frame / frames_per_step, trajectory_pc.shape[0] - 1)
        model_index = int(np.floor(model_position))
        phase = _current_phase(model_index, phase_index)
        for collection, head, trial in zip(lines, heads, chosen_trials):
            xy = _trajectory_until(trajectory_pc[:, trial, :], model_position)
            if len(xy) > 1:
                collection.set_segments(_make_segments(xy))
            else:
                collection.set_segments([])
            head.set_offsets(xy[-1:])
        recovery_position = min(frame / frames_per_step, recovery_pc.shape[0] - 1)
        recovery_scatter.set_offsets(_interp_time(recovery_pc, recovery_position))

        cue_activity = _interp_time(input_population[:, representative, :], model_position)
        output_activity = _interp_time(output_population[:, representative, :], model_position)
        input_line.set_data(preferred_degrees, cue_activity)
        output_line.set_data(preferred_degrees, output_activity)

        decoded = np.degrees(np.unwrap(decoded_over_time[: model_index + 1, representative]))
        target = target_degrees[representative]
        decoded = target + ((decoded - target + 180.0) % 360.0 - 180.0)
        if model_position > model_index and model_index < trajectory_pc.shape[0] - 1:
            next_decoded = np.degrees(np.unwrap(decoded_over_time[model_index : model_index + 2, representative]))[-1]
            next_decoded = target + ((next_decoded - target + 180.0) % 360.0 - 180.0)
            interp_decoded = decoded[-1] + (next_decoded - decoded[-1]) * (model_position - model_index)
            decoded_plot = np.concatenate([decoded, [interp_decoded]])
            time_plot = np.concatenate([time[: model_index + 1], [model_position]])
        else:
            decoded_plot = decoded
            time_plot = time[: model_index + 1]
        decoded_line.set_data(time_plot, decoded_plot)
        target_line.set_ydata([target, target])
        frame_line.set_xdata([model_position, model_position])

        err = float(_interp_time(angular_error_degrees[:, representative], model_position))
        status.set_text(
            f"time step: {model_position:05.1f}\n"
            f"phase: {phase}\n"
            f"target: {target:6.1f} deg\n"
            f"decoded: {decoded_plot[-1]:6.1f} deg\n"
            f"error: {err:6.2f} deg\n"
            f"perturb SD: {recovery_perturbation_scale:.2f}\n"
            f"perturb n: {recovery_pc.shape[1]}\n"
            f"source: {recovery_source_label}"
        )
        status.set_color(_phase_color(phase))
        return [
            *lines,
            *heads,
            recovery_scatter,
            input_line,
            output_line,
            decoded_line,
            target_line,
            frame_line,
            status,
        ]

    animation = FuncAnimation(fig, update, frames=visual_frames, interval=1000 / fps, blit=False)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.91, bottom=0.11, wspace=0.30, hspace=0.30)
    if output_format == "mp4":
        writer = FFMpegWriter(fps=fps, bitrate=2200)
    else:
        writer = PillowWriter(fps=fps)
    animation.save(path, writer=writer, dpi=130)
    plt.close(fig)


def main() -> None:
    """Parse command-line arguments and generate a hidden-state movie."""
    parser = argparse.ArgumentParser(description="Animate tuned RNN hidden-state dynamics.")
    parser.add_argument("--config", default="configs/yang_fixation_circular_working_memory.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint produced by training.")
    parser.add_argument("--n-trials", type=int, default=64, help="Number of trials to sample.")
    parser.add_argument("--example-trials", type=int, default=16, help="Number of simultaneous trajectories to draw.")
    parser.add_argument("--fps", type=int, default=6, help="Animation frames per second.")
    parser.add_argument("--format", choices=["gif", "mp4"], default="gif", help="Movie format.")
    parser.add_argument(
        "--delay-steps",
        type=int,
        help="Optional movie-only delay length override. Use 90 for a 100-step trial with the stable tuned config.",
    )
    parser.add_argument("--perturbation-scale", type=float, default=0.4, help="Hidden-state perturbation SD for the recovery panel.")
    parser.add_argument("--recovery-steps", type=int, help="Number of blank-dynamics recovery steps to animate.")
    parser.add_argument(
        "--recovery-source",
        choices=["landscape-perturbed", "wide-perturbed"],
        default="landscape-perturbed",
        help="Use saved landscape perturbations or sample a new wider perturbation cloud from movie trials.",
    )
    parser.add_argument(
        "--recovery-perturbations-per-trial",
        type=int,
        default=4,
        help="Perturbed starts per movie trial when --recovery-source wide-perturbed.",
    )
    parser.add_argument(
        "--frames-per-step",
        type=int,
        default=1,
        help="Number of interpolated video frames per real recurrent model step.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Override analysis device.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device
    result = run_hidden_state_movie(
        config,
        args.checkpoint,
        n_trials=args.n_trials,
        example_trials=args.example_trials,
        fps=args.fps,
        output_format=args.format,
        delay_steps=args.delay_steps,
        perturbation_scale=args.perturbation_scale,
        recovery_steps=args.recovery_steps,
        frames_per_step=args.frames_per_step,
        recovery_source=args.recovery_source,
        recovery_perturbations_per_trial=args.recovery_perturbations_per_trial,
    )
    print(f"movie={result.movie_path}")
    print(f"arrays={result.arrays_path}")
    print(f"summary={result.summary_path}")


if __name__ == "__main__":
    main()
