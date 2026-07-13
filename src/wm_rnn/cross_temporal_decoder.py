"""Cross-temporal decoding of circular memory content from RNN hidden states."""

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
from sklearn.linear_model import Ridge

from wm_rnn.config import load_config
from wm_rnn.device import select_device
from wm_rnn.io import ensure_run_dirs, write_json
from wm_rnn.training_utils import batch_to_tensors, fresh_model, generate_batch_for_task, task_config_from_dict
from wm_rnn.tuned_task import circular_angular_error


@dataclass(frozen=True)
class DecoderResult:
    """Artifacts produced by cross-temporal hidden-state decoding."""

    arrays_path: Path
    figure_path: Path
    summary_path: Path
    mean_error_degrees: np.ndarray


def decode_hidden_states_cross_temporally(
    hidden: np.ndarray,
    angles: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    ridge_alpha: float = 1.0,
) -> np.ndarray:
    """Return train-time by test-time mean circular decoding error in degrees."""
    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [time, trials, units]")
    if hidden.shape[1] != len(angles):
        raise ValueError("angles must contain one value per trial")

    targets = np.column_stack((np.cos(angles), np.sin(angles)))
    errors = np.empty((hidden.shape[0], hidden.shape[0]), dtype=np.float64)
    for train_time in range(hidden.shape[0]):
        decoder = Ridge(alpha=float(ridge_alpha))
        decoder.fit(hidden[train_time, train_indices], targets[train_indices])
        for test_time in range(hidden.shape[0]):
            vectors = decoder.predict(hidden[test_time, test_indices])
            predicted = np.arctan2(vectors[:, 1], vectors[:, 0]) % (2.0 * np.pi)
            error = circular_angular_error(predicted, angles[test_indices])
            errors[train_time, test_time] = float(np.degrees(error).mean())
    return errors


def run_cross_temporal_decoder(config: dict[str, Any], checkpoint_path: str | Path) -> DecoderResult:
    """Fit and evaluate circular hidden-state decoders across all trial times."""
    if str(config["task"].get("task_type", "categorical")) != "tuned":
        raise ValueError("cross-temporal circular decoding requires task.task_type: tuned")

    device_info = select_device(config["training"].get("device", "auto"))
    model = fresh_model(config, device_info.device)
    checkpoint = torch.load(checkpoint_path, map_location=device_info.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    decoder_config = config.get("decoder", {})
    n_trials = int(decoder_config.get("n_trials", 512))
    train_fraction = float(decoder_config.get("train_fraction", 0.75))
    ridge_alpha = float(decoder_config.get("ridge_alpha", 1.0))
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("decoder.train_fraction must be between zero and one")

    task_config = task_config_from_dict(config, seed_offset=50000, batch_size=n_trials)
    batch = generate_batch_for_task(task_config)
    with torch.no_grad():
        inputs, _, _ = batch_to_tensors(batch, device_info.device)
        _, hidden = model(inputs)
    hidden_np = hidden.detach().cpu().numpy()

    rng = np.random.default_rng(int(config["task"].get("seed", 0)) + 60000)
    permutation = rng.permutation(n_trials)
    split = int(round(n_trials * train_fraction))
    train_indices, test_indices = permutation[:split], permutation[split:]
    errors = decode_hidden_states_cross_temporally(
        hidden_np,
        np.asarray(batch.angles),
        train_indices,
        test_indices,
        ridge_alpha,
    )

    dirs = ensure_run_dirs(config["paths"]["output_dir"])
    run_name = config["paths"].get("run_name", "circular_working_memory")
    arrays_path = dirs["arrays"] / f"{run_name}_cross_temporal_decoder.npz"
    np.savez_compressed(
        arrays_path,
        mean_error_degrees=errors,
        diagonal_error_degrees=np.diag(errors),
        angles=np.asarray(batch.angles),
        train_indices=train_indices,
        test_indices=test_indices,
    )
    cue_slice = batch.phase_index["cue"]
    delay_slice = batch.phase_index["delay"]
    response_slice = batch.phase_index["response"]
    figure_path = dirs["figures"] / f"{run_name}_cross_temporal_decoder.png"
    _plot_decoder(
        errors,
        figure_path,
        phase_boundaries=[cue_slice.start, cue_slice.stop, delay_slice.stop, response_slice.stop],
    )
    delay_diagonal = np.diag(errors)[delay_slice]
    summary_path = write_json(
        dirs["metrics"] / f"{run_name}_cross_temporal_decoder_summary.json",
        {
            "checkpoint": str(checkpoint_path),
            "device": device_info.description,
            "n_trials": n_trials,
            "n_train_trials": int(len(train_indices)),
            "n_test_trials": int(len(test_indices)),
            "ridge_alpha": ridge_alpha,
            "mean_diagonal_error_degrees": float(np.diag(errors).mean()),
            "mean_delay_diagonal_error_degrees": float(delay_diagonal.mean()),
            "max_delay_diagonal_error_degrees": float(delay_diagonal.max()),
            "phase_boundaries": {
                "cue_start": int(cue_slice.start),
                "cue_end": int(cue_slice.stop),
                "delay_start": int(delay_slice.start),
                "delay_end": int(delay_slice.stop),
                "response_start": int(response_slice.start),
                "response_end": int(response_slice.stop),
            },
            "arrays": str(arrays_path),
            "figure": str(figure_path),
        },
    )
    return DecoderResult(arrays_path, figure_path, summary_path, errors)


def _plot_decoder(errors: np.ndarray, path: Path, phase_boundaries: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    image = ax.imshow(errors, origin="lower", aspect="auto", cmap="viridis_r", vmin=0.0)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Mean angular error (degrees)")
    ax.set_xlabel("Test time step")
    ax.set_ylabel("Decoder training time step")
    ax.set_title("Cross-temporal hidden-state decoding")
    for boundary in phase_boundaries:
        ax.axvline(boundary - 0.5, color="white", linewidth=0.8, alpha=0.8)
        ax.axhline(boundary - 0.5, color="white", linewidth=0.8, alpha=0.8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode circular memory content from hidden states across time.")
    parser.add_argument("--config", required=True, help="Path to the model YAML configuration.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained checkpoint.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Override analysis device.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device
    result = run_cross_temporal_decoder(config, args.checkpoint)
    print(f"summary={result.summary_path}")
    print(f"figure={result.figure_path}")


if __name__ == "__main__":
    main()
