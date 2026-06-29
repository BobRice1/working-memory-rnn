"""Hidden-state PCA analysis for trained working-memory RNN checkpoints."""

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
from sklearn.decomposition import PCA

from wm_rnn.config import load_config
from wm_rnn.device import select_device
from wm_rnn.io import ensure_run_dirs, write_json
from wm_rnn.task import generate_delay_batch
from wm_rnn.training_utils import batch_to_tensors, fresh_model, task_config_from_dict


@dataclass(frozen=True)
class PCAResult:
    """Output paths produced by hidden-state PCA analysis.

    Attributes:
        figure_path: PNG trajectory plot path.
        hidden_states_path: Compressed NumPy archive of hidden states and PCA
            projections.
        summary_path: JSON file containing PCA variance and output paths.
    """

    figure_path: Path
    hidden_states_path: Path
    summary_path: Path


def run_pca_analysis(config: dict[str, Any], checkpoint_path: str | Path) -> PCAResult:
    """Project hidden-state trajectories into PCA space and save outputs.

    Args:
        config: Experiment configuration dictionary.
        checkpoint_path: Path to a checkpoint produced by ``train_model``.

    Returns:
        ``PCAResult`` containing paths to the saved figure, arrays, and summary.
    """
    device_info = select_device(config["training"].get("device", "auto"))
    dirs = ensure_run_dirs(config["paths"]["output_dir"])
    model = fresh_model(config, device_info.device)
    checkpoint = torch.load(checkpoint_path, map_location=device_info.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    n_trials = int(config["analysis"].get("n_trials", 64))
    task_config = task_config_from_dict(config, seed_offset=20000, batch_size=n_trials)
    batch = generate_delay_batch(task_config)

    with torch.no_grad():
        inputs, _, _ = batch_to_tensors(batch, device_info.device)
        _, hidden = model(inputs)

    hidden_np = hidden.detach().cpu().numpy()
    flattened = hidden_np.reshape(-1, hidden_np.shape[-1])
    pca = PCA(n_components=int(config["analysis"].get("n_components", 2)))
    projected = pca.fit_transform(flattened).reshape(hidden_np.shape[0], hidden_np.shape[1], -1)

    run_name = config["paths"].get("run_name", "baseline_delay")
    hidden_states_path = dirs["arrays"] / f"{run_name}_hidden_states.npz"
    np.savez_compressed(hidden_states_path, hidden=hidden_np, projected=projected, cues=batch.cues)

    figure_path = dirs["figures"] / f"{run_name}_pca_trajectories.png"
    _plot_trajectories(projected, batch.cues, figure_path)
    summary_path = write_json(
        dirs["metrics"] / f"{run_name}_pca_summary.json",
        {
            "device": device_info.description,
            "checkpoint": str(checkpoint_path),
            "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
            "hidden_states": str(hidden_states_path),
            "figure": str(figure_path),
        },
    )
    return PCAResult(figure_path=figure_path, hidden_states_path=hidden_states_path, summary_path=summary_path)


def _plot_trajectories(projected: np.ndarray, cues: np.ndarray, figure_path: Path) -> None:
    """Save a 2D PCA trajectory plot colored by remembered cue class."""
    plt.figure(figsize=(6, 5))
    classes = np.unique(cues)
    cmap = plt.get_cmap("tab10")
    for class_idx in classes:
        trial_indices = np.where(cues == class_idx)[0][:8]
        for trial_idx in trial_indices:
            xy = projected[:, trial_idx, :2]
            plt.plot(xy[:, 0], xy[:, 1], marker="o", markersize=2, linewidth=1, alpha=0.55, color=cmap(int(class_idx)))
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("Delay-task hidden-state trajectories by cue")
    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=160)
    plt.close()


def main() -> None:
    """Parse command-line arguments and run hidden-state PCA analysis."""
    parser = argparse.ArgumentParser(description="Run PCA analysis for a trained working-memory RNN checkpoint.")
    parser.add_argument("--config", default="configs/baseline_delay.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint produced by training.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Override analysis device.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device
    result = run_pca_analysis(config, args.checkpoint)
    print(f"figure={result.figure_path}")
    print(f"hidden_states={result.hidden_states_path}")


if __name__ == "__main__":
    main()
