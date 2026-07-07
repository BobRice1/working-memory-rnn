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
from wm_rnn.training_utils import batch_to_tensors, fresh_model, generate_batch_for_task, task_config_from_dict


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
    task_type = str(config["task"].get("task_type", "categorical"))
    model = fresh_model(config, device_info.device)
    checkpoint = torch.load(checkpoint_path, map_location=device_info.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    n_trials = int(config["analysis"].get("n_trials", 64))
    task_config = task_config_from_dict(config, seed_offset=20000, batch_size=n_trials)
    batch = generate_batch_for_task(task_config)
    labels = batch.angles if task_type == "tuned" else batch.cues

    with torch.no_grad():
        inputs, _, _ = batch_to_tensors(batch, device_info.device)
        _, hidden = model(inputs)

    hidden_np = hidden.detach().cpu().numpy()
    flattened = hidden_np.reshape(-1, hidden_np.shape[-1])
    pca = PCA(n_components=int(config["analysis"].get("n_components", 2)))
    projected = pca.fit_transform(flattened).reshape(hidden_np.shape[0], hidden_np.shape[1], -1)

    run_name = config["paths"].get("run_name", "baseline_delay")
    hidden_states_path = dirs["arrays"] / f"{run_name}_hidden_states.npz"
    arrays = {"hidden": hidden_np, "projected": projected, "labels": labels, "task_type": task_type}
    if task_type == "categorical":
        arrays["cues"] = labels
    np.savez_compressed(hidden_states_path, **arrays)

    figure_path = dirs["figures"] / f"{run_name}_pca_trajectories.png"
    _plot_trajectories(projected, labels, figure_path, task_type)
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


def _plot_trajectories(projected: np.ndarray, labels: np.ndarray, figure_path: Path, task_type: str) -> None:
    """Save a 2D PCA trajectory plot colored by remembered label."""
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    if task_type == "tuned":
        cmap = plt.get_cmap("hsv")
        for trial_idx in range(min(projected.shape[1], labels.shape[0], 32)):
            xy = projected[:, trial_idx, :2]
            color = cmap(float(labels[trial_idx] % (2.0 * np.pi)) / (2.0 * np.pi))
            ax.plot(xy[:, 0], xy[:, 1], marker="o", markersize=2, linewidth=1, alpha=0.55, color=color)
            if trial_idx < 12:
                ax.scatter(xy[0, 0], xy[0, 1], marker="s", s=16, color="black", alpha=0.45)
                ax.scatter(xy[-1, 0], xy[-1, 1], marker="^", s=20, color=color, edgecolors="black", linewidths=0.25)
        title = "Tuned delay-task hidden-state trajectories by angle"
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=360.0))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Target angle (deg)")
        ax.scatter([], [], marker="s", s=28, color="black", alpha=0.45, label="trajectory start")
        ax.scatter([], [], marker="^", s=32, color="white", edgecolors="black", label="trajectory end")
        ax.legend(frameon=False, loc="best", fontsize=8)
    else:
        classes = np.unique(labels)
        cmap = plt.get_cmap("tab10")
        for class_idx in classes:
            trial_indices = np.where(labels == class_idx)[0][:8]
            for trial_idx in trial_indices:
                xy = projected[:, trial_idx, :2]
                ax.plot(
                    xy[:, 0],
                    xy[:, 1],
                    marker="o",
                    markersize=2,
                    linewidth=1,
                    alpha=0.55,
                    color=cmap(int(class_idx)),
                    label=f"cue {int(class_idx)}" if trial_idx == trial_indices[0] else None,
                )
        title = "Delay-task hidden-state trajectories by cue"
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    plt.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=160)
    plt.close(fig)


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
