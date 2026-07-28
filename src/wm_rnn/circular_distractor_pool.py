"""Train and evaluate the one-item circular distractor checkpoint family."""

from __future__ import annotations

import argparse
import csv
import hashlib
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wm_rnn.config import load_config
from wm_rnn.device import select_device
from wm_rnn.io import ensure_run_dirs, write_json
from wm_rnn.train import train_model
from wm_rnn.training_utils import (
    batch_to_tensors,
    fresh_model,
    task_config_from_dict,
)
from wm_rnn.tuned_task import (
    TunedDelayTaskConfig,
    circular_angular_error,
    decode_population_angle,
)


@dataclass(frozen=True)
class CircularConditionMetrics:
    """Held-out metrics for one clean or distractor condition."""

    condition: str
    mean_angular_error_degrees: float
    median_angular_error_degrees: float
    fixation_accuracy: float
    trials: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_trained_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    embedded_config = payload["config"]
    model = fresh_model(embedded_config, device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, embedded_config


def evaluate_condition(
    model: torch.nn.Module,
    config: dict[str, Any],
    condition: str,
    distractor_onset_fraction: float | None = None,
) -> CircularConditionMetrics:
    """Evaluate paired held-out circular trials for one task condition."""
    if condition not in {"clean", "distractor"}:
        raise ValueError("condition must be 'clean' or 'distractor'")
    evaluation = config["evaluation"]
    trials = int(evaluation["trials_per_condition"])
    batch_size = int(evaluation["batch_size"])
    if trials <= 0 or batch_size <= 0 or trials % batch_size != 0:
        raise ValueError(
            "evaluation trials must be positive and divisible by batch_size"
        )
    base_task = task_config_from_dict(config, batch_size=batch_size)
    if not isinstance(base_task, TunedDelayTaskConfig):
        raise TypeError("circular distractor evaluation requires a tuned task")
    configured_distractor_steps = int(base_task.distractor_steps)
    task = replace(
        base_task,
        pre_cue_steps=int(config["task"]["pre_cue_steps"]),
        cue_steps=int(config["task"]["cue_steps"]),
        delay_steps=int(config["task"]["delay_steps"]),
        distractor_steps=(
            configured_distractor_steps if condition == "distractor" else 0
        ),
        distractor_onset_fraction=(
            float(distractor_onset_fraction)
            if distractor_onset_fraction is not None
            else base_task.distractor_onset_fraction
        ),
    )
    device = next(model.parameters()).device
    response_transition = int(
        config["training"].get("response_transition_steps", 5)
    )
    ignore_initial = int(config["training"].get("ignore_initial_steps", 5))
    errors: list[np.ndarray] = []
    fixation_correct = 0
    fixation_total = 0
    n_batches = trials // batch_size

    with torch.inference_mode():
        for batch_index in range(n_batches):
            batch = generate_paired_batch(
                task,
                int(evaluation["seed_base"]) + batch_index,
            )
            inputs, targets, _ = batch_to_tensors(batch, device)
            outputs, _ = model(inputs)
            response = batch.phase_index["response"]
            scored_response = slice(
                response.start + response_transition,
                response.stop,
            )
            decoded = decode_population_angle(
                outputs[
                    scored_response, :, : task.n_tuned_units
                ].cpu().numpy(),
                batch.preferred_angles,
            )
            target_angles = np.broadcast_to(
                batch.angles[np.newaxis, :],
                decoded.shape,
            )
            trial_errors = np.degrees(
                circular_angular_error(decoded, target_angles)
            ).mean(axis=0)
            errors.append(trial_errors)

            fixation_predictions = (
                outputs[ignore_initial:, :, task.n_tuned_units] >= 0.5
            )
            fixation_targets = (
                targets[ignore_initial:, :, task.n_tuned_units] >= 0.5
            )
            fixation_correct += int(
                (fixation_predictions == fixation_targets).sum().item()
            )
            fixation_total += int(fixation_targets.numel())

    all_errors = np.concatenate(errors)
    return CircularConditionMetrics(
        condition=condition,
        mean_angular_error_degrees=float(np.mean(all_errors)),
        median_angular_error_degrees=float(np.median(all_errors)),
        fixation_accuracy=float(fixation_correct / fixation_total),
        trials=trials,
    )


def generate_paired_batch(
    task: TunedDelayTaskConfig,
    seed: int,
):
    """Generate a deterministic batch; matching seeds pair target angles."""
    from wm_rnn.tuned_task import generate_tuned_delay_batch

    return generate_tuned_delay_batch(replace(task, seed=int(seed)))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_pool(config: dict[str, Any]) -> tuple[Path, Path]:
    """Train every configured seed and evaluate clean/distractor competence."""
    seeds = [int(seed) for seed in config["pool"]["seeds"]]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("pool.seeds must be non-empty and unique")
    target_competent = int(
        config["pool"].get("target_competent_checkpoints", len(seeds))
    )
    if not 0 < target_competent <= len(seeds):
        raise ValueError(
            "pool.target_competent_checkpoints must lie within the seed pool"
        )
    output_dir = Path(config["paths"]["output_dir"])
    run_name = str(config["paths"]["run_name"])
    dirs = ensure_run_dirs(output_dir)
    selected = select_device(config["training"].get("device", "auto"))
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        seed_config = deepcopy(config)
        seed_output = output_dir / "seed_sweep" / f"seed_{seed}"
        seed_run_name = f"{run_name}_seed_{seed}"
        seed_config["task"]["seed"] = seed
        seed_config["paths"]["output_dir"] = str(seed_output)
        seed_config["paths"]["run_name"] = seed_run_name
        checkpoint = seed_output / "checkpoints" / f"{seed_run_name}.pt"
        if checkpoint.exists():
            checkpoint_payload = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            train_final_loss = float(checkpoint_payload["history"][-1]["loss"])
            checkpoint_path = checkpoint
            print(f"seed={seed} reusing existing checkpoint")
        else:
            train_result = train_model(seed_config)
            train_final_loss = float(train_result.history[-1]["loss"])
            checkpoint_path = train_result.checkpoint_path
        model, embedded_config = _load_trained_model(
            checkpoint_path,
            selected.device,
        )
        clean = evaluate_condition(model, embedded_config, "clean")
        distractor = evaluate_condition(model, embedded_config, "distractor")
        evaluation = config["evaluation"]
        passed = (
            clean.mean_angular_error_degrees
            <= float(evaluation["clean_max_mean_angular_error_degrees"])
            and distractor.mean_angular_error_degrees
            <= float(evaluation["distractor_max_mean_angular_error_degrees"])
            and min(clean.fixation_accuracy, distractor.fixation_accuracy)
            >= float(evaluation["fixation_min_accuracy"])
        )
        row = {
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "train_final_loss": train_final_loss,
            "clean_mean_angular_error_degrees": (
                clean.mean_angular_error_degrees
            ),
            "clean_median_angular_error_degrees": (
                clean.median_angular_error_degrees
            ),
            "distractor_mean_angular_error_degrees": (
                distractor.mean_angular_error_degrees
            ),
            "distractor_median_angular_error_degrees": (
                distractor.median_angular_error_degrees
            ),
            "distractor_cost_degrees": (
                distractor.mean_angular_error_degrees
                - clean.mean_angular_error_degrees
            ),
            "clean_fixation_accuracy": clean.fixation_accuracy,
            "distractor_fixation_accuracy": distractor.fixation_accuracy,
            "competence_passed": passed,
        }
        rows.append(row)
        print(
            f"seed={seed} clean_error="
            f"{clean.mean_angular_error_degrees:.3f} "
            f"distractor_error={distractor.mean_angular_error_degrees:.3f} "
            f"passed={passed}"
        )
        if sum(bool(item["competence_passed"]) for item in rows) >= (
            target_competent
        ):
            break

    csv_path = _write_csv(
        dirs["metrics"] / f"{run_name}_pool.csv",
        rows,
    )
    summary_path = write_json(
        dirs["metrics"] / f"{run_name}_pool_summary.json",
        {
            "model_family": "single_item_circular_with_trained_distractors",
            "device": selected.description,
            "seeds": seeds,
            "evaluated_seeds": [int(row["seed"]) for row in rows],
            "target_competent_checkpoints": target_competent,
            "training_trial_type_sampling": config["training"][
                "trial_type_sampling"
            ],
            "evaluation": config["evaluation"],
            "competent_checkpoints": int(
                sum(bool(row["competence_passed"]) for row in rows)
            ),
            "retained_checkpoint_seeds": [
                int(row["seed"])
                for row in rows
                if bool(row["competence_passed"])
            ][:target_competent],
            "results": rows,
        },
    )
    return csv_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the one-item circular distractor family."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/fixation_circular_distractor_working_memory.yaml",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        help="Override the configured device.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required because this command trains a checkpoint family.",
    )
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("training withheld; pass --execute to run the pool")
    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device
    csv_path, summary_path = run_pool(config)
    print(f"csv={csv_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
