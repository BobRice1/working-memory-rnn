"""Train a circular RNN family with balanced variable distractor timing."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wm_rnn.circular_distractor_pool import (
    _load_trained_model,
    _sha256,
    _write_csv,
    evaluate_condition,
)
from wm_rnn.config import load_config
from wm_rnn.device import select_device
from wm_rnn.io import ensure_run_dirs, write_json
from wm_rnn.train import draw_distractor_onset_fraction_block, train_model


def validate_variable_timing_config(
    config: dict[str, Any],
) -> tuple[list[int], list[float], int]:
    """Validate and resolve the preregistered pool and timing schedule."""
    seeds = [int(seed) for seed in config["pool"]["seeds"]]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("pool.seeds must be non-empty and unique")
    target = int(config["pool"]["target_competent_checkpoints"])
    if not 0 < target <= len(seeds):
        raise ValueError(
            "pool.target_competent_checkpoints must lie within the seed pool"
        )
    fractions = draw_distractor_onset_fraction_block(
        np.random.default_rng(0),
        config["task"]["distractor_onset_fraction_choices"],
    )
    fractions = sorted(fractions)
    evaluation_fractions = sorted(
        float(value)
        for value in config["evaluation"]["distractor_onset_fractions"]
    )
    if fractions != evaluation_fractions:
        raise ValueError(
            "training and evaluation distractor onset fractions must match"
        )
    return seeds, fractions, target


def _fraction_label(fraction: float) -> str:
    return f"f{round(100 * fraction):03d}"


def run_variable_timing_pool(
    config: dict[str, Any],
) -> tuple[Path, Path]:
    """Retain the first ten candidates competent at every frozen timing."""
    seeds, fractions, target = validate_variable_timing_config(config)
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
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            train_final_loss = float(payload["history"][-1]["loss"])
            checkpoint_path = checkpoint
            print(f"seed={seed} reusing existing checkpoint")
        else:
            result = train_model(seed_config)
            train_final_loss = float(result.history[-1]["loss"])
            checkpoint_path = result.checkpoint_path

        model, embedded_config = _load_trained_model(
            checkpoint_path, selected.device
        )
        clean = evaluate_condition(model, embedded_config, "clean")
        timing_metrics = {
            fraction: evaluate_condition(
                model,
                embedded_config,
                "distractor",
                distractor_onset_fraction=fraction,
            )
            for fraction in fractions
        }
        evaluation = config["evaluation"]
        distractor_errors = [
            timing_metrics[fraction].mean_angular_error_degrees
            for fraction in fractions
        ]
        fixation_accuracies = [clean.fixation_accuracy] + [
            timing_metrics[fraction].fixation_accuracy
            for fraction in fractions
        ]
        passed = (
            clean.mean_angular_error_degrees
            <= float(evaluation["clean_max_mean_angular_error_degrees"])
            and max(distractor_errors)
            <= float(
                evaluation[
                    "distractor_each_max_mean_angular_error_degrees"
                ]
            )
            and min(fixation_accuracies)
            >= float(evaluation["fixation_min_accuracy"])
        )
        row: dict[str, Any] = {
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
            "clean_fixation_accuracy": clean.fixation_accuracy,
            "distractor_mean_angular_error_degrees": float(
                np.mean(distractor_errors)
            ),
            "distractor_max_mean_angular_error_degrees": float(
                max(distractor_errors)
            ),
            "distractor_timing_range_degrees": float(
                max(distractor_errors) - min(distractor_errors)
            ),
            "minimum_fixation_accuracy": float(min(fixation_accuracies)),
            "competence_passed": passed,
        }
        for fraction in fractions:
            label = _fraction_label(fraction)
            metrics = timing_metrics[fraction]
            row[f"{label}_mean_angular_error_degrees"] = (
                metrics.mean_angular_error_degrees
            )
            row[f"{label}_median_angular_error_degrees"] = (
                metrics.median_angular_error_degrees
            )
            row[f"{label}_distractor_cost_degrees"] = (
                metrics.mean_angular_error_degrees
                - clean.mean_angular_error_degrees
            )
            row[f"{label}_fixation_accuracy"] = metrics.fixation_accuracy
        rows.append(row)
        print(
            f"seed={seed} clean_error="
            f"{clean.mean_angular_error_degrees:.3f} "
            f"worst_distractor_error={max(distractor_errors):.3f} "
            f"timing_range={row['distractor_timing_range_degrees']:.3f} "
            f"passed={passed}"
        )
        if sum(bool(item["competence_passed"]) for item in rows) >= target:
            break

    csv_path = _write_csv(
        dirs["metrics"] / f"{run_name}_pool.csv", rows
    )
    retained = [
        int(row["seed"])
        for row in rows
        if bool(row["competence_passed"])
    ][:target]
    summary_path = write_json(
        dirs["metrics"] / f"{run_name}_pool_summary.json",
        {
            "model_family": (
                "single_item_circular_variable_distractor_timing"
            ),
            "device": selected.description,
            "candidate_seed_schedule": seeds,
            "evaluated_seeds": [int(row["seed"]) for row in rows],
            "retention_rule": (
                "retain the first target_competent_checkpoints candidates "
                "passing every frozen competence gate"
            ),
            "target_competent_checkpoints": target,
            "training_distractor_onset_fractions": fractions,
            "evaluation": config["evaluation"],
            "competent_checkpoints": len(retained),
            "pool_complete": len(retained) == target,
            "retained_checkpoint_seeds": retained,
            "results": rows,
        },
    )
    return csv_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the variable-timing circular distractor checkpoint family."
        )
    )
    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "fixation_circular_variable_distractor_working_memory.yaml"
        ),
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"]
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
    csv_path, summary_path = run_variable_timing_pool(config)
    print(f"csv={csv_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
