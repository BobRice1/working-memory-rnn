"""Competence evaluation for shared 0-back/2-back checkpoints."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from wm_rnn.config import load_config
from wm_rnn.device import select_device
from wm_rnn.io import ensure_run_dirs, write_json
from wm_rnn.nback_metrics import nback_metrics
from wm_rnn.nback_task import NBackTaskConfig, generate_nback_batch
from wm_rnn.training_utils import (
    batch_to_tensors,
    fresh_model,
    task_config_from_dict,
)


@dataclass(frozen=True)
class NBackEvalResult:
    """Saved and in-memory N-back competence results."""

    metrics_path: Path
    metrics: dict[str, Any]
    passed: bool


def resolve_nback_bank_seed(
    config: dict[str, Any],
    section: str,
) -> int:
    """Resolve a checkpoint-specific base seed for a frozen data bank."""
    bank = config[section]
    checkpoint_seed = int(config["task"]["seed"])
    stride = int(bank.get("checkpoint_seed_stride", 1))
    if stride < 1:
        raise ValueError("checkpoint_seed_stride must be at least one")
    return int(bank.get("seed_offset", 0)) + stride * checkpoint_seed


def _metric_without_sequences(metrics: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(metrics)
    result.pop("sequence_cross_entropies", None)
    return result


def evaluate_nback_conditions(
    model: torch.nn.Module,
    config: dict[str, Any],
    *,
    seed: int,
    sequences_per_condition: int,
) -> dict[str, dict[str, Any]]:
    """Evaluate one shared model on fixed homogeneous 0- and 2-back banks."""
    base_task = task_config_from_dict(
        config, batch_size=sequences_per_condition
    )
    if not isinstance(base_task, NBackTaskConfig):
        raise ValueError("N-back evaluation requires task_type: n_back")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    results: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for offset, n_back in enumerate((0, 2)):
            task = replace(
                base_task,
                n_back=n_back,
                seed=int(seed) + offset,
            )
            batch = generate_nback_batch(task)
            inputs, targets, loss_mask = batch_to_tensors(batch, device)
            logits, _ = model(inputs)
            results[f"{n_back}-back"] = _metric_without_sequences(
                nback_metrics(logits, targets, loss_mask, batch)
            )
    model.train(was_training)
    return results


def competence_checks(
    metrics: dict[str, dict[str, Any]],
    gates: dict[str, float],
) -> dict[str, Any]:
    """Apply the frozen shared-task competence criteria."""
    zero = metrics["0-back"]
    two = metrics["2-back"]
    checks = {
        "zero_back_accuracy": zero["accuracy"]
        >= float(gates["accuracy"]),
        "two_back_accuracy": two["accuracy"]
        >= float(gates["accuracy"]),
        "zero_back_discriminability": zero["discriminability"]
        >= float(gates["discriminability"]),
        "two_back_discriminability": two["discriminability"]
        >= float(gates["discriminability"]),
        "two_back_lure_accuracy": two["one_back_lure_accuracy"]
        >= float(gates["lure_accuracy"]),
        "zero_back_has_both_classes": (
            zero["match_count"] > 0 and zero["nonmatch_count"] > 0
        ),
        "two_back_has_both_classes": (
            two["match_count"] > 0 and two["nonmatch_count"] > 0
        ),
        "two_back_has_lures": two["one_back_lure_count"] > 0,
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def stage_one_checks(
    metrics: dict[str, dict[str, Any]],
    gates: dict[str, float],
) -> dict[str, Any]:
    """Apply the frozen 0-back acquisition criteria."""
    zero = metrics["0-back"]
    checks = {
        "zero_back_accuracy": zero["accuracy"]
        >= float(gates["accuracy"]),
        "zero_back_discriminability": zero["discriminability"]
        >= float(gates["discriminability"]),
        "zero_back_has_both_classes": (
            zero["match_count"] > 0 and zero["nonmatch_count"] > 0
        ),
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def evaluate_nback_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    seed: int | None = None,
) -> NBackEvalResult:
    """Evaluate a saved N-back checkpoint on its fresh competence bank."""
    device_info = select_device(config["training"].get("device", "auto"))
    model = fresh_model(config, device_info.device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device_info.device
    )
    model.load_state_dict(checkpoint["model_state"])
    evaluation = config["evaluation"]
    resolved_seed = int(
        seed
        if seed is not None
        else resolve_nback_bank_seed(config, "evaluation")
    )
    metrics_by_condition = evaluate_nback_conditions(
        model,
        config,
        seed=resolved_seed,
        sequences_per_condition=int(
            evaluation["sequences_per_condition"]
        ),
    )
    acceptance = competence_checks(
        metrics_by_condition, config["competence"]
    )
    payload = {
        "device": device_info.description,
        "seed": resolved_seed,
        "sequences_per_condition": int(
            evaluation["sequences_per_condition"]
        ),
        "checkpoint": str(checkpoint_path),
        "conditions": metrics_by_condition,
        "acceptance": acceptance,
    }
    dirs = ensure_run_dirs(config["paths"]["output_dir"])
    run_name = config["paths"]["run_name"]
    metrics_path = write_json(
        dirs["metrics"] / f"{run_name}_nback_competence.json",
        payload,
    )
    return NBackEvalResult(
        metrics_path=metrics_path,
        metrics=payload,
        passed=bool(acceptance["passed"]),
    )


def main() -> None:
    """Evaluate a trained N-back checkpoint from the command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate a shared 0-back/2-back checkpoint."
    )
    parser.add_argument(
        "--config",
        default="configs/nback_working_memory.yaml",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device
    result = evaluate_nback_checkpoint(config, args.checkpoint)
    print(f"metrics={result.metrics_path}")
    print(f"passed={result.passed}")


if __name__ == "__main__":
    main()
