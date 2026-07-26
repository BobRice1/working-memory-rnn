"""Competence-gated training for a shared 0-back/2-back CTRNN."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wm_rnn.config import load_config
from wm_rnn.device import select_device
from wm_rnn.io import ensure_run_dirs, write_history_csv, write_json
from wm_rnn.nback_evaluation import (
    competence_checks,
    evaluate_nback_conditions,
    stage_one_checks,
)
from wm_rnn.nback_task import NBackTaskConfig, generate_nback_batch
from wm_rnn.training_utils import (
    batch_to_tensors,
    fresh_model,
    masked_cross_entropy,
    response_accuracy,
    task_config_from_dict,
)


@dataclass(frozen=True)
class NBackTrainResult:
    """Outputs from one competence-gated N-back training run."""

    checkpoint_path: Path
    metrics_path: Path
    history_path: Path
    history: list[dict[str, Any]]
    validation_history: list[dict[str, Any]]
    passed: bool


def draw_balanced_nback_block(
    rng: np.random.Generator,
) -> list[int]:
    """Return a reproducibly shuffled block containing 0-back and 2-back."""
    block = [0, 2]
    rng.shuffle(block)
    return block


def _validation_record(
    *,
    step: int,
    stage: int,
    metrics: dict[str, dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "step": int(step),
        "stage": int(stage),
        "passed": bool(gate["passed"]),
        "checks": gate["checks"],
        "conditions": metrics,
    }


def _run_validation(
    model: torch.nn.Module,
    config: dict[str, Any],
    *,
    step: int,
    stage: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    validation = config["validation"]
    metrics = evaluate_nback_conditions(
        model,
        config,
        seed=int(config["task"]["seed"])
        + int(validation.get("seed_offset", 100000)),
        sequences_per_condition=int(
            validation["sequences_per_condition"]
        ),
    )
    gate = (
        stage_one_checks(metrics, config["stage1_competence"])
        if stage == 1
        else competence_checks(metrics, config["competence"])
    )
    return metrics, gate


def train_nback_model(config: dict[str, Any]) -> NBackTrainResult:
    """Train one shared N-back model under the frozen two-stage curriculum."""
    if str(config["task"].get("task_type")) != "n_back":
        raise ValueError("train_nback_model requires task_type: n_back")
    device_info = select_device(config["training"].get("device", "auto"))
    print(device_info.description)
    base_seed = int(config["task"]["seed"])
    torch.manual_seed(base_seed)
    np_rng = np.random.default_rng(base_seed + 777777)
    task = task_config_from_dict(config)
    if not isinstance(task, NBackTaskConfig):
        raise ValueError("N-back task configuration did not resolve correctly")

    model = fresh_model(config, device_info.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
    )
    gradient_clip_norm = float(
        config["training"].get("gradient_clip_norm", 1.0)
    )
    stage1_max = int(config["training"]["stage1_max_steps"])
    stage2_max = int(config["training"]["stage2_max_steps"])
    stage1_min = int(config["training"].get("stage1_min_steps", 200))
    stage1_every = int(config["training"]["stage1_validate_every"])
    stage2_every = int(config["training"]["stage2_validate_every"])
    required_consecutive = int(
        config["training"].get("required_consecutive_passes", 2)
    )
    log_every = int(config["training"].get("log_every", 100))

    history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    global_step = 0
    consecutive = 0
    stage1_passed = False

    def train_step(n_back: int, stage: int) -> None:
        nonlocal global_step
        global_step += 1
        batch_task = replace(
            task,
            n_back=n_back,
            seed=base_seed + global_step,
        )
        batch = generate_nback_batch(batch_task)
        inputs, targets, loss_mask = batch_to_tensors(
            batch, device_info.device
        )
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(inputs)
        loss = masked_cross_entropy(logits, targets, loss_mask)
        if not torch.isfinite(loss):
            raise RuntimeError("N-back training loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), gradient_clip_norm
        )
        optimizer.step()
        accuracy = response_accuracy(
            logits.detach(), targets, loss_mask
        )
        history.append(
            {
                "step": global_step,
                "stage": stage,
                "n_back": n_back,
                "loss": float(loss.item()),
                "timestep_accuracy": accuracy,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if (
            global_step == 1
            or global_step % log_every == 0
        ):
            print(
                f"step={global_step} stage={stage} n_back={n_back} "
                f"loss={loss.item():.4f} accuracy={accuracy:.3f}"
            )

    for stage_step in range(1, stage1_max + 1):
        train_step(0, 1)
        if stage_step % stage1_every != 0:
            continue
        metrics, gate = _run_validation(
            model, config, step=global_step, stage=1
        )
        validation_history.append(
            _validation_record(
                step=global_step,
                stage=1,
                metrics=metrics,
                gate=gate,
            )
        )
        consecutive = consecutive + 1 if gate["passed"] else 0
        print(
            f"validation step={global_step} stage=1 "
            f"passed={gate['passed']} consecutive={consecutive}"
        )
        if (
            stage_step >= stage1_min
            and consecutive >= required_consecutive
        ):
            stage1_passed = True
            break

    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stage2_passed = False
    consecutive = 0
    mode_block: list[int] = []
    if stage1_passed:
        for stage_step in range(1, stage2_max + 1):
            if not mode_block:
                mode_block = draw_balanced_nback_block(np_rng)
            train_step(mode_block.pop(), 2)
            if stage_step % stage2_every != 0:
                continue
            metrics, gate = _run_validation(
                model, config, step=global_step, stage=2
            )
            validation_history.append(
                _validation_record(
                    step=global_step,
                    stage=2,
                    metrics=metrics,
                    gate=gate,
                )
            )
            if gate["passed"]:
                mean_loss = float(
                    np.mean(
                        [
                            metrics["0-back"]["mean_cross_entropy"],
                            metrics["2-back"]["mean_cross_entropy"],
                        ]
                    )
                )
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    best_state = deepcopy(model.state_dict())
                consecutive += 1
            else:
                consecutive = 0
            print(
                f"validation step={global_step} stage=2 "
                f"passed={gate['passed']} consecutive={consecutive}"
            )
            if consecutive >= required_consecutive:
                stage2_passed = True
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    dirs = ensure_run_dirs(config["paths"]["output_dir"])
    run_name = config["paths"]["run_name"]
    checkpoint_path = (
        dirs["checkpoints"] / f"{run_name}.pt"
    )
    history_path = (
        dirs["metrics"] / f"{run_name}_train_history.csv"
    )
    metrics_path = (
        dirs["metrics"] / f"{run_name}_train_metrics.json"
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "history": history,
            "validation_history": validation_history,
            "stage1_passed": stage1_passed,
            "stage2_passed": stage2_passed,
        },
        checkpoint_path,
    )
    write_history_csv(history_path, history)
    write_json(
        metrics_path,
        {
            "device": device_info.description,
            "seed": base_seed,
            "steps": global_step,
            "stage1_passed": stage1_passed,
            "stage2_passed": stage2_passed,
            "best_validation_cross_entropy": (
                best_loss if np.isfinite(best_loss) else None
            ),
            "validation_history": validation_history,
            "checkpoint": str(checkpoint_path),
        },
    )
    return NBackTrainResult(
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        history_path=history_path,
        history=history,
        validation_history=validation_history,
        passed=stage2_passed,
    )


def main() -> None:
    """Train a shared N-back checkpoint from the command line."""
    parser = argparse.ArgumentParser(
        description="Train a competence-gated shared 0-back/2-back CTRNN."
    )
    parser.add_argument(
        "--config",
        default="configs/nback_working_memory.yaml",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device
    result = train_nback_model(config)
    print(f"checkpoint={result.checkpoint_path}")
    print(f"metrics={result.metrics_path}")
    print(f"passed={result.passed}")


if __name__ == "__main__":
    main()
