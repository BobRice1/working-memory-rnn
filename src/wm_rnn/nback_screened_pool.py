"""Perturbation-blind competence screening for an N-back seed pool."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from wm_rnn.config import load_config
from wm_rnn.io import ensure_run_dirs, write_json
from wm_rnn.nback_evaluation import NBackEvalResult, evaluate_nback_checkpoint
from wm_rnn.nback_seed_sweep import config_for_nback_seed
from wm_rnn.train_nback import NBackTrainResult, train_nback_model

TrainFunction = Callable[[dict[str, Any]], NBackTrainResult]
EvaluateFunction = Callable[
    [dict[str, Any], str | Path], NBackEvalResult
]


@dataclass(frozen=True)
class NBackScreenedPoolResult:
    """Saved and in-memory results for a competence-screened seed pool."""

    summary_path: Path
    csv_path: Path
    results: list[dict[str, Any]]
    selected_seeds: list[int]
    passed: bool


def _failed_checks(
    competence_result: NBackEvalResult | None,
) -> list[str]:
    if competence_result is None:
        return []
    checks = competence_result.metrics.get("acceptance", {}).get(
        "checks", {}
    )
    return [
        str(name)
        for name, passed in checks.items()
        if not bool(passed)
    ]


def _result_row(
    seed: int,
    train_result: NBackTrainResult,
    competence_result: NBackEvalResult | None,
) -> dict[str, Any]:
    conditions = (
        competence_result.metrics.get("conditions", {})
        if competence_result is not None
        else {}
    )
    zero = conditions.get("0-back", {})
    two = conditions.get("2-back", {})
    competence_passed = bool(
        competence_result is not None and competence_result.passed
    )
    if not train_result.passed:
        status = "training_failed"
    elif not competence_passed:
        status = "competence_failed"
    else:
        status = "selected"
    return {
        "seed": int(seed),
        "status": status,
        "retained": competence_passed,
        "training_passed": bool(train_result.passed),
        "competence_passed": competence_passed,
        "failure_stage": (
            ""
            if competence_passed
            else "training"
            if not train_result.passed
            else "competence"
        ),
        "failed_checks": _failed_checks(competence_result),
        "steps": len(train_result.history),
        "checkpoint": str(train_result.checkpoint_path),
        "training_metrics": str(train_result.metrics_path),
        "competence_metrics": (
            str(competence_result.metrics_path)
            if competence_result is not None
            else ""
        ),
        "zero_back_accuracy": zero.get("accuracy", ""),
        "zero_back_discriminability": zero.get(
            "discriminability", ""
        ),
        "two_back_accuracy": two.get("accuracy", ""),
        "two_back_discriminability": two.get(
            "discriminability", ""
        ),
        "two_back_lure_accuracy": two.get(
            "one_back_lure_accuracy", ""
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "seed",
        "status",
        "retained",
        "training_passed",
        "competence_passed",
        "failure_stage",
        "failed_checks",
        "steps",
        "checkpoint",
        "training_metrics",
        "competence_metrics",
        "zero_back_accuracy",
        "zero_back_discriminability",
        "two_back_accuracy",
        "two_back_discriminability",
        "two_back_lure_accuracy",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable["failed_checks"] = ";".join(
                serializable["failed_checks"]
            )
            writer.writerow(serializable)
    return path


def _screening_spec(
    config: dict[str, Any],
    seeds: list[int] | None,
    target_count: int | None,
) -> tuple[list[int], int]:
    screening = config.get("screening", {})
    resolved_seeds = [
        int(seed)
        for seed in (
            seeds
            if seeds is not None
            else screening.get("candidate_seeds", [])
        )
    ]
    resolved_target = int(
        target_count
        if target_count is not None
        else screening.get("target_count", 0)
    )
    if not resolved_seeds:
        raise ValueError("at least one candidate seed is required")
    if len(set(resolved_seeds)) != len(resolved_seeds):
        raise ValueError("candidate seeds must be unique")
    if resolved_seeds != sorted(resolved_seeds):
        raise ValueError("candidate seeds must be in ascending order")
    if resolved_target < 1 or resolved_target > len(resolved_seeds):
        raise ValueError(
            "target_count must be between one and the pool size"
        )
    return resolved_seeds, resolved_target


def _pool_payload(
    candidate_seeds: list[int],
    target: int,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = [
        int(row["seed"]) for row in results if row["retained"]
    ]
    failed = [
        int(row["seed"]) for row in results if not row["retained"]
    ]
    unattempted = candidate_seeds[len(results) :]
    if len(selected) == target:
        stop_reason = "target_reached"
    elif len(selected) + len(unattempted) < target:
        stop_reason = "target_impossible"
    else:
        stop_reason = "in_progress"
    return {
        "candidate_seeds": candidate_seeds,
        "target_count": target,
        "attempted_seeds": [int(row["seed"]) for row in results],
        "retained_seeds": selected,
        "retained_checkpoints": [
            str(row["checkpoint"])
            for row in results
            if row["retained"]
        ],
        "failed_seeds": failed,
        "unattempted_seeds": unattempted,
        "n_passed": len(selected),
        "pool_pass_rate": (
            len(selected) / len(results) if results else 0.0
        ),
        "passed": len(selected) == target,
        "stop_reason": stop_reason,
        "target_impossible": stop_reason == "target_impossible",
        "results": results,
    }


def _persist_pool_state(
    *,
    dirs: dict[str, Path],
    base_name: str,
    candidate_seeds: list[int],
    target: int,
    results: list[dict[str, Any]],
) -> tuple[Path, Path, dict[str, Any]]:
    csv_path = _write_csv(
        dirs["metrics"] / f"{base_name}_screened_pool.csv",
        results,
    )
    payload = _pool_payload(candidate_seeds, target, results)
    summary_path = write_json(
        dirs["metrics"] / f"{base_name}_screened_pool_summary.json",
        payload,
    )
    return summary_path, csv_path, payload


def _load_existing_state(
    summary_path: Path,
    candidate_seeds: list[int],
    target: int,
) -> list[dict[str, Any]]:
    if not summary_path.exists():
        return []
    with summary_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("candidate_seeds") != candidate_seeds:
        raise ValueError(
            "existing pool summary has a different candidate seed order"
        )
    if int(payload.get("target_count", -1)) != target:
        raise ValueError(
            "existing pool summary has a different target_count"
        )
    results = list(payload.get("results", []))
    attempted = [int(row["seed"]) for row in results]
    if attempted != candidate_seeds[: len(attempted)]:
        raise ValueError(
            "existing pool results are not a prefix of candidate seeds"
        )
    return results


def run_nback_screened_pool(
    config: dict[str, Any],
    seeds: list[int] | None = None,
    *,
    target_count: int | None = None,
    train_fn: TrainFunction | None = None,
    evaluate_fn: EvaluateFunction | None = None,
) -> NBackScreenedPoolResult:
    """Retain the first competent checkpoints from a frozen seed pool."""
    candidate_seeds, target = _screening_spec(
        config, seeds, target_count
    )
    train = train_fn or train_nback_model
    evaluate = evaluate_fn or evaluate_nback_checkpoint
    base_output = Path(config["paths"]["output_dir"])
    base_name = str(config["paths"]["run_name"])
    dirs = ensure_run_dirs(base_output)
    summary_target = (
        dirs["metrics"] / f"{base_name}_screened_pool_summary.json"
    )
    results = _load_existing_state(
        summary_target, candidate_seeds, target
    )
    selected_seeds = [
        int(row["seed"]) for row in results if row["retained"]
    ]
    existing_payload = _pool_payload(
        candidate_seeds, target, results
    )
    if existing_payload["stop_reason"] != "in_progress":
        summary_path, csv_path, _ = _persist_pool_state(
            dirs=dirs,
            base_name=base_name,
            candidate_seeds=candidate_seeds,
            target=target,
            results=results,
        )
        return NBackScreenedPoolResult(
            summary_path=summary_path,
            csv_path=csv_path,
            results=results,
            selected_seeds=selected_seeds,
            passed=bool(existing_payload["passed"]),
        )

    for index in range(len(results), len(candidate_seeds)):
        seed = candidate_seeds[index]
        seed_config = config_for_nback_seed(config, seed)
        train_result = train(seed_config)
        competence_result = None
        if train_result.passed:
            competence_result = evaluate(
                seed_config, train_result.checkpoint_path
            )
        row = _result_row(seed, train_result, competence_result)
        results.append(row)
        if row["retained"]:
            selected_seeds.append(seed)

        summary_path, csv_path, payload = _persist_pool_state(
            dirs=dirs,
            base_name=base_name,
            candidate_seeds=candidate_seeds,
            target=target,
            results=results,
        )
        if len(selected_seeds) == target:
            break
        remaining = len(candidate_seeds) - index - 1
        if len(selected_seeds) + remaining < target:
            break

    passed = bool(payload["passed"])
    return NBackScreenedPoolResult(
        summary_path=summary_path,
        csv_path=csv_path,
        results=results,
        selected_seeds=selected_seeds,
        passed=passed,
    )


def main() -> None:
    """Run the frozen competence-screened N-back pool."""
    parser = argparse.ArgumentParser(
        description="Select competent N-back checkpoints from a seed pool."
    )
    parser.add_argument(
        "--config",
        default="configs/nback_working_memory_screened_final.yaml",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device
    result = run_nback_screened_pool(config)
    print(f"summary={result.summary_path}")
    print(f"csv={result.csv_path}")
    print(f"selected_seeds={result.selected_seeds}")
    print(f"passed={result.passed}")


if __name__ == "__main__":
    main()
