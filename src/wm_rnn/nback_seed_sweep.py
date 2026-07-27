"""Stop-on-failure training and competence evaluation across N-back seeds."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wm_rnn.config import load_config
from wm_rnn.io import ensure_run_dirs, write_json
from wm_rnn.nback_evaluation import evaluate_nback_checkpoint
from wm_rnn.train_nback import train_nback_model


@dataclass(frozen=True)
class NBackSeedSweepResult:
    """Saved and in-memory seed-family results."""

    summary_path: Path
    csv_path: Path
    results: list[dict[str, Any]]
    passed: bool


def config_for_nback_seed(
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Return an isolated per-seed configuration."""
    resolved = deepcopy(config)
    base_output = Path(config["paths"]["output_dir"])
    base_name = str(config["paths"]["run_name"])
    resolved["task"]["seed"] = int(seed)
    resolved["paths"]["output_dir"] = str(
        base_output / "seed_sweep" / f"seed_{seed}"
    )
    resolved["paths"]["run_name"] = f"{base_name}_seed_{seed}"
    return resolved


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "training_passed",
        "competence_passed",
        "steps",
        "checkpoint",
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
        writer.writerows(rows)
    return path


def run_nback_seed_sweep(
    config: dict[str, Any],
    seeds: list[int],
) -> NBackSeedSweepResult:
    """Train/evaluate seeds sequentially and stop on the first failed gate."""
    if not seeds:
        raise ValueError("at least one seed is required")
    base_output = Path(config["paths"]["output_dir"])
    base_name = str(config["paths"]["run_name"])
    dirs = ensure_run_dirs(base_output)
    results: list[dict[str, Any]] = []
    for seed in (int(value) for value in seeds):
        seed_config = config_for_nback_seed(config, seed)
        train_result = train_nback_model(seed_config)
        competence_result = None
        if train_result.passed:
            competence_result = evaluate_nback_checkpoint(
                seed_config, train_result.checkpoint_path
            )
        conditions = (
            competence_result.metrics["conditions"]
            if competence_result is not None
            else {}
        )
        zero = conditions.get("0-back", {})
        two = conditions.get("2-back", {})
        row = {
            "seed": seed,
            "training_passed": bool(train_result.passed),
            "competence_passed": bool(
                competence_result is not None
                and competence_result.passed
            ),
            "steps": len(train_result.history),
            "checkpoint": str(train_result.checkpoint_path),
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
        results.append(row)
        if not row["competence_passed"]:
            break

    passed = len(results) == len(seeds) and all(
        row["competence_passed"] for row in results
    )
    csv_path = _write_csv(
        dirs["metrics"] / f"{base_name}_seed_sweep.csv", results
    )
    summary_path = write_json(
        dirs["metrics"] / f"{base_name}_seed_sweep_summary.json",
        {
            "requested_seeds": [int(value) for value in seeds],
            "completed_seeds": [row["seed"] for row in results],
            "passed": passed,
            "stopped_early": len(results) != len(seeds),
            "results": results,
        },
    )
    return NBackSeedSweepResult(
        summary_path=summary_path,
        csv_path=csv_path,
        results=results,
        passed=passed,
    )


def main() -> None:
    """Run a stop-on-failure N-back seed family from the command line."""
    parser = argparse.ArgumentParser(
        description="Train/evaluate shared N-back models across seeds."
    )
    parser.add_argument(
        "--config",
        default="configs/nback_working_memory.yaml",
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["training"]["device"] = args.device
    result = run_nback_seed_sweep(config, args.seeds)
    print(f"summary={result.summary_path}")
    print(f"csv={result.csv_path}")
    print(f"passed={result.passed}")


if __name__ == "__main__":
    main()
