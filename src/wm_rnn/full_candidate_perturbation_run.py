"""Run the trained-distractor circular candidate evaluation and joint summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wm_rnn.circular_family_a_pilot import (
    FrozenCheckpoint,
    run_pilot as run_circular,
)
from wm_rnn.exploratory_pilot_summary import run_summary


CONFIG = Path(
    "configs/full_candidate_perturbation_trained_distractor_1024.yaml"
)
OUTPUT = Path("outputs/full_candidate_perturbation_trained_distractor_1024")
BASE_CONFIG = Path("configs/fixation_circular_distractor_working_memory.yaml")
BASE_CONFIG_SHA256 = (
    "95433330DD59F77BA0A6AE6E177C0FD2E44121F92E65036F9857FEEE1F4C4656"
)
POOL_MANIFEST = Path(
    "outputs/fixation_circular_distractor_working_memory/metrics/"
    "fixation_circular_distractor_working_memory_pool_summary.json"
)
NBACK_SIGNATURES = Path(
    "outputs/full_candidate_perturbation_1024/nback/pilot_signatures.csv"
)
CANDIDATE_OPERATORS = {
    "synaptic_drive_gain",
    "heterogeneous_drive_gain",
    "sensory_input_gain",
    "distractor_input_gain",
    "recurrent_gain",
    "state_persistence",
    "time_constant",
}


def trained_distractor_checkpoints(
    repo_root: str | Path,
    manifest_path: str | Path | None = None,
) -> tuple[FrozenCheckpoint, ...]:
    """Load the exact competent circular checkpoint set from its run manifest."""
    root = Path(repo_root).resolve()
    resolved_manifest = Path(manifest_path or POOL_MANIFEST)
    if not resolved_manifest.is_absolute():
        resolved_manifest = root / resolved_manifest
    with resolved_manifest.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    retained = tuple(int(seed) for seed in manifest["retained_checkpoint_seeds"])
    by_seed = {int(row["seed"]): row for row in manifest["results"]}
    checkpoints = []
    for seed in retained:
        row = by_seed[seed]
        if not bool(row["competence_passed"]):
            raise ValueError(f"retained checkpoint {seed} failed competence")
        checkpoints.append(
            FrozenCheckpoint(
                seed=seed,
                path=str(row["checkpoint"]),
                sha256=str(row["checkpoint_sha256"]),
            )
        )
    if len(checkpoints) != int(manifest["target_competent_checkpoints"]):
        raise ValueError("retained checkpoint count does not match pool target")
    return tuple(checkpoints)


def run(repo_root: str | Path = ".", device: str = "cuda") -> dict[str, object]:
    root = Path(repo_root).resolve()
    checkpoints = trained_distractor_checkpoints(root)
    circular_csv, circular_metadata = run_circular(
        root,
        pilot_config_path=root / CONFIG,
        output_dir=root / OUTPUT / "circular_trained_distractor",
        device=device,
        checkpoints=checkpoints,
        trials_per_cell=1024,
        required_operators=CANDIDATE_OPERATORS,
        base_config_path=root / BASE_CONFIG,
        expected_base_config_sha256=BASE_CONFIG_SHA256,
        output_stem="circular_trained_distractor",
        interpretive_limit=(
            "Descriptive candidate-only evaluation on circular checkpoints "
            "trained to filter distractors; not confirmatory inference and "
            "not a biological psilocybin model."
        ),
    )
    leader = run_summary(
        root / OUTPUT,
        circular_grid=circular_csv,
        nback_signature_table=root / NBACK_SIGNATURES,
    )
    return {
        "status": "descriptive_candidate_only",
        "circular_grid": str(circular_csv),
        "circular_metadata": str(circular_metadata),
        "nback_signature_source": str(root / NBACK_SIGNATURES),
        "joint_leading_profile": leader,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args.repo_root, args.device), indent=2))


if __name__ == "__main__":
    main()
