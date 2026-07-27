"""Run the full 1,024-sample candidate-only perturbation evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wm_rnn.circular_family_a_pilot import FULL_CHECKPOINTS, run_pilot as run_circular
from wm_rnn.nback_exploratory_pilot import PilotDesign, run_pilot as run_nback


CONFIG = Path("configs/full_candidate_perturbation_1024.yaml")
OUTPUT = Path("outputs/full_candidate_perturbation_1024")
NBACK_SEEDS = tuple(range(20260912, 20260922))
NBACK_PROFILES = (1, 4, 7, 9, 10, 12)
NBACK_DESIGN = PilotDesign(
    checkpoint_seeds=NBACK_SEEDS,
    profile_ids=NBACK_PROFILES,
    batch_size=128,
    n_batches=8,
    task_seed_base=161_000_000,
    noise_seed_base=162_000_000,
    expected_sequences_per_cell=1024,
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


def run(repo_root: str | Path = ".", device: str = "cuda") -> dict[str, object]:
    root = Path(repo_root).resolve()
    circular_csv, circular_metadata = run_circular(
        root,
        pilot_config_path=root / CONFIG,
        output_dir=root / OUTPUT / "circular_family_a",
        device=device,
        checkpoints=FULL_CHECKPOINTS,
        trials_per_cell=1024,
        required_operators=CANDIDATE_OPERATORS,
    )
    nback = run_nback(
        config_path=root / CONFIG,
        repo_root=root,
        output_dir=root / OUTPUT / "nback",
        device_override=device,
        design=NBACK_DESIGN,
    )
    return {
        "status": "descriptive_candidate_only",
        "circular_grid": str(circular_csv),
        "circular_metadata": str(circular_metadata),
        "nback": nback,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(args.repo_root, args.device), indent=2))


if __name__ == "__main__":
    main()
