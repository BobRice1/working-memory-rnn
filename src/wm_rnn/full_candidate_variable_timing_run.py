"""Rerun the candidate grid on the 10-seed variable-timing circular family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wm_rnn.circular_family_a_pilot import run_pilot as run_circular
from wm_rnn.config import load_config
from wm_rnn.exploratory_pilot_summary import run_summary
from wm_rnn.full_candidate_perturbation_run import (
    CANDIDATE_OPERATORS,
    trained_distractor_checkpoints,
)


CONFIG = Path(
    "configs/full_candidate_perturbation_variable_timing_1024.yaml"
)
OUTPUT = Path("outputs/full_candidate_perturbation_variable_timing_1024")
BASE_CONFIG = Path(
    "configs/fixation_circular_variable_distractor_working_memory.yaml"
)
BASE_CONFIG_SHA256 = (
    "C568B49FBF17504D6047454E150C00C54F3E8C9503CE9E4EDD50C2CDA5FA554D"
)
POOL_MANIFEST = Path(
    "outputs/fixation_circular_variable_distractor_working_memory/"
    "metrics/"
    "fixation_circular_variable_distractor_working_memory_pool_summary.json"
)
NBACK_SIGNATURES = Path(
    "outputs/full_candidate_perturbation_1024/nback/pilot_signatures.csv"
)


def run(
    repo_root: str | Path = ".",
    device: str = "cuda",
) -> dict[str, object]:
    """Execute the frozen family-replication grid and joint summary."""
    root = Path(repo_root).resolve()
    config = load_config(root / CONFIG)
    base_config = load_config(root / BASE_CONFIG)
    if config["pilot"]["distractor_timing"] != (
        "per_trial_stratified_uniform_all_valid_starts"
    ):
        raise ValueError("unexpected distractor timing specification")
    checkpoints = trained_distractor_checkpoints(
        root, root / POOL_MANIFEST
    )
    circular_csv, circular_metadata = run_circular(
        root,
        pilot_config_path=root / CONFIG,
        output_dir=root / OUTPUT / "circular_variable_timing",
        device=device,
        checkpoints=checkpoints,
        trials_per_cell=1024,
        required_operators=CANDIDATE_OPERATORS,
        base_config_path=root / BASE_CONFIG,
        expected_base_config_sha256=BASE_CONFIG_SHA256,
        output_stem="circular_variable_timing",
        interpretive_limit=(
            "Descriptive family-replication evaluation on 10 circular "
            "checkpoints trained with variable distractor timing. The "
            "distractor onset is assigned per trial from a frozen, balanced "
            "random permutation of every valid delay start. This is not "
            "matched-cost inference or a biological psilocybin model."
        ),
        randomize_distractor_onsets=True,
    )
    leader = run_summary(
        root / OUTPUT,
        circular_grid=circular_csv,
        nback_signature_table=root / NBACK_SIGNATURES,
    )
    return {
        "status": "descriptive_family_replication",
        "circular_grid": str(circular_csv),
        "circular_metadata": str(circular_metadata),
        "nback_signature_source": str(root / NBACK_SIGNATURES),
        "joint_leading_profile": leader,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required because this command evaluates the full grid.",
    )
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("evaluation withheld; pass --execute to run")
    print(json.dumps(run(args.repo_root, args.device), indent=2))


if __name__ == "__main__":
    main()
