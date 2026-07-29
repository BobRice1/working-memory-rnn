"""Dense state-persistence neighbourhood on circular and N-back families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wm_rnn.circular_family_a_pilot import run_pilot as run_circular
from wm_rnn.config import load_config
from wm_rnn.exploratory_pilot_summary import (
    circular_signatures,
    nback_signatures,
)
from wm_rnn.full_candidate_perturbation_run import (
    trained_distractor_checkpoints,
)
from wm_rnn.nback_exploratory_pilot import PilotDesign, run_pilot as run_nback
from wm_rnn.scientific_writeup_figures import persistence_dose_response


CONFIG = Path("configs/state_persistence_dense_variable_timing_1024.yaml")
OUTPUT = Path("outputs/state_persistence_dense_variable_timing_1024")
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
FIGURE_DIR = Path(
    "docs/reports/figures/full_candidate_perturbation/comparison"
)
PERSISTENCE_STRENGTHS = (
    0.80,
    0.85,
    0.88,
    0.89,
    0.90,
    0.91,
    0.92,
    0.93,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
    1.00,
)
NBACK_SEEDS = tuple(range(20260912, 20260922))


def dense_nback_design() -> PilotDesign:
    """Return the frozen 10-seed, 1,024-sequence persistence-only design."""
    return PilotDesign(
        checkpoint_seeds=NBACK_SEEDS,
        profile_ids=(10,),
        batch_size=128,
        n_batches=8,
        expected_sequences_per_cell=1024,
    )


def _validate_config(config: dict[str, object]) -> None:
    operators = config.get("operators")
    if not isinstance(operators, dict):
        raise ValueError("dense persistence config lacks operators")
    if set(operators) != {"state_persistence"}:
        raise ValueError("dense run admits only state_persistence")
    grid = tuple(float(value) for value in operators["state_persistence"])
    if grid != PERSISTENCE_STRENGTHS:
        raise ValueError("state_persistence grid does not match frozen design")
    pilot = config["pilot"]
    if pilot.get("distractor_timing") != (
        "per_trial_stratified_uniform_all_valid_starts"
    ):
        raise ValueError("unexpected distractor timing specification")


def run(
    repo_root: str | Path = ".",
    device: str = "cuda",
) -> dict[str, object]:
    """Execute the dense persistence neighbourhood and write the dose figure."""
    root = Path(repo_root).resolve()
    config = load_config(root / CONFIG)
    _validate_config(config)
    design = dense_nback_design()
    design.validate()

    checkpoints = trained_distractor_checkpoints(root, root / POOL_MANIFEST)
    circular_csv, circular_metadata = run_circular(
        root,
        pilot_config_path=root / CONFIG,
        output_dir=root / OUTPUT / "circular_variable_timing",
        device=device,
        checkpoints=checkpoints,
        trials_per_cell=1024,
        required_operators={"state_persistence"},
        base_config_path=root / BASE_CONFIG,
        expected_base_config_sha256=BASE_CONFIG_SHA256,
        output_stem="circular_variable_timing",
        interpretive_limit=(
            "Descriptive dense persistence neighbourhood on the 10-seed "
            "variable-timing circular family. Not matched-cost inference and "
            "not a biological psilocybin model."
        ),
        randomize_distractor_onsets=True,
    )
    nback_result = run_nback(
        config_path=root / CONFIG,
        repo_root=root,
        output_dir=root / OUTPUT / "nback",
        device_override=device,
        design=design,
    )
    circular = circular_signatures(circular_csv)
    nback = nback_signatures(Path(nback_result["signatures_path"]))
    figure_path = root / FIGURE_DIR / (
        "persistence_response_dense_10seed_variable_timing.png"
    )
    persistence_dose_response(figure_path, circular, nback)
    summary = {
        "status": "descriptive_dense_persistence_neighbourhood",
        "persistence_strengths": list(PERSISTENCE_STRENGTHS),
        "circular_grid": str(circular_csv),
        "circular_metadata": str(circular_metadata),
        "nback_signatures": nback_result["signatures_path"],
        "nback_metrics": nback_result["metrics_path"],
        "dose_response_figure": str(figure_path),
        "n_circular_signature_rows": len(circular),
        "n_nback_signature_rows": len(nback),
    }
    summary_path = root / OUTPUT / "summary" / "dense_persistence_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


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
        help="Required because this command evaluates the dense grid.",
    )
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("evaluation withheld; pass --execute to run")
    print(json.dumps(run(args.repo_root, args.device), indent=2))


if __name__ == "__main__":
    main()
