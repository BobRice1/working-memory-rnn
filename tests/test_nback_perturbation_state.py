"""Tests for hash-validated phased N-back run persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wm_rnn.nback_perturbation_state import (
    PerturbationStateError,
    atomic_write_json,
    atomic_write_npz,
    begin_phase,
    canonical_design_hash,
    complete_phase,
    initialize_or_resume_run,
    record_completed_cell,
    validate_resume,
)


PHASES = ("neutral_checks", "calibration", "cost_check", "outcomes")
CELLS = {
    "neutral_checks": ("neutral_seed_0",),
    "calibration": ("calibration_seed_0",),
    "cost_check": ("cost_seed_0",),
    "outcomes": ("outcome_seed_0",),
}


def _design(target: float = 0.05) -> dict[str, object]:
    return {
        "target_additive_cost": target,
        "grids": {"p5": [0.0, 0.01, 0.02]},
        "banks": {"calibration": 132000000},
    }


def _paths(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    checkpoint = tmp_path / "dummy_seed_20260912.pt"
    checkpoint.write_bytes(b"not-a-real-checkpoint")
    return (
        run_dir / "manifest.json",
        run_dir / "state.json",
        {"20260912": checkpoint},
    )


def _initialize(tmp_path: Path):
    manifest, state, checkpoints = _paths(tmp_path)
    snapshot = initialize_or_resume_run(
        manifest,
        state,
        design=_design(),
        checkpoints=checkpoints,
        phase_order=PHASES,
        expected_cells=CELLS,
    )
    return manifest, state, checkpoints, snapshot


def test_canonical_design_hash_is_key_order_invariant_and_value_sensitive() -> None:
    first = {"b": [2, 3], "a": {"x": 1}}
    reordered = {"a": {"x": 1}, "b": [2, 3]}
    changed = {"a": {"x": 2}, "b": [2, 3]}

    assert canonical_design_hash(first) == canonical_design_hash(reordered)
    assert canonical_design_hash(first) != canonical_design_hash(changed)
    with pytest.raises(ValueError, match="finite"):
        canonical_design_hash({"bad": float("nan")})


def test_atomic_json_and_npz_writes_are_complete_and_replaceable(
    tmp_path: Path,
) -> None:
    json_path = atomic_write_json(tmp_path / "artifact.json", {"value": 1})
    atomic_write_json(json_path, {"value": 2})
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"value": 2}

    npz_path = atomic_write_npz(
        tmp_path / "artifact.npz",
        values=np.arange(5),
    )
    atomic_write_npz(npz_path, values=np.arange(3))
    with np.load(npz_path, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["values"], np.arange(3))
    with pytest.raises(ValueError, match="object arrays"):
        atomic_write_npz(
            tmp_path / "unsafe.npz",
            values=np.asarray([object()], dtype=object),
        )


def test_manifest_is_immutable_while_state_advances_and_resume_reuses_cell(
    tmp_path: Path,
) -> None:
    manifest, state, checkpoints, initial = _initialize(tmp_path)
    manifest_bytes = manifest.read_bytes()
    assert initial.revision == 0

    begun = begin_phase(manifest, state, "neutral_checks")
    artifact = atomic_write_json(
        manifest.parent / "neutral_seed_0.json",
        {"passed": True},
    )
    recorded = record_completed_cell(
        manifest,
        state,
        phase="neutral_checks",
        cell_id="neutral_seed_0",
        artifacts=[artifact],
        metadata={"checkpoint_seed": 20260912},
    )
    repeated = record_completed_cell(
        manifest,
        state,
        phase="neutral_checks",
        cell_id="neutral_seed_0",
        artifacts=[artifact],
        metadata={"checkpoint_seed": 20260912},
    )

    assert begun.revision == 1
    assert recorded.reused is False
    assert repeated.reused is True
    assert repeated.snapshot.revision == recorded.snapshot.revision
    assert manifest.read_bytes() == manifest_bytes
    with pytest.raises(PerturbationStateError, match="persisted record"):
        record_completed_cell(
            manifest,
            state,
            phase="neutral_checks",
            cell_id="neutral_seed_0",
            artifacts=[artifact],
            metadata={"checkpoint_seed": 999},
        )
    resumed = validate_resume(
        manifest,
        state,
        design=_design(),
        checkpoints=checkpoints,
        phase_order=PHASES,
        expected_cells=CELLS,
    )
    assert resumed.reusable_cells["neutral_checks"] == ("neutral_seed_0",)


def test_phase_prerequisites_and_registered_cells_are_enforced(
    tmp_path: Path,
) -> None:
    manifest, state, _, _ = _initialize(tmp_path)

    with pytest.raises(PerturbationStateError, match="prerequisites"):
        begin_phase(manifest, state, "calibration")
    begin_phase(manifest, state, "neutral_checks")
    with pytest.raises(PerturbationStateError, match="missing completed cells"):
        complete_phase(manifest, state, "neutral_checks")
    artifact = atomic_write_json(
        manifest.parent / "neutral.json", {"passed": True}
    )
    with pytest.raises(PerturbationStateError, match="unregistered cell"):
        record_completed_cell(
            manifest,
            state,
            phase="neutral_checks",
            cell_id="not_registered",
            artifacts=[artifact],
        )
    record_completed_cell(
        manifest,
        state,
        phase="neutral_checks",
        cell_id="neutral_seed_0",
        artifacts=[artifact],
    )
    completed = complete_phase(manifest, state, "neutral_checks")
    assert completed.phase_statuses["neutral_checks"] == "completed"
    assert begin_phase(
        manifest, state, "calibration"
    ).phase_statuses["calibration"] == "active"


def test_changed_design_and_checkpoint_bytes_refuse_resume(
    tmp_path: Path,
) -> None:
    manifest, state, checkpoints, _ = _initialize(tmp_path / "design")
    with pytest.raises(PerturbationStateError, match="design changed"):
        validate_resume(
            manifest,
            state,
            design=_design(target=0.06),
            checkpoints=checkpoints,
            phase_order=PHASES,
            expected_cells=CELLS,
        )

    other_manifest, other_state, other_checkpoints, _ = _initialize(
        tmp_path / "checkpoint"
    )
    next(iter(other_checkpoints.values())).write_bytes(b"changed")
    with pytest.raises(PerturbationStateError, match="checkpoint"):
        validate_resume(
            other_manifest,
            other_state,
            design=_design(),
            checkpoints=other_checkpoints,
            phase_order=PHASES,
            expected_cells=CELLS,
        )


def test_corrupt_completed_artifact_and_manifest_refuse_resume(
    tmp_path: Path,
) -> None:
    manifest, state, checkpoints, _ = _initialize(tmp_path / "artifact")
    begin_phase(manifest, state, "neutral_checks")
    artifact = atomic_write_npz(
        manifest.parent / "neutral.npz",
        values=np.arange(4),
    )
    record_completed_cell(
        manifest,
        state,
        phase="neutral_checks",
        cell_id="neutral_seed_0",
        artifacts=[artifact],
    )
    artifact.write_bytes(b"corrupt")
    with pytest.raises(PerturbationStateError, match="artifact is corrupt"):
        validate_resume(
            manifest,
            state,
            design=_design(),
            checkpoints=checkpoints,
            phase_order=PHASES,
            expected_cells=CELLS,
        )

    other_manifest, other_state, other_checkpoints, _ = _initialize(
        tmp_path / "manifest"
    )
    payload = json.loads(other_manifest.read_text(encoding="utf-8"))
    payload["extra"] = "mutation"
    other_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PerturbationStateError, match="phase plan changed"):
        validate_resume(
            other_manifest,
            other_state,
            design=_design(),
            checkpoints=other_checkpoints,
            phase_order=PHASES,
            expected_cells=CELLS,
        )


def test_partial_identity_and_outside_artifact_are_refused(
    tmp_path: Path,
) -> None:
    manifest, state, checkpoints = _paths(tmp_path)
    atomic_write_json(manifest, {"partial": True})
    with pytest.raises(PerturbationStateError, match="partial run identity"):
        initialize_or_resume_run(
            manifest,
            state,
            design=_design(),
            checkpoints=checkpoints,
            phase_order=PHASES,
            expected_cells=CELLS,
        )

    other_manifest, other_state, _, _ = _initialize(tmp_path / "outside")
    begin_phase(other_manifest, other_state, "neutral_checks")
    outside = atomic_write_json(tmp_path / "outside.json", {"value": 1})
    with pytest.raises(PerturbationStateError, match="run directory"):
        record_completed_cell(
            other_manifest,
            other_state,
            phase="neutral_checks",
            cell_id="neutral_seed_0",
            artifacts=[outside],
        )
