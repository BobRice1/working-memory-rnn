"""Hash-validated persistence for phased N-back perturbation runs.

The immutable manifest records the complete design, checkpoint identities,
phase order, and expected cell IDs.  A separate mutable state file records
phase progress and completed-cell artifact hashes.  This separation permits
safe resume without rewriting the scientific identity of a run.

This module does not load models, construct perturbations, or evaluate any
experimental outcome.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 2
DEFAULT_PHASE_ORDER = (
    "neutral-calibration",
    "calibration",
    "cost-check",
    "neutral-confirmatory",
    "confirmatory",
    "dose",
    "finalize",
)
_PHASE_STATUSES = {"pending", "active", "completed"}


class PerturbationStateError(RuntimeError):
    """Raised when a run cannot be created, resumed, or safely advanced."""


@dataclass(frozen=True)
class ResumeSnapshot:
    """Validated view of one persisted run."""

    manifest_path: Path
    state_path: Path
    design_hash: str
    manifest_sha256: str
    revision: int
    phase_statuses: dict[str, str]
    reusable_cells: dict[str, tuple[str, ...]]
    phase_timings: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class CellRecordResult:
    """Result of recording, or reusing, a completed cell."""

    snapshot: ResumeSnapshot
    reused: bool


@dataclass(frozen=True)
class PhaseAttemptStart:
    """One newly opened active-execution timing attempt."""

    attempt_id: int
    snapshot: ResumeSnapshot


def _canonical_json_bytes(payload: Any) -> bytes:
    try:
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "payload must be finite and canonically JSON serializable"
        ) from error
    return text.encode("utf-8")


def _normalized_json(payload: Any) -> Any:
    """Return a detached JSON-compatible value with strict finite numbers."""
    return json.loads(_canonical_json_bytes(payload).decode("utf-8"))


def canonical_design_hash(design: Mapping[str, Any]) -> str:
    """Return the SHA-256 of a canonical JSON representation of ``design``."""
    if not isinstance(design, Mapping):
        raise TypeError("design must be a mapping")
    return hashlib.sha256(_canonical_json_bytes(dict(design))).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one regular file."""
    target = Path(path)
    if not target.is_file():
        raise PerturbationStateError(f"required file is missing: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically replace a JSON file with a fully flushed finite payload."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _normalized_json(dict(payload)),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def atomic_write_npz(
    path: str | Path,
    **arrays: np.ndarray,
) -> Path:
    """Atomically replace an NPZ file without permitting object arrays."""
    if not arrays:
        raise ValueError("at least one array is required")
    normalized: dict[str, np.ndarray] = {}
    for name, values in arrays.items():
        if not isinstance(name, str) or not name:
            raise ValueError("array names must be non-empty strings")
        array = np.asarray(values)
        if array.dtype.hasobject:
            raise ValueError("object arrays are not permitted in run artifacts")
        normalized[name] = array

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.",
        suffix=".npz",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **normalized)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _load_json(path: Path, kind: str) -> dict[str, Any]:
    if not path.is_file():
        raise PerturbationStateError(f"{kind} is missing: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise PerturbationStateError(
            f"{kind} is unreadable or corrupt: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise PerturbationStateError(f"{kind} must contain a JSON object")
    return payload


def _validated_phase_spec(
    phase_order: Sequence[str],
    expected_cells: Mapping[str, Sequence[str]],
) -> tuple[list[str], dict[str, list[str]]]:
    phases = [str(phase) for phase in phase_order]
    if not phases or any(not phase for phase in phases):
        raise ValueError("phase_order must contain non-empty phase names")
    if len(set(phases)) != len(phases):
        raise ValueError("phase_order must not contain duplicates")
    if set(expected_cells) != set(phases):
        raise ValueError(
            "expected_cells must contain exactly the registered phases"
        )
    cells: dict[str, list[str]] = {}
    all_ids: set[str] = set()
    for phase in phases:
        phase_cells = [str(cell) for cell in expected_cells[phase]]
        if any(not cell for cell in phase_cells):
            raise ValueError("cell IDs must be non-empty")
        if len(set(phase_cells)) != len(phase_cells):
            raise ValueError(f"duplicate cell ID in phase {phase}")
        overlap = all_ids.intersection(phase_cells)
        if overlap:
            raise ValueError(
                f"cell IDs must be globally unique: {sorted(overlap)}"
            )
        all_ids.update(phase_cells)
        cells[phase] = sorted(phase_cells)
    return phases, cells


def _checkpoint_records(
    checkpoints: Mapping[str, str | Path],
) -> list[dict[str, Any]]:
    if not checkpoints:
        raise ValueError("at least one checkpoint identity is required")
    records: list[dict[str, Any]] = []
    for checkpoint_id in sorted(checkpoints):
        identifier = str(checkpoint_id)
        if not identifier:
            raise ValueError("checkpoint IDs must be non-empty")
        path = Path(checkpoints[checkpoint_id]).resolve()
        if not path.is_file():
            raise PerturbationStateError(
                f"checkpoint is missing: {identifier}: {path}"
            )
        records.append(
            {
                "checkpoint_id": identifier,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _manifest_payload(
    design: Mapping[str, Any],
    checkpoints: Mapping[str, str | Path],
    phase_order: Sequence[str],
    expected_cells: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    phases, cells = _validated_phase_spec(phase_order, expected_cells)
    normalized_design = _normalized_json(dict(design))
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "nback_perturbation_run_manifest",
        "design_hash": canonical_design_hash(normalized_design),
        "design": normalized_design,
        "checkpoints": _checkpoint_records(checkpoints),
        "phase_order": phases,
        "expected_cells": cells,
    }


def _initial_state(manifest: Mapping[str, Any], manifest_hash: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "nback_perturbation_run_state",
        "manifest_sha256": manifest_hash,
        "design_hash": manifest["design_hash"],
        "revision": 0,
        "phases": {
            phase: {
                "status": "pending",
                "cells": {},
                "timing": {
                    "accumulated_seconds": 0.0,
                    "next_attempt_id": 0,
                    "active_attempt_id": None,
                    "attempts": [],
                },
            }
            for phase in manifest["phase_order"]
        },
    }


def _utc_now() -> str:
    """Return an audit-only UTC timestamp; durations use a monotonic clock."""
    return datetime.now(timezone.utc).isoformat()


def _finite_nonnegative_seconds(value: Any, name: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as error:
        raise PerturbationStateError(
            f"{name} must be a finite non-negative duration"
        ) from error
    if not np.isfinite(resolved) or resolved < 0.0:
        raise PerturbationStateError(
            f"{name} must be a finite non-negative duration"
        )
    return resolved


def _validate_phase_timing(
    timing: Any,
    *,
    phase: str,
    phase_status: str,
) -> dict[str, Any]:
    if not isinstance(timing, dict):
        raise PerturbationStateError(f"phase {phase} has invalid timing state")
    required = {
        "accumulated_seconds",
        "next_attempt_id",
        "active_attempt_id",
        "attempts",
    }
    if set(timing) != required:
        raise PerturbationStateError(f"phase {phase} has invalid timing state")
    accumulated = _finite_nonnegative_seconds(
        timing["accumulated_seconds"],
        f"phase {phase} accumulated timing",
    )
    next_attempt_id = timing["next_attempt_id"]
    active_attempt_id = timing["active_attempt_id"]
    attempts = timing["attempts"]
    if (
        not isinstance(next_attempt_id, int)
        or next_attempt_id < 0
        or (
            active_attempt_id is not None
            and (
                not isinstance(active_attempt_id, int)
                or active_attempt_id < 0
            )
        )
        or not isinstance(attempts, list)
    ):
        raise PerturbationStateError(f"phase {phase} has invalid timing state")

    ids: list[int] = []
    active_ids: list[int] = []
    committed_total = 0.0
    required_attempt = {
        "attempt_id",
        "status",
        "started_utc",
        "detected_interrupted_utc",
        "ended_utc",
        "committed_seconds",
        "last_committed_cell_id",
        "device",
        "cuda_synchronized",
    }
    for attempt in attempts:
        if not isinstance(attempt, dict) or set(attempt) != required_attempt:
            raise PerturbationStateError(
                f"phase {phase} has invalid timing attempt"
            )
        attempt_id = attempt["attempt_id"]
        status = attempt["status"]
        if (
            not isinstance(attempt_id, int)
            or attempt_id < 0
            or status not in {"active", "interrupted", "completed"}
            or not isinstance(attempt["started_utc"], str)
            or not attempt["started_utc"]
            or not isinstance(attempt["device"], str)
            or not attempt["device"]
            or not isinstance(attempt["cuda_synchronized"], bool)
        ):
            raise PerturbationStateError(
                f"phase {phase} has invalid timing attempt"
            )
        for key in ("detected_interrupted_utc", "ended_utc"):
            if attempt[key] is not None and (
                not isinstance(attempt[key], str) or not attempt[key]
            ):
                raise PerturbationStateError(
                    f"phase {phase} has invalid timing attempt"
                )
        last_cell = attempt["last_committed_cell_id"]
        if last_cell is not None and (
            not isinstance(last_cell, str) or not last_cell
        ):
            raise PerturbationStateError(
                f"phase {phase} has invalid timing attempt"
            )
        committed_total += _finite_nonnegative_seconds(
            attempt["committed_seconds"],
            f"phase {phase} attempt timing",
        )
        if status == "active":
            active_ids.append(attempt_id)
            if (
                attempt["detected_interrupted_utc"] is not None
                or attempt["ended_utc"] is not None
            ):
                raise PerturbationStateError(
                    f"phase {phase} active attempt is already closed"
                )
        elif status == "interrupted":
            if (
                attempt["detected_interrupted_utc"] is None
                or attempt["ended_utc"] is not None
            ):
                raise PerturbationStateError(
                    f"phase {phase} interrupted attempt is invalid"
                )
        elif (
            attempt["ended_utc"] is None
            or attempt["detected_interrupted_utc"] is not None
        ):
            raise PerturbationStateError(
                f"phase {phase} completed attempt is invalid"
            )
        ids.append(attempt_id)

    if ids != list(range(next_attempt_id)):
        raise PerturbationStateError(
            f"phase {phase} timing attempt IDs are not contiguous"
        )
    if active_ids != (
        [] if active_attempt_id is None else [active_attempt_id]
    ):
        raise PerturbationStateError(
            f"phase {phase} active timing attempt does not match"
        )
    if phase_status in {"pending", "completed"} and active_attempt_id is not None:
        raise PerturbationStateError(
            f"phase {phase} cannot retain an active timing attempt"
        )
    if phase_status == "pending" and attempts:
        raise PerturbationStateError(
            f"pending phase {phase} cannot contain timing attempts"
        )
    if not np.isclose(
        committed_total,
        accumulated,
        rtol=0.0,
        atol=1e-9,
    ):
        raise PerturbationStateError(
            f"phase {phase} accumulated timing does not match attempts"
        )
    return deepcopy(timing)


def _artifact_path_record(path: Path, run_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(run_root)
    except ValueError as error:
        raise PerturbationStateError(
            f"cell artifact must be inside the run directory: {resolved}"
        ) from error
    if not resolved.is_file():
        raise PerturbationStateError(f"cell artifact is missing: {resolved}")
    return {
        "path": relative.as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _validate_manifest_structure(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise PerturbationStateError("unsupported run-manifest schema")
    if manifest.get("kind") != "nback_perturbation_run_manifest":
        raise PerturbationStateError("unexpected run-manifest kind")
    design = manifest.get("design")
    if not isinstance(design, dict):
        raise PerturbationStateError("run manifest has no valid design")
    if manifest.get("design_hash") != canonical_design_hash(design):
        raise PerturbationStateError("run-manifest design hash is corrupt")
    phases = manifest.get("phase_order")
    cells = manifest.get("expected_cells")
    if not isinstance(phases, list) or not isinstance(cells, dict):
        raise PerturbationStateError("run manifest has an invalid phase spec")
    try:
        _validated_phase_spec(phases, cells)
    except ValueError as error:
        raise PerturbationStateError(
            "run manifest has an invalid phase spec"
        ) from error
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise PerturbationStateError("run manifest has no checkpoints")
    identifiers: set[str] = set()
    for record in checkpoints:
        if not isinstance(record, dict):
            raise PerturbationStateError("invalid checkpoint identity record")
        identifier = record.get("checkpoint_id")
        if not isinstance(identifier, str) or not identifier:
            raise PerturbationStateError("invalid checkpoint ID")
        if identifier in identifiers:
            raise PerturbationStateError("duplicate checkpoint ID")
        identifiers.add(identifier)
        path = Path(str(record.get("path", "")))
        if (
            not path.is_absolute()
            or not isinstance(record.get("sha256"), str)
            or not isinstance(record.get("size_bytes"), int)
        ):
            raise PerturbationStateError("invalid checkpoint identity record")


def _validate_checkpoint_files(manifest: Mapping[str, Any]) -> None:
    for record in manifest["checkpoints"]:
        path = Path(record["path"])
        if not path.is_file():
            raise PerturbationStateError(
                f"registered checkpoint is missing: {record['checkpoint_id']}"
            )
        if path.stat().st_size != record["size_bytes"]:
            raise PerturbationStateError(
                f"registered checkpoint changed: {record['checkpoint_id']}"
            )
        if sha256_file(path) != record["sha256"]:
            raise PerturbationStateError(
                f"registered checkpoint changed: {record['checkpoint_id']}"
            )


def _validate_state(
    manifest: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    manifest_hash: str,
    run_root: Path,
) -> ResumeSnapshot:
    if state.get("schema_version") != SCHEMA_VERSION:
        raise PerturbationStateError("unsupported run-state schema")
    if state.get("kind") != "nback_perturbation_run_state":
        raise PerturbationStateError("unexpected run-state kind")
    if state.get("manifest_sha256") != manifest_hash:
        raise PerturbationStateError(
            "immutable run manifest was changed after state creation"
        )
    if state.get("design_hash") != manifest["design_hash"]:
        raise PerturbationStateError("run-state design hash does not match")
    revision = state.get("revision")
    if not isinstance(revision, int) or revision < 0:
        raise PerturbationStateError("run-state revision is invalid")
    phases = state.get("phases")
    if (
        not isinstance(phases, dict)
        or set(phases) != set(manifest["phase_order"])
    ):
        raise PerturbationStateError("run-state phase order does not match")

    statuses: dict[str, str] = {}
    reusable: dict[str, tuple[str, ...]] = {}
    phase_timings: dict[str, dict[str, Any]] = {}
    completed_prefix = True
    active_seen = False
    for phase in manifest["phase_order"]:
        phase_state = phases.get(phase)
        if not isinstance(phase_state, dict):
            raise PerturbationStateError(f"invalid state for phase {phase}")
        status = phase_state.get("status")
        cells = phase_state.get("cells")
        if status not in _PHASE_STATUSES or not isinstance(cells, dict):
            raise PerturbationStateError(f"invalid state for phase {phase}")
        phase_timings[phase] = _validate_phase_timing(
            phase_state.get("timing"),
            phase=phase,
            phase_status=status,
        )
        if status == "completed":
            if not completed_prefix or active_seen:
                raise PerturbationStateError("completed phases are out of order")
        elif status == "active":
            if not completed_prefix or active_seen:
                raise PerturbationStateError("active phases are out of order")
            active_seen = True
            completed_prefix = False
        else:
            completed_prefix = False

        expected = set(manifest["expected_cells"][phase])
        if not set(cells).issubset(expected):
            raise PerturbationStateError(
                f"phase {phase} contains an unregistered cell"
            )
        if status == "pending" and cells:
            raise PerturbationStateError(
                f"pending phase {phase} cannot contain completed cells"
            )
        if status == "completed" and set(cells) != expected:
            raise PerturbationStateError(
                f"completed phase {phase} is missing cells"
            )
        for cell_id, cell in cells.items():
            if not isinstance(cell, dict) or cell.get("status") != "completed":
                raise PerturbationStateError(
                    f"cell {cell_id} in phase {phase} is invalid"
                )
            artifacts = cell.get("artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                raise PerturbationStateError(
                    f"completed cell {cell_id} has no artifacts"
                )
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    raise PerturbationStateError(
                        f"cell {cell_id} has an invalid artifact record"
                    )
                relative_path = Path(str(artifact.get("path", "")))
                if relative_path.is_absolute():
                    raise PerturbationStateError(
                        f"cell {cell_id} has an invalid artifact path"
                    )
                path = (run_root / relative_path).resolve()
                try:
                    path.relative_to(run_root)
                except ValueError as error:
                    raise PerturbationStateError(
                        f"cell {cell_id} has an invalid artifact path"
                    ) from error
                if not path.is_file():
                    raise PerturbationStateError(
                        f"completed-cell artifact is missing: {path}"
                    )
                if path.stat().st_size != artifact.get("size_bytes"):
                    raise PerturbationStateError(
                        f"completed-cell artifact is corrupt: {path}"
                    )
                if sha256_file(path) != artifact.get("sha256"):
                    raise PerturbationStateError(
                        f"completed-cell artifact is corrupt: {path}"
                    )
            try:
                _normalized_json(cell.get("metadata", {}))
            except ValueError as error:
                raise PerturbationStateError(
                    f"cell {cell_id} metadata is corrupt"
                ) from error
        statuses[phase] = status
        reusable[phase] = tuple(sorted(cells))

    return ResumeSnapshot(
        manifest_path=run_root / "unused",
        state_path=run_root / "unused",
        design_hash=str(manifest["design_hash"]),
        manifest_sha256=manifest_hash,
        revision=revision,
        phase_statuses=statuses,
        reusable_cells=reusable,
        phase_timings=phase_timings,
    )


def _validated_snapshot(
    manifest_path: Path,
    state_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], ResumeSnapshot]:
    manifest = _load_json(manifest_path, "run manifest")
    _validate_manifest_structure(manifest)
    _validate_checkpoint_files(manifest)
    state = _load_json(state_path, "run state")
    manifest_hash = sha256_file(manifest_path)
    snapshot = _validate_state(
        manifest,
        state,
        manifest_hash=manifest_hash,
        run_root=manifest_path.parent.resolve(),
    )
    snapshot = ResumeSnapshot(
        manifest_path=manifest_path.resolve(),
        state_path=state_path.resolve(),
        design_hash=snapshot.design_hash,
        manifest_sha256=snapshot.manifest_sha256,
        revision=snapshot.revision,
        phase_statuses=snapshot.phase_statuses,
        reusable_cells=snapshot.reusable_cells,
        phase_timings=snapshot.phase_timings,
    )
    return manifest, state, snapshot


def initialize_or_resume_run(
    manifest_path: str | Path,
    state_path: str | Path,
    *,
    design: Mapping[str, Any],
    checkpoints: Mapping[str, str | Path],
    expected_cells: Mapping[str, Sequence[str]],
    phase_order: Sequence[str] = DEFAULT_PHASE_ORDER,
) -> ResumeSnapshot:
    """Create a new run identity or validate and reuse an existing one."""
    manifest_target = Path(manifest_path).resolve()
    state_target = Path(state_path).resolve()
    if manifest_target.parent != state_target.parent:
        raise ValueError("manifest and state must share one run directory")
    expected_manifest = _manifest_payload(
        design,
        checkpoints,
        phase_order,
        expected_cells,
    )
    manifest_exists = manifest_target.exists()
    state_exists = state_target.exists()
    if manifest_exists != state_exists:
        raise PerturbationStateError(
            "partial run identity: manifest and state must both exist"
        )
    if not manifest_exists:
        atomic_write_json(manifest_target, expected_manifest)
        manifest_hash = sha256_file(manifest_target)
        atomic_write_json(
            state_target,
            _initial_state(expected_manifest, manifest_hash),
        )
    else:
        persisted = _load_json(manifest_target, "run manifest")
        if persisted != expected_manifest:
            if persisted.get("design_hash") != expected_manifest["design_hash"]:
                raise PerturbationStateError(
                    "refusing resume because the run design changed"
                )
            if persisted.get("checkpoints") != expected_manifest["checkpoints"]:
                raise PerturbationStateError(
                    "refusing resume because a checkpoint identity changed"
                )
            raise PerturbationStateError(
                "refusing resume because the registered phase plan changed"
            )
    _, _, snapshot = _validated_snapshot(manifest_target, state_target)
    return snapshot


def validate_resume(
    manifest_path: str | Path,
    state_path: str | Path,
    *,
    design: Mapping[str, Any],
    checkpoints: Mapping[str, str | Path],
    expected_cells: Mapping[str, Sequence[str]],
    phase_order: Sequence[str] = DEFAULT_PHASE_ORDER,
) -> ResumeSnapshot:
    """Validate an existing run without creating or modifying any file."""
    manifest_target = Path(manifest_path).resolve()
    state_target = Path(state_path).resolve()
    if not manifest_target.is_file() or not state_target.is_file():
        raise PerturbationStateError("run manifest or state is missing")
    expected = _manifest_payload(
        design,
        checkpoints,
        phase_order,
        expected_cells,
    )
    persisted = _load_json(manifest_target, "run manifest")
    if persisted != expected:
        if persisted.get("design_hash") != expected["design_hash"]:
            raise PerturbationStateError(
                "refusing resume because the run design changed"
            )
        if persisted.get("checkpoints") != expected["checkpoints"]:
            raise PerturbationStateError(
                "refusing resume because a checkpoint identity changed"
            )
        raise PerturbationStateError(
            "refusing resume because the registered phase plan changed"
        )
    _, _, snapshot = _validated_snapshot(manifest_target, state_target)
    return snapshot


def _write_state(
    state_path: Path,
    state: dict[str, Any],
) -> None:
    state["revision"] = int(state["revision"]) + 1
    atomic_write_json(state_path, state)


def begin_phase(
    manifest_path: str | Path,
    state_path: str | Path,
    phase: str,
) -> ResumeSnapshot:
    """Activate the next registered phase, idempotently on resume."""
    manifest_target = Path(manifest_path).resolve()
    state_target = Path(state_path).resolve()
    manifest, state, snapshot = _validated_snapshot(
        manifest_target, state_target
    )
    if phase not in manifest["phase_order"]:
        raise PerturbationStateError(f"unregistered phase: {phase}")
    phase_state = state["phases"][phase]
    if phase_state["status"] in {"active", "completed"}:
        return snapshot
    index = manifest["phase_order"].index(phase)
    prerequisites = manifest["phase_order"][:index]
    if any(
        state["phases"][prior]["status"] != "completed"
        for prior in prerequisites
    ):
        raise PerturbationStateError(
            f"phase prerequisites are incomplete for {phase}"
        )
    phase_state["status"] = "active"
    _write_state(state_target, state)
    return _validated_snapshot(manifest_target, state_target)[2]


def start_phase_attempt(
    manifest_path: str | Path,
    state_path: str | Path,
    *,
    phase: str,
    device: str,
    cuda_synchronized: bool,
    started_utc: str | None = None,
) -> PhaseAttemptStart:
    """Open one active-execution attempt, closing a crashed predecessor.

    An earlier still-active attempt is marked interrupted when resume detects
    it. No time between its last committed cell and this resume is counted.
    """
    manifest_target = Path(manifest_path).resolve()
    state_target = Path(state_path).resolve()
    manifest, state, _ = _validated_snapshot(manifest_target, state_target)
    if phase not in manifest["phase_order"]:
        raise PerturbationStateError(f"unregistered phase: {phase}")
    phase_state = state["phases"][phase]
    if phase_state["status"] != "active":
        raise PerturbationStateError(
            f"phase {phase} must be active before starting a timing attempt"
        )
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string")
    if not isinstance(cuda_synchronized, bool):
        raise ValueError("cuda_synchronized must be boolean")
    timestamp = _utc_now() if started_utc is None else str(started_utc)
    if not timestamp:
        raise ValueError("started_utc must be non-empty")

    timing = phase_state["timing"]
    active_id = timing["active_attempt_id"]
    if active_id is not None:
        prior = timing["attempts"][active_id]
        if prior["status"] != "active":
            raise PerturbationStateError(
                f"phase {phase} timing attempt state is inconsistent"
            )
        prior["status"] = "interrupted"
        prior["detected_interrupted_utc"] = timestamp
        timing["active_attempt_id"] = None

    attempt_id = int(timing["next_attempt_id"])
    timing["next_attempt_id"] = attempt_id + 1
    timing["active_attempt_id"] = attempt_id
    timing["attempts"].append(
        {
            "attempt_id": attempt_id,
            "status": "active",
            "started_utc": timestamp,
            "detected_interrupted_utc": None,
            "ended_utc": None,
            "committed_seconds": 0.0,
            "last_committed_cell_id": None,
            "device": device,
            "cuda_synchronized": cuda_synchronized,
        }
    )
    _write_state(state_target, state)
    snapshot = _validated_snapshot(manifest_target, state_target)[2]
    return PhaseAttemptStart(attempt_id=attempt_id, snapshot=snapshot)


def _apply_timing_delta(
    phase_state: dict[str, Any],
    *,
    phase: str,
    attempt_id: int,
    timing_delta_seconds: float,
    last_cell_id: str | None,
) -> None:
    timing = phase_state["timing"]
    if timing["active_attempt_id"] != attempt_id:
        raise PerturbationStateError(
            f"phase {phase} timing attempt is not active"
        )
    delta = _finite_nonnegative_seconds(
        timing_delta_seconds,
        f"phase {phase} timing delta",
    )
    attempt = timing["attempts"][attempt_id]
    if attempt["status"] != "active":
        raise PerturbationStateError(
            f"phase {phase} timing attempt is not active"
        )
    timing["accumulated_seconds"] = (
        float(timing["accumulated_seconds"]) + delta
    )
    attempt["committed_seconds"] = (
        float(attempt["committed_seconds"]) + delta
    )
    if last_cell_id is not None:
        attempt["last_committed_cell_id"] = last_cell_id


def finish_phase_attempt(
    manifest_path: str | Path,
    state_path: str | Path,
    *,
    phase: str,
    attempt_id: int,
    timing_delta_seconds: float,
    ended_utc: str | None = None,
) -> ResumeSnapshot:
    """Atomically add the final active-time delta and close one attempt."""
    manifest_target = Path(manifest_path).resolve()
    state_target = Path(state_path).resolve()
    manifest, state, _ = _validated_snapshot(manifest_target, state_target)
    if phase not in manifest["phase_order"]:
        raise PerturbationStateError(f"unregistered phase: {phase}")
    phase_state = state["phases"][phase]
    if phase_state["status"] != "active":
        raise PerturbationStateError(
            f"phase {phase} must be active before finishing a timing attempt"
        )
    resolved_attempt = int(attempt_id)
    _apply_timing_delta(
        phase_state,
        phase=phase,
        attempt_id=resolved_attempt,
        timing_delta_seconds=timing_delta_seconds,
        last_cell_id=None,
    )
    timestamp = _utc_now() if ended_utc is None else str(ended_utc)
    if not timestamp:
        raise ValueError("ended_utc must be non-empty")
    timing = phase_state["timing"]
    attempt = timing["attempts"][resolved_attempt]
    attempt["status"] = "completed"
    attempt["ended_utc"] = timestamp
    timing["active_attempt_id"] = None
    _write_state(state_target, state)
    return _validated_snapshot(manifest_target, state_target)[2]


def record_completed_cell(
    manifest_path: str | Path,
    state_path: str | Path,
    *,
    phase: str,
    cell_id: str,
    artifacts: Sequence[str | Path],
    metadata: Mapping[str, Any] | None = None,
    timing_attempt_id: int | None = None,
    timing_delta_seconds: float | None = None,
) -> CellRecordResult:
    """Record immutable hashes for one completed registered cell.

    Repeating the call for an already valid cell returns ``reused=True`` and
    does not rewrite state.
    """
    manifest_target = Path(manifest_path).resolve()
    state_target = Path(state_path).resolve()
    manifest, state, snapshot = _validated_snapshot(
        manifest_target, state_target
    )
    if phase not in manifest["phase_order"]:
        raise PerturbationStateError(f"unregistered phase: {phase}")
    if cell_id not in manifest["expected_cells"][phase]:
        raise PerturbationStateError(
            f"unregistered cell for phase {phase}: {cell_id}"
        )
    if not artifacts:
        raise ValueError("a completed cell must have at least one artifact")
    records = [
        _artifact_path_record(
            Path(artifact),
            manifest_target.parent.resolve(),
        )
        for artifact in artifacts
    ]
    artifact_paths = [record["path"] for record in records]
    if len(set(artifact_paths)) != len(artifact_paths):
        raise ValueError("cell artifact paths must be unique")
    requested_cell = {
        "status": "completed",
        "artifacts": sorted(records, key=lambda record: record["path"]),
        "metadata": _normalized_json(dict(metadata or {})),
    }
    phase_state = state["phases"][phase]
    existing = phase_state["cells"].get(cell_id)
    if existing is not None:
        if existing != requested_cell:
            raise PerturbationStateError(
                f"completed cell {cell_id} does not match its persisted record"
            )
        return CellRecordResult(snapshot=snapshot, reused=True)
    if phase_state["status"] != "active":
        raise PerturbationStateError(
            f"phase {phase} must be active before recording cells"
        )
    active_attempt = phase_state["timing"]["active_attempt_id"]
    if active_attempt is None:
        if timing_attempt_id is not None or timing_delta_seconds is not None:
            raise PerturbationStateError(
                f"phase {phase} has no active timing attempt"
            )
    else:
        if timing_attempt_id is None or timing_delta_seconds is None:
            raise PerturbationStateError(
                f"phase {phase} cell timing must be committed atomically"
            )
        _apply_timing_delta(
            phase_state,
            phase=phase,
            attempt_id=int(timing_attempt_id),
            timing_delta_seconds=timing_delta_seconds,
            last_cell_id=cell_id,
        )
    phase_state["cells"][cell_id] = requested_cell
    _write_state(state_target, state)
    return CellRecordResult(
        snapshot=_validated_snapshot(manifest_target, state_target)[2],
        reused=False,
    )


def complete_phase(
    manifest_path: str | Path,
    state_path: str | Path,
    phase: str,
) -> ResumeSnapshot:
    """Complete an active phase only after every registered cell is valid."""
    manifest_target = Path(manifest_path).resolve()
    state_target = Path(state_path).resolve()
    manifest, state, snapshot = _validated_snapshot(
        manifest_target, state_target
    )
    if phase not in manifest["phase_order"]:
        raise PerturbationStateError(f"unregistered phase: {phase}")
    phase_state = state["phases"][phase]
    if phase_state["status"] == "completed":
        return snapshot
    if phase_state["status"] != "active":
        raise PerturbationStateError(
            f"phase {phase} must be active before completion"
        )
    expected = set(manifest["expected_cells"][phase])
    completed = set(phase_state["cells"])
    missing = sorted(expected - completed)
    if missing:
        raise PerturbationStateError(
            f"phase {phase} is missing completed cells: {missing}"
        )
    if phase_state["timing"]["active_attempt_id"] is not None:
        raise PerturbationStateError(
            f"phase {phase} has an unfinished timing attempt"
        )
    phase_state["status"] = "completed"
    _write_state(state_target, state)
    return _validated_snapshot(manifest_target, state_target)[2]
