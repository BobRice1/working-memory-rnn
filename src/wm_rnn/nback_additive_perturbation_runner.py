"""Phased execution for the frozen additive-cost N-back perturbation study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from scipy import stats

from wm_rnn.config import load_config
from wm_rnn.device import SelectedDevice, select_device
from wm_rnn.nback_additive_calibration import (
    CALIBRATION_TARGET,
    CALIBRATION_TOLERANCE,
    CONFIRMATORY_PROFILE_IDS,
    COST_CHECK_DRAWS,
    COST_CHECK_SEQUENCES,
    OPERATOR_PROFILES,
    P2_VECTOR_SEEDS,
    P5_REPLICATE_LABELS,
    PROFILE_BY_ID,
    TASK_BANKS,
    AdditiveCalibrationResult,
    OperatorProfile,
    average_replicate_sequence_units,
    calibrate_profile,
    cost_check_bootstrap_seed,
    p2_vector_seed,
    p5_generator_seed,
    summarize_additive_cost,
    task_seed,
    validate_heldout_additive_cost,
)
from wm_rnn.nback_additive_cost_precision import (
    RetainedCheckpoint,
    load_retained_checkpoints,
)
from wm_rnn.nback_metrics import _settling_values, nback_metrics
from wm_rnn.nback_perturbation import (
    build_nback_operator,
    candidate_vs_p5_load_contrast,
)
from wm_rnn.nback_perturbation_state import (
    PerturbationStateError,
    ResumeSnapshot,
    atomic_write_json,
    atomic_write_npz,
    begin_phase,
    canonical_design_hash,
    complete_phase,
    finish_phase_attempt,
    initialize_or_resume_run,
    record_completed_cell,
    sha256_file,
    start_phase_attempt,
)
from wm_rnn.nback_task import (
    NBackBatch,
    NBackTaskConfig,
    generate_nback_batch,
)
from wm_rnn.training_utils import (
    batch_to_tensors,
    fresh_model,
    task_config_from_dict,
)


PREREGISTRATION_COMMIT = "fc99475"
PHASE_ORDER = (
    "neutral-calibration",
    "calibration",
    "cost-check",
    "neutral-confirmatory",
    "confirmatory",
    "dose",
    "finalize",
)
DEFAULT_CONFIG = Path("configs/nback_additive_perturbation.yaml")
DEFAULT_PREREGISTRATION = Path(
    "docs/preregistration/"
    "nback_additive_perturbation_preregistration.md"
)


class EvaluationValidityError(RuntimeError):
    """Expected scientific cell invalidity from nonfinite evaluation output."""


@dataclass(frozen=True)
class NeutralEquivalence:
    """Exact native-versus-neutral equality result."""

    exact: bool
    comparisons: int
    maximum_absolute_logit_difference: float
    maximum_absolute_hidden_difference: float
    additive_cost: float


@dataclass(frozen=True)
class EvaluationBundle:
    """One native or perturbed cell, retaining nested replicate results."""

    replicate_sequence_ce: np.ndarray
    replicate_metrics: tuple[dict[str, Any], ...]
    task_hashes: tuple[str, ...]
    replicate_settling_steps: np.ndarray | None = None
    replicate_behavioral_correct: np.ndarray | None = None

    def __post_init__(self) -> None:
        units = np.asarray(self.replicate_sequence_ce)
        if (
            units.ndim != 2
            or units.shape[0] not in {1, 3}
            or units.shape[1] == 0
            or not np.all(np.isfinite(units))
            or np.any(units < 0.0)
        ):
            raise EvaluationValidityError(
                "replicate_sequence_ce must be finite non-negative [1|3, n]"
            )
        if len(self.replicate_metrics) != units.shape[0]:
            raise ValueError("one pooled metric mapping is required per replicate")
        if not self.task_hashes or any(not value for value in self.task_hashes):
            raise ValueError("task_hashes must be non-empty")
        if self.replicate_settling_steps is not None:
            settling = np.asarray(self.replicate_settling_steps)
            if settling.ndim != 2 or settling.shape[0] != units.shape[0]:
                raise ValueError(
                    "replicate_settling_steps must align with replicates"
                )
        if self.replicate_behavioral_correct is not None:
            behavioral = np.asarray(self.replicate_behavioral_correct)
            if (
                behavioral.ndim != 2
                or behavioral.shape[0] != units.shape[0]
            ):
                raise ValueError(
                    "replicate_behavioral_correct must align with replicates"
                )
        if (
            self.replicate_settling_steps is None
        ) != (self.replicate_behavioral_correct is None):
            raise ValueError(
                "settling steps and behavioral correctness are paired"
            )
        if (
            self.replicate_settling_steps is not None
            and self.replicate_behavioral_correct is not None
            and self.replicate_settling_steps.shape
            != self.replicate_behavioral_correct.shape
        ):
            raise ValueError("settling audit arrays must have identical shapes")

    @property
    def n_replicates(self) -> int:
        return int(self.replicate_sequence_ce.shape[0])

    @property
    def n_sequences(self) -> int:
        return int(self.replicate_sequence_ce.shape[1])

    def averaged_sequence_ce(self) -> np.ndarray:
        if self.n_replicates == 1:
            return self.replicate_sequence_ce[0]
        return average_replicate_sequence_units(
            self.replicate_sequence_ce
        )

    def averaged_metrics(self) -> dict[str, Any]:
        return _average_metric_mappings(self.replicate_metrics)


class EvaluationBackend(Protocol):
    """Injectable model-evaluation boundary used by tests and CUDA runs."""

    def neutral_equivalence(
        self,
        checkpoint: RetainedCheckpoint,
        profile: OperatorProfile,
        *,
        phase: str,
        condition_code: int,
    ) -> NeutralEquivalence: ...

    def evaluate(
        self,
        checkpoint: RetainedCheckpoint,
        profile: OperatorProfile | None,
        strength: float | None,
        *,
        phase: str,
        condition_code: int,
    ) -> EvaluationBundle: ...

    def release_checkpoint(self, checkpoint: RetainedCheckpoint) -> None: ...


@dataclass(frozen=True)
class RunPaths:
    """Deterministic files and directories for one phased run."""

    root: Path
    state_dir: Path
    manifest_dir: Path
    manifest_path: Path
    state_path: Path


@dataclass(frozen=True)
class RunnerContext:
    """Validated immutable run identity plus its evaluation backend."""

    config: dict[str, Any]
    checkpoints: tuple[RetainedCheckpoint, ...]
    profiles: tuple[OperatorProfile, ...]
    paths: RunPaths
    design: dict[str, Any]
    expected_cells: dict[str, tuple[str, ...]]
    backend: EvaluationBackend
    device: SelectedDevice
    timing_runtime: PhaseTimingRuntimeSlot = field(
        default_factory=lambda: PhaseTimingRuntimeSlot()
    )


@dataclass
class PhaseTimingRuntime:
    """Process-local monotonic timing cursor for one phase attempt."""

    phase: str
    attempt_id: int
    started: float
    accounted: float
    synchronize_cuda: bool


@dataclass
class PhaseTimingRuntimeSlot:
    """Mutable timing slot retained inside the otherwise frozen context."""

    active: PhaseTimingRuntime | None = None


def _json_safe_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    result.pop("sequence_cross_entropies", None)
    return result


def _average_values(values: Sequence[Any]) -> Any:
    if any(value is None for value in values):
        return None
    first = values[0]
    if isinstance(first, Mapping):
        keys = set(first)
        if any(not isinstance(value, Mapping) or set(value) != keys for value in values):
            raise ValueError("replicate metric mappings must have identical keys")
        return {
            key: _average_values([value[key] for value in values])
            for key in sorted(keys)
        }
    if isinstance(first, bool):
        return bool(all(bool(value) for value in values))
    if isinstance(first, str):
        if any(value != first for value in values):
            raise ValueError("replicate metric labels must match")
        return first
    if isinstance(first, (int, float, np.integer, np.floating)):
        numeric = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(numeric)):
            raise EvaluationValidityError(
                "replicate metrics must be finite"
            )
        return float(np.mean(numeric))
    raise TypeError(f"unsupported replicate metric type: {type(first).__name__}")


def _average_metric_mappings(
    metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not metrics:
        raise ValueError("at least one replicate metric mapping is required")
    cleaned = tuple(_json_safe_metrics(metric) for metric in metrics)
    return _average_values(cleaned)


def _require_finite_evaluation_value(
    value: Any,
    *,
    path: str = "metrics",
) -> None:
    """Reject only nonfinite numerical evaluation results.

    Shape, schema, and type errors remain ordinary implementation errors and
    must not be converted into scientific NA cells.
    """
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require_finite_evaluation_value(
                child,
                path=f"{path}.{key}",
            )
        return
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise EvaluationValidityError(
                f"nonfinite evaluation value at {path}"
            )
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.all(
            np.isfinite(value)
        ):
            raise EvaluationValidityError(
                f"nonfinite evaluation value at {path}"
            )
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _require_finite_evaluation_value(
                child,
                path=f"{path}[{index}]",
            )
        return
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(float(value)):
            raise EvaluationValidityError(
                f"nonfinite evaluation value at {path}"
            )


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _combine_nback_batches(batches: Sequence[NBackBatch]) -> NBackBatch:
    if not batches:
        raise ValueError("at least one N-back batch is required")
    first = batches[0]
    for batch in batches[1:]:
        if (
            batch.n_back != first.n_back
            or batch.event_steps != first.event_steps
            or not np.array_equal(batch.event_onsets, first.event_onsets)
        ):
            raise ValueError("N-back batches are not pool-compatible")
    return NBackBatch(
        inputs=np.concatenate([batch.inputs for batch in batches], axis=1),
        targets=np.concatenate([batch.targets for batch in batches], axis=1),
        loss_mask=np.concatenate(
            [batch.loss_mask for batch in batches], axis=1
        ),
        stimuli=np.concatenate([batch.stimuli for batch in batches], axis=1),
        item_labels=np.concatenate(
            [batch.item_labels for batch in batches], axis=1
        ),
        item_scored=np.concatenate(
            [batch.item_scored for batch in batches], axis=1
        ),
        one_back_lures=np.concatenate(
            [batch.one_back_lures for batch in batches], axis=1
        ),
        event_onsets=first.event_onsets.copy(),
        event_steps=first.event_steps,
        n_back=first.n_back,
    )


def _profile_parameters(
    profile: OperatorProfile,
    strength: float,
    *,
    replicate_ordinal: int,
    phase: str,
    checkpoint_ordinal: int,
    condition_code: int,
    batch_index: int,
) -> dict[str, object]:
    operator = profile.operator
    if operator == "synaptic_drive_gain":
        return {"gain": strength, "bias_mode": profile.variant}
    if operator == "heterogeneous_drive_gain":
        return {
            "log_std": strength,
            "vector_seed": p2_vector_seed(replicate_ordinal),
            "bias_mode": profile.variant,
        }
    if operator == "sensory_input_gain":
        return {"gain": strength}
    if operator == "recurrent_gain":
        return {"gain": strength}
    if operator == "state_persistence":
        return {"persistence_gain": strength}
    if operator == "time_constant":
        return {"tau_scale": strength}
    if operator == "gaussian_state_noise":
        return {
            "sigma": strength,
            "generator_seed": p5_generator_seed(
                phase,
                checkpoint_ordinal,
                condition_code,
                replicate_ordinal,
                batch_index,
            ),
        }
    raise KeyError(f"unsupported registered operator: {operator}")


class TorchNBackBackend:
    """Checkpoint-backed implementation of the frozen evaluation boundary."""

    def __init__(
        self,
        config: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self._loaded_seed: int | None = None
        self._model: torch.nn.Module | None = None
        if float(config["model"].get("recurrent_noise_std", 0.0)) != 0.0:
            raise ValueError("registered checkpoints require zero recurrent noise")

    def _model_for(
        self, checkpoint: RetainedCheckpoint
    ) -> torch.nn.Module:
        if self._loaded_seed == checkpoint.seed and self._model is not None:
            return self._model
        self.release_checkpoint(checkpoint)
        saved = torch.load(checkpoint.path, map_location=self.device)
        embedded = saved.get("config", {})
        embedded_seed = embedded.get("task", {}).get("seed")
        embedded_task = embedded.get("task", {})
        embedded_model = embedded.get("model", {})
        embedded_noise = embedded_model.get(
            "recurrent_noise_std", 0.0
        )
        if int(embedded_seed) != checkpoint.seed:
            raise ValueError("checkpoint seed does not match retained manifest")
        if float(embedded_noise) != 0.0:
            raise ValueError("checkpoint recurrent_noise_std must equal zero")
        expected_architecture = {
            "hidden_size": 64,
            "dt": 20.0,
            "tau": 100.0,
            "activation": "tanh",
        }
        for field, expected in expected_architecture.items():
            actual = embedded_model.get(field)
            if actual != expected:
                raise ValueError(
                    f"checkpoint architecture mismatch for {field}"
                )
        if (
            embedded_task.get("task_type") != "n_back"
            or int(embedded_task.get("n_stimuli", -1)) != 6
        ):
            raise ValueError("checkpoint task architecture must be 8-input N-back")
        model = fresh_model(self.config, self.device)
        if (
            model.config.input_size != 8
            or model.config.output_size != 2
            or model.config.hidden_size != 64
        ):
            raise ValueError("constructed model architecture is not frozen")
        model.load_state_dict(saved["model_state"])
        model.eval()
        self._loaded_seed = checkpoint.seed
        self._model = model
        return model

    def _task_batch(
        self,
        checkpoint: RetainedCheckpoint,
        phase: str,
        condition_code: int,
        batch_index: int,
    ) -> NBackBatch:
        base = task_config_from_dict(self.config, batch_size=128)
        if not isinstance(base, NBackTaskConfig):
            raise ValueError("runner requires task_type: n_back")
        task = replace(
            base,
            n_back=condition_code * 2,
            seed=task_seed(
                phase,
                checkpoint.ordinal,
                condition_code,
                batch_index,
            ),
        )
        batch = generate_nback_batch(task)
        contexts = batch.inputs[:, :, 6:8]
        if (
            contexts.shape[-1] != 2
            or not np.all((contexts == 0.0) | (contexts == 1.0))
            or not np.all(np.sum(contexts, axis=-1) == 1.0)
            or np.any(contexts[:, :, 0] * contexts[:, :, 1] != 0.0)
        ):
            raise ValueError(
                "inputs 6 and 7 must be mutually exclusive rule contexts"
            )
        return batch

    @staticmethod
    def _replicate_count(profile: OperatorProfile | None) -> int:
        if profile is not None and profile.operator in {
            "heterogeneous_drive_gain",
            "gaussian_state_noise",
        }:
            return 3
        return 1

    @torch.no_grad()
    def evaluate(
        self,
        checkpoint: RetainedCheckpoint,
        profile: OperatorProfile | None,
        strength: float | None,
        *,
        phase: str,
        condition_code: int,
    ) -> EvaluationBundle:
        model = self._model_for(checkpoint)
        bank = TASK_BANKS[phase]
        if condition_code not in bank.condition_codes:
            raise ValueError("condition is not available in this phase")
        if profile is None and strength is not None:
            raise ValueError("native baseline cannot have a strength")
        if profile is not None and strength is None:
            raise ValueError("perturbed evaluation requires a strength")

        replicate_units: list[np.ndarray] = []
        replicate_metrics: list[dict[str, Any]] = []
        replicate_settling: list[np.ndarray] = []
        replicate_behavioral_correct: list[np.ndarray] = []
        reference_hashes: tuple[str, ...] | None = None
        for replicate in range(self._replicate_count(profile)):
            logits_by_batch: list[torch.Tensor] = []
            batches: list[NBackBatch] = []
            hashes: list[str] = []
            static_forward = None
            if (
                profile is not None
                and profile.operator != "gaussian_state_noise"
            ):
                static_forward = build_nback_operator(
                    model,
                    profile.operator,
                    **_profile_parameters(
                        profile,
                        float(strength),
                        replicate_ordinal=replicate,
                        phase=phase,
                        checkpoint_ordinal=checkpoint.ordinal,
                        condition_code=condition_code,
                        batch_index=0,
                    ),
                )
            for batch_index in range(bank.n_batches):
                batch = self._task_batch(
                    checkpoint, phase, condition_code, batch_index
                )
                inputs, _, _ = batch_to_tensors(batch, self.device)
                if profile is None:
                    forward = model
                elif profile.operator == "gaussian_state_noise":
                    forward = build_nback_operator(
                        model,
                        profile.operator,
                        **_profile_parameters(
                            profile,
                            float(strength),
                            replicate_ordinal=replicate,
                            phase=phase,
                            checkpoint_ordinal=checkpoint.ordinal,
                            condition_code=condition_code,
                            batch_index=batch_index,
                        ),
                    )
                else:
                    forward = static_forward
                if forward is None:
                    raise RuntimeError("operator forward was not constructed")
                logits, _ = forward(inputs)
                if not torch.isfinite(logits).all():
                    raise EvaluationValidityError(
                        "operator produced nonfinite logits"
                    )
                logits_by_batch.append(logits.detach().cpu())
                batches.append(batch)
                hashes.append(
                    _hash_arrays(batch.inputs, batch.targets, batch.loss_mask)
                )
            pooled_batch = _combine_nback_batches(batches)
            pooled_logits = torch.cat(logits_by_batch, dim=1)
            targets = torch.from_numpy(pooled_batch.targets).long()
            mask = torch.from_numpy(pooled_batch.loss_mask).float()
            metrics = nback_metrics(
                pooled_logits, targets, mask, pooled_batch
            )
            _require_finite_evaluation_value(metrics)
            settling, behavioral_correct = _settling_values(
                pooled_logits,
                pooled_batch,
                probability_threshold=0.80,
                margin_threshold=0.60,
                consecutive_steps=3,
            )
            replicate_units.append(
                np.asarray(
                    metrics["sequence_cross_entropies"],
                    dtype=np.float64,
                )
            )
            replicate_metrics.append(_json_safe_metrics(metrics))
            replicate_settling.append(settling)
            replicate_behavioral_correct.append(behavioral_correct)
            current_hashes = tuple(hashes)
            if reference_hashes is None:
                reference_hashes = current_hashes
            elif current_hashes != reference_hashes:
                raise RuntimeError("replicates did not reuse identical tasks")
        matrix = np.stack(replicate_units)
        expected = bank.n_batches * 128
        if matrix.shape != (self._replicate_count(profile), expected):
            raise RuntimeError("evaluation produced an unexpected CE shape")
        return EvaluationBundle(
            replicate_sequence_ce=matrix,
            replicate_metrics=tuple(replicate_metrics),
            task_hashes=reference_hashes or (),
            replicate_settling_steps=np.stack(replicate_settling),
            replicate_behavioral_correct=np.stack(
                replicate_behavioral_correct
            ),
        )

    @torch.no_grad()
    def neutral_equivalence(
        self,
        checkpoint: RetainedCheckpoint,
        profile: OperatorProfile,
        *,
        phase: str,
        condition_code: int,
    ) -> NeutralEquivalence:
        model = self._model_for(checkpoint)
        neutral_strength = profile.ordered_grid[0]
        bank = TASK_BANKS[phase]
        maximum_logit = 0.0
        maximum_hidden = 0.0
        comparisons = 0
        baseline_units: list[np.ndarray] = []
        neutral_units_by_replicate: list[list[np.ndarray]] = [
            [] for _ in range(self._replicate_count(profile))
        ]
        exact = True
        for batch_index in range(bank.n_batches):
            batch = self._task_batch(
                checkpoint, phase, condition_code, batch_index
            )
            inputs, _, _ = batch_to_tensors(batch, self.device)
            native_logits, native_hidden = model(inputs)
            if not torch.isfinite(native_logits).all():
                raise EvaluationValidityError(
                    "native model produced nonfinite logits"
                )
            native_metrics = nback_metrics(
                native_logits,
                torch.from_numpy(batch.targets).long().to(self.device),
                torch.from_numpy(batch.loss_mask).float().to(self.device),
                batch,
            )
            _require_finite_evaluation_value(native_metrics)
            baseline_units.append(
                np.asarray(
                    native_metrics["sequence_cross_entropies"],
                    dtype=np.float64,
                )
            )
            for replicate in range(self._replicate_count(profile)):
                forward = build_nback_operator(
                    model,
                    profile.operator,
                    **_profile_parameters(
                        profile,
                        neutral_strength,
                        replicate_ordinal=replicate,
                        phase=phase,
                        checkpoint_ordinal=checkpoint.ordinal,
                        condition_code=condition_code,
                        batch_index=batch_index,
                    ),
                )
                logits, hidden = forward(inputs)
                if not torch.isfinite(logits).all():
                    raise EvaluationValidityError(
                        "neutral operator produced nonfinite logits"
                    )
                logit_equal = torch.equal(native_logits, logits)
                hidden_equal = torch.equal(native_hidden, hidden)
                exact = exact and logit_equal and hidden_equal
                maximum_logit = max(
                    maximum_logit,
                    float(torch.max(torch.abs(native_logits - logits)).item()),
                )
                maximum_hidden = max(
                    maximum_hidden,
                    float(torch.max(torch.abs(native_hidden - hidden)).item()),
                )
                metrics = nback_metrics(
                    logits,
                    torch.from_numpy(batch.targets).long().to(self.device),
                    torch.from_numpy(batch.loss_mask).float().to(self.device),
                    batch,
                )
                _require_finite_evaluation_value(metrics)
                neutral_units_by_replicate[replicate].append(
                    np.asarray(
                        metrics["sequence_cross_entropies"],
                        dtype=np.float64,
                    )
                )
                comparisons += 1
        baseline = np.concatenate(baseline_units)
        neutral_matrix = np.stack(
            [
                np.concatenate(replicate_units)
                for replicate_units in neutral_units_by_replicate
            ]
        )
        neutral = (
            neutral_matrix[0]
            if neutral_matrix.shape[0] == 1
            else average_replicate_sequence_units(neutral_matrix)
        )
        additive_cost = summarize_additive_cost(
            baseline, neutral
        ).additive_cost
        return NeutralEquivalence(
            exact=bool(exact and additive_cost == 0.0),
            comparisons=comparisons,
            maximum_absolute_logit_difference=maximum_logit,
            maximum_absolute_hidden_difference=maximum_hidden,
            additive_cost=additive_cost,
        )

    def release_checkpoint(self, checkpoint: RetainedCheckpoint) -> None:
        del checkpoint
        self._model = None
        self._loaded_seed = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def expected_cells(
    checkpoints: Sequence[RetainedCheckpoint],
    profiles: Sequence[OperatorProfile] = OPERATOR_PROFILES,
) -> dict[str, tuple[str, ...]]:
    """Materialize every registered cell before execution."""
    cells: dict[str, list[str]] = {phase: [] for phase in PHASE_ORDER}
    for checkpoint in checkpoints:
        cp = checkpoint.ordinal
        for profile in profiles:
            pid = profile.profile_id
            cells["neutral-calibration"].append(
                f"neutral-calibration:cp{cp:02d}:p{pid:02d}"
            )
            cells["calibration"].append(
                f"calibration:cp{cp:02d}:p{pid:02d}"
            )
            cells["cost-check"].append(
                f"cost-check:cp{cp:02d}:p{pid:02d}"
            )
            cells["dose"].append(f"dose:cp{cp:02d}:p{pid:02d}")
            for condition in (0, 1):
                cells["neutral-confirmatory"].append(
                    "neutral-confirmatory:"
                    f"cp{cp:02d}:p{pid:02d}:c{condition}"
                )
                cells["confirmatory"].append(
                    f"confirmatory:cp{cp:02d}:p{pid:02d}:c{condition}"
                )
        for condition in (0, 1):
            cells["confirmatory"].append(
                f"confirmatory:baseline:cp{cp:02d}:c{condition}"
            )
    cells["finalize"].append("finalize:summary")
    for phase in PHASE_ORDER:
        cells[phase].append(f"{phase}:timing")
    return {phase: tuple(values) for phase, values in cells.items()}


def _run_paths(root: str | Path) -> RunPaths:
    resolved = Path(root).resolve()
    state_dir = resolved / "state"
    manifest_dir = resolved / "manifest"
    for name in (
        "neutral",
        "calibration",
        "cost_check",
        "confirmatory",
        "dose",
        "metrics",
        "state",
        "manifest",
    ):
        (resolved / name).mkdir(parents=True, exist_ok=True)
    return RunPaths(
        root=resolved,
        state_dir=state_dir,
        manifest_dir=manifest_dir,
        # The persistence layer anchors artifact containment to the common
        # parent of these files, so the run identity must live at run root.
        manifest_path=resolved / "run_manifest.json",
        state_path=resolved / "run_state.json",
    )


def _cell_stem(cell_id: str) -> str:
    return cell_id.replace(":", "__").replace("-", "_")


def _cell_paths(
    context: RunnerContext,
    phase: str,
    cell_id: str,
    *,
    with_arrays: bool,
) -> tuple[Path, Path | None]:
    directory = (
        context.paths.root / "neutral"
        if phase.startswith("neutral-")
        else context.paths.root / phase.replace("-", "_")
    )
    json_path = directory / "cells" / f"{_cell_stem(cell_id)}.json"
    arrays_path = (
        directory / "arrays" / f"{_cell_stem(cell_id)}.npz"
        if with_arrays
        else None
    )
    return json_path, arrays_path


def _record_cell(
    context: RunnerContext,
    *,
    phase: str,
    cell_id: str,
    payload: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray] | None = None,
) -> None:
    json_path, arrays_path = _cell_paths(
        context, phase, cell_id, with_arrays=arrays is not None
    )
    atomic_write_json(json_path, dict(payload))
    artifacts: list[Path] = [json_path]
    if arrays is not None:
        if arrays_path is None:
            raise RuntimeError("array path was not created")
        atomic_write_npz(arrays_path, **dict(arrays))
        artifacts.append(arrays_path)
    _record_completed_artifacts(
        context,
        phase=phase,
        cell_id=cell_id,
        artifacts=artifacts,
        metadata={
            "valid": bool(payload.get("valid", False)),
            "status": str(payload.get("status", "completed")),
        },
    )


def _record_completed_artifacts(
    context: RunnerContext,
    *,
    phase: str,
    cell_id: str,
    artifacts: Sequence[str | Path],
    metadata: Mapping[str, Any],
) -> None:
    """Record artifacts and the current active-time delta in one state write."""
    runtime = context.timing_runtime.active
    timing_arguments: dict[str, Any] = {}
    elapsed: float | None = None
    if runtime is not None:
        if runtime.phase != phase:
            raise RuntimeError("active timing attempt belongs to another phase")
        if runtime.synchronize_cuda:
            torch.cuda.synchronize(context.device.device)
        elapsed = time.perf_counter() - runtime.started
        delta = elapsed - runtime.accounted
        if not np.isfinite(delta) or delta < 0.0:
            raise RuntimeError("phase timing clock produced an invalid delta")
        timing_arguments = {
            "timing_attempt_id": runtime.attempt_id,
            "timing_delta_seconds": delta,
        }
    record_completed_cell(
        context.paths.manifest_path,
        context.paths.state_path,
        phase=phase,
        cell_id=cell_id,
        artifacts=artifacts,
        metadata=metadata,
        **timing_arguments,
    )
    if runtime is not None and elapsed is not None:
        runtime.accounted = elapsed


def _record_evaluation_invalid(
    context: RunnerContext,
    *,
    phase: str,
    cell_id: str,
    checkpoint: RetainedCheckpoint,
    error: EvaluationValidityError,
    profile: OperatorProfile | None = None,
    condition: int | None = None,
    baseline: bool = False,
) -> None:
    """Persist an expected nonfinite evaluation as a registered NA cell."""
    payload: dict[str, Any] = {
        "checkpoint_seed": checkpoint.seed,
        "checkpoint_ordinal": checkpoint.ordinal,
        "valid": False,
        "status": "invalid",
        "invalid_reason": (
            "nonfinite_baseline_evaluation"
            if baseline
            else "nonfinite_operator_evaluation"
        ),
        "invalid_detail": str(error),
    }
    if profile is not None:
        payload["profile_id"] = profile.profile_id
    if condition is not None:
        payload["condition_code"] = condition
    _record_cell(
        context,
        phase=phase,
        cell_id=cell_id,
        payload=payload,
    )


def _bundle_audit_arrays(
    bundle: EvaluationBundle,
    *,
    prefix: str,
) -> dict[str, np.ndarray]:
    arrays = {
        f"{prefix}_replicate_sequence_ce": bundle.replicate_sequence_ce
    }
    if bundle.replicate_settling_steps is not None:
        arrays[f"{prefix}_replicate_settling_steps"] = (
            bundle.replicate_settling_steps
        )
    if bundle.replicate_behavioral_correct is not None:
        arrays[f"{prefix}_replicate_behavioral_correct"] = (
            bundle.replicate_behavioral_correct
        )
    return arrays


def _load_cell(
    context: RunnerContext,
    phase: str,
    cell_id: str,
) -> dict[str, Any]:
    path = _cell_paths(
        context, phase, cell_id, with_arrays=False
    )[0]
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _is_reusable(
    snapshot: ResumeSnapshot,
    phase: str,
    cell_id: str,
) -> bool:
    return cell_id in snapshot.reusable_cells.get(phase, ())


def _neutral_cell_id(
    phase: str,
    checkpoint: RetainedCheckpoint,
    profile: OperatorProfile,
    condition: int | None = None,
) -> str:
    suffix = (
        "" if condition is None else f":c{condition}"
    )
    return (
        f"{phase}:cp{checkpoint.ordinal:02d}:"
        f"p{profile.profile_id:02d}{suffix}"
    )


def _profile_cell_id(
    phase: str,
    checkpoint: RetainedCheckpoint,
    profile: OperatorProfile,
    condition: int | None = None,
) -> str:
    return _neutral_cell_id(phase, checkpoint, profile, condition)


def _run_neutral_phase(
    context: RunnerContext,
    phase: str,
    snapshot: ResumeSnapshot,
) -> None:
    evaluation_phase = (
        "calibration"
        if phase == "neutral-calibration"
        else "confirmatory"
    )
    conditions = (0,) if phase == "neutral-calibration" else (0, 1)
    for checkpoint in context.checkpoints:
        try:
            for profile in context.profiles:
                for condition in conditions:
                    cell_id = _neutral_cell_id(
                        phase, checkpoint, profile, condition if len(conditions) > 1 else None
                    )
                    if _is_reusable(snapshot, phase, cell_id):
                        continue
                    result = context.backend.neutral_equivalence(
                        checkpoint,
                        profile,
                        phase=evaluation_phase,
                        condition_code=condition,
                    )
                    _record_cell(
                        context,
                        phase=phase,
                        cell_id=cell_id,
                        payload={
                            "phase": phase,
                            "checkpoint_seed": checkpoint.seed,
                            "checkpoint_ordinal": checkpoint.ordinal,
                            "profile_id": profile.profile_id,
                            "condition_code": condition,
                            **asdict(result),
                            "valid": result.exact,
                            "status": (
                                "passed" if result.exact else "failed"
                            ),
                        },
                    )
        finally:
            context.backend.release_checkpoint(checkpoint)


def _calibration_valid(
    context: RunnerContext,
    checkpoint: RetainedCheckpoint,
    profile: OperatorProfile,
) -> bool:
    cell = _load_cell(
        context,
        "calibration",
        _profile_cell_id("calibration", checkpoint, profile),
    )
    return bool(cell.get("valid"))


def _global_neutral_calibration_valid(
    context: RunnerContext,
    profile: OperatorProfile,
) -> bool:
    """Require neutral equivalence at every checkpoint before calibration."""
    return all(
        bool(
            _load_cell(
                context,
                "neutral-calibration",
                _neutral_cell_id(
                    "neutral-calibration",
                    checkpoint,
                    profile,
                ),
            ).get("valid")
        )
        for checkpoint in context.checkpoints
    )


def _run_calibration(
    context: RunnerContext,
    snapshot: ResumeSnapshot,
) -> None:
    phase = "calibration"
    neutral_validity = {
        profile.profile_id: _global_neutral_calibration_valid(
            context, profile
        )
        for profile in context.profiles
    }
    for checkpoint in context.checkpoints:
        pending = [
            profile
            for profile in context.profiles
            if not _is_reusable(
                snapshot,
                phase,
                _profile_cell_id(phase, checkpoint, profile),
            )
        ]
        if not pending:
            continue
        eligible: list[OperatorProfile] = []
        for profile in pending:
            if neutral_validity[profile.profile_id]:
                eligible.append(profile)
                continue
            _record_cell(
                context,
                phase=phase,
                cell_id=_profile_cell_id(phase, checkpoint, profile),
                payload={
                    "checkpoint_seed": checkpoint.seed,
                    "checkpoint_ordinal": checkpoint.ordinal,
                    "profile_id": profile.profile_id,
                    "valid": False,
                    "status": "skipped",
                    "invalid_reason": (
                        "global_neutral_calibration_failure"
                    ),
                },
            )
        if not eligible:
            continue
        try:
            try:
                baseline = context.backend.evaluate(
                    checkpoint,
                    None,
                    None,
                    phase="calibration",
                    condition_code=0,
                )
                baseline_units = baseline.averaged_sequence_ce()
            except EvaluationValidityError as error:
                for profile in eligible:
                    _record_evaluation_invalid(
                        context,
                        phase=phase,
                        cell_id=_profile_cell_id(
                            phase, checkpoint, profile
                        ),
                        checkpoint=checkpoint,
                        profile=profile,
                        error=error,
                        baseline=True,
                    )
                continue
            for profile in eligible:
                cell_id = _profile_cell_id(phase, checkpoint, profile)
                bundles: dict[float, EvaluationBundle] = {}

                def cost_function(strength: float) -> float:
                    bundle = context.backend.evaluate(
                        checkpoint,
                        profile,
                        strength,
                        phase="calibration",
                        condition_code=0,
                    )
                    if bundle.task_hashes != baseline.task_hashes:
                        raise RuntimeError(
                            "calibration settings did not reuse baseline tasks"
                        )
                    bundles[float(strength)] = bundle
                    return summarize_additive_cost(
                        baseline_units,
                        bundle.averaged_sequence_ce(),
                    ).additive_cost

                try:
                    result = calibrate_profile(profile, cost_function)
                except EvaluationValidityError as error:
                    _record_evaluation_invalid(
                        context,
                        phase=phase,
                        cell_id=cell_id,
                        checkpoint=checkpoint,
                        profile=profile,
                        error=error,
                    )
                    continue
                strengths = np.asarray(
                    [item.strength for item in result.evaluations],
                    dtype=np.float64,
                )
                costs = np.asarray(
                    [item.additive_cost for item in result.evaluations],
                    dtype=np.float64,
                )
                replicate_units = np.stack(
                    [
                        bundles[float(strength)].replicate_sequence_ce
                        for strength in strengths
                    ]
                )
                _record_cell(
                    context,
                    phase=phase,
                    cell_id=cell_id,
                    payload={
                        "checkpoint_seed": checkpoint.seed,
                        "checkpoint_ordinal": checkpoint.ordinal,
                        "profile": asdict(profile),
                        "calibration": {
                            key: value
                            for key, value in asdict(result).items()
                            if key != "evaluations"
                        },
                        "task_hashes": list(baseline.task_hashes),
                        "valid": result.converged,
                        "status": (
                            "selected" if result.converged else "failed"
                        ),
                        "invalid_reason": (
                            None if result.converged else result.note
                        ),
                    },
                    arrays={
                        "baseline_sequence_ce": baseline_units,
                        "evaluated_strengths": strengths,
                        "evaluated_additive_costs": costs,
                        "perturbed_replicate_sequence_ce": replicate_units,
                    },
                )
        finally:
            context.backend.release_checkpoint(checkpoint)


def _selected_strength(
    context: RunnerContext,
    checkpoint: RetainedCheckpoint,
    profile: OperatorProfile,
) -> float | None:
    payload = _load_cell(
        context,
        "calibration",
        _profile_cell_id("calibration", checkpoint, profile),
    )
    if not bool(payload.get("valid")):
        return None
    return float(payload["calibration"]["selected_strength"])


def _cost_payload(
    context: RunnerContext,
    checkpoint: RetainedCheckpoint,
    profile: OperatorProfile,
) -> dict[str, Any]:
    return _load_cell(
        context,
        "cost-check",
        _profile_cell_id("cost-check", checkpoint, profile),
    )


def _run_cost_check(
    context: RunnerContext,
    snapshot: ResumeSnapshot,
) -> None:
    phase = "cost-check"
    ordered_profiles = sorted(
        context.profiles,
        key=lambda profile: (profile.profile_id != 14, profile.profile_id),
    )
    p5_profile = PROFILE_BY_ID[14]
    if p5_profile not in context.profiles:
        raise ValueError("cost-check phase requires registered P5 profile 14")
    for checkpoint in context.checkpoints:
        pending = [
            profile
            for profile in ordered_profiles
            if not _is_reusable(
                snapshot,
                phase,
                _profile_cell_id(phase, checkpoint, profile),
            )
        ]
        if not pending:
            continue
        try:
            try:
                baseline = context.backend.evaluate(
                    checkpoint,
                    None,
                    None,
                    phase="cost_check",
                    condition_code=0,
                )
                baseline_units = baseline.averaged_sequence_ce()
            except EvaluationValidityError as error:
                for profile in pending:
                    _record_evaluation_invalid(
                        context,
                        phase=phase,
                        cell_id=_profile_cell_id(
                            phase, checkpoint, profile
                        ),
                        checkpoint=checkpoint,
                        profile=profile,
                        error=error,
                        baseline=True,
                    )
                continue
            for profile in pending:
                cell_id = _profile_cell_id(phase, checkpoint, profile)
                strength = _selected_strength(
                    context, checkpoint, profile
                )
                if strength is None:
                    _record_cell(
                        context,
                        phase=phase,
                        cell_id=cell_id,
                        payload={
                            "checkpoint_seed": checkpoint.seed,
                            "profile_id": profile.profile_id,
                            "valid": False,
                            "status": "skipped",
                            "invalid_reason": "unavailable_calibrated_strength",
                        },
                    )
                    continue
                p5_payload = None
                if profile.profile_id != 14:
                    p5_payload = _cost_payload(
                        context, checkpoint, p5_profile
                    )
                    if p5_payload.get("point_cost") is None:
                        _record_cell(
                            context,
                            phase=phase,
                            cell_id=cell_id,
                            payload={
                                "checkpoint_seed": checkpoint.seed,
                                "profile_id": profile.profile_id,
                                "valid": False,
                                "status": "skipped",
                                "invalid_reason": "p5_reference_invalid",
                            },
                        )
                        continue
                try:
                    bundle = context.backend.evaluate(
                        checkpoint,
                        profile,
                        strength,
                        phase="cost_check",
                        condition_code=0,
                    )
                except EvaluationValidityError as error:
                    _record_evaluation_invalid(
                        context,
                        phase=phase,
                        cell_id=cell_id,
                        checkpoint=checkpoint,
                        profile=profile,
                        error=error,
                    )
                    continue
                if bundle.task_hashes != baseline.task_hashes:
                    raise RuntimeError(
                        "cost setting did not reuse baseline tasks"
                    )
                check = validate_heldout_additive_cost(
                    baseline_units,
                    bundle.averaged_sequence_ce(),
                    bootstrap_seed=cost_check_bootstrap_seed(
                        checkpoint.ordinal, profile.profile_id
                    ),
                    p5_point_cost=(
                        None
                        if p5_payload is None
                        else float(p5_payload["point_cost"])
                    ),
                    p5_reference_valid=(
                        None
                        if p5_payload is None
                        else bool(p5_payload["valid"])
                    ),
                    draws=COST_CHECK_DRAWS,
                )
                _record_cell(
                    context,
                    phase=phase,
                    cell_id=cell_id,
                    payload={
                        "checkpoint_seed": checkpoint.seed,
                        "checkpoint_ordinal": checkpoint.ordinal,
                        "profile_id": profile.profile_id,
                        "strength": strength,
                        **asdict(check),
                        "task_hashes": list(baseline.task_hashes),
                        "valid": check.cost_match_valid,
                        "status": (
                            "passed" if check.cost_match_valid else "failed"
                        ),
                    },
                    arrays={
                        "baseline_sequence_ce": baseline_units,
                        "perturbed_replicate_sequence_ce": (
                            bundle.replicate_sequence_ce
                        ),
                        "perturbed_mean_sequence_ce": (
                            bundle.averaged_sequence_ce()
                        ),
                        "paired_differences": (
                            bundle.averaged_sequence_ce() - baseline_units
                        ),
                    },
                )
        finally:
            context.backend.release_checkpoint(checkpoint)


def _baseline_cell_id(
    checkpoint: RetainedCheckpoint, condition: int
) -> str:
    return (
        f"confirmatory:baseline:cp{checkpoint.ordinal:02d}:c{condition}"
    )


def _baseline_transport_valid(
    condition: int, metrics: Mapping[str, Any]
) -> bool:
    common = (
        float(metrics["accuracy"]) >= 0.95
        and float(metrics["discriminability"]) >= 0.90
        and float(metrics["match_count"]) > 0
        and float(metrics["nonmatch_count"]) > 0
    )
    if condition == 0:
        return common
    return (
        common
        and metrics["one_back_lure_accuracy"] is not None
        and float(metrics["one_back_lure_accuracy"]) >= 0.90
        and float(metrics["one_back_lure_count"]) > 0
    )


def _global_cost_valid(
    context: RunnerContext,
    profile: OperatorProfile,
) -> bool:
    p5 = PROFILE_BY_ID[14]
    for checkpoint in context.checkpoints:
        candidate = _cost_payload(context, checkpoint, profile)
        p5_cell = _cost_payload(context, checkpoint, p5)
        if not bool(candidate.get("valid")) or not bool(p5_cell.get("valid")):
            return False
    return True


def _global_neutral_confirmatory_valid(
    context: RunnerContext,
    profile: OperatorProfile,
) -> bool:
    return all(
        bool(
            _load_cell(
                context,
                "neutral-confirmatory",
                _neutral_cell_id(
                    "neutral-confirmatory",
                    checkpoint,
                    profile,
                    condition,
                ),
            ).get("valid")
        )
        for checkpoint in context.checkpoints
        for condition in (0, 1)
    )


def _global_confirmatory_valid(
    context: RunnerContext,
    profile: OperatorProfile,
) -> bool:
    p5 = PROFILE_BY_ID[14]
    required_profiles = (
        (p5,) if profile.profile_id == 14 else (profile, p5)
    )
    return all(
        bool(
            _load_cell(
                context,
                "confirmatory",
                _profile_cell_id(
                    "confirmatory",
                    checkpoint,
                    required,
                    condition,
                ),
            ).get("valid")
        )
        for checkpoint in context.checkpoints
        for required in required_profiles
        for condition in (0, 1)
    )


def _run_confirmatory(
    context: RunnerContext,
    snapshot: ResumeSnapshot,
) -> None:
    phase = "confirmatory"
    # Global baseline-transport sub-barrier.
    for checkpoint in context.checkpoints:
        try:
            for condition in (0, 1):
                cell_id = _baseline_cell_id(checkpoint, condition)
                if _is_reusable(snapshot, phase, cell_id):
                    continue
                try:
                    bundle = context.backend.evaluate(
                        checkpoint,
                        None,
                        None,
                        phase="confirmatory",
                        condition_code=condition,
                    )
                except EvaluationValidityError as error:
                    _record_evaluation_invalid(
                        context,
                        phase=phase,
                        cell_id=cell_id,
                        checkpoint=checkpoint,
                        error=error,
                        condition=condition,
                        baseline=True,
                    )
                    continue
                metrics = bundle.averaged_metrics()
                valid = _baseline_transport_valid(condition, metrics)
                _record_cell(
                    context,
                    phase=phase,
                    cell_id=cell_id,
                    payload={
                        "checkpoint_seed": checkpoint.seed,
                        "condition_code": condition,
                        "metrics": metrics,
                        "task_hashes": list(bundle.task_hashes),
                        "valid": valid,
                        "status": "passed" if valid else "failed",
                        "invalid_reason": (
                            None
                            if valid
                            else "baseline_transport_failure"
                        ),
                    },
                    arrays={
                        **_bundle_audit_arrays(bundle, prefix="native")
                    },
                )
        finally:
            context.backend.release_checkpoint(checkpoint)

    transport_valid = all(
        bool(
            _load_cell(
                context,
                phase,
                _baseline_cell_id(checkpoint, condition),
            ).get("valid")
        )
        for checkpoint in context.checkpoints
        for condition in (0, 1)
    )
    profile_validity = {
        profile.profile_id: (
            transport_valid
            and _global_cost_valid(context, profile)
            and _global_neutral_confirmatory_valid(context, profile)
            and _global_neutral_confirmatory_valid(
                context, PROFILE_BY_ID[14]
            )
        )
        for profile in context.profiles
    }
    runtime_invalid_profiles: set[int] = set()
    ordered_profiles = sorted(
        context.profiles,
        key=lambda profile: (profile.profile_id != 14, profile.profile_id),
    )
    for profile in ordered_profiles:
        for checkpoint in context.checkpoints:
            try:
                for condition in (0, 1):
                    cell_id = _profile_cell_id(
                        phase, checkpoint, profile, condition
                    )
                    if _is_reusable(snapshot, phase, cell_id):
                        continue
                    runtime_invalid = (
                        profile.profile_id in runtime_invalid_profiles
                        or (
                            profile.profile_id != 14
                            and 14 in runtime_invalid_profiles
                        )
                    )
                    if (
                        runtime_invalid
                        or not profile_validity[profile.profile_id]
                    ):
                        _record_cell(
                            context,
                            phase=phase,
                            cell_id=cell_id,
                            payload={
                                "checkpoint_seed": checkpoint.seed,
                                "profile_id": profile.profile_id,
                                "condition_code": condition,
                                "valid": False,
                                "status": "skipped",
                                "invalid_reason": (
                                    "nonfinite_p5_reference_evaluation"
                                    if (
                                        profile.profile_id != 14
                                        and 14 in runtime_invalid_profiles
                                    )
                                    else (
                                        "profile_global_evaluation_failure"
                                        if runtime_invalid
                                        else (
                                            "baseline_transport_failure"
                                            if not transport_valid
                                            else (
                                                "global_cost_validity_failure"
                                                if not _global_cost_valid(
                                                    context, profile
                                                )
                                                else "neutral_equivalence_failure"
                                            )
                                        )
                                    )
                                ),
                            },
                        )
                        continue
                    neutral = _load_cell(
                        context,
                        "neutral-confirmatory",
                        _neutral_cell_id(
                            "neutral-confirmatory",
                            checkpoint,
                            profile,
                            condition,
                        ),
                    )
                    if not bool(neutral.get("valid")):
                        _record_cell(
                            context,
                            phase=phase,
                            cell_id=cell_id,
                            payload={
                                "checkpoint_seed": checkpoint.seed,
                                "profile_id": profile.profile_id,
                                "condition_code": condition,
                                "valid": False,
                                "status": "skipped",
                                "invalid_reason": "neutral_equivalence_failure",
                            },
                        )
                        continue
                    strength = _selected_strength(
                        context, checkpoint, profile
                    )
                    if strength is None:
                        raise RuntimeError(
                            "globally valid profile has no selected strength"
                        )
                    try:
                        bundle = context.backend.evaluate(
                            checkpoint,
                            profile,
                            strength,
                            phase="confirmatory",
                            condition_code=condition,
                        )
                    except EvaluationValidityError as error:
                        runtime_invalid_profiles.add(profile.profile_id)
                        _record_evaluation_invalid(
                            context,
                            phase=phase,
                            cell_id=cell_id,
                            checkpoint=checkpoint,
                            profile=profile,
                            error=error,
                            condition=condition,
                        )
                        continue
                    baseline = _load_cell(
                        context,
                        phase,
                        _baseline_cell_id(checkpoint, condition),
                    )
                    if list(bundle.task_hashes) != baseline["task_hashes"]:
                        raise RuntimeError(
                            "outcome setting did not reuse native tasks"
                        )
                    _record_cell(
                        context,
                        phase=phase,
                        cell_id=cell_id,
                        payload={
                            "checkpoint_seed": checkpoint.seed,
                            "profile_id": profile.profile_id,
                            "condition_code": condition,
                            "strength": strength,
                            "metrics": bundle.averaged_metrics(),
                            "replicate_metrics": list(
                                bundle.replicate_metrics
                            ),
                            "task_hashes": list(bundle.task_hashes),
                            "valid": True,
                            "status": "completed",
                        },
                        arrays={
                            **_bundle_audit_arrays(
                                bundle, prefix="perturbed"
                            )
                        },
                    )
            finally:
                context.backend.release_checkpoint(checkpoint)


def _farther_strengths(
    profile: OperatorProfile, matched_strength: float
) -> tuple[float, ...]:
    neutral = profile.ordered_grid[0]
    matched_distance = abs(matched_strength - neutral)
    return tuple(
        strength
        for strength in profile.ordered_grid[1:]
        if abs(strength - neutral) > matched_distance + 1e-12
    )


def _run_dose(
    context: RunnerContext,
    snapshot: ResumeSnapshot,
) -> None:
    phase = "dose"
    runtime_invalid_profiles: set[int] = set()
    ordered_profiles = sorted(
        context.profiles,
        key=lambda profile: (profile.profile_id != 14, profile.profile_id),
    )
    for profile in ordered_profiles:
        for checkpoint in context.checkpoints:
            try:
                cell_id = _profile_cell_id(phase, checkpoint, profile)
                if _is_reusable(snapshot, phase, cell_id):
                    continue
                if (
                    profile.profile_id in runtime_invalid_profiles
                    or (
                        profile.profile_id != 14
                        and 14 in runtime_invalid_profiles
                    )
                ):
                    _record_cell(
                        context,
                        phase=phase,
                        cell_id=cell_id,
                        payload={
                            "checkpoint_seed": checkpoint.seed,
                            "profile_id": profile.profile_id,
                            "valid": False,
                            "status": "skipped",
                            "invalid_reason": (
                                "nonfinite_p5_reference_evaluation"
                                if (
                                    profile.profile_id != 14
                                    and 14 in runtime_invalid_profiles
                                )
                                else "profile_global_evaluation_failure"
                            ),
                        },
                    )
                    continue
                if not _global_confirmatory_valid(context, profile):
                    _record_cell(
                        context,
                        phase=phase,
                        cell_id=cell_id,
                        payload={
                            "checkpoint_seed": checkpoint.seed,
                            "profile_id": profile.profile_id,
                            "valid": False,
                            "status": "skipped",
                            "invalid_reason": (
                                "global_confirmatory_validity_failure"
                            ),
                        },
                    )
                    continue
                matched = _selected_strength(
                    context, checkpoint, profile
                )
                if matched is None:
                    raise RuntimeError("valid dose profile lacks strength")
                if profile.profile_id == 14:
                    strengths = profile.ordered_grid
                else:
                    farther = _farther_strengths(profile, matched)
                    if len(farther) < 2:
                        _record_cell(
                            context,
                            phase=phase,
                            cell_id=cell_id,
                            payload={
                                "checkpoint_seed": checkpoint.seed,
                                "profile_id": profile.profile_id,
                                "valid": False,
                                "status": "skipped",
                                "invalid_reason": (
                                    "insufficient_registered_grid"
                                ),
                            },
                        )
                        continue
                    strengths = (matched, farther[0], farther[1])
                arrays: list[np.ndarray] = []
                settling_arrays: list[np.ndarray] = []
                behavioral_arrays: list[np.ndarray] = []
                metrics: list[dict[str, Any]] = []
                try:
                    for strength in strengths:
                        for condition in (0, 1):
                            bundle = context.backend.evaluate(
                                checkpoint,
                                profile,
                                strength,
                                phase="confirmatory",
                                condition_code=condition,
                            )
                            arrays.append(bundle.replicate_sequence_ce)
                            if bundle.replicate_settling_steps is not None:
                                settling_arrays.append(
                                    bundle.replicate_settling_steps
                                )
                            if (
                                bundle.replicate_behavioral_correct
                                is not None
                            ):
                                behavioral_arrays.append(
                                    bundle.replicate_behavioral_correct
                                )
                            metrics.append(
                                {
                                    "strength": strength,
                                    "condition_code": condition,
                                    "metrics": bundle.averaged_metrics(),
                                }
                            )
                except EvaluationValidityError as error:
                    runtime_invalid_profiles.add(profile.profile_id)
                    _record_evaluation_invalid(
                        context,
                        phase=phase,
                        cell_id=cell_id,
                        checkpoint=checkpoint,
                        profile=profile,
                        error=error,
                    )
                    continue
                _record_cell(
                    context,
                    phase=phase,
                    cell_id=cell_id,
                    payload={
                        "checkpoint_seed": checkpoint.seed,
                        "profile_id": profile.profile_id,
                        "strengths": list(strengths),
                        "metrics": metrics,
                        "valid": True,
                        "status": "completed",
                    },
                    arrays={
                        "replicate_sequence_ce": np.stack(arrays),
                        **(
                            {
                                "replicate_settling_steps": np.stack(
                                    settling_arrays
                                ),
                                "replicate_behavioral_correct": np.stack(
                                    behavioral_arrays
                                ),
                            }
                            if settling_arrays and behavioral_arrays
                            else {}
                        ),
                    },
                )
            finally:
                context.backend.release_checkpoint(checkpoint)


def _atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return path


def _one_sample_t_summary(values: Sequence[float]) -> dict[str, Any]:
    """Return the registered one-sided t summary, including zero-SD limits."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("checkpoint values must be a finite non-empty vector")
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    result: dict[str, Any] = {
        "n_checkpoints": int(array.size),
        "mean_c2_nback": mean,
        "sample_sd": sd,
        "positive_checkpoint_count": int(np.sum(array > 0.0)),
        "positive_checkpoint_fraction": float(np.mean(array > 0.0)),
    }
    if array.size < 2:
        return {
            **result,
            "paired_dz": None,
            "t_statistic": None,
            "one_sided_p_value": None,
            "ci95_lower": None,
            "ci95_upper": None,
        }
    if sd == 0.0:
        if mean > 0.0:
            t_statistic: float | str = "infinity"
            dz: float | str = "infinity"
            p_value = 0.0
        elif mean < 0.0:
            t_statistic = "-infinity"
            dz = "-infinity"
            p_value = 1.0
        else:
            t_statistic = 0.0
            dz = 0.0
            p_value = 0.5
        return {
            **result,
            "paired_dz": dz,
            "t_statistic": t_statistic,
            "one_sided_p_value": p_value,
            "ci95_lower": mean,
            "ci95_upper": mean,
        }
    t_statistic_value = mean / (sd / np.sqrt(array.size))
    critical = float(stats.t.ppf(0.975, df=array.size - 1))
    margin = critical * sd / np.sqrt(array.size)
    return {
        **result,
        "paired_dz": mean / sd,
        "t_statistic": t_statistic_value,
        "one_sided_p_value": float(
            stats.t.sf(t_statistic_value, df=array.size - 1)
        ),
        "ci95_lower": mean - margin,
        "ci95_upper": mean + margin,
    }


def _classify_dose_selectivities(
    selectivities: Sequence[float],
) -> str:
    """Apply the registered three-point dose-ordering categories."""
    values = np.asarray(selectivities, dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(
            "dose ordering requires three finite selectivity values"
        )
    matched = float(values[0])
    farther = values[1:]
    if matched <= 0.0 or np.any(farther < 0.0):
        return "scrambled"
    if np.all(values > 0.0) and np.all(values[1:] >= values[:-1]):
        return "preserved"
    if np.any(farther > 0.0):
        return "degraded"
    return "scrambled"


def _dose_summaries(
    context: RunnerContext,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    classifications: list[dict[str, Any]] = []
    p5_curves: list[dict[str, Any]] = []
    p5 = PROFILE_BY_ID[14]
    for checkpoint in context.checkpoints:
        baseline = {
            condition: _load_cell(
                context,
                "confirmatory",
                _baseline_cell_id(checkpoint, condition),
            )
            for condition in (0, 1)
        }
        p5_cell = _load_cell(
            context,
            "dose",
            _profile_cell_id("dose", checkpoint, p5),
        )
        if bool(p5_cell.get("valid")):
            by_strength: dict[float, dict[int, dict[str, Any]]] = {}
            for row in p5_cell["metrics"]:
                by_strength.setdefault(float(row["strength"]), {})[
                    int(row["condition_code"])
                ] = row["metrics"]
            for strength, condition_metrics in sorted(by_strength.items()):
                p5_curves.append(
                    {
                        "checkpoint_seed": checkpoint.seed,
                        "strength": strength,
                        "zero_back_mean_ce": condition_metrics[0][
                            "mean_cross_entropy"
                        ],
                        "two_back_mean_ce": condition_metrics[1][
                            "mean_cross_entropy"
                        ],
                        "zero_back_additive_ce_cost": (
                            float(
                                condition_metrics[0][
                                    "mean_cross_entropy"
                                ]
                            )
                            - float(
                                baseline[0]["metrics"][
                                    "mean_cross_entropy"
                                ]
                            )
                        ),
                        "zero_back_discriminability": condition_metrics[0][
                            "discriminability"
                        ],
                        "two_back_discriminability": condition_metrics[1][
                            "discriminability"
                        ],
                    }
                )
        for profile in context.profiles:
            if profile.profile_id == 14:
                continue
            dose = _load_cell(
                context,
                "dose",
                _profile_cell_id("dose", checkpoint, profile),
            )
            if not bool(dose.get("valid")):
                classifications.append(
                    {
                        "checkpoint_seed": checkpoint.seed,
                        "profile_id": profile.profile_id,
                        "classification": "NA",
                        "reason": dose.get("invalid_reason"),
                        "load_selectivities": None,
                    }
                )
                continue
            by_strength = {}
            for row in dose["metrics"]:
                by_strength.setdefault(float(row["strength"]), {})[
                    int(row["condition_code"])
                ] = row["metrics"]
            selectivities: list[float] = []
            for strength in dose["strengths"]:
                metrics = by_strength[float(strength)]
                zero_impairment = (
                    float(
                        baseline[0]["metrics"]["discriminability"]
                    )
                    - float(metrics[0]["discriminability"])
                ) / float(baseline[0]["metrics"]["discriminability"])
                two_impairment = (
                    float(
                        baseline[1]["metrics"]["discriminability"]
                    )
                    - float(metrics[1]["discriminability"])
                ) / float(baseline[1]["metrics"]["discriminability"])
                selectivities.append(two_impairment - zero_impairment)
            classification = _classify_dose_selectivities(selectivities)
            classifications.append(
                {
                    "checkpoint_seed": checkpoint.seed,
                    "profile_id": profile.profile_id,
                    "classification": classification,
                    "reason": None,
                    "strengths": dose["strengths"],
                    "load_selectivities": selectivities,
                }
            )
    profile_classifications: list[dict[str, Any]] = []
    for profile in context.profiles:
        if profile.profile_id == 14:
            continue
        rows = [
            row
            for row in classifications
            if row["profile_id"] == profile.profile_id
        ]
        valid_rows = [
            row for row in rows if row["load_selectivities"] is not None
        ]
        if len(valid_rows) != len(context.checkpoints):
            profile_classifications.append(
                {
                    "profile_id": profile.profile_id,
                    "classification": "NA",
                    "reason": "not_all_checkpoints_valid",
                    "mean_load_selectivities": None,
                }
            )
            continue
        matrix = np.asarray(
            [row["load_selectivities"] for row in valid_rows],
            dtype=np.float64,
        )
        means = np.mean(matrix, axis=0)
        classification = _classify_dose_selectivities(means)
        profile_classifications.append(
            {
                "profile_id": profile.profile_id,
                "classification": classification,
                "reason": None,
                "mean_load_selectivities": means.tolist(),
            }
        )
    return classifications, profile_classifications, p5_curves


_LATENCY_CONTRAST_FIELDS = (
    "candidate_zero_back_rmst_change",
    "candidate_two_back_rmst_change",
    "p5_zero_back_rmst_change",
    "p5_two_back_rmst_change",
    "candidate_load_rmst_interaction",
    "p5_load_rmst_interaction",
    "excess_load_rmst_interaction",
)

_DESCRIPTIVE_MEAN_FIELDS = (
    "candidate_zero_back_discriminability_change",
    "candidate_two_back_discriminability_change",
    "p5_zero_back_discriminability_change",
    "p5_two_back_discriminability_change",
    "candidate_load_did_change",
    "p5_load_did_change",
    "excess_load_did_change",
    "candidate_load_did_impairment",
    "p5_load_did_impairment",
    "excess_load_did_impairment",
    "candidate_zero_back_additive_ce_cost",
    "candidate_two_back_additive_ce_cost",
    "p5_zero_back_additive_ce_cost",
    "p5_two_back_additive_ce_cost",
    "candidate_ce_interaction",
    "p5_ce_interaction",
    "excess_ce_interaction",
    "baseline_zero_back_failure_rate",
    "baseline_two_back_failure_rate",
    "candidate_zero_back_failure_rate",
    "candidate_two_back_failure_rate",
    "p5_zero_back_failure_rate",
    "p5_two_back_failure_rate",
    "candidate_zero_back_failure_rate_change",
    "candidate_two_back_failure_rate_change",
    "p5_zero_back_failure_rate_change",
    "p5_two_back_failure_rate_change",
    "candidate_failure_rate_interaction",
    "p5_failure_rate_interaction",
    "excess_failure_rate_interaction",
    "baseline_zero_back_fraction_settled",
    "baseline_two_back_fraction_settled",
    "candidate_zero_back_fraction_settled",
    "candidate_two_back_fraction_settled",
    "p5_zero_back_fraction_settled",
    "p5_two_back_fraction_settled",
    *_LATENCY_CONTRAST_FIELDS,
)


def _finite_metric(
    metrics: Mapping[str, Any],
    key: str,
    *,
    nested: str | None = None,
) -> float:
    source = metrics if nested is None else metrics.get(nested)
    if not isinstance(source, Mapping) or key not in source:
        location = key if nested is None else f"{nested}.{key}"
        raise ValueError(f"confirmatory metrics are missing {location}")
    value = float(source[key])
    if not np.isfinite(value):
        location = key if nested is None else f"{nested}.{key}"
        raise ValueError(f"confirmatory metric {location} must be finite")
    return value


def _checkpoint_descriptive_outcomes(
    baseline: Mapping[int, Mapping[str, Any]],
    candidate: Mapping[int, Mapping[str, Any]],
    p5: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return all registered descriptive outcomes for one checkpoint."""
    if set(baseline) != {0, 1} or set(candidate) != {0, 1} or set(p5) != {
        0,
        1,
    }:
        raise ValueError("descriptive outcomes require both N-back conditions")

    baseline_d = {
        condition: _finite_metric(metrics, "discriminability")
        for condition, metrics in baseline.items()
    }
    candidate_d = {
        condition: _finite_metric(metrics, "discriminability")
        for condition, metrics in candidate.items()
    }
    p5_d = {
        condition: _finite_metric(metrics, "discriminability")
        for condition, metrics in p5.items()
    }
    candidate_d_change = {
        condition: candidate_d[condition] - baseline_d[condition]
        for condition in (0, 1)
    }
    p5_d_change = {
        condition: p5_d[condition] - baseline_d[condition]
        for condition in (0, 1)
    }
    candidate_load_did_change = (
        candidate_d_change[1] - candidate_d_change[0]
    )
    p5_load_did_change = p5_d_change[1] - p5_d_change[0]

    baseline_ce = {
        condition: _finite_metric(metrics, "mean_cross_entropy")
        for condition, metrics in baseline.items()
    }
    candidate_ce = {
        condition: _finite_metric(metrics, "mean_cross_entropy")
        for condition, metrics in candidate.items()
    }
    p5_ce = {
        condition: _finite_metric(metrics, "mean_cross_entropy")
        for condition, metrics in p5.items()
    }
    candidate_ce_cost = {
        condition: candidate_ce[condition] - baseline_ce[condition]
        for condition in (0, 1)
    }
    p5_ce_cost = {
        condition: p5_ce[condition] - baseline_ce[condition]
        for condition in (0, 1)
    }

    baseline_failure = {
        condition: _finite_metric(metrics, "failure_rate")
        for condition, metrics in baseline.items()
    }
    candidate_failure = {
        condition: _finite_metric(metrics, "failure_rate")
        for condition, metrics in candidate.items()
    }
    p5_failure = {
        condition: _finite_metric(metrics, "failure_rate")
        for condition, metrics in p5.items()
    }
    candidate_failure_change = {
        condition: candidate_failure[condition] - baseline_failure[condition]
        for condition in (0, 1)
    }
    p5_failure_change = {
        condition: p5_failure[condition] - baseline_failure[condition]
        for condition in (0, 1)
    }

    baseline_fraction = {
        condition: _finite_metric(
            metrics, "fraction_settled", nested="settling_all"
        )
        for condition, metrics in baseline.items()
    }
    candidate_fraction = {
        condition: _finite_metric(
            metrics, "fraction_settled", nested="settling_all"
        )
        for condition, metrics in candidate.items()
    }
    p5_fraction = {
        condition: _finite_metric(
            metrics, "fraction_settled", nested="settling_all"
        )
        for condition, metrics in p5.items()
    }
    fractions = (
        *baseline_fraction.values(),
        *candidate_fraction.values(),
        *p5_fraction.values(),
    )
    if any(not 0.0 <= value <= 1.0 for value in fractions):
        raise ValueError("fraction_settled must lie in [0, 1]")
    joint_latency_valid = all(value >= 0.80 for value in fractions)

    baseline_rmst = {
        condition: _finite_metric(
            metrics,
            "restricted_mean_settling_steps",
            nested="settling_all",
        )
        for condition, metrics in baseline.items()
    }
    candidate_rmst = {
        condition: _finite_metric(
            metrics,
            "restricted_mean_settling_steps",
            nested="settling_all",
        )
        for condition, metrics in candidate.items()
    }
    p5_rmst = {
        condition: _finite_metric(
            metrics,
            "restricted_mean_settling_steps",
            nested="settling_all",
        )
        for condition, metrics in p5.items()
    }

    candidate_ce_interaction = candidate_ce_cost[1] - candidate_ce_cost[0]
    p5_ce_interaction = p5_ce_cost[1] - p5_ce_cost[0]
    candidate_failure_interaction = (
        candidate_failure_change[1] - candidate_failure_change[0]
    )
    p5_failure_interaction = (
        p5_failure_change[1] - p5_failure_change[0]
    )
    result: dict[str, Any] = {
        "baseline_zero_back_discriminability": baseline_d[0],
        "baseline_two_back_discriminability": baseline_d[1],
        "candidate_zero_back_discriminability": candidate_d[0],
        "candidate_two_back_discriminability": candidate_d[1],
        "p5_zero_back_discriminability": p5_d[0],
        "p5_two_back_discriminability": p5_d[1],
        "candidate_zero_back_discriminability_change": candidate_d_change[0],
        "candidate_two_back_discriminability_change": candidate_d_change[1],
        "p5_zero_back_discriminability_change": p5_d_change[0],
        "p5_two_back_discriminability_change": p5_d_change[1],
        "candidate_load_did_change": candidate_load_did_change,
        "p5_load_did_change": p5_load_did_change,
        "excess_load_did_change": (
            candidate_load_did_change - p5_load_did_change
        ),
        "candidate_load_did_impairment": -candidate_load_did_change,
        "p5_load_did_impairment": -p5_load_did_change,
        "excess_load_did_impairment": (
            -candidate_load_did_change + p5_load_did_change
        ),
        "baseline_zero_back_mean_ce": baseline_ce[0],
        "baseline_two_back_mean_ce": baseline_ce[1],
        "candidate_zero_back_mean_ce": candidate_ce[0],
        "candidate_two_back_mean_ce": candidate_ce[1],
        "p5_zero_back_mean_ce": p5_ce[0],
        "p5_two_back_mean_ce": p5_ce[1],
        "candidate_zero_back_additive_ce_cost": candidate_ce_cost[0],
        "candidate_two_back_additive_ce_cost": candidate_ce_cost[1],
        "p5_zero_back_additive_ce_cost": p5_ce_cost[0],
        "p5_two_back_additive_ce_cost": p5_ce_cost[1],
        "candidate_ce_interaction": candidate_ce_interaction,
        "p5_ce_interaction": p5_ce_interaction,
        "excess_ce_interaction": (
            candidate_ce_interaction - p5_ce_interaction
        ),
        "baseline_zero_back_failure_rate": baseline_failure[0],
        "baseline_two_back_failure_rate": baseline_failure[1],
        "candidate_zero_back_failure_rate": candidate_failure[0],
        "candidate_two_back_failure_rate": candidate_failure[1],
        "p5_zero_back_failure_rate": p5_failure[0],
        "p5_two_back_failure_rate": p5_failure[1],
        "candidate_zero_back_failure_rate_change": (
            candidate_failure_change[0]
        ),
        "candidate_two_back_failure_rate_change": (
            candidate_failure_change[1]
        ),
        "p5_zero_back_failure_rate_change": p5_failure_change[0],
        "p5_two_back_failure_rate_change": p5_failure_change[1],
        "candidate_failure_rate_interaction": candidate_failure_interaction,
        "p5_failure_rate_interaction": p5_failure_interaction,
        "excess_failure_rate_interaction": (
            candidate_failure_interaction - p5_failure_interaction
        ),
        "baseline_zero_back_fraction_settled": baseline_fraction[0],
        "baseline_two_back_fraction_settled": baseline_fraction[1],
        "candidate_zero_back_fraction_settled": candidate_fraction[0],
        "candidate_two_back_fraction_settled": candidate_fraction[1],
        "p5_zero_back_fraction_settled": p5_fraction[0],
        "p5_two_back_fraction_settled": p5_fraction[1],
        "baseline_zero_back_rmst": baseline_rmst[0],
        "baseline_two_back_rmst": baseline_rmst[1],
        "candidate_zero_back_rmst": candidate_rmst[0],
        "candidate_two_back_rmst": candidate_rmst[1],
        "p5_zero_back_rmst": p5_rmst[0],
        "p5_two_back_rmst": p5_rmst[1],
        "joint_latency_valid": joint_latency_valid,
        "dynamics_outcome": (
            "latency" if joint_latency_valid else "failure_rate"
        ),
        "latency_invalid_reason": (
            None if joint_latency_valid else "fraction_settled_below_0_80"
        ),
    }
    if joint_latency_valid:
        candidate_rmst_change = {
            condition: candidate_rmst[condition] - baseline_rmst[condition]
            for condition in (0, 1)
        }
        p5_rmst_change = {
            condition: p5_rmst[condition] - baseline_rmst[condition]
            for condition in (0, 1)
        }
        candidate_rmst_interaction = (
            candidate_rmst_change[1] - candidate_rmst_change[0]
        )
        p5_rmst_interaction = p5_rmst_change[1] - p5_rmst_change[0]
        result.update(
            {
                "candidate_zero_back_rmst_change": candidate_rmst_change[0],
                "candidate_two_back_rmst_change": candidate_rmst_change[1],
                "p5_zero_back_rmst_change": p5_rmst_change[0],
                "p5_two_back_rmst_change": p5_rmst_change[1],
                "candidate_load_rmst_interaction": (
                    candidate_rmst_interaction
                ),
                "p5_load_rmst_interaction": p5_rmst_interaction,
                "excess_load_rmst_interaction": (
                    candidate_rmst_interaction - p5_rmst_interaction
                ),
            }
        )
    else:
        result.update({field: None for field in _LATENCY_CONTRAST_FIELDS})
    return result


def _profile_descriptive_outcomes(
    checkpoint_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Average registered descriptive checkpoint outcomes within profile."""
    if not checkpoint_rows:
        raise ValueError("profile descriptive summaries require checkpoints")
    result: dict[str, Any] = {}
    for field in _DESCRIPTIVE_MEAN_FIELDS:
        values = [row.get(field) for row in checkpoint_rows]
        if any(value is None for value in values):
            result[f"mean_{field}"] = None
            continue
        numeric = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"checkpoint field {field} must be finite")
        result[f"mean_{field}"] = float(np.mean(numeric))
    latency_count = sum(
        bool(row.get("joint_latency_valid")) for row in checkpoint_rows
    )
    result.update(
        {
            "n_latency_valid_checkpoints": latency_count,
            "joint_latency_valid_all_checkpoints": (
                latency_count == len(checkpoint_rows)
            ),
            "dynamics_outcome": (
                "latency"
                if latency_count == len(checkpoint_rows)
                else "failure_rate"
            ),
        }
    )
    return result


def _run_finalize(
    context: RunnerContext,
    snapshot: ResumeSnapshot,
) -> None:
    phase = "finalize"
    cell_id = "finalize:summary"
    if _is_reusable(snapshot, phase, cell_id):
        return
    p5 = PROFILE_BY_ID[14]
    rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    for profile in context.profiles:
        if profile.profile_id == 14:
            continue
        values: list[float] = []
        valid = True
        for checkpoint in context.checkpoints:
            candidate_metrics = {}
            p5_metrics = {}
            baseline_metrics = {}
            for condition in (0, 1):
                candidate = _load_cell(
                    context,
                    "confirmatory",
                    _profile_cell_id(
                        "confirmatory", checkpoint, profile, condition
                    ),
                )
                comparator = _load_cell(
                    context,
                    "confirmatory",
                    _profile_cell_id(
                        "confirmatory", checkpoint, p5, condition
                    ),
                )
                baseline = _load_cell(
                    context,
                    "confirmatory",
                    _baseline_cell_id(checkpoint, condition),
                )
                if not all(
                    bool(payload.get("valid"))
                    for payload in (candidate, comparator, baseline)
                ):
                    valid = False
                    break
                candidate_metrics[condition] = candidate["metrics"]
                p5_metrics[condition] = comparator["metrics"]
                baseline_metrics[condition] = baseline["metrics"]
            if not valid:
                break
            contrast = candidate_vs_p5_load_contrast(
                baseline_zero_back=float(
                    baseline_metrics[0]["discriminability"]
                ),
                baseline_two_back=float(
                    baseline_metrics[1]["discriminability"]
                ),
                candidate_zero_back=float(
                    candidate_metrics[0]["discriminability"]
                ),
                candidate_two_back=float(
                    candidate_metrics[1]["discriminability"]
                ),
                p5_zero_back=float(p5_metrics[0]["discriminability"]),
                p5_two_back=float(p5_metrics[1]["discriminability"]),
            )
            values.append(contrast.c2_nback)
            rows.append(
                {
                    "profile_id": profile.profile_id,
                    "checkpoint_seed": checkpoint.seed,
                    **asdict(contrast),
                    **_checkpoint_descriptive_outcomes(
                        baseline_metrics,
                        candidate_metrics,
                        p5_metrics,
                    ),
                }
            )
        if valid and len(values) == len(context.checkpoints):
            profile_checkpoint_rows = [
                row
                for row in rows
                if row["profile_id"] == profile.profile_id
            ]
            profiles.append(
                {
                    "profile_id": profile.profile_id,
                    "profile_class": profile.profile_class,
                    **_one_sample_t_summary(values),
                    **_profile_descriptive_outcomes(
                        profile_checkpoint_rows
                    ),
                    "registered_sign_criterion_met": (
                        int(np.sum(np.asarray(values) > 0.0)) >= 8
                        if (
                            profile.profile_id
                            in CONFIRMATORY_PROFILE_IDS
                            and len(values) == 10
                        )
                        else None
                    ),
                    "valid": True,
                }
            )
        else:
            profiles.append(
                {
                    "profile_id": profile.profile_id,
                    "profile_class": profile.profile_class,
                    "n_checkpoints": len(values),
                    "registered_sign_criterion_met": None,
                    "valid": False,
                    "invalid_reason": "not_testable_validity",
                }
            )
    metrics_dir = context.paths.root / "metrics"
    checkpoint_fields = (
        list(rows[0])
        if rows
        else [
            "profile_id",
            "checkpoint_seed",
            "candidate_zero_back_impairment",
            "candidate_two_back_impairment",
            "candidate_load_selectivity",
            "p5_zero_back_impairment",
            "p5_two_back_impairment",
            "p5_load_selectivity",
            "c2_nback",
        ]
    )
    csv_path = _atomic_write_csv(
        metrics_dir / "nback_c2_checkpoint.csv",
        rows,
        checkpoint_fields,
    )
    profile_fields: list[str] = []
    for profile_row in profiles:
        for field in profile_row:
            if field not in profile_fields:
                profile_fields.append(field)
    profile_csv_path = _atomic_write_csv(
        metrics_dir / "nback_c2_profile.csv",
        profiles,
        profile_fields,
    )
    (
        dose_ordering_checkpoints,
        dose_ordering_profiles,
        p5_curve,
    ) = _dose_summaries(context)
    payload = {
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "design_hash": canonical_design_hash(context.design),
        "confirmatory_profile_ids": list(CONFIRMATORY_PROFILE_IDS),
        "c2_p_value_role": (
            "diagnostic_component_not_independent_confirmatory_discovery"
        ),
        "profiles": profiles,
        "dose_ordering_checkpoints": dose_ordering_checkpoints,
        "dose_ordering_profiles": dose_ordering_profiles,
        "p5_registered_strength_curve": p5_curve,
        "valid": True,
        "status": "completed",
    }
    json_path = metrics_dir / "nback_c2_summary.json"
    atomic_write_json(json_path, payload)
    _record_completed_artifacts(
        context,
        phase=phase,
        cell_id=cell_id,
        artifacts=[json_path, csv_path, profile_csv_path],
        metadata={"valid": True, "status": "completed"},
    )


def run_phase(context: RunnerContext, phase: str) -> ResumeSnapshot:
    """Run or safely resume exactly one registered phase."""
    _validate_runtime_hashes(context)
    if phase not in PHASE_ORDER:
        raise ValueError(f"phase must be one of {PHASE_ORDER}")
    snapshot = begin_phase(
        context.paths.manifest_path,
        context.paths.state_path,
        phase,
    )
    if snapshot.phase_statuses[phase] == "completed":
        return snapshot
    timing_cell = f"{phase}:timing"
    if _is_reusable(snapshot, phase, timing_cell):
        return complete_phase(
            context.paths.manifest_path,
            context.paths.state_path,
            phase,
        )
    synchronize_cuda = context.device.device.type == "cuda"
    if synchronize_cuda:
        torch.cuda.synchronize(context.device.device)
    attempt = start_phase_attempt(
        context.paths.manifest_path,
        context.paths.state_path,
        phase=phase,
        device=context.device.description,
        cuda_synchronized=synchronize_cuda,
    )
    runtime = PhaseTimingRuntime(
        phase=phase,
        attempt_id=attempt.attempt_id,
        started=time.perf_counter(),
        accounted=0.0,
        synchronize_cuda=synchronize_cuda,
    )
    context.timing_runtime.active = runtime
    try:
        if phase in {"neutral-calibration", "neutral-confirmatory"}:
            _run_neutral_phase(context, phase, snapshot)
        elif phase == "calibration":
            _run_calibration(context, snapshot)
        elif phase == "cost-check":
            _run_cost_check(context, snapshot)
        elif phase == "confirmatory":
            _run_confirmatory(context, snapshot)
        elif phase == "dose":
            _run_dose(context, snapshot)
        elif phase == "finalize":
            _run_finalize(context, snapshot)
        if synchronize_cuda:
            torch.cuda.synchronize(context.device.device)
        elapsed = time.perf_counter() - runtime.started
        final_delta = elapsed - runtime.accounted
        if not np.isfinite(final_delta) or final_delta < 0.0:
            raise RuntimeError("phase timing clock produced an invalid delta")
        finished = finish_phase_attempt(
            context.paths.manifest_path,
            context.paths.state_path,
            phase=phase,
            attempt_id=runtime.attempt_id,
            timing_delta_seconds=final_delta,
        )
    except BaseException:
        context.timing_runtime.active = None
        raise
    context.timing_runtime.active = None
    timing = finished.phase_timings[phase]
    if not _is_reusable(finished, phase, timing_cell):
        interrupted = sum(
            attempt_row["status"] == "interrupted"
            for attempt_row in timing["attempts"]
        )
        _record_cell(
            context,
            phase=phase,
            cell_id=timing_cell,
            payload={
                "phase": phase,
                "wall_time_seconds": timing["accumulated_seconds"],
                "active_execution_wall_time_seconds": (
                    timing["accumulated_seconds"]
                ),
                "attempt_count": len(timing["attempts"]),
                "interrupted_attempt_count": interrupted,
                "attempts": timing["attempts"],
                "cuda_synchronized": synchronize_cuda,
                "device": context.device.description,
                "valid": True,
                "status": "completed",
            },
        )
    return complete_phase(
        context.paths.manifest_path,
        context.paths.state_path,
        phase,
    )


def _profile_manifest_payload(
    profiles: Sequence[OperatorProfile],
) -> dict[str, Any]:
    return {
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "profiles": [asdict(profile) for profile in profiles],
        "confirmatory_profile_ids": list(CONFIRMATORY_PROFILE_IDS),
    }


def _seed_manifest_payload(
    checkpoints: Sequence[RetainedCheckpoint],
    profiles: Sequence[OperatorProfile],
) -> dict[str, Any]:
    task_rows = []
    p5_rows = []
    bootstrap_rows = []
    for checkpoint in checkpoints:
        for phase, bank in TASK_BANKS.items():
            for condition in bank.condition_codes:
                for batch in range(bank.n_batches):
                    task_rows.append(
                        {
                            "phase": phase,
                            "checkpoint_ordinal": checkpoint.ordinal,
                            "condition_code": condition,
                            "batch_index": batch,
                            "task_seed": task_seed(
                                phase,
                                checkpoint.ordinal,
                                condition,
                                batch,
                            ),
                        }
                    )
                    for replicate in range(3):
                        p5_rows.append(
                            {
                                "phase": phase,
                                "checkpoint_ordinal": checkpoint.ordinal,
                                "condition_code": condition,
                                "replicate_ordinal": replicate,
                                "replicate_label": P5_REPLICATE_LABELS[
                                    replicate
                                ],
                                "batch_index": batch,
                                "generator_seed": p5_generator_seed(
                                    phase,
                                    checkpoint.ordinal,
                                    condition,
                                    replicate,
                                    batch,
                                ),
                            }
                        )
        for profile in profiles:
            bootstrap_rows.append(
                {
                    "checkpoint_ordinal": checkpoint.ordinal,
                    "profile_id": profile.profile_id,
                    "bootstrap_seed": cost_check_bootstrap_seed(
                        checkpoint.ordinal, profile.profile_id
                    ),
                }
            )
    return {
        "task_rows": task_rows,
        "p2_literal_vector_seeds": list(P2_VECTOR_SEEDS),
        "p5_rows": p5_rows,
        "cost_check_bootstrap_rows": bootstrap_rows,
    }


def _write_or_validate_json(path: Path, payload: Mapping[str, Any]) -> None:
    normalized = json.loads(
        json.dumps(
            dict(payload),
            sort_keys=True,
            allow_nan=False,
        )
    )
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            persisted = json.load(handle)
        if persisted != normalized:
            raise PerturbationStateError(
                f"immutable auxiliary manifest changed: {path}"
            )
    else:
        atomic_write_json(path, normalized)


def _implementation_source_paths() -> dict[str, Path]:
    module_dir = Path(__file__).resolve().parent
    return {
        "runner": Path(__file__).resolve(),
        "calibration": module_dir / "nback_additive_calibration.py",
        "cost_precision": module_dir / "nback_additive_cost_precision.py",
        "metrics": module_dir / "nback_metrics.py",
        "adapter": module_dir / "nback_perturbation.py",
        "state": module_dir / "nback_perturbation_state.py",
        "perturbation_operators": module_dir / "perturbation_operators.py",
        "nback_task": module_dir / "nback_task.py",
        "model": module_dir / "model.py",
        "training_utils": module_dir / "training_utils.py",
        "config": module_dir / "config.py",
        "device": module_dir / "device.py",
        "outcomes": module_dir / "nback_additive_outcomes.py",
    }


def _hashed_path_records(
    paths: Mapping[str, str | Path],
) -> dict[str, dict[str, str]]:
    return {
        name: {
            "path": str(Path(path).resolve()),
            "sha256": sha256_file(Path(path).resolve()),
        }
        for name, path in sorted(paths.items())
    }


def _validate_runtime_hashes(context: RunnerContext) -> None:
    """Revalidate every frozen computation input before every phase call."""
    integrity = context.design.get("runtime_integrity")
    if not isinstance(integrity, Mapping):
        raise PerturbationStateError("runtime integrity manifest is missing")
    for category in (
        "implementation_sources",
        "auxiliary_manifests",
        "external_prerequisites",
        "checkpoints",
    ):
        records = integrity.get(category)
        if not isinstance(records, Mapping):
            raise PerturbationStateError(
                f"runtime integrity category is missing: {category}"
            )
        for name, raw_record in records.items():
            if not isinstance(raw_record, Mapping):
                raise PerturbationStateError(
                    f"invalid runtime integrity record: {category}.{name}"
                )
            path = Path(str(raw_record.get("path", "")))
            expected = str(raw_record.get("sha256", ""))
            if not path.is_file() or sha256_file(path) != expected:
                raise PerturbationStateError(
                    "frozen runtime input changed: "
                    f"{category}.{name} ({path})"
                )
    implementation_records = integrity["implementation_sources"]
    current_implementation_hashes = {
        str(name): str(record["sha256"])
        for name, record in implementation_records.items()
    }
    expected_design_hash = str(
        context.design.get("implementation_design_hash", "")
    )
    if (
        canonical_design_hash(current_implementation_hashes)
        != expected_design_hash
    ):
        raise PerturbationStateError(
            "implementation source design hash mismatch"
        )


def initialize_runner_context(
    *,
    config: dict[str, Any],
    checkpoints: Sequence[RetainedCheckpoint],
    output_dir: str | Path,
    backend: EvaluationBackend,
    device: SelectedDevice,
    design_evidence: Mapping[str, Any],
    runtime_prerequisites: Mapping[str, str | Path] | None = None,
    profiles: Sequence[OperatorProfile] = OPERATOR_PROFILES,
) -> RunnerContext:
    """Create or validate the complete immutable run identity."""
    resolved_checkpoints = tuple(checkpoints)
    resolved_profiles = tuple(profiles)
    if (
        len({profile.profile_id for profile in resolved_profiles})
        != len(resolved_profiles)
        or any(
            PROFILE_BY_ID.get(profile.profile_id) != profile
            for profile in resolved_profiles
        )
    ):
        raise ValueError("profiles must be unique exact registered rows")
    if 14 not in {profile.profile_id for profile in resolved_profiles}:
        raise ValueError("registered P5 profile 14 is required")
    paths = _run_paths(output_dir)
    profile_path = paths.manifest_dir / "profile_manifest.json"
    seed_path = paths.manifest_dir / "seed_manifest.json"
    _write_or_validate_json(
        profile_path, _profile_manifest_payload(resolved_profiles)
    )
    _write_or_validate_json(
        seed_path,
        _seed_manifest_payload(resolved_checkpoints, resolved_profiles),
    )
    implementation_records = _hashed_path_records(
        _implementation_source_paths()
    )
    implementation_hashes = {
        name: record["sha256"]
        for name, record in implementation_records.items()
    }
    implementation_design_hash = canonical_design_hash(
        implementation_hashes
    )
    auxiliary_records = _hashed_path_records(
        {
            "profile_manifest": profile_path,
            "seed_manifest": seed_path,
        }
    )
    external_records = _hashed_path_records(runtime_prerequisites or {})
    checkpoint_records = _hashed_path_records(
        {
            str(checkpoint.seed): checkpoint.path
            for checkpoint in resolved_checkpoints
        }
    )
    design = {
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "phase_order": list(PHASE_ORDER),
        "profiles": [asdict(profile) for profile in resolved_profiles],
        "checkpoint_order": [
            {
                "ordinal": checkpoint.ordinal,
                "seed": checkpoint.seed,
            }
            for checkpoint in resolved_checkpoints
        ],
        "task_banks": {
            phase: asdict(bank) for phase, bank in TASK_BANKS.items()
        },
        "constants": {
            "calibration_target": CALIBRATION_TARGET,
            "calibration_tolerance": CALIBRATION_TOLERANCE,
            "cost_check_sequences": COST_CHECK_SEQUENCES,
            "cost_check_draws": COST_CHECK_DRAWS,
        },
        "profile_manifest_sha256": sha256_file(profile_path),
        "seed_manifest_sha256": sha256_file(seed_path),
        "implementation_source_sha256": implementation_hashes,
        "implementation_design_hash": implementation_design_hash,
        "runtime_integrity": {
            "implementation_sources": implementation_records,
            "auxiliary_manifests": auxiliary_records,
            "external_prerequisites": external_records,
            "checkpoints": checkpoint_records,
        },
        "device": device.description,
        **dict(design_evidence),
    }
    cells = expected_cells(resolved_checkpoints, resolved_profiles)
    initialize_or_resume_run(
        paths.manifest_path,
        paths.state_path,
        design=design,
        checkpoints={
            str(checkpoint.seed): checkpoint.path
            for checkpoint in resolved_checkpoints
        },
        expected_cells=cells,
        phase_order=PHASE_ORDER,
    )
    return RunnerContext(
        config=config,
        checkpoints=resolved_checkpoints,
        profiles=resolved_profiles,
        paths=paths,
        design=design,
        expected_cells=cells,
        backend=backend,
        device=device,
    )


def _resolve_repo_path(repo_root: Path, raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _validate_frozen_config(config: Mapping[str, Any]) -> None:
    task = config["task"]
    model = config["model"]
    perturbation = config["perturbation"]
    expected_task = {
        "task_type": "n_back",
        "n_stimuli": 6,
        "n_back": 0,
        "sequence_items": 20,
        "stimulus_steps": 3,
        "interstimulus_steps": 6,
        "scored_start_item": 2,
        "target_identity": 0,
        "matches_per_sequence": 6,
        "min_one_back_lures": 3,
        "batch_size": 128,
    }
    expected_model = {
        "hidden_size": 64,
        "dt": 20.0,
        "tau": 100.0,
        "activation": "tanh",
        "recurrent_noise_std": 0.0,
    }
    expected_perturbation = {
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "n_cost_check": 1024,
        "calibration_target": 0.050,
        "calibration_tolerance": 0.0025,
        "calibration_sequences": 512,
        "cost_check_sequences": 1024,
        "confirmatory_sequences_per_condition": 1024,
        "bootstrap_draws": 10_000,
        "maximum_half_width": 0.005,
        "maximum_p5_gap": 0.005,
    }
    for section_name, section, expected_values in (
        ("task", task, expected_task),
        ("model", model, expected_model),
        ("perturbation", perturbation, expected_perturbation),
    ):
        for field, expected in expected_values.items():
            if section.get(field) != expected:
                raise ValueError(
                    f"frozen {section_name}.{field} does not match registration"
                )
    if [float(value) for value in perturbation.get("cost_band", [])] != [
        0.040,
        0.060,
    ]:
        raise ValueError(
            "frozen perturbation.cost_band does not match registration"
        )


def load_frozen_runner_context(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    repo_root: str | Path = ".",
    device_override: str | None = None,
    output_dir: str | Path | None = None,
) -> RunnerContext:
    """Validate every frozen input and initialize the real execution context."""
    root = Path(repo_root).resolve()
    resolved_config = _resolve_repo_path(root, config_path)
    config = load_config(resolved_config)
    _validate_frozen_config(config)
    if device_override is not None:
        config["training"]["device"] = device_override
    device = select_device(config["training"].get("device", "auto"))
    perturbation = config["perturbation"]
    if perturbation["preregistration_commit"] != PREREGISTRATION_COMMIT:
        raise ValueError("configuration preregistration commit is not frozen")
    manifest_path = _resolve_repo_path(
        root, perturbation["checkpoint_manifest"]
    )
    precision_summary_path = _resolve_repo_path(
        root, perturbation["precision_summary"]
    )
    precision_arrays_path = _resolve_repo_path(
        root, perturbation["precision_arrays"]
    )
    preregistration_path = _resolve_repo_path(
        root, DEFAULT_PREREGISTRATION
    )
    if (
        sha256_file(preregistration_path)
        != str(perturbation["preregistration_sha256"])
    ):
        raise PerturbationStateError(
            "frozen preregistration file hash mismatch"
        )
    expected_hashes = {
        manifest_path: perturbation["checkpoint_manifest_sha256"],
        precision_summary_path: perturbation["precision_summary_sha256"],
        precision_arrays_path: perturbation["precision_arrays_sha256"],
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != str(expected):
            raise PerturbationStateError(
                f"frozen input hash mismatch: {path}"
            )
    with precision_summary_path.open(encoding="utf-8") as handle:
        precision = json.load(handle)
    if (
        not bool(precision.get("passed"))
        or int(precision["planning"]["n_cost_check"]) != 1024
        or int(perturbation["n_cost_check"]) != 1024
    ):
        raise PerturbationStateError(
            "additive precision prerequisite is not the frozen pass"
        )
    checkpoints = load_retained_checkpoints(
        manifest_path, repo_root=root
    )
    recorded_checkpoint_hashes = precision["checkpoint_sha256"]
    for checkpoint in checkpoints:
        if (
            sha256_file(checkpoint.path)
            != recorded_checkpoint_hashes[str(checkpoint.seed)]
        ):
            raise PerturbationStateError(
                f"checkpoint hash mismatch: {checkpoint.seed}"
            )
    if float(config["model"].get("recurrent_noise_std", 0.0)) != 0.0:
        raise ValueError("frozen model recurrent_noise_std must be zero")
    if (
        int(perturbation["calibration_sequences"]) != 512
        or int(perturbation["cost_check_sequences"]) != 1024
        or int(perturbation["confirmatory_sequences_per_condition"])
        != 1024
        or int(perturbation["bootstrap_draws"]) != COST_CHECK_DRAWS
    ):
        raise ValueError("configuration phase sizes do not match registration")
    resolved_output = _resolve_repo_path(
        root,
        output_dir
        if output_dir is not None
        else perturbation["output_dir"],
    )
    backend = TorchNBackBackend(config, device.device)
    return initialize_runner_context(
        config=config,
        checkpoints=checkpoints,
        output_dir=resolved_output,
        backend=backend,
        device=device,
        runtime_prerequisites={
            "preregistration": preregistration_path,
            "screened_pool_manifest": manifest_path,
            "precision_summary": precision_summary_path,
            "precision_arrays": precision_arrays_path,
            "config": resolved_config,
        },
        design_evidence={
            "preregistration_path": str(preregistration_path),
            "preregistration_sha256": sha256_file(preregistration_path),
            "screened_pool_manifest_path": str(manifest_path),
            "screened_pool_manifest_sha256": sha256_file(manifest_path),
            "precision_summary_path": str(precision_summary_path),
            "precision_summary_sha256": sha256_file(
                precision_summary_path
            ),
            "precision_arrays_path": str(precision_arrays_path),
            "precision_arrays_sha256": sha256_file(
                precision_arrays_path
            ),
            "config_path": str(resolved_config),
            "config_sha256": sha256_file(resolved_config),
        },
    )


def main() -> None:
    """Run one explicitly requested registered phase."""
    parser = argparse.ArgumentParser(
        description="Run frozen additive N-back perturbation phases."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--phase", choices=PHASE_ORDER, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    context = load_frozen_runner_context(
        args.config,
        repo_root=Path.cwd(),
        device_override=args.device,
        output_dir=args.output_dir,
    )
    snapshot = run_phase(context, args.phase)
    print(f"phase={args.phase}")
    print(f"status={snapshot.phase_statuses[args.phase]}")
    print(f"device={context.device.description}")
    print(f"state={context.paths.state_path}")


if __name__ == "__main__":
    main()
