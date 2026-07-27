"""Focused synthetic tests for the phased additive N-back runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import wm_rnn.nback_additive_perturbation_runner as runner_module

from wm_rnn.config import load_config
from wm_rnn.device import SelectedDevice
from wm_rnn.nback_additive_calibration import (
    OPERATOR_PROFILES,
    PROFILE_BY_ID,
    TASK_BANKS,
    OperatorProfile,
)
from wm_rnn.nback_additive_cost_precision import RetainedCheckpoint
from wm_rnn.nback_additive_perturbation_runner import (
    PHASE_ORDER,
    EvaluationBundle,
    EvaluationValidityError,
    NeutralEquivalence,
    _checkpoint_descriptive_outcomes,
    _classify_dose_selectivities,
    _combine_nback_batches,
    _implementation_source_paths,
    _profile_parameters,
    _one_sample_t_summary,
    _validate_frozen_config,
    expected_cells,
    initialize_runner_context,
    run_phase,
)
from wm_rnn.nback_perturbation import build_nback_operator
from wm_rnn.nback_metrics import nback_metrics
from wm_rnn.nback_perturbation_state import (
    PerturbationStateError,
    begin_phase,
    canonical_design_hash,
    record_completed_cell,
    sha256_file,
)
from wm_rnn.nback_task import NBackTaskConfig, generate_nback_batch
from wm_rnn.training_utils import fresh_model


class FakeBackend:
    """Deterministic synthetic backend with no model or operator calls."""

    def __init__(
        self,
        *,
        neutral_failures: set[int] | None = None,
        neutral_confirmatory_failures: set[tuple[int, int]] | None = None,
        neutral_checkpoint_failures: (
            set[tuple[str, int, int, int]] | None
        ) = None,
        evaluation_failures: (
            set[tuple[str, int, int | None, int, float | None]] | None
        ) = None,
        transport_failure: bool = False,
    ) -> None:
        self.neutral_failures = neutral_failures or set()
        self.neutral_confirmatory_failures = (
            neutral_confirmatory_failures or set()
        )
        self.neutral_checkpoint_failures = (
            neutral_checkpoint_failures or set()
        )
        self.evaluation_failures = evaluation_failures or set()
        self.transport_failure = transport_failure
        self.calls: list[tuple[object, ...]] = []

    @staticmethod
    def _metrics(
        condition: int,
        *,
        discriminability: float,
        transport_failure: bool,
    ) -> dict[str, object]:
        accuracy = 0.80 if transport_failure else 0.99
        return {
            "condition": f"{condition * 2}-back",
            "accuracy": accuracy,
            "mean_cross_entropy": 0.01,
            "hit_count": 6000,
            "false_alarm_count": 100,
            "match_count": 6144,
            "nonmatch_count": 12288,
            "hit_rate": 0.98,
            "false_alarm_rate": 0.01,
            "specificity": 0.99,
            "balanced_accuracy": 0.985,
            "discriminability": discriminability,
            "response_bias": 0.1,
            "d_prime": 3.0,
            "one_back_lure_count": 3072 if condition else 0,
            "one_back_lure_false_alarm_count": 10 if condition else 0,
            "one_back_lure_false_alarm_rate": (
                0.003 if condition else None
            ),
            "one_back_lure_accuracy": 0.997 if condition else None,
            "ordinary_nonmatch_count": 9216,
            "ordinary_nonmatch_false_alarm_count": 90,
            "ordinary_nonmatch_false_alarm_rate": 0.01,
            "ordinary_nonmatch_accuracy": 0.99,
            "settling_all": {
                "count": 18432,
                "fraction_settled": 0.99,
                "failure_rate": 0.01,
                "median_settling_steps": 1.0,
                "restricted_mean_settling_steps": 1.1,
            },
            "settling_correct_decisions": {
                "count": 18000,
                "fraction_settled": 0.995,
                "failure_rate": 0.005,
                "median_settling_steps": 1.0,
                "restricted_mean_settling_steps": 1.05,
            },
            "failure_rate": 0.01,
            "settling_valid": True,
        }

    def neutral_equivalence(
        self,
        checkpoint: RetainedCheckpoint,
        profile: object,
        *,
        phase: str,
        condition_code: int,
    ) -> NeutralEquivalence:
        profile_id = profile.profile_id  # type: ignore[attr-defined]
        self.calls.append(
            ("neutral", phase, checkpoint.ordinal, profile_id, condition_code)
        )
        exact = (
            profile_id not in self.neutral_failures
            and (
                phase,
                profile_id,
                checkpoint.ordinal,
                condition_code,
            )
            not in self.neutral_checkpoint_failures
            and not (
                phase == "confirmatory"
                and (profile_id, condition_code)
                in self.neutral_confirmatory_failures
            )
        )
        return NeutralEquivalence(
            exact=exact,
            comparisons=3,
            maximum_absolute_logit_difference=0.0 if exact else 1.0,
            maximum_absolute_hidden_difference=0.0 if exact else 1.0,
            additive_cost=0.0 if exact else 0.01,
        )

    def evaluate(
        self,
        checkpoint: RetainedCheckpoint,
        profile: object | None,
        strength: float | None,
        *,
        phase: str,
        condition_code: int,
    ) -> EvaluationBundle:
        profile_id = (
            None if profile is None else profile.profile_id  # type: ignore[attr-defined]
        )
        self.calls.append(
            (
                "evaluate",
                phase,
                checkpoint.ordinal,
                profile_id,
                condition_code,
                strength,
            )
        )
        if (
            phase,
            checkpoint.ordinal,
            profile_id,
            condition_code,
            None if strength is None else float(strength),
        ) in self.evaluation_failures:
            raise EvaluationValidityError("synthetic nonfinite evaluation")
        n_sequences = TASK_BANKS[phase].n_batches * 128
        baseline = 0.01
        if profile_id is None:
            cost = 0.0
            discriminability = 0.95 if condition_code == 0 else 0.92
            replicates = 1
        elif profile_id == 14:
            cost = float(strength)
            discriminability = 0.90 if condition_code == 0 else 0.80
            replicates = 3
        else:
            cost = abs(float(strength) - 1.0)
            discriminability = 0.90 if condition_code == 0 else 0.70
            replicates = 1
        units = np.full(
            (replicates, n_sequences),
            baseline + cost,
            dtype=np.float64,
        )
        metrics = tuple(
            self._metrics(
                condition_code,
                discriminability=discriminability,
                transport_failure=(
                    self.transport_failure and profile_id is None
                ),
            )
            for _ in range(replicates)
        )
        hashes = tuple(
            f"{phase}:{checkpoint.ordinal}:{condition_code}:{batch}"
            for batch in range(TASK_BANKS[phase].n_batches)
        )
        return EvaluationBundle(
            replicate_sequence_ce=units,
            replicate_metrics=metrics,
            task_hashes=hashes,
        )

    def release_checkpoint(self, checkpoint: RetainedCheckpoint) -> None:
        self.calls.append(("release", checkpoint.ordinal))


def _context(
    tmp_path: Path,
    backend: FakeBackend,
    *,
    n_checkpoints: int = 1,
    selected_device: SelectedDevice | None = None,
) -> object:
    checkpoints = []
    for ordinal in range(n_checkpoints):
        checkpoint = tmp_path / f"checkpoint_{ordinal}.pt"
        checkpoint.write_bytes(
            f"synthetic-checkpoint-{ordinal}".encode("ascii")
        )
        checkpoints.append(
            RetainedCheckpoint(
                ordinal=ordinal,
                seed=20260912 + ordinal,
                path=checkpoint,
            )
        )
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    prerequisite_paths: dict[str, Path] = {}
    for name in (
        "preregistration",
        "screened_pool_manifest",
        "precision_summary",
        "precision_arrays",
        "config",
    ):
        path = evidence_dir / f"{name}.input"
        if not path.exists():
            path.write_text(f"synthetic-{name}", encoding="utf-8")
        prerequisite_paths[name] = path
    config = load_config("configs/nback_additive_perturbation.yaml")
    profiles = (PROFILE_BY_ID[1], PROFILE_BY_ID[14])
    return initialize_runner_context(
        config=config,
        checkpoints=tuple(checkpoints),
        output_dir=tmp_path / "run",
        backend=backend,
        device=(
            selected_device
            if selected_device is not None
            else SelectedDevice(torch.device("cpu"), "synthetic CPU")
        ),
        design_evidence={
            "preregistration_sha256": "synthetic-prereg",
            "screened_pool_manifest_sha256": "synthetic-screen",
            "precision_summary_sha256": "synthetic-summary",
            "precision_arrays_sha256": "synthetic-arrays",
            "config_sha256": "synthetic-config",
        },
        runtime_prerequisites=prerequisite_paths,
        profiles=profiles,
    )


def test_frozen_config_and_exact_phase_plan() -> None:
    config = load_config("configs/nback_additive_perturbation.yaml")

    assert PHASE_ORDER == (
        "neutral-calibration",
        "calibration",
        "cost-check",
        "neutral-confirmatory",
        "confirmatory",
        "dose",
        "finalize",
    )
    assert config["perturbation"]["preregistration_commit"] == "fc99475"
    assert (
        config["perturbation"]["preregistration_sha256"]
        == "712ff2b3cf5139482724ab53437a70e14fd0d70514111ae0e8159a77b957312b"
    )
    assert config["perturbation"]["calibration_sequences"] == 512
    assert config["perturbation"]["n_cost_check"] == 1024
    assert (
        config["perturbation"]["confirmatory_sequences_per_condition"]
        == 1024
    )
    changed = json.loads(json.dumps(config))
    changed["task"]["matches_per_sequence"] = 5
    with pytest.raises(ValueError, match="matches_per_sequence"):
        _validate_frozen_config(changed)


def test_expected_cells_are_global_and_complete(tmp_path: Path) -> None:
    checkpoint = RetainedCheckpoint(0, 11, tmp_path / "unused")
    cells = expected_cells(
        (checkpoint,), (PROFILE_BY_ID[1], PROFILE_BY_ID[14])
    )

    assert set(cells) == set(PHASE_ORDER)
    assert len(cells["neutral-calibration"]) == 3
    assert len(cells["calibration"]) == 3
    assert len(cells["cost-check"]) == 3
    assert len(cells["neutral-confirmatory"]) == 5
    assert len(cells["confirmatory"]) == 7
    assert len(cells["dose"]) == 3
    assert cells["finalize"] == (
        "finalize:summary",
        "finalize:timing",
    )
    flattened = [cell for phase in PHASE_ORDER for cell in cells[phase]]
    assert len(flattened) == len(set(flattened))


def test_pooled_batches_and_nested_replicate_metrics() -> None:
    batches = [
        generate_nback_batch(
            NBackTaskConfig(n_back=2, batch_size=128, seed=100 + index)
        )
        for index in range(8)
    ]
    pooled = _combine_nback_batches(batches)
    targets = torch.from_numpy(pooled.targets)
    logits = torch.full((*pooled.targets.shape, 2), -8.0)
    logits.scatter_(2, targets.unsqueeze(-1), 8.0)
    metrics = nback_metrics(
        logits,
        targets,
        torch.from_numpy(pooled.loss_mask),
        pooled,
    )

    assert pooled.inputs.shape[1] == 1024
    assert metrics["match_count"] == 6144
    assert metrics["nonmatch_count"] == 12288
    assert metrics["discriminability"] == 1.0

    replicate_metrics = (
        FakeBackend._metrics(1, discriminability=0.70, transport_failure=False),
        FakeBackend._metrics(1, discriminability=0.80, transport_failure=False),
        FakeBackend._metrics(1, discriminability=0.90, transport_failure=False),
    )
    bundle = EvaluationBundle(
        replicate_sequence_ce=np.ones((3, 1024)),
        replicate_metrics=replicate_metrics,
        task_hashes=("same",),
    )
    averaged = bundle.averaged_metrics()
    assert averaged["discriminability"] == pytest.approx(0.80)
    assert averaged["match_count"] == 6144.0


@pytest.mark.parametrize("profile", OPERATOR_PROFILES)
def test_every_registered_profile_is_exactly_neutral(
    profile: OperatorProfile,
) -> None:
    config = load_config("configs/nback_additive_perturbation.yaml")
    model = fresh_model(config, torch.device("cpu"))
    model.eval()
    batch = generate_nback_batch(
        NBackTaskConfig(n_back=0, batch_size=2, seed=12345)
    )
    inputs = torch.from_numpy(batch.inputs)
    with torch.no_grad():
        native_logits, native_hidden = model(inputs)
    replicate_count = (
        3
        if profile.operator
        in {"heterogeneous_drive_gain", "gaussian_state_noise"}
        else 1
    )
    for replicate in range(replicate_count):
        parameters = _profile_parameters(
            profile,
            profile.ordered_grid[0],
            replicate_ordinal=replicate,
            phase="calibration",
            checkpoint_ordinal=0,
            condition_code=0,
            batch_index=0,
        )
        forward = build_nback_operator(
            model,
            profile.operator,
            **parameters,
        )
        with torch.no_grad():
            logits, hidden = forward(inputs)
        assert torch.equal(logits, native_logits)
        assert torch.equal(hidden, native_hidden)


def test_phase_barrier_artifact_containment_and_complete_resume(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    context = _context(tmp_path, backend)

    with pytest.raises(PerturbationStateError, match="prerequisites"):
        run_phase(context, "calibration")

    begin_phase(
        context.paths.manifest_path,
        context.paths.state_path,
        "neutral-calibration",
    )
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(PerturbationStateError, match="inside"):
        record_completed_cell(
            context.paths.manifest_path,
            context.paths.state_path,
            phase="neutral-calibration",
            cell_id="neutral-calibration:cp00:p01",
            artifacts=[outside],
        )

    for phase in PHASE_ORDER:
        snapshot = run_phase(context, phase)
        assert snapshot.phase_statuses[phase] == "completed"

    calls_after_first_run = list(backend.calls)
    run_phase(context, "confirmatory")
    assert backend.calls == calls_after_first_run
    resumed = _context(tmp_path, FakeBackend())
    assert resumed.paths.manifest_path == context.paths.manifest_path

    run_manifest = json.loads(
        context.paths.manifest_path.read_text(encoding="utf-8")
    )
    evidence = run_manifest["design"]
    assert evidence["preregistration_sha256"] == "synthetic-prereg"
    assert evidence["screened_pool_manifest_sha256"] == "synthetic-screen"
    assert evidence["precision_summary_sha256"] == "synthetic-summary"
    assert evidence["precision_arrays_sha256"] == "synthetic-arrays"
    assert evidence["config_sha256"] == "synthetic-config"
    assert len(run_manifest["checkpoints"][0]["sha256"]) == 64
    assert len(evidence["implementation_design_hash"]) == 64
    assert evidence["checkpoint_order"] == [
        {"ordinal": 0, "seed": 20260912}
    ]

    summary = json.loads(
        (
            context.paths.root / "metrics" / "nback_c2_summary.json"
        ).read_text(encoding="utf-8")
    )
    candidate = next(
        profile
        for profile in summary["profiles"]
        if profile["profile_id"] == 1
    )
    assert candidate["valid"] is True
    assert candidate["mean_c2_nback"] > 0.0
    assert "mean_excess_load_did_impairment" in candidate
    assert "mean_excess_ce_interaction" in candidate
    assert "mean_excess_failure_rate_interaction" in candidate
    assert "mean_candidate_two_back_failure_rate" in candidate
    assert "mean_candidate_two_back_fraction_settled" in candidate
    assert candidate["dynamics_outcome"] == "latency"
    dose = next(
        row
        for row in summary["dose_ordering_profiles"]
        if row["profile_id"] == 1
    )
    assert dose["classification"] == "preserved"
    assert summary["p5_registered_strength_curve"]
    assert all(
        "zero_back_additive_ce_cost" in row
        for row in summary["p5_registered_strength_curve"]
    )
    checkpoint_csv = (
        context.paths.root / "metrics" / "nback_c2_checkpoint.csv"
    )
    with checkpoint_csv.open(newline="", encoding="utf-8") as handle:
        checkpoint_rows = list(csv.DictReader(handle))
    candidate_checkpoint = next(
        row for row in checkpoint_rows if row["profile_id"] == "1"
    )
    assert "candidate_two_back_discriminability_change" in candidate_checkpoint
    assert "candidate_two_back_additive_ce_cost" in candidate_checkpoint
    assert "excess_load_did_change" in candidate_checkpoint
    assert "excess_failure_rate_interaction" in candidate_checkpoint
    assert "excess_load_rmst_interaction" in candidate_checkpoint
    assert candidate_checkpoint["joint_latency_valid"] == "True"
    profile_csv = context.paths.root / "metrics" / "nback_c2_profile.csv"
    assert profile_csv.is_file()
    with profile_csv.open(newline="", encoding="utf-8") as handle:
        profile_rows = list(csv.DictReader(handle))
    assert any(row["profile_id"] == "1" for row in profile_rows)
    for phase in PHASE_ORDER:
        timing = (
            context.paths.root
            / (
                "neutral"
                if phase.startswith("neutral-")
                else phase.replace("-", "_")
            )
            / "cells"
            / f"{phase.replace('-', '_')}__timing.json"
        )
        assert timing.is_file()


def test_neutral_failure_prevents_candidate_non_neutral_calibration(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(neutral_failures={1})
    context = _context(tmp_path, backend)

    run_phase(context, "neutral-calibration")
    run_phase(context, "calibration")

    candidate_non_neutral = [
        call
        for call in backend.calls
        if call[0] == "evaluate"
        and call[1] == "calibration"
        and call[3] == 1
    ]
    assert candidate_non_neutral == []
    payload = json.loads(
        (
            context.paths.root
            / "calibration"
            / "cells"
            / "calibration__cp00__p01.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        payload["invalid_reason"]
        == "global_neutral_calibration_failure"
    )


def test_neutral_calibration_failure_is_profile_global_across_checkpoints(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        neutral_checkpoint_failures={("calibration", 1, 1, 0)}
    )
    context = _context(tmp_path, backend, n_checkpoints=2)

    run_phase(context, "neutral-calibration")
    run_phase(context, "calibration")

    candidate_non_neutral = [
        call
        for call in backend.calls
        if call[0] == "evaluate"
        and call[1] == "calibration"
        and call[3] == 1
    ]
    assert candidate_non_neutral == []
    for ordinal in (0, 1):
        payload = json.loads(
            (
                context.paths.root
                / "calibration"
                / "cells"
                / f"calibration__cp{ordinal:02d}__p01.json"
            ).read_text(encoding="utf-8")
        )
        assert (
            payload["invalid_reason"]
            == "global_neutral_calibration_failure"
        )


def test_nonfinite_calibration_is_an_invalid_cell_not_a_crash(
    tmp_path: Path,
) -> None:
    failures = {
        ("calibration", 0, 1, 0, float(strength))
        for strength in PROFILE_BY_ID[1].ordered_grid
    }
    backend = FakeBackend(evaluation_failures=failures)
    context = _context(tmp_path, backend)

    run_phase(context, "neutral-calibration")
    run_phase(context, "calibration")

    payload = json.loads(
        (
            context.paths.root
            / "calibration"
            / "cells"
            / "calibration__cp00__p01.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["status"] == "invalid"
    assert payload["invalid_reason"] == "nonfinite_operator_evaluation"


def test_nonfinite_p5_cost_check_prevents_candidate_evaluation(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        evaluation_failures={("cost_check", 0, 14, 0, 0.05)}
    )
    context = _context(tmp_path, backend)
    for phase in PHASE_ORDER[:3]:
        run_phase(context, phase)

    candidate_cost_calls = [
        call
        for call in backend.calls
        if call[0] == "evaluate"
        and call[1] == "cost_check"
        and call[3] == 1
    ]
    assert candidate_cost_calls == []
    p5 = json.loads(
        (
            context.paths.root
            / "cost_check"
            / "cells"
            / "cost_check__cp00__p14.json"
        ).read_text(encoding="utf-8")
    )
    candidate = json.loads(
        (
            context.paths.root
            / "cost_check"
            / "cells"
            / "cost_check__cp00__p01.json"
        ).read_text(encoding="utf-8")
    )
    assert p5["status"] == "invalid"
    assert p5["invalid_reason"] == "nonfinite_operator_evaluation"
    assert candidate["invalid_reason"] == "p5_reference_invalid"


def test_nonfinite_confirmatory_baseline_invalidates_outcomes(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        evaluation_failures={
            ("confirmatory", 0, None, 1, None)
        }
    )
    context = _context(tmp_path, backend)
    for phase in PHASE_ORDER[:5]:
        run_phase(context, phase)

    payload = json.loads(
        (
            context.paths.root
            / "confirmatory"
            / "cells"
            / "confirmatory__baseline__cp00__c1.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["status"] == "invalid"
    assert payload["invalid_reason"] == "nonfinite_baseline_evaluation"
    assert not any(
        call[0] == "evaluate"
        and call[1] == "confirmatory"
        and call[3] is not None
        for call in backend.calls
    )


def test_nonfinite_confirmatory_profile_propagates_to_dose(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        evaluation_failures={
            ("confirmatory", 0, 1, 0, 1.05)
        }
    )
    context = _context(tmp_path, backend)
    for phase in PHASE_ORDER[:6]:
        run_phase(context, phase)

    confirmatory = json.loads(
        (
            context.paths.root
            / "confirmatory"
            / "cells"
            / "confirmatory__cp00__p01__c0.json"
        ).read_text(encoding="utf-8")
    )
    dose = json.loads(
        (
            context.paths.root
            / "dose"
            / "cells"
            / "dose__cp00__p01.json"
        ).read_text(encoding="utf-8")
    )
    assert confirmatory["status"] == "invalid"
    assert (
        confirmatory["invalid_reason"]
        == "nonfinite_operator_evaluation"
    )
    assert (
        dose["invalid_reason"]
        == "global_confirmatory_validity_failure"
    )


def test_nonfinite_dose_profile_is_invalid_not_partial(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        evaluation_failures={
            ("confirmatory", 0, 1, 0, 1.10)
        }
    )
    context = _context(tmp_path, backend)
    for phase in PHASE_ORDER[:6]:
        run_phase(context, phase)

    payload = json.loads(
        (
            context.paths.root
            / "dose"
            / "cells"
            / "dose__cp00__p01.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["status"] == "invalid"
    assert payload["invalid_reason"] == "nonfinite_operator_evaluation"
    assert not (
        context.paths.root
        / "dose"
        / "arrays"
        / "dose__cp00__p01.npz"
    ).exists()


def test_runtime_manifest_covers_transitive_computation_sources() -> None:
    required = {
        "runner",
        "calibration",
        "cost_precision",
        "metrics",
        "adapter",
        "state",
        "perturbation_operators",
        "nback_task",
        "model",
        "training_utils",
        "config",
        "device",
        "outcomes",
    }
    paths = _implementation_source_paths()

    assert required == set(paths)
    assert all(path.is_file() for path in paths.values())


@pytest.mark.parametrize(
    "mutation_target",
    ("implementation", "profile", "seed", "external", "checkpoint"),
)
def test_every_phase_call_refuses_mutated_runtime_evidence(
    tmp_path: Path,
    mutation_target: str,
) -> None:
    backend = FakeBackend()
    context = _context(tmp_path, backend)
    integrity = context.design["runtime_integrity"]
    if mutation_target == "implementation":
        synthetic_source = tmp_path / "synthetic_source.py"
        synthetic_source.write_text("VALUE = 1\n", encoding="utf-8")
        record = {
            "path": str(synthetic_source.resolve()),
            "sha256": sha256_file(synthetic_source),
        }
        integrity["implementation_sources"]["runner"] = record
        hashes = {
            name: source_record["sha256"]
            for name, source_record in integrity[
                "implementation_sources"
            ].items()
        }
        context.design["implementation_design_hash"] = (
            canonical_design_hash(hashes)
        )
        target = synthetic_source
    elif mutation_target == "profile":
        target = Path(
            integrity["auxiliary_manifests"]["profile_manifest"]["path"]
        )
    elif mutation_target == "seed":
        target = Path(
            integrity["auxiliary_manifests"]["seed_manifest"]["path"]
        )
    elif mutation_target == "external":
        target = Path(
            integrity["external_prerequisites"]["config"]["path"]
        )
    else:
        target = Path(
            integrity["checkpoints"]["20260912"]["path"]
        )
    target.write_bytes(target.read_bytes() + b"\nmutation")

    with pytest.raises(
        PerturbationStateError, match="frozen runtime input changed"
    ):
        run_phase(context, "neutral-calibration")
    assert backend.calls == []


def test_baseline_transport_failure_generates_no_non_neutral_outcomes(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(transport_failure=True)
    context = _context(tmp_path, backend)
    for phase in PHASE_ORDER[:5]:
        run_phase(context, phase)
    run_phase(context, "dose")

    confirmatory_operator_calls = [
        call
        for call in backend.calls
        if call[0] == "evaluate"
        and call[1] == "confirmatory"
        and call[3] is not None
    ]
    assert confirmatory_operator_calls == []
    candidate = json.loads(
        (
            context.paths.root
            / "confirmatory"
            / "cells"
            / "confirmatory__cp00__p01__c0.json"
        ).read_text(encoding="utf-8")
    )
    assert candidate["invalid_reason"] == "baseline_transport_failure"
    dose_operator_calls = [
        call
        for call in backend.calls
        if call[0] == "evaluate"
        and call[1] == "confirmatory"
        and call[3] is not None
    ]
    assert dose_operator_calls == []


def test_one_neutral_confirmatory_failure_stops_profile_globally(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        neutral_confirmatory_failures={(1, 0)}
    )
    context = _context(tmp_path, backend)
    for phase in PHASE_ORDER[:5]:
        run_phase(context, phase)

    candidate_outcomes = [
        call
        for call in backend.calls
        if call[0] == "evaluate"
        and call[1] == "confirmatory"
        and call[3] == 1
    ]
    assert candidate_outcomes == []
    for condition in (0, 1):
        payload = json.loads(
            (
                context.paths.root
                / "confirmatory"
                / "cells"
                / f"confirmatory__cp00__p01__c{condition}.json"
            ).read_text(encoding="utf-8")
        )
        assert payload["invalid_reason"] == "neutral_equivalence_failure"


def test_zero_sd_one_sided_t_limits_are_registered() -> None:
    positive = _one_sample_t_summary([0.2] * 10)
    zero = _one_sample_t_summary([0.0] * 10)
    negative = _one_sample_t_summary([-0.2] * 10)

    assert positive["t_statistic"] == "infinity"
    assert positive["paired_dz"] == "infinity"
    assert positive["one_sided_p_value"] == 0.0
    assert positive["ci95_lower"] == positive["ci95_upper"] == 0.2
    assert zero["t_statistic"] == 0.0
    assert zero["one_sided_p_value"] == 0.5
    assert negative["t_statistic"] == "-infinity"
    assert negative["one_sided_p_value"] == 1.0


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([0.1, 0.2, 0.3], "preserved"),
        ([0.1, 0.2, 0.15], "degraded"),
        ([0.1, 0.0, 0.2], "degraded"),
        ([0.1, 0.0, 0.0], "scrambled"),
        ([0.1, 0.2, -0.01], "scrambled"),
        ([0.1, -0.01, 0.2], "scrambled"),
        ([-0.01, 0.2, 0.3], "scrambled"),
    ],
)
def test_dose_classification_applies_reversal_rule_at_any_level(
    values: list[float],
    expected: str,
) -> None:
    assert _classify_dose_selectivities(values) == expected


def test_joint_latency_guard_substitutes_failure_rate_outcomes() -> None:
    baseline = {
        condition: FakeBackend._metrics(
            condition,
            discriminability=0.95 if condition == 0 else 0.92,
            transport_failure=False,
        )
        for condition in (0, 1)
    }
    candidate = {
        condition: FakeBackend._metrics(
            condition,
            discriminability=0.90 if condition == 0 else 0.70,
            transport_failure=False,
        )
        for condition in (0, 1)
    }
    p5 = {
        condition: FakeBackend._metrics(
            condition,
            discriminability=0.90 if condition == 0 else 0.80,
            transport_failure=False,
        )
        for condition in (0, 1)
    }
    candidate[1]["settling_all"]["fraction_settled"] = 0.79
    candidate[1]["settling_all"]["failure_rate"] = 0.21
    candidate[1]["failure_rate"] = 0.21

    outcomes = _checkpoint_descriptive_outcomes(
        baseline, candidate, p5
    )

    assert outcomes["joint_latency_valid"] is False
    assert outcomes["dynamics_outcome"] == "failure_rate"
    assert outcomes["latency_invalid_reason"] == (
        "fraction_settled_below_0_80"
    )
    assert all(
        outcomes[field] is None
        for field in (
            "candidate_zero_back_rmst_change",
            "candidate_two_back_rmst_change",
            "p5_zero_back_rmst_change",
            "p5_two_back_rmst_change",
            "candidate_load_rmst_interaction",
            "p5_load_rmst_interaction",
            "excess_load_rmst_interaction",
        )
    )
    assert outcomes["candidate_two_back_failure_rate"] == pytest.approx(
        0.21
    )
    assert outcomes["candidate_failure_rate_interaction"] == pytest.approx(
        0.20
    )


def test_phase_timing_accumulates_across_interrupted_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailsOnceBackend(FakeBackend):
        failed = False

        def neutral_equivalence(self, checkpoint, profile, **kwargs):
            if profile.profile_id == 14 and not self.failed:
                self.failed = True
                raise RuntimeError("synthetic interruption")
            return super().neutral_equivalence(
                checkpoint, profile, **kwargs
            )

    backend = FailsOnceBackend()
    context = _context(tmp_path, backend)
    ticks = iter((10.0, 11.0, 20.0, 21.0, 22.0))
    monkeypatch.setattr(
        runner_module.time, "perf_counter", lambda: next(ticks)
    )

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_phase(context, "neutral-calibration")
    first_profile_calls = [
        call
        for call in backend.calls
        if call[:2] == ("neutral", "calibration") and call[3] == 1
    ]
    assert len(first_profile_calls) == 1

    completed = run_phase(context, "neutral-calibration")
    timing = completed.phase_timings["neutral-calibration"]
    assert timing["accumulated_seconds"] == pytest.approx(3.0)
    assert [row["status"] for row in timing["attempts"]] == [
        "interrupted",
        "completed",
    ]
    resumed_profile_calls = [
        call
        for call in backend.calls
        if call[:2] == ("neutral", "calibration") and call[3] == 1
    ]
    assert len(resumed_profile_calls) == 1
    timing_payload = json.loads(
        (
            context.paths.root
            / "neutral"
            / "cells"
            / "neutral_calibration__timing.json"
        ).read_text(encoding="utf-8")
    )
    assert timing_payload["wall_time_seconds"] == pytest.approx(3.0)
    assert timing_payload["attempt_count"] == 2
    assert timing_payload["interrupted_attempt_count"] == 1

    calls_before_resume = list(backend.calls)
    assert run_phase(
        context, "neutral-calibration"
    ).phase_timings["neutral-calibration"] == timing
    assert backend.calls == calls_before_resume


def test_cuda_phase_timing_synchronizes_at_every_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend()
    context = _context(
        tmp_path,
        backend,
        selected_device=SelectedDevice(
            torch.device("cuda"), "synthetic CUDA"
        ),
    )
    sync_calls: list[torch.device] = []
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: sync_calls.append(device),
    )
    ticks = iter((1.0, 2.0, 3.0, 4.0))
    monkeypatch.setattr(
        runner_module.time, "perf_counter", lambda: next(ticks)
    )

    run_phase(context, "neutral-calibration")

    assert sync_calls == [torch.device("cuda")] * 4
