"""End-to-end and algebra tests for the perturbation experiment harness."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from wm_rnn.model import RNNConfig, WorkingMemoryRNN
from wm_rnn.perturbation_calibration import (
    calibrate_strength,
    validate_cost_match,
)
from wm_rnn.perturbation_experiment import (
    P5_REPLICATES,
    SIGNATURE_COLUMNS,
    average_p5_replicates,
    compute_excess_constraints,
    frozen_batch_seed,
    run_smoke_experiment,
)
from wm_rnn.perturbation_metrics import assess_settling_validity
from wm_rnn.tuned_task import TunedDelayTaskConfig


def _tiny_model_and_task() -> tuple[WorkingMemoryRNN, TunedDelayTaskConfig]:
    task = TunedDelayTaskConfig(
        n_tuned_units=8,
        pre_cue_steps=1,
        cue_steps=2,
        delay_steps=10,
        response_steps=5,
        batch_size=8,
        seed=123,
        fixation_gated=True,
        distractor_steps=2,
        distractor_angle_mode="fixed_offset",
    )
    model = WorkingMemoryRNN(
        RNNConfig(
            input_size=task.input_size,
            hidden_size=8,
            output_size=task.output_size,
            dt=20.0,
            tau=100.0,
            activation="tanh",
        )
    )
    model.eval()
    return model, task


def test_smoke_grid_runs_end_to_end_with_exact_schema_and_neutral_deltas(
    tmp_path,
) -> None:
    model, task = _tiny_model_and_task()

    result = run_smoke_experiment(model, task, tmp_path, family="A")

    assert result.grid_path.exists()
    assert result.metadata_path.exists()
    with result.grid_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == SIGNATURE_COLUMNS
    assert len(rows) == 8
    for row in rows:
        assert float(row["delta_angular_error_degrees"]) == pytest.approx(
            0.0, abs=1e-6
        )
        assert float(
            row["delta_restricted_mean_settling_steps"]
        ) == pytest.approx(0.0, abs=1e-6)
        assert float(row["delta_failure_rate"]) == pytest.approx(
            0.0, abs=1e-6
        )


def test_frozen_seed_scheme_reuses_trials_across_operators() -> None:
    first = frozen_batch_seed(202607300, 2, 40, 3)
    repeat = frozen_batch_seed(202607300, 2, 40, 3)
    other_condition = frozen_batch_seed(202607300, 3, 40, 3)

    assert first == repeat
    assert first != other_condition


def test_cost_check_does_not_change_calibrated_strength_and_has_distinct_failures() -> None:
    calibration = calibrate_strength(
        lambda strength: strength,
        [0.0, 0.5, 1.0],
        target_proportional_cost=0.30,
        tolerance=1e-4,
    )
    selected_strength = calibration.strength
    baseline = np.linspace(3.0, 5.0, 64)

    outside = validate_cost_match(
        baseline,
        baseline * 1.5,
        maximum_half_width=1.0,
        draws=200,
    )
    imprecise = validate_cost_match(
        baseline,
        baseline + 1.2,
        maximum_half_width=0.0,
        draws=200,
    )

    assert calibration.strength == selected_strength
    assert outside.invalid_reason == "cost_band_failure"
    assert imprecise.invalid_reason == "cost_precision_failure"


def test_settling_gates_retain_distinct_validity_reasons() -> None:
    fixation = assess_settling_validity(0.89, 1.0, 1.0)
    response = assess_settling_validity(0.95, 1.0, 0.49)

    assert fixation["settling_validity_reason"] == "fixation_failure"
    assert fixation["latency_valid"] is False
    assert response["settling_validity_reason"] == "low_fraction_settled"
    assert response["latency_valid"] is False
    assert response["primary_response_outcome"] == "failure_rate"


def test_synthetic_condition_effects_recover_known_excess_constraints() -> None:
    baseline = {
        "load1_clean": 10.0,
        "load2_clean": 20.0,
        "load1_distractor": 25.0,
    }
    candidate = {
        "load1_clean": 13.0,
        "load2_clean": 32.0,
        "load1_distractor": 47.5,
    }
    p5 = {
        "load1_clean": 13.0,
        "load2_clean": 26.0,
        "load1_distractor": 32.5,
    }

    contrasts = compute_excess_constraints(
        baseline,
        candidate,
        p5,
        baseline_rmst=5.0,
        candidate_rmst=9.0,
        p5_rmst=7.0,
    )

    assert contrasts["x1"] == pytest.approx(2.0)
    assert contrasts["x2"] == pytest.approx(0.30)
    assert contrasts["x3"] == pytest.approx(0.60)


def test_p5_replicates_are_averaged_within_checkpoint_only() -> None:
    rows = []
    for checkpoint_seed in (101, 102):
        for index, replicate in enumerate(P5_REPLICATES):
            rows.append(
                {
                    "family": "B",
                    "operator": "gaussian_state_noise",
                    "variant": "generic_control",
                    "strength": 0.05,
                    "strength_kind": "grid",
                    "condition": "load1_clean",
                    "delay_steps": 20,
                    "seed": checkpoint_seed,
                    "item_position": "pooled",
                    "noise_replicate": replicate,
                    "mean_angular_error_degrees": checkpoint_seed + index,
                }
            )

    averaged = average_p5_replicates(rows)

    assert len(averaged) == 2
    assert {row["seed"] for row in averaged} == {101, 102}
    assert all(row["n_noise_replicates"] == 3 for row in averaged)
    assert averaged[0]["noise_replicate"] == ""
