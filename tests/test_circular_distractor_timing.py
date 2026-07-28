from pathlib import Path

import numpy as np
import pytest

from wm_rnn.circular_distractor_timing import (
    TIMING_CONDITIONS,
    assert_midpoint_reproduction,
    checkpoint_comparisons,
    generate_paired_banks,
    load_design,
    load_midpoint_references,
    main,
    relative_distractor_start,
    resolve_timing_task,
    summarize_comparisons,
)
from wm_rnn.tuned_task import TunedDelayTaskConfig


def test_timing_conditions_resolve_to_frozen_delay_starts():
    base = TunedDelayTaskConfig(
        pre_cue_steps=25,
        cue_steps=20,
        delay_steps=20,
        response_steps=25,
        batch_size=8,
        fixation_gated=True,
        distractor_steps=5,
    )
    expected = {
        "clean": (0, None),
        "start": (5, 0),
        "quarter": (5, 4),
        "midpoint": (5, 8),
        "three_quarter": (5, 11),
        "end": (5, 15),
    }
    for label, (steps, relative_start) in expected.items():
        task = resolve_timing_task(base, label)
        batch_start = round(
            (task.delay_steps - task.distractor_steps)
            * task.distractor_onset_fraction
        )
        assert task.distractor_steps == steps
        if relative_start is not None:
            assert batch_start == relative_start
    assert tuple(TIMING_CONDITIONS) == tuple(expected)


def test_paired_banks_hold_angles_constant_and_move_only_timing():
    base = TunedDelayTaskConfig(
        n_tuned_units=8,
        pre_cue_steps=3,
        cue_steps=4,
        delay_steps=20,
        response_steps=5,
        batch_size=16,
        seed=99,
        fixation_gated=True,
        distractor_steps=5,
    )
    banks = generate_paired_banks(base, seed=123)
    reference_targets = banks["clean"].angles
    reference_distractors = banks["midpoint"].distractor_angles
    for batch in banks.values():
        assert np.array_equal(batch.angles, reference_targets)
    for label in TIMING_CONDITIONS:
        if label != "clean":
            assert np.array_equal(
                banks[label].distractor_angles,
                reference_distractors,
            )
    assert banks["start"].phase_index["distractor"].start != (
        banks["midpoint"].phase_index["distractor"].start
    )


def test_checkpoint_comparisons_use_clean_cost_and_midpoint_reference():
    metrics = {
        "clean": {"mean_angular_error_degrees": 2.0},
        "start": {"mean_angular_error_degrees": 4.0},
        "quarter": {"mean_angular_error_degrees": 3.5},
        "midpoint": {"mean_angular_error_degrees": 3.0},
        "three_quarter": {"mean_angular_error_degrees": 3.25},
        "end": {"mean_angular_error_degrees": 4.5},
    }
    rows = checkpoint_comparisons(123, metrics)
    by_condition = {row["condition"]: row for row in rows}
    assert by_condition["midpoint"]["distractor_cost_degrees"] == 1.0
    assert by_condition["start"]["timing_minus_midpoint_degrees"] == 1.0
    assert by_condition["quarter"]["timing_minus_midpoint_degrees"] == 0.5
    assert by_condition["three_quarter"]["timing_minus_midpoint_degrees"] == 0.25
    assert by_condition["end"]["timing_minus_midpoint_degrees"] == 1.5


def test_summarize_comparisons_uses_checkpoint_as_unit():
    rows = [
        {
            "checkpoint_seed": seed,
            "condition": "start",
            "distractor_cost_degrees": value + 2.0,
            "timing_minus_midpoint_degrees": value,
        }
        for seed, value in enumerate((1.0, 2.0, 3.0, 4.0, 5.0), start=1)
    ]
    summary = summarize_comparisons(rows)[0]
    assert summary["condition"] == "start"
    assert summary["n_checkpoints"] == 5
    assert summary["mean_timing_minus_midpoint_degrees"] == 3.0
    assert summary["sd_timing_minus_midpoint_degrees"] == pytest.approx(
        1.5811388300841898
    )
    assert summary["positive_checkpoints"] == 5
    assert summary["ci95_low"] < 3.0 < summary["ci95_high"]


def test_design_loads_without_checkpoint_execution():
    design = load_design(
        Path("configs/circular_distractor_timing_generalisation.yaml")
    )
    assert design["evaluation"]["trials_per_condition"] == 1024
    assert list(design["evaluation"]["onset_fractions"]) == list(
        TIMING_CONDITIONS
    )


def test_cli_requires_execute(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["circular_distractor_timing"],
    )
    main()
    assert "design_only" in capsys.readouterr().out


def test_midpoint_references_select_native_persistence_rows(tmp_path):
    source = tmp_path / "grid.csv"
    source.write_text(
        "checkpoint_seed,condition,delay_steps,operator,strength,"
        "mean_angular_error_degrees\n"
        "1,distractor,20,state_persistence,0.95,9.0\n"
        "1,distractor,20,state_persistence,1.0,3.5\n"
        "2,distractor,20,state_persistence,1.0,4.5\n",
        encoding="utf-8",
    )
    assert load_midpoint_references(source) == {1: 3.5, 2: 4.5}


def test_midpoint_reproduction_allows_only_sub_microdegree_roundoff():
    assert_midpoint_reproduction(4.4365174, 4.4365170)
    with pytest.raises(AssertionError):
        assert_midpoint_reproduction(4.4366170, 4.4365170)


def test_relative_distractor_start_is_blank_for_clean_trials():
    phases = {
        "delay": slice(45, 65),
        "distractor": slice(55, 55),
    }
    assert relative_distractor_start("clean", phases) == ""
    phases["distractor"] = slice(53, 58)
    assert relative_distractor_start("midpoint", phases) == 8
