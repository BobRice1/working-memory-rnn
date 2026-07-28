"""Tests for the current trained-distractor candidate runner."""

from __future__ import annotations

import json

import pytest

import wm_rnn.full_candidate_perturbation_run as runner


def test_checkpoint_manifest_retains_only_competent_target_set(
    tmp_path,
    monkeypatch,
) -> None:
    manifest = {
        "target_competent_checkpoints": 2,
        "retained_checkpoint_seeds": [11, 13],
        "results": [
            {
                "seed": 11,
                "checkpoint": "seed11.pt",
                "checkpoint_sha256": "A" * 64,
                "competence_passed": True,
            },
            {
                "seed": 12,
                "checkpoint": "seed12.pt",
                "checkpoint_sha256": "B" * 64,
                "competence_passed": False,
            },
            {
                "seed": 13,
                "checkpoint": "seed13.pt",
                "checkpoint_sha256": "C" * 64,
                "competence_passed": True,
            },
        ],
    }
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(runner, "POOL_MANIFEST", path)

    checkpoints = runner.trained_distractor_checkpoints(tmp_path)

    assert [checkpoint.seed for checkpoint in checkpoints] == [11, 13]


def test_checkpoint_manifest_rejects_failed_retained_seed(
    tmp_path,
    monkeypatch,
) -> None:
    manifest = {
        "target_competent_checkpoints": 1,
        "retained_checkpoint_seeds": [11],
        "results": [
            {
                "seed": 11,
                "checkpoint": "seed11.pt",
                "checkpoint_sha256": "A" * 64,
                "competence_passed": False,
            }
        ],
    }
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(runner, "POOL_MANIFEST", path)

    with pytest.raises(ValueError, match="failed competence"):
        runner.trained_distractor_checkpoints(tmp_path)
