"""Tests for item-level N-back behavioural and settling metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wm_rnn.nback_metrics import nback_metrics
from wm_rnn.nback_task import NBackTaskConfig, generate_nback_batch
from wm_rnn.training_utils import masked_cross_entropy


def _perfect_logits(
    targets: np.ndarray,
    *,
    correct_logit: float = 8.0,
) -> torch.Tensor:
    target_tensor = torch.from_numpy(targets).long()
    logits = torch.full((*targets.shape, 2), -correct_logit)
    logits.scatter_(2, target_tensor.unsqueeze(-1), correct_logit)
    return logits


def test_perfect_metrics_use_one_observation_per_item() -> None:
    batch = generate_nback_batch(
        NBackTaskConfig(n_back=2, batch_size=4, seed=3)
    )
    logits = _perfect_logits(batch.targets)
    metrics = nback_metrics(
        logits,
        torch.from_numpy(batch.targets),
        torch.from_numpy(batch.loss_mask),
        batch,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["hit_rate"] == 1.0
    assert metrics["false_alarm_rate"] == 0.0
    assert metrics["discriminability"] == 1.0
    assert metrics["response_bias"] is None
    assert metrics["match_count"] == 4 * 6
    assert metrics["nonmatch_count"] == 4 * 12
    assert metrics["one_back_lure_accuracy"] == 1.0
    assert metrics["settling_all"]["fraction_settled"] == 1.0
    assert metrics["settling_all"]["restricted_mean_settling_steps"] == 0.0


def test_final_event_prediction_defines_hit_and_false_alarm_counts() -> None:
    batch = generate_nback_batch(
        NBackTaskConfig(n_back=0, batch_size=1, seed=9)
    )
    logits = _perfect_logits(batch.targets)
    scored_items = np.flatnonzero(batch.item_scored[:, 0])
    match_item = next(
        item
        for item in scored_items
        if batch.item_labels[item, 0] == 1
    )
    nonmatch_item = next(
        item
        for item in scored_items
        if batch.item_labels[item, 0] == 0
    )
    for item in (match_item, nonmatch_item):
        final_step = int(batch.event_onsets[item] + batch.event_steps - 1)
        logits[final_step, 0] *= -1

    metrics = nback_metrics(
        logits,
        torch.from_numpy(batch.targets),
        torch.from_numpy(batch.loss_mask),
        batch,
    )

    assert metrics["hit_count"] == 5
    assert metrics["false_alarm_count"] == 1
    assert metrics["match_count"] == 6
    assert metrics["nonmatch_count"] == 12
    assert metrics["hit_rate"] == pytest.approx(5 / 6)
    assert metrics["false_alarm_rate"] == pytest.approx(1 / 12)
    assert metrics["discriminability"] == pytest.approx(0.75)
    assert metrics["response_bias"] == pytest.approx(1 / 3)


def test_nonsettling_outputs_are_capped_and_invalid() -> None:
    batch = generate_nback_batch(
        NBackTaskConfig(n_back=2, batch_size=2, seed=11)
    )
    logits = torch.zeros((*batch.targets.shape, 2))
    metrics = nback_metrics(
        logits,
        torch.from_numpy(batch.targets),
        torch.from_numpy(batch.loss_mask),
        batch,
    )

    assert metrics["settling_all"]["fraction_settled"] == 0.0
    assert metrics["settling_all"]["median_settling_steps"] is None
    assert (
        metrics["settling_all"]["restricted_mean_settling_steps"]
        == batch.event_steps
    )
    assert metrics["settling_valid"] is False


def test_class_weighted_cross_entropy_matches_hand_calculation() -> None:
    logits = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    targets = torch.tensor([[0, 1]])
    mask = torch.ones((1, 2))
    weights = torch.tensor([1.0, 2.0])

    loss = masked_cross_entropy(
        logits, targets, mask, class_weights=weights
    )

    assert loss.item() == pytest.approx(np.log(2.0))
