"""Item-level behavioural and settling metrics for N-back outputs."""

from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wm_rnn.nback_task import NBackBatch


def _validate_sequence_loss_inputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> None:
    """Validate the binary sequence-loss tensors before reduction."""
    if (
        logits.ndim != 3
        or logits.shape[-1] != 2
        or targets.shape != logits.shape[:2]
        or loss_mask.shape != logits.shape[:2]
    ):
        raise ValueError(
            "logits, targets, and loss_mask must have shapes "
            "[time, batch, 2], [time, batch], and [time, batch]"
        )
    if not torch.isfinite(logits).all():
        raise ValueError("logits must contain only finite values")
    if not torch.isfinite(loss_mask).all():
        raise ValueError("loss_mask must contain only finite values")
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("targets must contain only binary class labels")
    if not torch.all((loss_mask == 0) | (loss_mask == 1)):
        raise ValueError("loss_mask must contain only zeros and ones")
    if not torch.all(loss_mask.sum(dim=0) > 0):
        raise ValueError("every sequence must contain scored timesteps")


def _validate_batch_metric_inputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    batch: NBackBatch,
) -> None:
    """Require metric tensors to reproduce the registered generated batch."""
    _validate_sequence_loss_inputs(logits, targets, loss_mask)
    target_values = targets.detach().cpu().numpy()
    mask_values = loss_mask.detach().cpu().numpy()
    if not np.array_equal(target_values, batch.targets):
        raise ValueError("targets must exactly match batch.targets")
    if not np.array_equal(mask_values, batch.loss_mask):
        raise ValueError("loss_mask must exactly match batch.loss_mask")


def _item_arrays(
    logits: torch.Tensor,
    batch: NBackBatch,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return final-event predictions, labels, and scored mask."""
    if (
        logits.ndim != 3
        or logits.shape[:2] != batch.targets.shape
        or logits.shape[-1] != 2
    ):
        raise ValueError(
            "logits must have shape [time, batch, 2] matching the batch"
        )
    if not torch.isfinite(logits).all():
        raise ValueError("logits must contain only finite values")
    event_final_steps = torch.as_tensor(
        batch.event_onsets + batch.event_steps - 1,
        device=logits.device,
        dtype=torch.long,
    )
    final_logits = logits.index_select(0, event_final_steps)
    final_logits = final_logits.detach().cpu().numpy()
    predictions = final_logits.argmax(axis=-1)
    return predictions, batch.item_labels, batch.item_scored


def per_sequence_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> np.ndarray:
    """Return mean scored time-point cross-entropy for each sequence."""
    _validate_sequence_loss_inputs(logits, targets, loss_mask)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape)
    mask = loss_mask.to(dtype=losses.dtype)
    denominators = mask.sum(dim=0).clamp_min(1.0)
    values = (losses * mask).sum(dim=0) / denominators
    return values.detach().cpu().numpy()


def _rates(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | int | None]:
    matches = labels == 1
    nonmatches = labels == 0
    hit_count = int(((predictions == 1) & matches).sum())
    false_alarm_count = int(((predictions == 1) & nonmatches).sum())
    match_count = int(matches.sum())
    nonmatch_count = int(nonmatches.sum())
    hit_rate = hit_count / match_count if match_count else None
    false_alarm_rate = (
        false_alarm_count / nonmatch_count if nonmatch_count else None
    )
    if hit_rate is None or false_alarm_rate is None:
        specificity = None
        balanced_accuracy = None
        discriminability = None
        response_bias = None
        d_prime = None
    else:
        specificity = 1.0 - false_alarm_rate
        balanced_accuracy = 0.5 * (hit_rate + specificity)
        discriminability = hit_rate - false_alarm_rate
        denominator = 1.0 - discriminability
        response_bias = (
            false_alarm_rate / denominator
            if denominator > np.finfo(float).eps
            else None
        )
        corrected_hit = (hit_count + 0.5) / (match_count + 1.0)
        corrected_false_alarm = (false_alarm_count + 0.5) / (
            nonmatch_count + 1.0
        )
        normal = NormalDist()
        d_prime = float(
            normal.inv_cdf(corrected_hit)
            - normal.inv_cdf(corrected_false_alarm)
        )
    return {
        "hit_count": hit_count,
        "false_alarm_count": false_alarm_count,
        "match_count": match_count,
        "nonmatch_count": nonmatch_count,
        "hit_rate": hit_rate,
        "false_alarm_rate": false_alarm_rate,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "discriminability": discriminability,
        "response_bias": response_bias,
        "d_prime": d_prime,
    }


def _settling_values(
    logits: torch.Tensor,
    batch: NBackBatch,
    *,
    probability_threshold: float,
    margin_threshold: float,
    consecutive_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    settled_steps: list[int] = []
    behavioral_correct: list[bool] = []
    predictions, labels, scored = _item_arrays(logits, batch)
    for item_idx, onset in enumerate(batch.event_onsets):
        if not np.any(scored[item_idx]):
            continue
        event_probabilities = probabilities[
            onset : onset + batch.event_steps
        ]
        for batch_idx in range(event_probabilities.shape[1]):
            correct_class = int(labels[item_idx, batch_idx])
            correct_probability = event_probabilities[
                :, batch_idx, correct_class
            ]
            incorrect_probability = event_probabilities[
                :, batch_idx, 1 - correct_class
            ]
            passing = (
                (correct_probability >= probability_threshold)
                & (
                    correct_probability - incorrect_probability
                    >= margin_threshold
                )
            )
            settled = batch.event_steps
            for step in range(
                0, batch.event_steps - consecutive_steps + 1
            ):
                if np.all(passing[step : step + consecutive_steps]):
                    settled = step
                    break
            settled_steps.append(settled)
            behavioral_correct.append(
                bool(predictions[item_idx, batch_idx] == correct_class)
            )
    return (
        np.asarray(settled_steps, dtype=np.int64),
        np.asarray(behavioral_correct, dtype=bool),
    )


def _summarize_settling(
    values: np.ndarray,
    event_cap: int,
) -> dict[str, float | int | None]:
    if values.size == 0:
        return {
            "count": 0,
            "fraction_settled": None,
            "failure_rate": None,
            "median_settling_steps": None,
            "restricted_mean_settling_steps": None,
        }
    settled = values < event_cap
    fraction_settled = float(settled.mean())
    return {
        "count": int(values.size),
        "fraction_settled": fraction_settled,
        "failure_rate": float(1.0 - fraction_settled),
        "median_settling_steps": (
            float(np.median(values[settled])) if settled.any() else None
        ),
        "restricted_mean_settling_steps": float(values.mean()),
    }


def nback_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    batch: NBackBatch,
    *,
    probability_threshold: float = 0.80,
    margin_threshold: float = 0.60,
    consecutive_steps: int = 3,
) -> dict[str, Any]:
    """Compute one-observation-per-item N-back metrics."""
    _validate_batch_metric_inputs(logits, targets, loss_mask, batch)
    if (
        not np.isfinite(probability_threshold)
        or not 0.0 <= probability_threshold <= 1.0
    ):
        raise ValueError("probability_threshold must lie in [0, 1]")
    if (
        not np.isfinite(margin_threshold)
        or not 0.0 <= margin_threshold <= 1.0
    ):
        raise ValueError("margin_threshold must lie in [0, 1]")
    if consecutive_steps <= 0 or consecutive_steps > batch.event_steps:
        raise ValueError("consecutive_steps must fit within one event")
    predictions, labels, scored = _item_arrays(logits, batch)
    scored_predictions = predictions[scored]
    scored_labels = labels[scored]
    rates = _rates(scored_predictions, scored_labels)
    accuracy = float(np.mean(scored_predictions == scored_labels))
    sequence_losses = per_sequence_cross_entropy(
        logits, targets, loss_mask
    )
    settling, behavioral_correct = _settling_values(
        logits,
        batch,
        probability_threshold=probability_threshold,
        margin_threshold=margin_threshold,
        consecutive_steps=consecutive_steps,
    )
    all_settling = _summarize_settling(settling, batch.event_steps)
    correct_settling = _summarize_settling(
        settling[behavioral_correct], batch.event_steps
    )

    lure_mask = batch.one_back_lures & scored
    lure_count = int(lure_mask.sum())
    lure_predictions = predictions[lure_mask]
    lure_false_alarms = int((lure_predictions == 1).sum())
    lure_accuracy = (
        float(np.mean(lure_predictions == 0)) if lure_count else None
    )
    ordinary_nonmatch_mask = (
        scored & (labels == 0) & ~batch.one_back_lures
    )
    ordinary_nonmatch_count = int(ordinary_nonmatch_mask.sum())
    ordinary_nonmatch_predictions = predictions[ordinary_nonmatch_mask]
    ordinary_nonmatch_false_alarms = int(
        (ordinary_nonmatch_predictions == 1).sum()
    )
    ordinary_nonmatch_false_alarm_rate = (
        ordinary_nonmatch_false_alarms / ordinary_nonmatch_count
        if ordinary_nonmatch_count
        else None
    )

    return {
        "condition": batch.condition,
        "accuracy": accuracy,
        "mean_cross_entropy": float(sequence_losses.mean()),
        "sequence_cross_entropies": [
            float(value) for value in sequence_losses
        ],
        **rates,
        "one_back_lure_count": lure_count,
        "one_back_lure_false_alarm_count": lure_false_alarms,
        "one_back_lure_false_alarm_rate": (
            lure_false_alarms / lure_count if lure_count else None
        ),
        "one_back_lure_accuracy": lure_accuracy,
        "ordinary_nonmatch_count": ordinary_nonmatch_count,
        "ordinary_nonmatch_false_alarm_count": (
            ordinary_nonmatch_false_alarms
        ),
        "ordinary_nonmatch_false_alarm_rate": (
            ordinary_nonmatch_false_alarm_rate
        ),
        "ordinary_nonmatch_accuracy": (
            1.0 - ordinary_nonmatch_false_alarm_rate
            if ordinary_nonmatch_false_alarm_rate is not None
            else None
        ),
        "settling_all": all_settling,
        "settling_correct_decisions": correct_settling,
        "failure_rate": all_settling["failure_rate"],
        "settling_valid": (
            all_settling["fraction_settled"] is not None
            and all_settling["fraction_settled"] >= 0.80
        ),
    }
