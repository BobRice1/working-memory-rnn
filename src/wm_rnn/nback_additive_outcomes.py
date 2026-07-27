"""Bounded confirmatory summaries for additive-cost N-back outcomes.

This module performs no model loading, task generation, perturbation, or file
I/O. It only pools already-computed batch metrics, averages the registered
three-replicate P2/P5 summaries, forms checkpoint-level C2 contrasts, and
summarizes the frozen ten-checkpoint inferential vector.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np
from scipy import stats

from wm_rnn.nback_perturbation import (
    NBackLoadContrast,
    candidate_vs_p5_load_contrast,
)


NBACK_CHECKPOINT_SEEDS = (
    20260912,
    20260913,
    20260914,
    20260915,
    20260916,
    20260917,
    20260918,
    20260919,
    20260920,
    20260921,
)

_COUNT_FIELDS = (
    "hit_count",
    "false_alarm_count",
    "match_count",
    "nonmatch_count",
    "one_back_lure_count",
    "one_back_lure_false_alarm_count",
    "ordinary_nonmatch_count",
    "ordinary_nonmatch_false_alarm_count",
)

_REPLICATE_SCALAR_FIELDS = (
    "accuracy",
    "hit_rate",
    "false_alarm_rate",
    "specificity",
    "balanced_accuracy",
    "discriminability",
    "d_prime",
    "mean_cross_entropy",
    "ordinary_nonmatch_false_alarm_rate",
    "ordinary_nonmatch_accuracy",
    "failure_rate",
)


@dataclass(frozen=True)
class C2CheckpointTest:
    """One-sided Student-t summary of exactly ten checkpoint C2 values."""

    checkpoint_seeds: tuple[int, ...]
    checkpoint_values: tuple[float, ...]
    n: int
    mean: float
    sample_sd: float
    paired_dz: float
    sign_count: int
    sign_fraction: float
    t_statistic: float
    one_sided_p_value: float
    ci_95_lower: float
    ci_95_upper: float


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{context} is missing required field {key!r}")
    return mapping[key]


def _finite_float(value: Any, name: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if not np.isfinite(resolved):
        raise ValueError(f"{name} must be a finite scalar")
    return resolved


def _count(value: Any, name: str) -> int:
    resolved = _finite_float(value, name)
    if resolved < 0.0 or not resolved.is_integer():
        raise ValueError(f"{name} must be a non-negative integer")
    return int(resolved)


def _pooled_rates(
    *,
    hit_count: int,
    false_alarm_count: int,
    match_count: int,
    nonmatch_count: int,
) -> dict[str, float | None]:
    if match_count <= 0 or nonmatch_count <= 0:
        raise ValueError("pooled conditions require nonzero class counts")
    if hit_count > match_count or false_alarm_count > nonmatch_count:
        raise ValueError("response counts cannot exceed task class counts")
    hit_rate = hit_count / match_count
    false_alarm_rate = false_alarm_count / nonmatch_count
    specificity = 1.0 - false_alarm_rate
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
    return {
        "hit_rate": hit_rate,
        "false_alarm_rate": false_alarm_rate,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (hit_rate + specificity),
        "discriminability": discriminability,
        "response_bias": response_bias,
        "d_prime": float(
            normal.inv_cdf(corrected_hit)
            - normal.inv_cdf(corrected_false_alarm)
        ),
    }


def _pool_settling(
    batch_metrics: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, float | int | None]:
    total_count = 0
    settled_total = 0.0
    restricted_total = 0.0
    for index, metrics in enumerate(batch_metrics):
        context = f"batch {index} {key}"
        summary = _required(metrics, key, f"batch {index}")
        if not isinstance(summary, Mapping):
            raise ValueError(f"{context} must be a mapping")
        count = _count(_required(summary, "count", context), f"{context}.count")
        fraction = _required(summary, "fraction_settled", context)
        restricted = _required(
            summary, "restricted_mean_settling_steps", context
        )
        if count == 0:
            if fraction is not None or restricted is not None:
                raise ValueError(
                    f"{context} zero-count summaries must use None values"
                )
            continue
        fraction_value = _finite_float(
            fraction, f"{context}.fraction_settled"
        )
        restricted_value = _finite_float(
            restricted,
            f"{context}.restricted_mean_settling_steps",
        )
        if not 0.0 <= fraction_value <= 1.0 or restricted_value < 0.0:
            raise ValueError(f"{context} contains an invalid settling value")
        failure = _finite_float(
            _required(summary, "failure_rate", context),
            f"{context}.failure_rate",
        )
        if not np.isclose(
            failure, 1.0 - fraction_value, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                f"{context}.failure_rate must equal 1 - fraction_settled"
            )
        total_count += count
        settled_total += count * fraction_value
        restricted_total += count * restricted_value
    if total_count == 0:
        return {
            "count": 0,
            "fraction_settled": None,
            "failure_rate": None,
            "median_settling_steps": None,
            "restricted_mean_settling_steps": None,
        }
    fraction_settled = settled_total / total_count
    return {
        "count": total_count,
        "fraction_settled": fraction_settled,
        "failure_rate": 1.0 - fraction_settled,
        # Exact medians cannot be reconstructed from batch-level summaries.
        "median_settling_steps": None,
        "restricted_mean_settling_steps": restricted_total / total_count,
    }


def pool_condition_batch_metrics(
    batch_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pool raw counts across batches before calculating condition rates."""
    if not batch_metrics:
        raise ValueError("batch_metrics must contain at least one batch")
    if any(not isinstance(item, Mapping) for item in batch_metrics):
        raise ValueError("every batch metric must be a mapping")
    condition = str(_required(batch_metrics[0], "condition", "batch 0"))
    if condition not in {"0-back", "2-back"}:
        raise ValueError("condition must be '0-back' or '2-back'")

    totals = {field: 0 for field in _COUNT_FIELDS}
    sequence_losses: list[float] = []
    for index, metrics in enumerate(batch_metrics):
        context = f"batch {index}"
        if str(_required(metrics, "condition", context)) != condition:
            raise ValueError("all batches must have the same condition")
        for field in _COUNT_FIELDS:
            totals[field] += _count(
                _required(metrics, field, context),
                f"{context}.{field}",
            )
        losses = _required(metrics, "sequence_cross_entropies", context)
        if (
            isinstance(losses, (str, bytes))
            or not isinstance(losses, Sequence)
            or len(losses) == 0
        ):
            raise ValueError(
                f"{context}.sequence_cross_entropies must be non-empty"
            )
        for value in losses:
            loss = _finite_float(
                value, f"{context}.sequence_cross_entropies"
            )
            if loss < 0.0:
                raise ValueError("sequence cross-entropies must be non-negative")
            sequence_losses.append(loss)

    if (
        totals["one_back_lure_false_alarm_count"]
        > totals["one_back_lure_count"]
        or totals["ordinary_nonmatch_false_alarm_count"]
        > totals["ordinary_nonmatch_count"]
    ):
        raise ValueError("subset false alarms cannot exceed subset counts")
    if (
        totals["one_back_lure_count"]
        + totals["ordinary_nonmatch_count"]
        != totals["nonmatch_count"]
    ):
        raise ValueError(
            "lure and ordinary-nonmatch counts must partition nonmatches"
        )
    if (
        totals["one_back_lure_false_alarm_count"]
        + totals["ordinary_nonmatch_false_alarm_count"]
        != totals["false_alarm_count"]
    ):
        raise ValueError(
            "lure and ordinary false alarms must partition false alarms"
        )

    rates = _pooled_rates(
        hit_count=totals["hit_count"],
        false_alarm_count=totals["false_alarm_count"],
        match_count=totals["match_count"],
        nonmatch_count=totals["nonmatch_count"],
    )
    correct_count = (
        totals["hit_count"]
        + totals["nonmatch_count"]
        - totals["false_alarm_count"]
    )
    decision_count = totals["match_count"] + totals["nonmatch_count"]
    lure_count = totals["one_back_lure_count"]
    ordinary_count = totals["ordinary_nonmatch_count"]
    lure_false_alarm_rate = (
        totals["one_back_lure_false_alarm_count"] / lure_count
        if lure_count
        else None
    )
    ordinary_false_alarm_rate = (
        totals["ordinary_nonmatch_false_alarm_count"] / ordinary_count
        if ordinary_count
        else None
    )
    settling_all = _pool_settling(batch_metrics, "settling_all")
    settling_correct = _pool_settling(
        batch_metrics, "settling_correct_decisions"
    )

    return {
        "condition": condition,
        "n_batches": len(batch_metrics),
        "n_sequences": len(sequence_losses),
        **totals,
        "accuracy": correct_count / decision_count,
        **rates,
        "mean_cross_entropy": float(np.mean(sequence_losses)),
        "sequence_cross_entropies": sequence_losses,
        "one_back_lure_false_alarm_rate": lure_false_alarm_rate,
        "one_back_lure_accuracy": (
            1.0 - lure_false_alarm_rate
            if lure_false_alarm_rate is not None
            else None
        ),
        "ordinary_nonmatch_false_alarm_rate": ordinary_false_alarm_rate,
        "ordinary_nonmatch_accuracy": (
            1.0 - ordinary_false_alarm_rate
            if ordinary_false_alarm_rate is not None
            else None
        ),
        "settling_all": settling_all,
        "settling_correct_decisions": settling_correct,
        "failure_rate": settling_all["failure_rate"],
        "settling_valid": bool(
            settling_all["fraction_settled"] is not None
            and settling_all["fraction_settled"] >= 0.80
        ),
    }


def aggregate_three_replicate_condition_metrics(
    replicate_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Average three scalar metrics after each replicate was pooled."""
    if len(replicate_metrics) != 3:
        raise ValueError("exactly three pooled replicate metrics are required")
    if any(not isinstance(item, Mapping) for item in replicate_metrics):
        raise ValueError("every replicate metric must be a mapping")
    conditions = [
        str(_required(item, "condition", f"replicate {index}"))
        for index, item in enumerate(replicate_metrics)
    ]
    if len(set(conditions)) != 1 or conditions[0] not in {
        "0-back",
        "2-back",
    }:
        raise ValueError("all replicates must have the same valid condition")
    for index, item in enumerate(replicate_metrics):
        if _count(
            _required(item, "n_batches", f"replicate {index}"),
            f"replicate {index}.n_batches",
        ) <= 0:
            raise ValueError("each replicate must already pool complete batches")
        n_sequences = _count(
            _required(item, "n_sequences", f"replicate {index}"),
            f"replicate {index}.n_sequences",
        )
        losses = _required(
            item, "sequence_cross_entropies", f"replicate {index}"
        )
        if (
            isinstance(losses, (str, bytes))
            or not isinstance(losses, Sequence)
            or len(losses) != n_sequences
            or n_sequences == 0
        ):
            raise ValueError(
                "each replicate must retain all pooled sequence losses"
            )
        if any(
            _finite_float(value, "replicate sequence cross-entropy") < 0.0
            for value in losses
        ):
            raise ValueError(
                "replicate sequence cross-entropies must be non-negative"
            )

    result: dict[str, Any] = {
        "condition": conditions[0],
        "n_replicates": 3,
    }
    for field in _REPLICATE_SCALAR_FIELDS:
        values = [
            _finite_float(
                _required(item, field, f"replicate {index}"),
                f"replicate {index}.{field}",
            )
            for index, item in enumerate(replicate_metrics)
        ]
        result[field] = float(np.mean(values))

    if conditions[0] == "2-back":
        for field in (
            "one_back_lure_false_alarm_rate",
            "one_back_lure_accuracy",
        ):
            values = [
                _finite_float(
                    _required(item, field, f"replicate {index}"),
                    f"replicate {index}.{field}",
                )
                for index, item in enumerate(replicate_metrics)
            ]
            result[field] = float(np.mean(values))
    else:
        for index, item in enumerate(replicate_metrics):
            if _count(
                _required(
                    item, "one_back_lure_count", f"replicate {index}"
                ),
                f"replicate {index}.one_back_lure_count",
            ) != 0:
                raise ValueError("0-back replicates must not contain lures")
        result["one_back_lure_false_alarm_rate"] = None
        result["one_back_lure_accuracy"] = None

    response_biases = [
        _required(item, "response_bias", f"replicate {index}")
        for index, item in enumerate(replicate_metrics)
    ]
    result["response_bias"] = (
        None
        if any(value is None for value in response_biases)
        else float(
            np.mean(
                [
                    _finite_float(value, "replicate response_bias")
                    for value in response_biases
                ]
            )
        )
    )

    for field in _COUNT_FIELDS:
        values = [
            _count(
                _required(item, field, f"replicate {index}"),
                f"replicate {index}.{field}",
            )
            for index, item in enumerate(replicate_metrics)
        ]
        result[field] = float(np.mean(values))
        result[f"{field}_by_replicate"] = values

    for denominator in (
        "match_count",
        "nonmatch_count",
        "one_back_lure_count",
        "ordinary_nonmatch_count",
    ):
        values = result[f"{denominator}_by_replicate"]
        if len(set(values)) != 1:
            raise ValueError(
                f"{denominator} must match across identical replicate tasks"
            )
        result[denominator] = values[0]

    for key in ("settling_all", "settling_correct_decisions"):
        summaries = [
            _required(item, key, f"replicate {index}")
            for index, item in enumerate(replicate_metrics)
        ]
        if any(not isinstance(item, Mapping) for item in summaries):
            raise ValueError(f"{key} must be a mapping in every replicate")
        averaged: dict[str, float | None] = {}
        for field in (
            "count",
            "fraction_settled",
            "failure_rate",
            "restricted_mean_settling_steps",
        ):
            raw = [
                _required(item, field, f"replicate {index} {key}")
                for index, item in enumerate(summaries)
            ]
            averaged[field] = (
                None
                if any(value is None for value in raw)
                else float(
                    np.mean(
                        [
                            _finite_float(value, f"{key}.{field}")
                            for value in raw
                        ]
                    )
                )
            )
        averaged["median_settling_steps"] = None
        result[key] = averaged
    result["settling_valid"] = bool(
        result["settling_all"]["fraction_settled"] is not None
        and result["settling_all"]["fraction_settled"] >= 0.80
    )
    return result


def baseline_transport_checks(
    zero_back: Mapping[str, Any],
    two_back: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen confirmatory-bank native competence gates."""
    if str(_required(zero_back, "condition", "0-back")) != "0-back":
        raise ValueError("zero_back metrics must have condition '0-back'")
    if str(_required(two_back, "condition", "2-back")) != "2-back":
        raise ValueError("two_back metrics must have condition '2-back'")

    zero_accuracy = _finite_float(
        _required(zero_back, "accuracy", "0-back"), "0-back accuracy"
    )
    zero_d = _finite_float(
        _required(zero_back, "discriminability", "0-back"),
        "0-back discriminability",
    )
    two_accuracy = _finite_float(
        _required(two_back, "accuracy", "2-back"), "2-back accuracy"
    )
    two_d = _finite_float(
        _required(two_back, "discriminability", "2-back"),
        "2-back discriminability",
    )
    lure_accuracy = _finite_float(
        _required(two_back, "one_back_lure_accuracy", "2-back"),
        "2-back one_back_lure_accuracy",
    )
    checks = {
        "zero_back_accuracy": zero_accuracy >= 0.95,
        "zero_back_discriminability": zero_d >= 0.90,
        "two_back_accuracy": two_accuracy >= 0.95,
        "two_back_discriminability": two_d >= 0.90,
        "two_back_lure_accuracy": lure_accuracy >= 0.90,
        "zero_back_has_matches": _count(
            _required(zero_back, "match_count", "0-back"),
            "0-back match_count",
        )
        > 0,
        "zero_back_has_nonmatches": _count(
            _required(zero_back, "nonmatch_count", "0-back"),
            "0-back nonmatch_count",
        )
        > 0,
        "two_back_has_matches": _count(
            _required(two_back, "match_count", "2-back"),
            "2-back match_count",
        )
        > 0,
        "two_back_has_nonmatches": _count(
            _required(two_back, "nonmatch_count", "2-back"),
            "2-back nonmatch_count",
        )
        > 0,
        "two_back_has_lures": _count(
            _required(two_back, "one_back_lure_count", "2-back"),
            "2-back one_back_lure_count",
        )
        > 0,
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def checkpoint_candidate_vs_p5_c2(
    *,
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    p5: Mapping[str, Mapping[str, Any]],
) -> NBackLoadContrast:
    """Calculate one checkpoint C2 from complete condition summaries."""
    expected = {"0-back", "2-back"}
    for name, conditions in (
        ("baseline", baseline),
        ("candidate", candidate),
        ("p5", p5),
    ):
        if set(conditions) != expected:
            raise ValueError(f"{name} must contain exactly 0-back and 2-back")
        for condition in expected:
            metric_condition = str(
                _required(
                    conditions[condition],
                    "condition",
                    f"{name} {condition}",
                )
            )
            if metric_condition != condition:
                raise ValueError(
                    f"{name} {condition} has mismatched condition metadata"
                )

    discriminability = {
        (name, condition): _finite_float(
            _required(metrics[condition], "discriminability", name),
            f"{name} {condition} discriminability",
        )
        for name, metrics in (
            ("baseline", baseline),
            ("candidate", candidate),
            ("p5", p5),
        )
        for condition in expected
    }
    return candidate_vs_p5_load_contrast(
        baseline_zero_back=discriminability[("baseline", "0-back")],
        baseline_two_back=discriminability[("baseline", "2-back")],
        candidate_zero_back=discriminability[("candidate", "0-back")],
        candidate_two_back=discriminability[("candidate", "2-back")],
        p5_zero_back=discriminability[("p5", "0-back")],
        p5_two_back=discriminability[("p5", "2-back")],
    )


def one_sided_c2_checkpoint_test(
    checkpoint_values: Mapping[int, float],
) -> C2CheckpointTest:
    """Summarize the frozen ten-checkpoint C2 vector."""
    if set(checkpoint_values) != set(NBACK_CHECKPOINT_SEEDS):
        raise ValueError(
            "checkpoint_values must contain exactly the ten frozen checkpoints"
        )
    sample = np.asarray(
        [
            _finite_float(
                checkpoint_values[seed], f"checkpoint {seed} C2"
            )
            for seed in NBACK_CHECKPOINT_SEEDS
        ],
        dtype=np.float64,
    )
    n = sample.size
    mean = float(np.mean(sample))
    sample_sd = float(np.std(sample, ddof=1))
    sign_count = int(np.sum(sample > 0.0))
    if sample_sd == 0.0:
        if mean > 0.0:
            t_statistic = np.inf
            p_value = 0.0
            paired_dz = np.inf
        elif mean < 0.0:
            t_statistic = -np.inf
            p_value = 1.0
            paired_dz = -np.inf
        else:
            t_statistic = 0.0
            p_value = 0.5
            paired_dz = 0.0
        ci_lower = mean
        ci_upper = mean
    else:
        standard_error = sample_sd / np.sqrt(n)
        t_statistic = mean / standard_error
        p_value = float(stats.t.sf(t_statistic, df=n - 1))
        paired_dz = mean / sample_sd
        margin = float(stats.t.ppf(0.975, df=n - 1) * standard_error)
        ci_lower = mean - margin
        ci_upper = mean + margin
    return C2CheckpointTest(
        checkpoint_seeds=NBACK_CHECKPOINT_SEEDS,
        checkpoint_values=tuple(float(value) for value in sample),
        n=n,
        mean=mean,
        sample_sd=sample_sd,
        paired_dz=float(paired_dz),
        sign_count=sign_count,
        sign_fraction=sign_count / n,
        t_statistic=float(t_statistic),
        one_sided_p_value=float(p_value),
        ci_95_lower=float(ci_lower),
        ci_95_upper=float(ci_upper),
    )
