"""Tests for bounded additive-cost N-back confirmatory summaries."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from scipy import stats

from wm_rnn.nback_additive_outcomes import (
    NBACK_CHECKPOINT_SEEDS,
    aggregate_three_replicate_condition_metrics,
    baseline_transport_checks,
    checkpoint_candidate_vs_p5_c2,
    one_sided_c2_checkpoint_test,
    pool_condition_batch_metrics,
)


def _batch_metric(
    *,
    condition: str = "2-back",
    hit_count: int = 5,
    match_count: int = 6,
    false_alarm_count: int = 1,
    nonmatch_count: int = 12,
    lure_count: int = 3,
    lure_false_alarms: int = 0,
    losses: tuple[float, ...] = (0.1, 0.2),
) -> dict[str, object]:
    ordinary_count = nonmatch_count - lure_count
    ordinary_false_alarms = false_alarm_count - lure_false_alarms
    fraction = 0.75
    correct_count = hit_count + nonmatch_count - false_alarm_count
    return {
        "condition": condition,
        "hit_count": hit_count,
        "false_alarm_count": false_alarm_count,
        "match_count": match_count,
        "nonmatch_count": nonmatch_count,
        "one_back_lure_count": lure_count,
        "one_back_lure_false_alarm_count": lure_false_alarms,
        "ordinary_nonmatch_count": ordinary_count,
        "ordinary_nonmatch_false_alarm_count": ordinary_false_alarms,
        "sequence_cross_entropies": list(losses),
        "settling_all": {
            "count": match_count + nonmatch_count,
            "fraction_settled": fraction,
            "failure_rate": 1.0 - fraction,
            "median_settling_steps": 2.0,
            "restricted_mean_settling_steps": 3.0,
        },
        "settling_correct_decisions": {
            "count": correct_count,
            "fraction_settled": fraction if correct_count else None,
            "failure_rate": 1.0 - fraction if correct_count else None,
            "median_settling_steps": 2.0 if correct_count else None,
            "restricted_mean_settling_steps": (
                2.5 if correct_count else None
            ),
        },
    }


def test_pooling_uses_raw_counts_not_mean_batch_rates() -> None:
    small_perfect = _batch_metric(
        hit_count=1,
        match_count=1,
        false_alarm_count=0,
        nonmatch_count=1,
        lure_count=0,
        losses=(0.1,),
    )
    large_failed = _batch_metric(
        hit_count=0,
        match_count=9,
        false_alarm_count=9,
        nonmatch_count=9,
        lure_count=0,
        losses=(0.2, 0.3, 0.4),
    )

    pooled = pool_condition_batch_metrics(
        [small_perfect, large_failed]
    )

    assert pooled["hit_rate"] == pytest.approx(1 / 10)
    assert pooled["false_alarm_rate"] == pytest.approx(9 / 10)
    assert pooled["discriminability"] == pytest.approx(-0.8)
    assert pooled["accuracy"] == pytest.approx(2 / 20)
    assert pooled["hit_rate"] != pytest.approx(0.5)
    assert pooled["mean_cross_entropy"] == pytest.approx(0.25)


def test_pooling_recomputes_registered_subset_and_settling_metrics() -> None:
    first = _batch_metric()
    second = _batch_metric(
        hit_count=6,
        false_alarm_count=3,
        lure_false_alarms=1,
        losses=(0.3,),
    )
    second["settling_all"]["fraction_settled"] = 0.5
    second["settling_all"]["failure_rate"] = 0.5
    second["settling_all"]["restricted_mean_settling_steps"] = 5.0

    pooled = pool_condition_batch_metrics([first, second])

    assert pooled["match_count"] == 12
    assert pooled["one_back_lure_count"] == 6
    assert pooled["one_back_lure_false_alarm_rate"] == pytest.approx(1 / 6)
    assert pooled["ordinary_nonmatch_false_alarm_rate"] == pytest.approx(3 / 18)
    assert pooled["settling_all"]["fraction_settled"] == pytest.approx(
        0.625
    )
    assert pooled["failure_rate"] == pytest.approx(0.375)
    assert pooled["settling_all"][
        "restricted_mean_settling_steps"
    ] == pytest.approx(4.0)


def test_three_replicates_average_scalar_metrics_after_pooling() -> None:
    replicates = []
    for hit_count, false_alarms in ((6, 0), (5, 1), (3, 3)):
        replicates.append(
            pool_condition_batch_metrics(
                [
                    _batch_metric(
                        hit_count=hit_count,
                        false_alarm_count=false_alarms,
                        lure_false_alarms=0,
                    )
                ]
            )
        )

    result = aggregate_three_replicate_condition_metrics(replicates)

    assert result["n_replicates"] == 3
    assert result["discriminability"] == pytest.approx(
        np.mean([item["discriminability"] for item in replicates])
    )
    assert result["d_prime"] == pytest.approx(
        np.mean([item["d_prime"] for item in replicates])
    )
    concatenated_d_prime = pool_condition_batch_metrics(
        [
            _batch_metric(
                hit_count=sum((6, 5, 3)),
                match_count=18,
                false_alarm_count=sum((0, 1, 3)),
                nonmatch_count=36,
                lure_count=9,
                lure_false_alarms=0,
            )
        ]
    )["d_prime"]
    assert result["d_prime"] != pytest.approx(concatenated_d_prime)


def test_replicate_aggregation_requires_three_complete_finite_replicates() -> None:
    pooled = pool_condition_batch_metrics([_batch_metric()])
    with pytest.raises(ValueError, match="exactly three"):
        aggregate_three_replicate_condition_metrics([pooled, pooled])

    invalid = deepcopy(pooled)
    invalid["discriminability"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        aggregate_three_replicate_condition_metrics(
            [pooled, pooled, invalid]
        )


def test_zero_back_replicates_preserve_structural_lure_na() -> None:
    pooled = pool_condition_batch_metrics(
        [
            _batch_metric(
                condition="0-back",
                hit_count=6,
                false_alarm_count=0,
                lure_count=0,
                losses=(0.01,),
            )
        ]
    )

    result = aggregate_three_replicate_condition_metrics(
        [pooled, pooled, pooled]
    )

    assert result["one_back_lure_count"] == 0
    assert result["one_back_lure_accuracy"] is None
    assert result["one_back_lure_false_alarm_rate"] is None


def _condition(condition: str, discriminability: float) -> dict[str, object]:
    return {
        "condition": condition,
        "discriminability": discriminability,
    }


def test_checkpoint_c2_uses_registered_positive_sign() -> None:
    result = checkpoint_candidate_vs_p5_c2(
        baseline={
            "0-back": _condition("0-back", 0.90),
            "2-back": _condition("2-back", 0.80),
        },
        candidate={
            "0-back": _condition("0-back", 0.81),
            "2-back": _condition("2-back", 0.56),
        },
        p5={
            "0-back": _condition("0-back", 0.72),
            "2-back": _condition("2-back", 0.60),
        },
    )

    assert result.c2_nback == pytest.approx(0.15)


def test_baseline_transport_includes_raw_class_and_lure_guards() -> None:
    zero = pool_condition_batch_metrics(
        [
            _batch_metric(
                condition="0-back",
                hit_count=6,
                false_alarm_count=0,
                lure_count=0,
                losses=(0.01,),
            )
        ]
    )
    two = pool_condition_batch_metrics(
        [
            _batch_metric(
                hit_count=6,
                false_alarm_count=0,
                lure_false_alarms=0,
                losses=(0.02,),
            )
        ]
    )

    passed = baseline_transport_checks(zero, two)
    assert passed["passed"] is True

    no_lures = dict(two)
    no_lures["one_back_lure_count"] = 0
    failed = baseline_transport_checks(zero, no_lures)
    assert failed["passed"] is False
    assert failed["checks"]["two_back_has_lures"] is False


def test_one_sided_test_requires_exact_frozen_checkpoint_family() -> None:
    values = {
        seed: 0.01 * (index + 1)
        for index, seed in enumerate(NBACK_CHECKPOINT_SEEDS)
    }
    result = one_sided_c2_checkpoint_test(values)
    sample = np.asarray(list(values.values()), dtype=float)
    expected_t = sample.mean() / (
        sample.std(ddof=1) / np.sqrt(sample.size)
    )

    assert result.n == 10
    assert result.checkpoint_seeds == NBACK_CHECKPOINT_SEEDS
    assert result.mean == pytest.approx(sample.mean())
    assert result.sample_sd == pytest.approx(sample.std(ddof=1))
    assert result.paired_dz == pytest.approx(
        sample.mean() / sample.std(ddof=1)
    )
    assert result.sign_count == 10
    assert result.sign_fraction == 1.0
    assert result.t_statistic == pytest.approx(expected_t)
    assert result.one_sided_p_value == pytest.approx(
        stats.t.sf(expected_t, df=9)
    )
    assert result.ci_95_lower < result.mean < result.ci_95_upper

    incomplete = dict(values)
    incomplete.pop(NBACK_CHECKPOINT_SEEDS[-1])
    with pytest.raises(ValueError, match="exactly the ten"):
        one_sided_c2_checkpoint_test(incomplete)


def test_missing_and_nonfinite_primary_metrics_are_rejected() -> None:
    baseline = {
        "0-back": _condition("0-back", 0.9),
        "2-back": _condition("2-back", 0.8),
    }
    candidate = {
        "0-back": _condition("0-back", 0.8),
        "2-back": _condition("2-back", np.nan),
    }
    p5 = {
        "0-back": _condition("0-back", 0.8),
        "2-back": _condition("2-back", 0.7),
    }
    with pytest.raises(ValueError, match="finite"):
        checkpoint_candidate_vs_p5_c2(
            baseline=baseline,
            candidate=candidate,
            p5=p5,
        )

    values = {seed: 0.1 for seed in NBACK_CHECKPOINT_SEEDS}
    values[NBACK_CHECKPOINT_SEEDS[4]] = np.inf
    with pytest.raises(ValueError, match="finite"):
        one_sided_c2_checkpoint_test(values)
