"""Outcome-free tests for N-back perturbation integration."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wm_rnn.model import RNNConfig, WorkingMemoryRNN
from wm_rnn.nback_perturbation import (
    NBACK_OPERATOR_NAMES,
    build_nback_operator,
    build_neutral_nback_operator,
    candidate_vs_p5_load_contrast,
    sequence_log_loss_units,
    summarize_sequence_log_loss_cost,
)
from wm_rnn.nback_task import NBackTaskConfig, generate_nback_batch


def _model() -> WorkingMemoryRNN:
    torch.manual_seed(41)
    model = WorkingMemoryRNN(
        RNNConfig(
            input_size=8,
            hidden_size=11,
            output_size=2,
            dt=20.0,
            tau=100.0,
            activation="tanh",
        )
    )
    model.eval()
    return model


@pytest.mark.parametrize("operator", NBACK_OPERATOR_NAMES)
def test_neutral_operators_reproduce_native_nback_forward(
    operator: str,
) -> None:
    model = _model()
    inputs = torch.randn(
        (13, 4, 8),
        generator=torch.Generator().manual_seed(73),
    )

    expected = model(inputs)
    actual = build_neutral_nback_operator(model, operator)(inputs)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_nback_sensory_gain_changes_stimuli_but_not_rule_contexts() -> None:
    model = _model()
    with torch.no_grad():
        model.rnn.h2h.weight.zero_()
        model.rnn.h2h.bias.zero_()
    forward = build_nback_operator(
        model,
        "sensory_input_gain",
        gain=1.7,
    )
    context_only = torch.zeros((1, 3, 8))
    context_only[:, :, 6:] = torch.tensor([1.0, 0.0])
    stimulus_only = torch.zeros((1, 3, 8))
    stimulus_only[:, :, 0] = 1.0

    native_context = model(context_only)
    gained_context = forward(context_only)
    native_stimulus = model(stimulus_only)
    gained_stimulus = forward(stimulus_only)

    torch.testing.assert_close(gained_context[0], native_context[0])
    torch.testing.assert_close(gained_context[1], native_context[1])
    assert not torch.equal(gained_stimulus[1], native_stimulus[1])


def test_cost_units_are_one_log_loss_observation_per_sequence() -> None:
    batch = generate_nback_batch(
        NBackTaskConfig(n_back=0, batch_size=3, seed=101)
    )
    logits = torch.zeros((*batch.targets.shape, 2))

    units = sequence_log_loss_units(logits, batch)
    summary = summarize_sequence_log_loss_cost(units, units + 0.05)

    assert units.shape == (3,)
    np.testing.assert_allclose(units, np.log(2.0))
    assert summary.n_sequences == 3
    assert summary.baseline_mean_log_loss == pytest.approx(np.log(2.0))
    assert summary.perturbed_mean_log_loss == pytest.approx(
        np.log(2.0) + 0.05
    )
    assert summary.additive_cost == pytest.approx(0.05)


def test_candidate_vs_p5_load_contrast_has_registered_positive_sign() -> None:
    contrast = candidate_vs_p5_load_contrast(
        baseline_zero_back=0.90,
        baseline_two_back=0.80,
        candidate_zero_back=0.81,
        candidate_two_back=0.56,
        p5_zero_back=0.72,
        p5_two_back=0.60,
    )

    assert contrast.candidate_zero_back_impairment == pytest.approx(0.10)
    assert contrast.candidate_two_back_impairment == pytest.approx(0.30)
    assert contrast.candidate_load_selectivity == pytest.approx(0.20)
    assert contrast.p5_zero_back_impairment == pytest.approx(0.20)
    assert contrast.p5_two_back_impairment == pytest.approx(0.25)
    assert contrast.p5_load_selectivity == pytest.approx(0.05)
    assert contrast.c2_nback == pytest.approx(0.15)


def test_nback_adapter_rejects_context_gain_and_wrong_architecture() -> None:
    model = _model()
    with pytest.raises(ValueError, match="exactly six"):
        build_nback_operator(
            model,
            "sensory_input_gain",
            gain=1.1,
            n_tuned_units=8,
        )

    wrong = WorkingMemoryRNN(
        RNNConfig(input_size=10, hidden_size=11, output_size=2)
    )
    with pytest.raises(ValueError, match="8-input, 2-output"):
        build_neutral_nback_operator(wrong, "recurrent_gain")
