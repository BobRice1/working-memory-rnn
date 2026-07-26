"""Tests for explicitly placed post-training perturbation operators."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wm_rnn.model import RNNConfig, WorkingMemoryRNN
from wm_rnn.perturbation_operators import (
    OPERATORS,
    distractor_input_gain,
    gaussian_state_noise,
    heterogeneous_drive_gain,
    heterogeneous_gain_vector,
    heterogeneous_input_recurrent_hybrid,
    neutral_parameters,
    recurrent_gain,
    sensory_input_gain,
    state_persistence,
    state_persistence_coefficients,
    synaptic_drive_gain,
    time_constant,
    time_constant_coefficients,
)


def _model() -> WorkingMemoryRNN:
    torch.manual_seed(7)
    model = WorkingMemoryRNN(
        RNNConfig(
            input_size=10,
            hidden_size=12,
            output_size=9,
            dt=20.0,
            tau=100.0,
            activation="tanh",
        )
    )
    model.eval()
    return model


def _inputs() -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    return torch.randn((9, 5, 10), generator=generator)


def _neutral_forward(name: str, model: WorkingMemoryRNN):
    parameters = neutral_parameters(name)
    if name in {"sensory_input_gain", "distractor_input_gain"}:
        parameters["n_tuned_units"] = 8
    if name == "distractor_input_gain":
        parameters["distractor_slice"] = slice(3, 5)
    return OPERATORS[name](model, **parameters)


@pytest.mark.parametrize("name", sorted(OPERATORS))
def test_every_neutral_operator_reproduces_native_forward(name: str) -> None:
    model = _model()
    inputs = _inputs()
    expected_predictions, expected_hidden = model(inputs)

    actual_predictions, actual_hidden = _neutral_forward(name, model)(inputs)

    torch.testing.assert_close(actual_predictions, expected_predictions)
    torch.testing.assert_close(actual_hidden, expected_hidden)


def test_distractor_gain_is_exact_identity_without_distractor_window() -> None:
    model = _model()
    inputs = _inputs()
    expected = model(inputs)

    actual = distractor_input_gain(
        model,
        gain=1.5,
        n_tuned_units=8,
        distractor_slice=slice(4, 4),
    )(inputs)

    torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("factory_name", ["sensory", "distractor"])
def test_input_gain_leaves_fixation_and_probe_only_input_unchanged(
    factory_name: str,
) -> None:
    model = _model()
    inputs = torch.zeros((7, 4, 10))
    inputs[:, :, 8] = 1.0
    inputs[5:, :, 9] = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    expected = model(inputs)
    if factory_name == "sensory":
        forward = sensory_input_gain(
            model, gain=1.7, n_tuned_units=8
        )
    else:
        forward = distractor_input_gain(
            model,
            gain=1.7,
            n_tuned_units=8,
            distractor_slice=slice(2, 5),
        )

    actual = forward(inputs)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_heterogeneous_gain_vector_is_positive_fixed_and_mean_one() -> None:
    vector = heterogeneous_gain_vector(12, log_std=0.1, vector_seed=3101)
    repeat = heterogeneous_gain_vector(12, log_std=0.1, vector_seed=3101)
    different = heterogeneous_gain_vector(12, log_std=0.1, vector_seed=3102)

    assert np.all(vector > 0.0)
    assert np.mean(vector) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_array_equal(vector, repeat)
    assert not np.array_equal(vector, different)

    forward = heterogeneous_drive_gain(
        _model(),
        log_std=0.1,
        vector_seed=3101,
        bias_mode="bias_outside",
    )
    np.testing.assert_array_equal(forward.gain_vector, vector)  # type: ignore[attr-defined]


@pytest.mark.parametrize("tau_scale", [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25])
def test_time_constant_conserves_integrator_coefficients(
    tau_scale: float,
) -> None:
    carried, drive = time_constant_coefficients(_model(), tau_scale)

    assert 0.0 < drive <= 1.0
    assert carried >= 0.0
    assert carried + drive == pytest.approx(1.0, abs=1e-12)


def test_state_persistence_changes_only_carried_coefficient() -> None:
    model = _model()
    native_carried = model.rnn.oneminusalpha
    native_drive = model.rnn.alpha

    carried, drive = state_persistence_coefficients(
        model, persistence_gain=0.9
    )

    assert carried == pytest.approx(native_carried * 0.9)
    assert drive == pytest.approx(native_drive)
    assert carried + drive != pytest.approx(1.0)


def test_gaussian_noise_is_deterministic_for_fixed_seed() -> None:
    model = _model()
    inputs = _inputs()
    forward = gaussian_state_noise(
        model, sigma=0.05, generator_seed=4101
    )

    first = forward(inputs)
    second = forward(inputs)
    other = gaussian_state_noise(
        model, sigma=0.05, generator_seed=4102
    )(inputs)

    torch.testing.assert_close(first[0], second[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(first[1], second[1], rtol=0.0, atol=0.0)
    assert not torch.equal(first[1], other[1])


@pytest.mark.parametrize(
    ("factory", "kwargs", "match"),
    [
        (synaptic_drive_gain, {"gain": 0.0}, "gain"),
        (
            synaptic_drive_gain,
            {"gain": 1.1, "bias_mode": "unknown"},
            "bias_mode",
        ),
        (
            heterogeneous_drive_gain,
            {"log_std": -0.1, "vector_seed": 1},
            "log_std",
        ),
        (
            heterogeneous_drive_gain,
            {"gain_vector": np.ones(11)},
            "gain_vector",
        ),
        (
            sensory_input_gain,
            {"gain": 1.1, "n_tuned_units": 11},
            "n_tuned_units",
        ),
        (recurrent_gain, {"gain": -1.0}, "gain"),
        (
            gaussian_state_noise,
            {"sigma": -0.01, "generator_seed": 1},
            "sigma",
        ),
        (
            state_persistence,
            {"persistence_gain": 0.0},
            "persistence_gain",
        ),
        (time_constant, {"tau_scale": 0.0}, "tau_scale"),
    ],
)
def test_invalid_operator_parameters_raise(
    factory: object, kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        factory(_model(), **kwargs)  # type: ignore[operator]


def test_registry_and_neutral_parameters_are_complete() -> None:
    assert set(OPERATORS) == {
        "synaptic_drive_gain",
        "heterogeneous_drive_gain",
        "sensory_input_gain",
        "distractor_input_gain",
        "recurrent_gain",
        "gaussian_state_noise",
        "state_persistence",
        "time_constant",
    }
    with pytest.raises(KeyError, match="unknown operator"):
        neutral_parameters("not_an_operator")


@pytest.mark.parametrize("bias_mode", ["bias_outside", "bias_inside"])
def test_hybrid_q_one_reproduces_p2_component(bias_mode: str) -> None:
    model = _model()
    inputs = _inputs()
    p2 = heterogeneous_drive_gain(
        model,
        log_std=0.1,
        vector_seed=3101,
        bias_mode=bias_mode,
    )
    hybrid = heterogeneous_input_recurrent_hybrid(
        model,
        q=1.0,
        n_tuned_units=8,
        log_std=0.1,
        vector_seed=3101,
        bias_mode=bias_mode,
    )

    expected = p2(inputs)
    actual = hybrid(inputs)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_hybrid_records_ratio_and_inverse_placement() -> None:
    hybrid = heterogeneous_input_recurrent_hybrid(
        _model(),
        q=1.1,
        n_tuned_units=8,
        log_std=0.1,
        vector_seed=3101,
    )

    assert hybrid.component_gains == {  # type: ignore[attr-defined]
        "sensory": 1.1,
        "control": 1.0,
        "recurrent": pytest.approx(1.0 / 1.1),
    }


def test_hybrid_q_does_not_change_control_only_input_when_recurrence_is_silenced() -> None:
    model = _model()
    with torch.no_grad():
        model.rnn.h2h.weight.zero_()
    inputs = torch.zeros((1, 4, 10))
    inputs[:, :, 8:] = torch.tensor([1.0, -1.0])
    q_low = heterogeneous_input_recurrent_hybrid(
        model,
        q=0.9,
        n_tuned_units=8,
        log_std=0.1,
        vector_seed=3101,
    )
    q_high = heterogeneous_input_recurrent_hybrid(
        model,
        q=1.1,
        n_tuned_units=8,
        log_std=0.1,
        vector_seed=3101,
    )

    low = q_low(inputs)
    high = q_high(inputs)

    torch.testing.assert_close(low[0], high[0])
    torch.testing.assert_close(low[1], high[1])
