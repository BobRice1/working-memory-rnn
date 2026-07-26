"""Explicit post-training perturbation operators for the signature experiment.

Each factory returns a no-gradient forward function with the same public output
as :class:`wm_rnn.model.WorkingMemoryRNN`.  The timestep loop is intentionally
spelled out so that the placement of every perturbation relative to the leak,
synaptic drive, biases, and nonlinearity remains inspectable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wm_rnn.model import WorkingMemoryRNN


ForwardFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
OperatorFactory = Callable[..., ForwardFn]
_BIAS_MODES = {"bias_outside", "bias_inside"}


def _validate_model(model: WorkingMemoryRNN) -> None:
    if not isinstance(model, WorkingMemoryRNN):
        raise TypeError("model must be a WorkingMemoryRNN")


def _validate_positive(value: float, name: str) -> float:
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return resolved


def _validate_bias_mode(bias_mode: str) -> str:
    if bias_mode not in _BIAS_MODES:
        raise ValueError(f"bias_mode must be one of {sorted(_BIAS_MODES)}")
    return bias_mode


def _validate_tuned_units(model: WorkingMemoryRNN, n_tuned_units: int) -> int:
    resolved = int(n_tuned_units)
    if not 0 < resolved <= model.config.input_size:
        raise ValueError("n_tuned_units must lie within the model input size")
    return resolved


def _bias_sum(model: WorkingMemoryRNN) -> torch.Tensor:
    return model.rnn.input2h.bias + model.rnn.h2h.bias


def _weight_drive(
    model: WorkingMemoryRNN,
    input_t: torch.Tensor,
    hidden: torch.Tensor,
) -> torch.Tensor:
    return F.linear(input_t, model.rnn.input2h.weight) + F.linear(
        hidden, model.rnn.h2h.weight
    )


def _native_drive(
    model: WorkingMemoryRNN,
    input_t: torch.Tensor,
    hidden: torch.Tensor,
) -> torch.Tensor:
    return model.rnn.input2h(input_t) + model.rnn.h2h(hidden)


def _run_explicit(
    model: WorkingMemoryRNN,
    inputs: torch.Tensor,
    update: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if inputs.ndim != 3 or inputs.shape[-1] != model.config.input_size:
        raise ValueError("inputs must have shape [time, batch, model input_size]")
    hidden = model.rnn.init_hidden(inputs.shape, inputs.device)
    states = []
    for step in range(inputs.shape[0]):
        hidden = update(step, inputs[step], hidden)
        states.append(hidden)
    hidden_states = torch.stack(states, dim=0)
    return model.readout(hidden_states), hidden_states


def heterogeneous_gain_vector(
    hidden_size: int,
    log_std: float,
    vector_seed: int,
) -> np.ndarray:
    """Return the fixed positive P2 gain vector with exact arithmetic mean one."""
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    resolved_log_std = float(log_std)
    if not np.isfinite(resolved_log_std) or resolved_log_std < 0.0:
        raise ValueError("log_std must be finite and non-negative")
    rng = np.random.default_rng(int(vector_seed))
    vector = np.exp(
        rng.normal(0.0, resolved_log_std, size=int(hidden_size))
    ).astype(np.float64)
    vector /= np.mean(vector)
    return vector


def time_constant_coefficients(
    model: WorkingMemoryRNN, tau_scale: float
) -> tuple[float, float]:
    """Return P7 carried-state and drive coefficients."""
    _validate_model(model)
    scale = _validate_positive(tau_scale, "tau_scale")
    alpha_prime = min(float(model.rnn.alpha) / scale, 1.0)
    return 1.0 - alpha_prime, alpha_prime


def state_persistence_coefficients(
    model: WorkingMemoryRNN, persistence_gain: float
) -> tuple[float, float]:
    """Return P6 carried-state and unchanged drive coefficients."""
    _validate_model(model)
    gain = _validate_positive(persistence_gain, "persistence_gain")
    return float(model.rnn.oneminusalpha) * gain, float(model.rnn.alpha)


def synaptic_drive_gain(
    model: WorkingMemoryRNN,
    *,
    gain: float,
    bias_mode: str = "bias_outside",
) -> ForwardFn:
    """Create P1 scalar synaptic-drive gain (not exact F-I response gain)."""
    _validate_model(model)
    resolved_gain = _validate_positive(gain, "gain")
    resolved_bias_mode = _validate_bias_mode(bias_mode)

    @torch.no_grad()
    def forward(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if resolved_gain == 1.0:
            return model(inputs)

        def update(
            _step: int, input_t: torch.Tensor, hidden: torch.Tensor
        ) -> torch.Tensor:
            weight_drive = _weight_drive(model, input_t, hidden)
            if resolved_bias_mode == "bias_outside":
                drive = resolved_gain * weight_drive + _bias_sum(model)
            else:
                drive = resolved_gain * (weight_drive + _bias_sum(model))
            pre_activation = (
                model.rnn.oneminusalpha * hidden + model.rnn.alpha * drive
            )
            return model.rnn._activation(pre_activation)

        return _run_explicit(model, inputs, update)

    return forward


def heterogeneous_drive_gain(
    model: WorkingMemoryRNN,
    *,
    log_std: float | None = None,
    vector_seed: int = 3101,
    bias_mode: str = "bias_outside",
    gain_vector: np.ndarray | None = None,
) -> ForwardFn:
    """Create P2 fixed heterogeneous synaptic-drive gain."""
    _validate_model(model)
    resolved_bias_mode = _validate_bias_mode(bias_mode)
    if gain_vector is not None:
        vector = np.asarray(gain_vector, dtype=np.float64)
        if vector.shape != (model.config.hidden_size,):
            raise ValueError("gain_vector must have shape [hidden_size]")
        if not np.all(np.isfinite(vector)) or np.any(vector <= 0.0):
            raise ValueError("gain_vector must contain finite positive values")
        if not np.isclose(np.mean(vector), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("gain_vector must have population mean 1")
        vector = vector.copy()
    else:
        if log_std is None:
            raise ValueError("log_std is required when gain_vector is not supplied")
        vector = heterogeneous_gain_vector(
            model.config.hidden_size, log_std, vector_seed
        )
    is_neutral = np.array_equal(vector, np.ones_like(vector))

    @torch.no_grad()
    def forward(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if is_neutral:
            return model(inputs)
        gain_vector = torch.as_tensor(
            vector, dtype=inputs.dtype, device=inputs.device
        )

        def update(
            _step: int, input_t: torch.Tensor, hidden: torch.Tensor
        ) -> torch.Tensor:
            weight_drive = _weight_drive(model, input_t, hidden)
            if resolved_bias_mode == "bias_outside":
                drive = gain_vector * weight_drive + _bias_sum(model)
            else:
                drive = gain_vector * (weight_drive + _bias_sum(model))
            pre_activation = (
                model.rnn.oneminusalpha * hidden + model.rnn.alpha * drive
            )
            return model.rnn._activation(pre_activation)

        return _run_explicit(model, inputs, update)

    forward.gain_vector = vector.copy()  # type: ignore[attr-defined]
    return forward


def _sensory_scaled_input_drive(
    model: WorkingMemoryRNN,
    input_t: torch.Tensor,
    n_tuned_units: int,
    gain: float,
) -> torch.Tensor:
    sensory = F.linear(
        input_t[:, :n_tuned_units],
        model.rnn.input2h.weight[:, :n_tuned_units],
    )
    control = F.linear(
        input_t[:, n_tuned_units:],
        model.rnn.input2h.weight[:, n_tuned_units:],
    )
    return gain * sensory + control + model.rnn.input2h.bias


def sensory_input_gain(
    model: WorkingMemoryRNN,
    *,
    gain: float,
    n_tuned_units: int,
) -> ForwardFn:
    """Create P3a gain on tuned sensory input contribution only."""
    _validate_model(model)
    resolved_gain = _validate_positive(gain, "gain")
    tuned_units = _validate_tuned_units(model, n_tuned_units)

    @torch.no_grad()
    def forward(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if resolved_gain == 1.0:
            return model(inputs)

        def update(
            _step: int, input_t: torch.Tensor, hidden: torch.Tensor
        ) -> torch.Tensor:
            input_drive = _sensory_scaled_input_drive(
                model, input_t, tuned_units, resolved_gain
            )
            drive = input_drive + model.rnn.h2h(hidden)
            pre_activation = (
                model.rnn.oneminusalpha * hidden + model.rnn.alpha * drive
            )
            return model.rnn._activation(pre_activation)

        return _run_explicit(model, inputs, update)

    return forward


def distractor_input_gain(
    model: WorkingMemoryRNN,
    *,
    gain: float,
    n_tuned_units: int,
    distractor_slice: slice | None,
) -> ForwardFn:
    """Create P3b gain on tuned input during the distractor slice only."""
    _validate_model(model)
    resolved_gain = _validate_positive(gain, "gain")
    tuned_units = _validate_tuned_units(model, n_tuned_units)
    if distractor_slice is not None and not isinstance(distractor_slice, slice):
        raise TypeError("distractor_slice must be a slice or None")

    @torch.no_grad()
    def forward(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            resolved_gain == 1.0
            or distractor_slice is None
            or len(range(*distractor_slice.indices(inputs.shape[0]))) == 0
        ):
            return model(inputs)
        active_steps = set(range(*distractor_slice.indices(inputs.shape[0])))

        def update(
            step: int, input_t: torch.Tensor, hidden: torch.Tensor
        ) -> torch.Tensor:
            step_gain = resolved_gain if step in active_steps else 1.0
            input_drive = _sensory_scaled_input_drive(
                model, input_t, tuned_units, step_gain
            )
            drive = input_drive + model.rnn.h2h(hidden)
            pre_activation = (
                model.rnn.oneminusalpha * hidden + model.rnn.alpha * drive
            )
            return model.rnn._activation(pre_activation)

        return _run_explicit(model, inputs, update)

    return forward


def recurrent_gain(
    model: WorkingMemoryRNN,
    *,
    gain: float,
) -> ForwardFn:
    """Create P4 gain on recurrent weights with recurrent bias unscaled."""
    _validate_model(model)
    resolved_gain = _validate_positive(gain, "gain")

    @torch.no_grad()
    def forward(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if resolved_gain == 1.0:
            return model(inputs)

        def update(
            _step: int, input_t: torch.Tensor, hidden: torch.Tensor
        ) -> torch.Tensor:
            recurrent_drive = (
                resolved_gain * F.linear(hidden, model.rnn.h2h.weight)
                + model.rnn.h2h.bias
            )
            drive = model.rnn.input2h(input_t) + recurrent_drive
            pre_activation = (
                model.rnn.oneminusalpha * hidden + model.rnn.alpha * drive
            )
            return model.rnn._activation(pre_activation)

        return _run_explicit(model, inputs, update)

    return forward


def gaussian_state_noise(
    model: WorkingMemoryRNN,
    *,
    sigma: float,
    generator_seed: int,
) -> ForwardFn:
    """Create P5 seeded Gaussian noise on the pre-activation update."""
    _validate_model(model)
    resolved_sigma = float(sigma)
    if not np.isfinite(resolved_sigma) or resolved_sigma < 0.0:
        raise ValueError("sigma must be finite and non-negative")
    resolved_seed = int(generator_seed)

    @torch.no_grad()
    def forward(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if resolved_sigma == 0.0:
            return model(inputs)
        generator = torch.Generator(device=inputs.device)
        generator.manual_seed(resolved_seed)

        def update(
            _step: int, input_t: torch.Tensor, hidden: torch.Tensor
        ) -> torch.Tensor:
            drive = _native_drive(model, input_t, hidden)
            pre_activation = (
                model.rnn.oneminusalpha * hidden + model.rnn.alpha * drive
            )
            noise = torch.randn(
                hidden.shape,
                dtype=hidden.dtype,
                device=hidden.device,
                generator=generator,
            )
            return model.rnn._activation(
                pre_activation + resolved_sigma * noise
            )

        return _run_explicit(model, inputs, update)

    return forward


def state_persistence(
    model: WorkingMemoryRNN,
    *,
    persistence_gain: float,
) -> ForwardFn:
    """Create P6 asymmetric carried-state persistence gain."""
    carried_coefficient, drive_coefficient = state_persistence_coefficients(
        model, persistence_gain
    )

    @torch.no_grad()
    def forward(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if float(persistence_gain) == 1.0:
            return model(inputs)

        def update(
            _step: int, input_t: torch.Tensor, hidden: torch.Tensor
        ) -> torch.Tensor:
            drive = _native_drive(model, input_t, hidden)
            pre_activation = (
                carried_coefficient * hidden + drive_coefficient * drive
            )
            return model.rnn._activation(pre_activation)

        return _run_explicit(model, inputs, update)

    return forward


def time_constant(
    model: WorkingMemoryRNN,
    *,
    tau_scale: float,
) -> ForwardFn:
    """Create P7 conserved-integrator time-constant scaling."""
    carried_coefficient, drive_coefficient = time_constant_coefficients(
        model, tau_scale
    )

    @torch.no_grad()
    def forward(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if float(tau_scale) == 1.0:
            return model(inputs)

        def update(
            _step: int, input_t: torch.Tensor, hidden: torch.Tensor
        ) -> torch.Tensor:
            drive = _native_drive(model, input_t, hidden)
            pre_activation = (
                carried_coefficient * hidden + drive_coefficient * drive
            )
            return model.rnn._activation(pre_activation)

        return _run_explicit(model, inputs, update)

    return forward


def heterogeneous_input_recurrent_hybrid(
    model: WorkingMemoryRNN,
    *,
    q: float,
    n_tuned_units: int,
    log_std: float | None = None,
    vector_seed: int = 3101,
    bias_mode: str = "bias_outside",
    gain_vector: np.ndarray | None = None,
) -> ForwardFn:
    """Create the Phase 9 P2 plus sensory/recurrent ratio hybrid.

    Tuned sensory weights receive ``q`` and recurrent weights receive ``1/q``.
    Control-channel weights remain at one. The fixed P2 gain vector is applied
    after this component reweighting, with biases handled by the selected D2
    variant.
    """
    _validate_model(model)
    ratio = _validate_positive(q, "q")
    tuned_units = _validate_tuned_units(model, n_tuned_units)
    resolved_bias_mode = _validate_bias_mode(bias_mode)
    if gain_vector is None:
        if log_std is None:
            raise ValueError("log_std is required when gain_vector is not supplied")
        vector = heterogeneous_gain_vector(
            model.config.hidden_size, log_std, vector_seed
        )
    else:
        vector = np.asarray(gain_vector, dtype=np.float64)
        if vector.shape != (model.config.hidden_size,):
            raise ValueError("gain_vector must have shape [hidden_size]")
        if not np.all(np.isfinite(vector)) or np.any(vector <= 0.0):
            raise ValueError("gain_vector must contain finite positive values")
        if not np.isclose(np.mean(vector), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("gain_vector must have population mean 1")

    @torch.no_grad()
    def forward(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gain_tensor = torch.as_tensor(
            vector, dtype=inputs.dtype, device=inputs.device
        )

        def update(
            _step: int, input_t: torch.Tensor, hidden: torch.Tensor
        ) -> torch.Tensor:
            sensory = F.linear(
                input_t[:, :tuned_units],
                model.rnn.input2h.weight[:, :tuned_units],
            )
            control = F.linear(
                input_t[:, tuned_units:],
                model.rnn.input2h.weight[:, tuned_units:],
            )
            recurrent = F.linear(hidden, model.rnn.h2h.weight)
            reweighted = ratio * sensory + control + recurrent / ratio
            if resolved_bias_mode == "bias_outside":
                drive = gain_tensor * reweighted + _bias_sum(model)
            else:
                drive = gain_tensor * (reweighted + _bias_sum(model))
            pre_activation = (
                model.rnn.oneminusalpha * hidden + model.rnn.alpha * drive
            )
            return model.rnn._activation(pre_activation)

        return _run_explicit(model, inputs, update)

    forward.gain_vector = np.asarray(vector).copy()  # type: ignore[attr-defined]
    forward.component_gains = {  # type: ignore[attr-defined]
        "sensory": ratio,
        "control": 1.0,
        "recurrent": 1.0 / ratio,
    }
    return forward


OPERATORS: dict[str, OperatorFactory] = {
    "synaptic_drive_gain": synaptic_drive_gain,
    "heterogeneous_drive_gain": heterogeneous_drive_gain,
    "sensory_input_gain": sensory_input_gain,
    "distractor_input_gain": distractor_input_gain,
    "recurrent_gain": recurrent_gain,
    "gaussian_state_noise": gaussian_state_noise,
    "state_persistence": state_persistence,
    "time_constant": time_constant,
}


def neutral_parameters(name: str) -> dict[str, Any]:
    """Return the frozen neutral strength parameters for an operator."""
    parameters: dict[str, dict[str, Any]] = {
        "synaptic_drive_gain": {"gain": 1.0, "bias_mode": "bias_outside"},
        "heterogeneous_drive_gain": {
            "log_std": 0.0,
            "vector_seed": 3101,
            "bias_mode": "bias_outside",
        },
        "sensory_input_gain": {"gain": 1.0},
        "distractor_input_gain": {"gain": 1.0},
        "recurrent_gain": {"gain": 1.0},
        "gaussian_state_noise": {"sigma": 0.0, "generator_seed": 4101},
        "state_persistence": {"persistence_gain": 1.0},
        "time_constant": {"tau_scale": 1.0},
    }
    if name not in parameters:
        raise KeyError(f"unknown operator: {name}")
    return parameters[name].copy()
