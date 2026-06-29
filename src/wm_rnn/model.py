"""Neural network definitions for the baseline working-memory RNN."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RNNConfig:
    """Configuration for the continuous-time recurrent model.

    Attributes:
        input_size: Number of input channels per time step.
        hidden_size: Number of recurrent hidden units.
        output_size: Number of categorical readout classes.
        dt: Simulation time step used to compute the leak factor.
        tau: Recurrent time constant used to compute the leak factor.
    """

    input_size: int
    hidden_size: int
    output_size: int
    dt: float = 20.0
    tau: float = 100.0


class CTRNN(nn.Module):
    """Continuous-time ReLU RNN using a ``dt / tau`` leak update.

    This is a simple rate-style recurrent layer. It is not spiking,
    E/I-constrained, or biologically detailed; the time constant is included so
    later experiments can vary memory stability in an interpretable way.
    """

    def __init__(self, config: RNNConfig):
        """Initialize recurrent input, recurrent, and leak parameters.

        Args:
            config: Model sizes and time constants.

        Raises:
            ValueError: If ``dt`` or ``tau`` cannot produce a valid leak factor.
        """
        super().__init__()
        if config.tau <= 0:
            raise ValueError("tau must be positive")
        if config.dt <= 0:
            raise ValueError("dt must be positive")
        alpha = config.dt / config.tau
        if not 0 < alpha <= 1:
            raise ValueError("dt / tau must be in (0, 1]")

        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.alpha = alpha
        self.oneminusalpha = 1.0 - alpha
        self.input2h = nn.Linear(config.input_size, config.hidden_size)
        self.h2h = nn.Linear(config.hidden_size, config.hidden_size)

    def init_hidden(self, input_shape: torch.Size, device: torch.device) -> torch.Tensor:
        """Create a zero initial hidden state for an input sequence.

        Args:
            input_shape: Shape of a sequence tensor ``(time, batch, input)``.
            device: Torch device where the hidden state should live.

        Returns:
            Tensor shaped ``(batch, hidden_size)``.
        """
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def recurrence(self, input_t: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Advance hidden activity by one time step.

        Args:
            input_t: Current input tensor shaped ``(batch, input_size)``.
            hidden: Previous hidden state shaped ``(batch, hidden_size)``.

        Returns:
            Updated nonnegative hidden state.
        """
        pre_activation = self.input2h(input_t) + self.h2h(hidden)
        return torch.relu(hidden * self.oneminusalpha + pre_activation * self.alpha)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the recurrent layer over a full sequence.

        Args:
            inputs: Sequence tensor shaped ``(time, batch, input_size)``.
            hidden: Optional initial hidden state.

        Returns:
            Pair of all hidden states ``(time, batch, hidden_size)`` and the
            final hidden state ``(batch, hidden_size)``.
        """
        if hidden is None:
            hidden = self.init_hidden(inputs.shape, inputs.device)

        states = []
        for step in range(inputs.size(0)):
            hidden = self.recurrence(inputs[step], hidden)
            states.append(hidden)
        return torch.stack(states, dim=0), hidden


class WorkingMemoryRNN(nn.Module):
    """Baseline continuous-time RNN with a linear class readout."""

    def __init__(self, config: RNNConfig):
        """Initialize recurrent layer and output readout.

        Args:
            config: Model dimensions and continuous-time update settings.
        """
        super().__init__()
        self.config = config
        self.rnn = CTRNN(config)
        self.readout = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict cue class logits for every time step.

        Args:
            inputs: Sequence tensor shaped ``(time, batch, input_size)``.

        Returns:
            Pair of logits ``(time, batch, output_size)`` and hidden states
            ``(time, batch, hidden_size)``.
        """
        hidden_states, _ = self.rnn(inputs)
        logits = self.readout(hidden_states)
        return logits, hidden_states
