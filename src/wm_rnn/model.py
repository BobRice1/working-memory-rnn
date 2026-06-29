from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RNNConfig:
    input_size: int
    hidden_size: int
    output_size: int
    dt: float = 20.0
    tau: float = 100.0


class CTRNN(nn.Module):
    """Continuous-time ReLU RNN using a dt/tau leak update."""

    def __init__(self, config: RNNConfig):
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
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def recurrence(self, input_t: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        pre_activation = self.input2h(input_t) + self.h2h(hidden)
        return torch.relu(hidden * self.oneminusalpha + pre_activation * self.alpha)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
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
        super().__init__()
        self.config = config
        self.rnn = CTRNN(config)
        self.readout = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, _ = self.rnn(inputs)
        logits = self.readout(hidden_states)
        return logits, hidden_states
