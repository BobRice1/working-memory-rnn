from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DelayTaskConfig:
    n_classes: int = 4
    cue_steps: int = 5
    delay_steps: int = 20
    response_steps: int = 5
    batch_size: int = 64
    seed: int | None = None

    @property
    def input_size(self) -> int:
        return self.n_classes + 1

    @property
    def seq_len(self) -> int:
        return self.cue_steps + self.delay_steps + self.response_steps


@dataclass(frozen=True)
class DelayBatch:
    inputs: np.ndarray
    targets: np.ndarray
    loss_mask: np.ndarray
    cues: np.ndarray
    phase_index: dict[str, slice]


def generate_delay_batch(config: DelayTaskConfig) -> DelayBatch:
    """Generate a categorical delayed-response working-memory batch.

    Inputs have shape (time, batch, classes + fixation channel). The final
    channel is a constant fixation/context signal. Targets are the remembered
    cue class at every time point, but the loss mask scores only response steps.
    """
    if config.n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    if min(config.cue_steps, config.delay_steps, config.response_steps, config.batch_size) <= 0:
        raise ValueError("cue_steps, delay_steps, response_steps, and batch_size must be positive")

    rng = np.random.default_rng(config.seed)
    cues = rng.integers(0, config.n_classes, size=config.batch_size, dtype=np.int64)

    inputs = np.zeros((config.seq_len, config.batch_size, config.input_size), dtype=np.float32)
    targets = np.tile(cues, (config.seq_len, 1)).astype(np.int64)
    loss_mask = np.zeros((config.seq_len, config.batch_size), dtype=np.float32)

    cue_slice = slice(0, config.cue_steps)
    delay_slice = slice(config.cue_steps, config.cue_steps + config.delay_steps)
    response_slice = slice(config.cue_steps + config.delay_steps, config.seq_len)

    for trial_idx, cue in enumerate(cues):
        inputs[cue_slice, trial_idx, cue] = 1.0

    inputs[:, :, -1] = 1.0
    loss_mask[response_slice, :] = 1.0

    return DelayBatch(
        inputs=inputs,
        targets=targets,
        loss_mask=loss_mask,
        cues=cues,
        phase_index={"cue": cue_slice, "delay": delay_slice, "response": response_slice},
    )
