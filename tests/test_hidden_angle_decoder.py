"""Tests for hidden-state circular angle decoding."""

from __future__ import annotations

import numpy as np

from wm_rnn.hidden_angle_decoder import (
    decode_angles_from_hidden,
    fit_hidden_angle_decoder,
    resolve_angle_decode_method,
)


def test_fit_and_decode_recovers_angles() -> None:
    rng = np.random.default_rng(0)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=64)
    # Construct an explicit circular code in 8 units.
    prefs = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    hidden = np.cos(angles[:, None] - prefs[None, :])
    weights = fit_hidden_angle_decoder(hidden, angles, ridge_alpha=1e-3)
    decoded = decode_angles_from_hidden(hidden, weights)
    err = np.abs((decoded - angles + np.pi) % (2 * np.pi) - np.pi)
    assert float(np.degrees(err).mean()) < 1.0


def test_resolve_method_for_yang_config() -> None:
    assert (
        resolve_angle_decode_method({"task": {"task_type": "tuned", "fixation_gated": True}})
        == "hidden_ridge"
    )
    assert (
        resolve_angle_decode_method({"task": {"task_type": "tuned", "fixation_gated": False}})
        == "population_readout"
    )
