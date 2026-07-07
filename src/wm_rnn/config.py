"""Configuration loading and default settings for baseline experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def default_config() -> dict[str, Any]:
    """Return the default baseline delayed-response experiment config."""
    return {
        "task": {
            "task_type": "categorical",
            "n_classes": 4,
            "cue_steps": 5,
            "delay_steps": 20,
            "response_steps": 5,
            "batch_size": 64,
            "seed": 20260629,
        },
        "model": {
            "hidden_size": 64,
            "dt": 20.0,
            "tau": 100.0,
            "activation": "tanh",
        },
        "training": {
            "steps": 1000,
            "learning_rate": 0.001,
            "log_every": 50,
            "device": "auto",
        },
        "evaluation": {
            "batches": 20,
        },
        "analysis": {
            "n_components": 2,
            "n_trials": 64,
        },
        "paths": {
            "output_dir": "outputs/baseline_delay",
            "run_name": "baseline_delay",
        },
    }


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config and merge it onto package defaults.

    Args:
        path: Optional YAML file path. If omitted, only defaults are returned.

    Returns:
        Nested configuration dictionary with default values filled in.
    """
    config = default_config()
    if path is None:
        return config

    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return _deep_merge(config, loaded)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` values into ``base`` without mutation."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
