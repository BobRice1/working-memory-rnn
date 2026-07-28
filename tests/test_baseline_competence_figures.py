"""Smoke tests for baseline competence figure generation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wm_rnn.baseline_competence_figures import (
    plot_circular_delay_sweep,
    plot_circular_pool_competence,
    plot_nback_pool_competence,
)


def test_circular_pool_competence_plot(tmp_path: Path) -> None:
    pool = {
        "retained_checkpoint_seeds": [1, 2],
        "results": [
            {
                "seed": 1,
                "clean_mean_angular_error_degrees": 2.0,
                "distractor_mean_angular_error_degrees": 3.0,
            },
            {
                "seed": 2,
                "clean_mean_angular_error_degrees": 2.5,
                "distractor_mean_angular_error_degrees": 4.0,
            },
            {
                "seed": 3,
                "clean_mean_angular_error_degrees": 12.0,
                "distractor_mean_angular_error_degrees": 16.0,
            },
        ],
    }
    path = plot_circular_pool_competence(pool, tmp_path / "circular_pool.png")
    assert path.exists()
    assert path.stat().st_size > 0


def test_nback_pool_competence_plot(tmp_path: Path) -> None:
    pool = {
        "results": [
            {
                "seed": 10,
                "retained": True,
                "zero_back_accuracy": 1.0,
                "two_back_accuracy": 0.97,
                "zero_back_discriminability": 1.0,
                "two_back_discriminability": 0.95,
                "two_back_lure_accuracy": 0.98,
            },
            {
                "seed": 11,
                "retained": True,
                "zero_back_accuracy": 0.99,
                "two_back_accuracy": 0.96,
                "zero_back_discriminability": 0.99,
                "two_back_discriminability": 0.94,
                "two_back_lure_accuracy": 0.97,
            },
        ]
    }
    path = plot_nback_pool_competence(pool, tmp_path / "nback_pool.png")
    assert path.exists()


def test_circular_delay_sweep_plot(tmp_path: Path) -> None:
    rows = [
        {
            "condition": "clean",
            "delay_steps": 10,
            "mean_angular_error_degrees": 2.0,
        },
        {
            "condition": "clean",
            "delay_steps": 20,
            "mean_angular_error_degrees": 2.2,
        },
        {
            "condition": "distractor",
            "delay_steps": 20,
            "mean_angular_error_degrees": 3.1,
        },
    ]
    path = plot_circular_delay_sweep(rows, tmp_path / "delay.png")
    assert path.exists()
    assert np.isfinite(rows[0]["mean_angular_error_degrees"])
