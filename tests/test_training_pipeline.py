from pathlib import Path

import numpy as np

from wm_rnn.analysis import run_pca_analysis
from wm_rnn.config import default_config
from wm_rnn.evaluate import evaluate_model
from wm_rnn.train import train_model


def tiny_config(tmp_path: Path):
    config = default_config()
    config["task"].update(
        {
            "n_classes": 3,
            "cue_steps": 2,
            "delay_steps": 3,
            "response_steps": 2,
            "batch_size": 8,
            "seed": 7,
        }
    )
    config["model"].update({"hidden_size": 12, "dt": 20.0, "tau": 100.0})
    config["training"].update({"steps": 3, "learning_rate": 0.01, "log_every": 1, "device": "cpu"})
    config["evaluation"].update({"batches": 2})
    config["paths"].update({"output_dir": str(tmp_path)})
    return config


def test_short_training_run_writes_checkpoint_and_metrics(tmp_path):
    result = train_model(tiny_config(tmp_path))

    assert result.checkpoint_path.exists()
    assert result.metrics_path.exists()
    assert len(result.history) == 3
    assert result.history[-1]["loss"] >= 0.0


def test_evaluation_and_pca_write_outputs(tmp_path):
    train_result = train_model(tiny_config(tmp_path))

    eval_result = evaluate_model(tiny_config(tmp_path), train_result.checkpoint_path)
    pca_result = run_pca_analysis(tiny_config(tmp_path), train_result.checkpoint_path)

    assert eval_result.metrics_path.exists()
    assert 0.0 <= eval_result.metrics["accuracy"] <= 1.0
    assert pca_result.figure_path.exists()
    assert pca_result.hidden_states_path.exists()
    assert np.load(pca_result.hidden_states_path)["hidden"].ndim == 3
