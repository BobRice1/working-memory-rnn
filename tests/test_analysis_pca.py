import numpy as np
import torch

from wm_rnn.analysis import run_pca_analysis
from wm_rnn.training_utils import fresh_model


def test_run_pca_analysis_saves_tuned_labels_and_task_type(tmp_path):
    config = {
        "task": {
            "task_type": "tuned",
            "n_tuned_units": 8,
            "tuning_kappa": 4.0,
            "cue_steps": 1,
            "delay_steps": 1,
            "response_steps": 1,
            "batch_size": 4,
            "seed": 3,
        },
        "model": {
            "hidden_size": 4,
            "dt": 20.0,
            "tau": 100.0,
            "activation": "tanh",
        },
        "training": {"device": "cpu"},
        "analysis": {"n_trials": 5, "n_components": 2},
        "paths": {"output_dir": str(tmp_path), "run_name": "tuned_pca_test"},
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    model = fresh_model(config, torch.device("cpu"))
    torch.save({"model_state": model.state_dict()}, checkpoint_path)

    result = run_pca_analysis(config, checkpoint_path)

    assert result.figure_path.exists()
    with np.load(result.hidden_states_path) as archive:
        assert archive["task_type"].item() == "tuned"
        assert archive["labels"].shape == (5,)
        assert np.issubdtype(archive["labels"].dtype, np.floating)
