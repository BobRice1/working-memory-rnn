import json

from wm_rnn.config import load_config
from wm_rnn.train import train_model


def test_tuned_training_metrics_do_not_report_categorical_accuracy(tmp_path):
    config = load_config("configs/tuned_delay.yaml")
    config["model"]["hidden_size"] = 8
    config["task"]["n_tuned_units"] = 8
    config["task"]["batch_size"] = 4
    config["training"]["device"] = "cpu"
    config["training"]["steps"] = 1
    config["training"]["log_every"] = 1
    config["paths"]["output_dir"] = str(tmp_path / "tuned_train")
    config["paths"]["run_name"] = "tuned_train"

    result = train_model(config)

    assert "accuracy" not in result.history[-1]
    assert result.history[-1]["population_mse"] == result.history[-1]["loss"]

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert "final_accuracy" not in metrics
    assert metrics["final_population_mse"] == result.history[-1]["population_mse"]
