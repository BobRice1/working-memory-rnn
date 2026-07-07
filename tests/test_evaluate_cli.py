from pathlib import Path

from wm_rnn import evaluate


def test_main_prints_tuned_metrics_without_accuracy(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "wm_rnn.evaluate",
            "--config",
            "configs/tuned_delay.yaml",
            "--checkpoint",
            "checkpoint.pt",
            "--device",
            "cpu",
        ],
    )
    monkeypatch.setattr(evaluate, "load_config", lambda _path: {"training": {}})
    monkeypatch.setattr(
        evaluate,
        "evaluate_model",
        lambda _config, _checkpoint: evaluate.EvalResult(
            metrics_path=tmp_path / "metrics.json",
            confusion_path=None,
            metrics={
                "mean_angular_error_degrees": 12.3456,
                "population_mse": 0.01234,
            },
        ),
    )

    evaluate.main()

    assert capsys.readouterr().out.splitlines() == [
        f"metrics={Path(tmp_path / 'metrics.json')}",
        "mean_angular_error_degrees=12.346",
        "population_mse=0.012",
    ]


def test_main_keeps_categorical_accuracy_output(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "wm_rnn.evaluate",
            "--config",
            "configs/baseline_delay.yaml",
            "--checkpoint",
            "checkpoint.pt",
        ],
    )
    monkeypatch.setattr(evaluate, "load_config", lambda _path: {"training": {}})
    monkeypatch.setattr(
        evaluate,
        "evaluate_model",
        lambda _config, _checkpoint: evaluate.EvalResult(
            metrics_path=tmp_path / "metrics.json",
            confusion_path=tmp_path / "confusion.csv",
            metrics={"accuracy": 0.9876},
        ),
    )

    evaluate.main()

    assert capsys.readouterr().out.splitlines() == [
        f"metrics={Path(tmp_path / 'metrics.json')}",
        "accuracy=0.988",
    ]
