from pathlib import Path

import torch

from wm_rnn.config import load_config
from wm_rnn.device import select_device


def test_load_config_merges_yaml_with_defaults(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "task:\n"
        "  n_classes: 5\n"
        "training:\n"
        "  steps: 12\n"
        "model:\n"
        "  hidden_size: 32\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["task"]["n_classes"] == 5
    assert config["task"]["cue_steps"] == 5
    assert config["training"]["steps"] == 12
    assert config["model"]["hidden_size"] == 32
    assert config["analysis"]["n_components"] == 2


def test_select_device_auto_uses_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=0: "NVIDIA GeForce RTX 3060 Laptop GPU")

    selected = select_device("auto")

    assert selected.device.type == "cuda"
    assert "RTX 3060" in selected.description


def test_select_device_falls_back_to_cpu_for_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    selected = select_device("cuda")

    assert selected.device.type == "cpu"
    assert "CUDA requested but unavailable" in selected.description
