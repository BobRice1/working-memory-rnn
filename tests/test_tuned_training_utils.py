import math

import torch
import pytest

from wm_rnn.training_utils import model_config_from_dict, task_config_from_dict, tuned_response_metrics
from wm_rnn.task import DelayBatch, DelayTaskConfig, generate_delay_batch
from wm_rnn.tuned_task import TunedDelayTaskConfig, circular_preferred_angles, encode_circular_population
from wm_rnn.tuned_task import TunedDelayBatch


def test_task_config_from_dict_builds_tuned_task_config():
    config = {
        "task": {
            "task_type": "tuned",
            "n_tuned_units": 32,
            "tuning_kappa": 8.0,
            "cue_steps": 5,
            "delay_steps": 20,
            "response_steps": 5,
            "batch_size": 64,
            "seed": 10,
        }
    }

    task_config = task_config_from_dict(config, seed_offset=2, batch_size=7)

    assert isinstance(task_config, TunedDelayTaskConfig)
    assert task_config.seed == 12
    assert task_config.batch_size == 7
    assert task_config.input_size == 33
    assert task_config.output_size == 32


def test_task_config_from_dict_defaults_to_categorical_task_config():
    config = {
        "task": {
            "n_classes": 6,
            "cue_steps": 3,
            "delay_steps": 4,
            "response_steps": 2,
            "batch_size": 11,
            "seed": 20,
        }
    }

    task_config = task_config_from_dict(config, seed_offset=5, batch_size=7)

    assert isinstance(task_config, DelayTaskConfig)
    assert task_config.n_classes == 6
    assert task_config.seed == 25
    assert task_config.batch_size == 7
    assert task_config.input_size == 7


def test_task_config_from_dict_rejects_unknown_task_type():
    config = {
        "task": {
            "task_type": "unsupported",
            "batch_size": 4,
        }
    }

    with pytest.raises(ValueError, match="unknown task_type: unsupported"):
        task_config_from_dict(config)


def test_model_config_from_dict_uses_tuned_input_and_output_sizes():
    config = {
        "task": {
            "task_type": "tuned",
            "n_tuned_units": 32,
            "tuning_kappa": 8.0,
            "cue_steps": 5,
            "delay_steps": 20,
            "response_steps": 5,
            "batch_size": 64,
            "seed": 10,
        },
        "model": {
            "hidden_size": 64,
            "dt": 20.0,
            "tau": 100.0,
            "activation": "tanh",
        },
    }

    model_config = model_config_from_dict(config)

    assert model_config.input_size == 33
    assert model_config.output_size == 32


def test_model_config_from_dict_uses_categorical_input_and_output_sizes_by_default():
    config = {
        "task": {
            "n_classes": 7,
            "cue_steps": 5,
            "delay_steps": 20,
            "response_steps": 5,
            "batch_size": 64,
            "seed": 10,
        },
        "model": {
            "hidden_size": 64,
            "dt": 20.0,
            "tau": 100.0,
            "activation": "tanh",
        },
    }

    model_config = model_config_from_dict(config)

    assert model_config.input_size == 8
    assert model_config.output_size == 7


def test_batch_to_tensors_preserves_categorical_target_dtype():
    from wm_rnn.training_utils import batch_to_tensors

    task_config = DelayTaskConfig(
        n_classes=4,
        cue_steps=2,
        delay_steps=3,
        response_steps=2,
        batch_size=5,
        seed=1,
    )
    batch = generate_delay_batch(task_config)

    _, targets, _ = batch_to_tensors(batch, torch.device("cpu"))

    assert targets.dtype == torch.long


def test_masked_cross_entropy_still_scores_categorical_logits():
    from wm_rnn.training_utils import masked_cross_entropy

    logits = torch.tensor(
        [
            [[-20.0, 20.0, -20.0], [20.0, -20.0, -20.0]],
            [[-20.0, -20.0, 20.0], [20.0, -20.0, -20.0]],
        ]
    )
    targets = torch.tensor(
        [
            [0, 1],
            [2, 0],
        ],
        dtype=torch.long,
    )
    loss_mask = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )

    loss = masked_cross_entropy(logits, targets, loss_mask)

    assert loss < 1e-6


def test_generate_batch_for_task_dispatches_categorical_and_tuned_batches():
    from wm_rnn.training_utils import generate_batch_for_task

    categorical_config = DelayTaskConfig(
        n_classes=4,
        cue_steps=2,
        delay_steps=3,
        response_steps=2,
        batch_size=5,
        seed=1,
    )
    tuned_config = TunedDelayTaskConfig(
        n_tuned_units=8,
        tuning_kappa=4.0,
        cue_steps=2,
        delay_steps=3,
        response_steps=2,
        batch_size=5,
        seed=1,
    )

    categorical_batch = generate_batch_for_task(categorical_config)
    tuned_batch = generate_batch_for_task(tuned_config)

    assert isinstance(categorical_batch, DelayBatch)
    assert isinstance(tuned_batch, TunedDelayBatch)


def test_masked_mse_ignores_unscored_time_steps():
    from wm_rnn.training_utils import masked_mse

    predictions = torch.tensor(
        [
            [[100.0, 100.0]],
            [[2.0, 4.0]],
        ]
    )
    targets = torch.tensor(
        [
            [[0.0, 0.0]],
            [[1.0, 1.0]],
        ]
    )
    loss_mask = torch.tensor([[0.0], [1.0]])

    loss = masked_mse(predictions, targets, loss_mask)

    assert torch.isclose(loss, torch.tensor(5.0))


def test_tuned_response_metrics_report_low_error_for_matching_population():
    preferred = circular_preferred_angles(16)
    target_angles = torch.tensor([0.0, math.pi / 2])
    target_population = torch.from_numpy(
        encode_circular_population(target_angles.numpy(), preferred, tuning_kappa=8.0)
    ).float()
    predictions = target_population.unsqueeze(0)
    targets = target_population.unsqueeze(0)
    loss_mask = torch.ones(1, 2)

    metrics = tuned_response_metrics(predictions, targets, loss_mask, preferred, target_angles.numpy())

    assert metrics["mean_angular_error_degrees"] < 1.0
    assert metrics["population_mse"] == 0.0
