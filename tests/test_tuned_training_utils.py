import math

import numpy as np
import torch
import pytest

from wm_rnn.training_utils import (
    circular_distribution_loss,
    model_config_from_dict,
    task_config_from_dict,
    tuned_response_metrics,
    weighted_tuned_mse,
)
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


def test_tuned_response_metrics_report_fixation_quality():
    preferred = circular_preferred_angles(4)
    population = encode_circular_population(np.array([0.0]), preferred, tuning_kappa=8.0)
    predictions = torch.tensor([[[*population[0], 0.0]], [[*population[0], 1.0]]])
    targets = predictions.clone()
    loss_mask = torch.tensor([[0.0], [1.0]])

    metrics = tuned_response_metrics(
        predictions, targets, loss_mask, preferred, np.array([0.0], dtype=np.float32)
    )

    assert metrics["fixation_mse"] == 0.0
    assert metrics["fixation_accuracy"] == 1.0


def test_weighted_tuned_mse_emphasizes_fixation_channel():
    predictions = torch.zeros((2, 1, 3))
    targets = torch.zeros_like(predictions)
    predictions[1, 0, -1] = 1.0
    time_weights = torch.tensor([[0.0], [5.0]])

    loss = weighted_tuned_mse(predictions, targets, time_weights, fixation_weight=2.0)

    assert loss.item() == pytest.approx(0.5)


def _distribution_loss_inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    preferred = circular_preferred_angles(8)
    target_population = torch.from_numpy(
        encode_circular_population(
            np.array([math.pi / 3], dtype=np.float32),
            preferred,
            tuning_kappa=8.0,
        )
    ).float()
    targets = torch.zeros((1, 1, 9), dtype=torch.float32)
    targets[0, 0, :8] = target_population[0]
    predictions = torch.zeros_like(targets)
    response_mask = torch.ones((1, 1), dtype=torch.float32)
    fixation_mask = torch.ones((1, 1), dtype=torch.float32)
    return predictions, targets, response_mask, fixation_mask


def test_circular_distribution_uniform_logits_have_nonzero_gradient():
    predictions, targets, response_mask, fixation_mask = (
        _distribution_loss_inputs()
    )
    predictions.requires_grad_(True)

    total, circular, fixation = circular_distribution_loss(
        predictions,
        targets,
        response_mask,
        fixation_mask,
        n_tuned_units=8,
        fixation_weight=2.0,
    )
    total.backward()

    assert torch.isfinite(total)
    assert torch.isfinite(circular)
    assert torch.isfinite(fixation)
    assert predictions.grad is not None
    assert torch.linalg.vector_norm(predictions.grad[..., :8]) > 0.0


def test_aligned_logits_beat_uniform_circular_distribution_loss():
    uniform, targets, response_mask, fixation_mask = (
        _distribution_loss_inputs()
    )
    aligned = uniform.clone()
    aligned[..., :8] = torch.log(targets[..., :8].clamp_min(1e-12))

    _, uniform_circular, _ = circular_distribution_loss(
        uniform,
        targets,
        response_mask,
        fixation_mask,
        n_tuned_units=8,
    )
    _, aligned_circular, _ = circular_distribution_loss(
        aligned,
        targets,
        response_mask,
        fixation_mask,
        n_tuned_units=8,
    )

    assert aligned_circular < uniform_circular


def test_circular_distribution_loss_is_rotation_equivariant():
    predictions, targets, response_mask, fixation_mask = (
        _distribution_loss_inputs()
    )
    predictions[..., :8] = torch.linspace(-1.0, 1.0, 8)
    _, original, _ = circular_distribution_loss(
        predictions,
        targets,
        response_mask,
        fixation_mask,
        n_tuned_units=8,
    )
    rotated_predictions = predictions.clone()
    rotated_targets = targets.clone()
    rotated_predictions[..., :8] = torch.roll(
        predictions[..., :8], shifts=3, dims=-1
    )
    rotated_targets[..., :8] = torch.roll(
        targets[..., :8], shifts=3, dims=-1
    )
    _, rotated, _ = circular_distribution_loss(
        rotated_predictions,
        rotated_targets,
        response_mask,
        fixation_mask,
        n_tuned_units=8,
    )

    assert torch.allclose(original, rotated, atol=1e-6)


def test_circular_and_fixation_loss_components_are_separate():
    predictions, targets, response_mask, fixation_mask = (
        _distribution_loss_inputs()
    )
    _, circular_before, fixation_before = circular_distribution_loss(
        predictions,
        targets,
        response_mask,
        fixation_mask,
        n_tuned_units=8,
    )
    changed_fixation = predictions.clone()
    changed_fixation[..., -1] = 1.0
    _, circular_after, fixation_after = circular_distribution_loss(
        changed_fixation,
        targets,
        response_mask,
        fixation_mask,
        n_tuned_units=8,
    )
    changed_population = predictions.clone()
    changed_population[..., 0] = 5.0
    _, circular_population, fixation_population = (
        circular_distribution_loss(
            changed_population,
            targets,
            response_mask,
            fixation_mask,
            n_tuned_units=8,
        )
    )

    assert torch.allclose(circular_before, circular_after)
    assert fixation_after > fixation_before
    assert circular_population != circular_before
    assert torch.allclose(fixation_population, fixation_before)


def test_softmax_tuned_metrics_report_confidence_and_cross_entropy():
    preferred = circular_preferred_angles(8)
    target_angle = np.array([math.pi / 4], dtype=np.float32)
    target_population = encode_circular_population(
        target_angle, preferred, tuning_kappa=8.0
    )
    targets = torch.zeros((1, 1, 9), dtype=torch.float32)
    targets[0, 0, :8] = torch.from_numpy(target_population[0])
    predictions = torch.zeros_like(targets)
    predictions[0, 0, :8] = torch.log(targets[0, 0, :8].clamp_min(1e-12))

    metrics = tuned_response_metrics(
        predictions,
        targets,
        torch.ones((1, 1)),
        preferred,
        target_angle,
        population_normalization="softmax",
    )

    assert metrics["mean_angular_error_degrees"] < 1.0
    assert metrics["mean_response_cross_entropy"] > 0.0
    assert 0.0 < metrics["mean_population_resultant_length"] <= 1.0
    assert metrics["population_normalization"] == "softmax"
