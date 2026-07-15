import pytest
import torch

from wm_rnn.model import CTRNN, RNNConfig
from wm_rnn.noise_structure_experiment import generate_perturbations, topology_covariance


def test_optional_perturbation_preserves_default_and_zero_path():
    rnn = CTRNN(RNNConfig(3, 4, 2))
    rnn.eval()
    inputs = torch.randn(5, 2, 3)
    baseline, _ = rnn(inputs)
    zero, _ = rnn(inputs, perturbations=torch.zeros(5, 2, 4))
    assert torch.equal(baseline, zero)


def test_perturbation_shape_is_checked():
    rnn = CTRNN(RNNConfig(3, 4, 2))
    with pytest.raises(ValueError):
        rnn(torch.randn(5, 2, 3), perturbations=torch.zeros(5, 2, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only experiment")
def test_noise_is_masked_and_rms_matched_on_cuda():
    device = torch.device("cuda")
    weight = torch.randn(8, 8, device=device)
    mask = torch.tensor([False, True, True, True, False], device=device)
    for condition in ("independent_gaussian", "temporally_correlated", "context_topology_correlated"):
        generator = torch.Generator(device=device).manual_seed(7)
        noise = generate_perturbations(condition, (5, 128, 8), .05, mask, generator, weight)
        assert torch.count_nonzero(noise[~mask]) == 0
        assert float(noise[mask].square().mean().sqrt()) == pytest.approx(.05, abs=1e-6)


def test_topology_covariance_has_unit_mean_diagonal():
    covariance = topology_covariance(torch.randn(8, 8))
    assert float(torch.diagonal(covariance).mean()) == pytest.approx(1.0, abs=1e-6)
    assert torch.linalg.eigvalsh(covariance).min() > 0
