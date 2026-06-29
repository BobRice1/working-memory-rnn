import torch

from wm_rnn.model import RNNConfig, WorkingMemoryRNN


def test_model_forward_returns_logits_and_hidden_states():
    model = WorkingMemoryRNN(RNNConfig(input_size=5, hidden_size=16, output_size=4, dt=20.0, tau=100.0))
    inputs = torch.zeros(10, 7, 5)

    logits, hidden = model(inputs)

    assert logits.shape == (10, 7, 4)
    assert hidden.shape == (10, 7, 16)
    assert model.rnn.alpha == 0.2


def test_recurrent_step_keeps_hidden_nonnegative_with_relu():
    model = WorkingMemoryRNN(RNNConfig(input_size=2, hidden_size=3, output_size=2))
    inputs = torch.ones(4, 1, 2)

    _, hidden = model(inputs)

    assert torch.all(hidden >= 0.0)
