import numpy as np

from wm_rnn.cross_temporal_decoder import decode_hidden_states_cross_temporally


def test_cross_temporal_decoder_recovers_stable_circular_code():
    angles = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    circular_code = np.column_stack((np.cos(angles), np.sin(angles)))
    hidden = np.stack([circular_code, circular_code, circular_code])
    train_indices = np.arange(0, 80, 2)
    test_indices = np.arange(1, 80, 2)

    errors = decode_hidden_states_cross_temporally(
        hidden,
        angles,
        train_indices,
        test_indices,
        ridge_alpha=1e-6,
    )

    assert errors.shape == (3, 3)
    assert float(errors.max()) < 0.01


def test_cross_temporal_decoder_validates_trial_count():
    hidden = np.zeros((2, 4, 3), dtype=np.float32)
    angles = np.zeros(3, dtype=np.float32)

    try:
        decode_hidden_states_cross_temporally(hidden, angles, np.array([0, 1]), np.array([2, 3]))
    except ValueError as exc:
        assert "one value per trial" in str(exc)
    else:
        raise AssertionError("expected trial-count validation error")
