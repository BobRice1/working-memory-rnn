import numpy as np

from wm_rnn.task import DelayTaskConfig, generate_delay_batch


def test_delay_batch_shapes_and_masks():
    config = DelayTaskConfig(
        n_classes=4,
        cue_steps=3,
        delay_steps=5,
        response_steps=2,
        batch_size=7,
        seed=123,
    )

    batch = generate_delay_batch(config)

    assert batch.inputs.shape == (10, 7, 5)
    assert batch.targets.shape == (10, 7)
    assert batch.loss_mask.shape == (10, 7)
    assert batch.cues.shape == (7,)
    assert batch.phase_index["cue"] == slice(0, 3)
    assert batch.phase_index["delay"] == slice(3, 8)
    assert batch.phase_index["response"] == slice(8, 10)

    assert np.all(batch.loss_mask[:8] == 0.0)
    assert np.all(batch.loss_mask[8:] == 1.0)
    assert np.all(batch.inputs[:, :, -1] == 1.0)


def test_cue_is_present_only_during_cue_period():
    config = DelayTaskConfig(
        n_classes=3,
        cue_steps=2,
        delay_steps=4,
        response_steps=3,
        batch_size=5,
        seed=999,
    )

    batch = generate_delay_batch(config)
    class_inputs = batch.inputs[:, :, :3]

    for trial_idx, cue in enumerate(batch.cues):
        assert np.all(class_inputs[:2, trial_idx, cue] == 1.0)
        assert np.all(class_inputs[:2, trial_idx].sum(axis=1) == 1.0)
        assert np.all(class_inputs[2:, trial_idx] == 0.0)
        assert np.all(batch.targets[:, trial_idx] == cue)
