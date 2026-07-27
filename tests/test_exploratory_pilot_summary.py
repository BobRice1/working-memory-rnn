from wm_rnn.exploratory_pilot_summary import cross_task_summary


def test_cross_task_summary_requires_two_of_three_and_positive_mean():
    circular = []
    nback = []
    for seed, slowing, delay, distractor, load in (
        (1, True, 0.3, 0.2, 0.4),
        (2, True, 0.2, 0.1, 0.3),
        (3, False, -0.1, -0.05, -0.1),
    ):
        circular.append(
            {
                "checkpoint_seed": seed,
                "operator": "state_persistence",
                "strength": 0.95,
                "clean20_proportional_error_impairment": 0.05,
                "clean20_settling_delta": 0.2,
                "slowing_with_preservation": slowing,
                "delay_selectivity": delay,
                "distractor_selectivity": distractor,
            }
        )
        nback.append(
            {
                "checkpoint_seed": seed,
                "operator": "state_persistence",
                "strength": 0.95,
                "load_selectivity": load,
            }
        )
    row = cross_task_summary(circular, nback)[0]
    assert row["slowing_with_preservation"]
    assert row["delay_selective"]
    assert row["distractor_selective"]
    assert row["load_selective"]
    assert row["complete_primary_pattern"]
