from wm_rnn.evaluate import aggregate_tuned_metrics


def test_tuned_evaluation_uses_global_median_not_mean_of_batch_medians():
    metrics = aggregate_tuned_metrics(
        [
            {
                "angular_errors_degrees": [1.0, 100.0],
                "population_squared_errors": [0.1, 0.2],
            },
            {
                "angular_errors_degrees": [2.0],
                "population_squared_errors": [0.4],
            },
        ]
    )

    assert metrics["mean_angular_error_degrees"] == (1.0 + 100.0 + 2.0) / 3.0
    assert metrics["median_angular_error_degrees"] == 2.0
    assert metrics["population_mse"] == (0.1 + 0.2 + 0.4) / 3.0
