import csv
import json

from wm_rnn.plot_seed_sweeps import plot_seed_sweeps


def test_plot_seed_sweeps_supports_tuned_angular_error(tmp_path):
    csv_path = tmp_path / "seed.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["delay_steps", "mean_angular_error_degrees"])
        writer.writeheader()
        writer.writerows(
            [
                {"delay_steps": 10, "mean_angular_error_degrees": 2.0},
                {"delay_steps": 20, "mean_angular_error_degrees": 3.0},
            ]
        )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "task_type": "tuned",
                "base_output_dir": str(tmp_path),
                "base_run_name": "circular_test",
                "trained_delay_steps": 20,
                "results": [{"seed": 1, "delay_sweep_csv": str(csv_path)}],
            }
        ),
        encoding="utf-8",
    )

    result = plot_seed_sweeps(summary_path)

    assert result.figure_path.exists()
    assert result.seed_count == 1
    assert result.delays == [10, 20]
