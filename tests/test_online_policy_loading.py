import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from models.model import create_model
from runner.evaluate_online import (
    load_demo_initial_state,
    load_policy,
    predict_action,
    resolve_task_horizon,
    seed_environment,
    summarize_online_results,
)


class OnlineFedAvgPolicyTest(unittest.TestCase):
    def test_loads_and_runs_fedavg_checkpoint(self):
        model = create_model(3, hidden_size=4, num_layers=1, num_classes=2, dropout=0.0)
        task = {
            "task_id": 0,
            "name": "lift_ph",
            "type": "regression",
            "num_classes": 2,
            "obs_keys": ["robot0_eef_pos"],
            "normalization": {
                "obs_mean": [1.0, 2.0, 3.0],
                "obs_std": [1.0, 1.0, 1.0],
                "action_mean": [0.5, -0.5],
                "action_std": [2.0, 2.0],
            },
        }
        checkpoint = {
            "algorithm": "fedavg",
            "tasks": [task],
            "task_models": {
                "lift_ph": {
                    "model_config": {
                        "num_features": 3,
                        "hidden_size": 4,
                        "num_layers": 1,
                        "output_dim": 2,
                        "dropout": 0.0,
                    },
                    "state_dict": model.state_dict(),
                }
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fedavg.pth"
            torch.save(checkpoint, path)
            policy, server, obs_keys, loaded_task, algorithm = load_policy(
                path, "lift_ph", torch.device("cpu")
            )
            action = predict_action(
                policy,
                server,
                np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                loaded_task,
                torch.device("cpu"),
            )

        self.assertIsNone(server)
        self.assertEqual(obs_keys, ["robot0_eef_pos"])
        self.assertEqual(algorithm, "fedavg")
        self.assertEqual(action.shape, (2,))
        self.assertTrue(np.isfinite(action).all())

    def test_resolves_per_task_default_horizons(self):
        self.assertEqual(resolve_task_horizon("lift_ph"), 400)
        self.assertEqual(resolve_task_horizon("can_mh"), 400)
        self.assertEqual(resolve_task_horizon("square_mg"), 400)
        self.assertEqual(resolve_task_horizon("tool_hang_ph"), 800)
        self.assertEqual(resolve_task_horizon("transport_ph"), 800)
        self.assertEqual(resolve_task_horizon("transport_ph", horizon_override=123), 123)

    def test_seed_environment_ignores_non_callable_seed_attribute(self):
        class EnvWithNoneSeed:
            seed = None

        seed_environment(EnvWithNoneSeed(), 42)

    def test_load_demo_initial_state_cycles_sorted_demos(self):
        task = {"source_file": ""}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.hdf5"
            task["source_file"] = str(path)
            with h5py.File(path, "w") as handle:
                data = handle.create_group("data")
                for key, value in [("demo_10", 10.0), ("demo_2", 2.0)]:
                    demo = data.create_group(key)
                    demo.attrs["model_file"] = f"<xml>{key}</xml>"
                    demo.create_dataset("states", data=np.asarray([[value, value + 1.0]]))

            initial = load_demo_initial_state(task, episode=0)
            cycled = load_demo_initial_state(task, episode=2)

        self.assertEqual(initial["demo_key"], "demo_2")
        self.assertEqual(cycled["demo_key"], "demo_2")
        np.testing.assert_allclose(initial["state"], np.asarray([2.0, 3.0]))

    def test_summarizes_only_completed_tasks(self):
        summary = summarize_online_results(
            "checkpoint.pth",
            {
                "lift_ph": {
                    "success_rate": 1.0,
                    "mean_return": 10.0,
                    "mean_steps": 20.0,
                    "action_stats": {"saturation_fraction": 0.1},
                },
                "transport_ph": {"error": "boom"},
            },
        )

        self.assertEqual(summary["macro_success_rate"], 1.0)
        self.assertEqual(summary["macro_mean_return"], 10.0)
        self.assertEqual(summary["macro_mean_steps"], 20.0)
        self.assertEqual(summary["macro_saturation_fraction"], 0.1)
        self.assertEqual(summary["completed_tasks"], ["lift_ph"])


if __name__ == "__main__":
    unittest.main()
