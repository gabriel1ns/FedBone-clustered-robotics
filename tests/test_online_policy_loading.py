import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from models.model import create_model
from runner.evaluate_online import load_policy, predict_action


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


if __name__ == "__main__":
    unittest.main()
