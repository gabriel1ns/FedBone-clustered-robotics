import unittest

import numpy as np
import torch

from data.datasets import RoboMimicTaskDataset, get_dataloader
from federated.robomimic_baselines import (
    RoboMimicRegressionClient,
    run_clustered_regression,
    run_fedavg_regression,
)
from models.model import create_model


class RoboMimicBaselineSmokeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.device = torch.device("cpu")
        self.task_info = {
            "task_id": 0,
            "name": "synthetic_lift",
            "type": "regression",
            "num_classes": 2,
            "success_threshold": 2.0,
            "normalization": {
                "action_mean": [0.0, 0.0],
                "action_std": [1.0, 1.0],
            },
        }
        rng = np.random.default_rng(7)
        self.datasets = []
        for client_id in range(3):
            inputs = rng.normal(client_id * 0.2, 1.0, size=(12, 1, 3))
            actions = inputs[:, 0, :2] * 0.5
            self.datasets.append(RoboMimicTaskDataset(inputs, actions))
        test_inputs = rng.normal(size=(10, 1, 3))
        test_actions = test_inputs[:, 0, :2] * 0.5
        self.test_loader = get_dataloader(
            RoboMimicTaskDataset(test_inputs, test_actions), 5, shuffle=False
        )

    def model(self):
        return create_model(3, hidden_size=4, num_layers=1, num_classes=2, dropout=0.0)

    def clients(self, model):
        return [
            RoboMimicRegressionClient(
                index, self.model(), dataset, self.device, 4, 0.01, 1
            )
            for index, dataset in enumerate(self.datasets)
        ]

    def test_fedavg_one_round(self):
        model = self.model()
        history = run_fedavg_regression(
            self.clients(model),
            model,
            self.test_loader,
            self.device,
            self.task_info,
            num_rounds=1,
            clients_per_round=2,
        )
        self.assertEqual(history["rounds"], [1])
        self.assertGreaterEqual(history["rmse"][0], 0.0)

    def test_clustered_one_round(self):
        model = self.model()
        history, models = run_clustered_regression(
            self.clients(model),
            model,
            self.test_loader,
            self.device,
            self.task_info,
            num_rounds=1,
            num_clusters=2,
            clustering_method="kmeans",
            reclustering_interval=0,
        )
        self.assertEqual(history["rounds"], [1])
        self.assertEqual(len(models), 2)


if __name__ == "__main__":
    unittest.main()
