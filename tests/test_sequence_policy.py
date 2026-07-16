import unittest

import numpy as np
import torch

from data.robomimic_loader import _window_demo
from federated.fedbone_fl import gaussian_nll_loss, policy_action_mean
from models.fedbone_model import TaskHead, create_fedbone_client


class SequencePolicyTest(unittest.TestCase):
    def test_window_demo_uses_causal_padding_and_targets_last_action(self):
        observations = np.asarray(
            [[0.0], [1.0], [2.0], [3.0]],
            dtype=np.float32,
        )
        actions = np.asarray(
            [[10.0], [11.0], [12.0], [13.0]],
            dtype=np.float32,
        )

        windows, targets = _window_demo(observations, actions, sequence_length=3)

        self.assertEqual(windows.shape, (4, 3, 1))
        np.testing.assert_allclose(windows[0, :, 0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(windows[1, :, 0], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(windows[2, :, 0], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(windows[3, :, 0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(targets[:, 0], [10.0, 11.0, 12.0, 13.0])

    def test_regression_head_uses_last_timestep(self):
        head = TaskHead(hidden_size=4, num_classes=1, task_type="regression")
        head.head = torch.nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            head.head.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))

        sequence = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [5.0, 0.0, 0.0, 0.0],
                ]
            ]
        )

        output = head(sequence)

        torch.testing.assert_close(output, torch.tensor([[5.0]]))

    def test_gaussian_policy_head_outputs_mean_and_log_std(self):
        model = create_fedbone_client(
            num_features=3,
            embed_dim=4,
            hidden_size=8,
            num_classes=2,
            task_type="regression",
            policy_type="gaussian",
        )
        features = torch.randn(5, 10, 8)

        outputs = model(None, general_features=features)
        mean = policy_action_mean(outputs, "gaussian")
        loss = gaussian_nll_loss(outputs, torch.zeros(5, 2))

        self.assertEqual(outputs.shape, (5, 4))
        self.assertEqual(mean.shape, (5, 2))
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
