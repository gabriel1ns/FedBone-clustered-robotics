import unittest

import torch

from federated.gp_aggregation import GPAggregation


class GPAttentionTest(unittest.TestCase):
    def setUp(self):
        self.aggregator = GPAggregation(gradient_dim=2)

    def test_first_round_attention_is_neutral(self):
        gradients = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
        attention = self.aggregator.compute_gradient_attentions(gradients, None)
        torch.testing.assert_close(attention, torch.ones(2))

    def test_attention_compares_clients_jointly(self):
        gradients = [
            torch.tensor([1.0, 0.0]),
            torch.tensor([-1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
        ]
        history = torch.tensor([1.0, 0.0])
        attention = self.aggregator.compute_gradient_attentions(
            gradients, history
        )

        self.assertGreater(attention[0].item(), attention[2].item())
        self.assertGreater(attention[2].item(), attention[1].item())
        self.assertAlmostEqual(attention.mean().item(), 1.0, places=6)
        self.assertGreater(torch.std(attention).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
