import unittest
from types import SimpleNamespace

import torch

from federated.controlled_comparison import (
    aggregate_selected_task_states,
    build_stratified_schedule,
    initialize_task_states,
    synchronize_clients_to_task_states,
    weighted_state_average,
)


class ControlledProtocolTest(unittest.TestCase):
    def test_schedule_has_equal_task_participation(self):
        clients = [
            SimpleNamespace(client_id=task_id * 10 + robot_id, task_id=task_id)
            for robot_id in range(10)
            for task_id in range(5)
        ]
        schedule = build_stratified_schedule(
            clients, num_rounds=4, clients_per_round=10, seed=42
        )

        self.assertEqual(len(schedule), 4)
        for selected in schedule:
            self.assertEqual(len(selected), 10)
            self.assertEqual(len(set(selected)), 10)
            counts = {task_id: 0 for task_id in range(5)}
            by_id = {client.client_id: client for client in clients}
            for client_id in selected:
                counts[by_id[client_id].task_id] += 1
            self.assertEqual(counts, {0: 2, 1: 2, 2: 2, 3: 2, 4: 2})

    def test_weighted_state_average(self):
        states = [
            {"weight": torch.tensor([1.0, 3.0])},
            {"weight": torch.tensor([5.0, 7.0])},
        ]
        averaged = weighted_state_average(states, [1, 3])
        torch.testing.assert_close(averaged["weight"], torch.tensor([4.0, 6.0]))

    def test_task_modules_are_synchronized_and_aggregated(self):
        clients = []
        for client_id in range(3):
            model = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                model.weight.fill_(float(client_id + 1))
            clients.append(SimpleNamespace(
                client_id=client_id,
                task_id=0,
                client_model=model,
                num_samples=1,
            ))

        task_states = initialize_task_states(clients)
        synchronize_clients_to_task_states(clients, task_states)
        self.assertTrue(all(
            client.client_model.weight.item() == 1.0 for client in clients
        ))

        with torch.no_grad():
            clients[0].client_model.weight.fill_(2.0)
            clients[1].client_model.weight.fill_(4.0)
        aggregated = aggregate_selected_task_states(
            {client.client_id: client for client in clients},
            [0, 1],
        )
        self.assertEqual(aggregated[0]["weight"].item(), 3.0)


if __name__ == "__main__":
    unittest.main()
