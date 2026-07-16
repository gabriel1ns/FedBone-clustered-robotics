import copy
import os
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch
import torch.optim as optim

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from sklearn.cluster import KMeans

from federated.fedbone_fl import (
    compute_policy_loss,
    evaluate_fedbone_multitask,
    policy_action_mean,
)
from federated.gp_aggregation import (
    GPAggregation,
    compute_gradient_conflict_score,
    simple_average_aggregation,
)
from utils.robotics_metrics import SplitCommunicationMeter, rounds_to_convergence


METHODS = (
    "shared_backbone_fedavg",
    "shared_backbone_clustered",
    "fedbone_simple",
    "fedbone_gp",
)


def build_stratified_schedule(clients, num_rounds, clients_per_round, seed):
    by_task = {}
    for client in clients:
        by_task.setdefault(int(client.task_id), []).append(int(client.client_id))

    task_ids = sorted(by_task)
    if clients_per_round < len(task_ids):
        raise ValueError(
            "clients_per_round must be at least the number of tasks for "
            "stratified participation."
        )

    base = clients_per_round // len(task_ids)
    remainder = clients_per_round % len(task_ids)
    rng = np.random.default_rng(seed)
    schedule = []
    for round_index in range(num_rounds):
        selected = []
        for position, task_id in enumerate(task_ids):
            count = base + int(position < remainder)
            pool = by_task[task_id]
            if count > len(pool):
                raise ValueError(f"Task {task_id} has only {len(pool)} clients.")
            chosen = rng.choice(pool, size=count, replace=False)
            selected.extend(int(value) for value in chosen)
        rng.shuffle(selected)
        schedule.append(selected)
    return schedule


def clone_state_dict(state_dict, device=None):
    return OrderedDict(
        (key, value.detach().clone().to(device) if device else value.detach().clone())
        for key, value in state_dict.items()
    )


def weighted_state_average(states, weights):
    total = float(sum(weights))
    averaged = OrderedDict()
    for key in states[0]:
        if states[0][key].is_floating_point():
            averaged[key] = sum(
                state[key] * (weight / total)
                for state, weight in zip(states, weights)
            )
        else:
            averaged[key] = states[0][key].clone()
    return averaged


def initialize_task_states(clients):
    task_states = {}
    for client in clients:
        task_id = int(client.task_id)
        if task_id not in task_states:
            task_states[task_id] = clone_state_dict(
                client.client_model.state_dict()
            )
    return task_states


def synchronize_clients_to_task_states(clients, task_states):
    for client in clients:
        client.client_model.load_state_dict(task_states[int(client.task_id)])


def aggregate_selected_task_states(clients_by_id, selected_ids):
    by_task = {}
    for client_id in selected_ids:
        client = clients_by_id[client_id]
        by_task.setdefault(int(client.task_id), []).append(client)

    return {
        task_id: weighted_state_average(
            [clone_state_dict(client.client_model.state_dict()) for client in matching],
            [client.num_samples for client in matching],
        )
        for task_id, matching in by_task.items()
    }


def state_dict_nbytes(state):
    return int(sum(value.numel() * value.element_size() for value in state.values()))


def state_delta(final_state, initial_state):
    return OrderedDict(
        (key, final_state[key] - initial_state[key])
        for key in initial_state
        if initial_state[key].is_floating_point()
    )


def flatten_update(update):
    return torch.cat([value.detach().reshape(-1).cpu() for value in update.values()])


def clustered_state_aggregation(initial_state, final_states, sample_counts, num_clusters, seed):
    deltas = [state_delta(state, initial_state) for state in final_states]
    representations = np.stack([flatten_update(delta).numpy() for delta in deltas])
    cluster_count = min(num_clusters, len(final_states))
    labels = KMeans(
        n_clusters=cluster_count,
        random_state=seed,
        n_init=10,
    ).fit_predict(representations)

    cluster_states = []
    distribution = {}
    for cluster_id in sorted(set(int(value) for value in labels)):
        indices = [index for index, label in enumerate(labels) if int(label) == cluster_id]
        cluster_states.append(weighted_state_average(
            [final_states[index] for index in indices],
            [sample_counts[index] for index in indices],
        ))
        distribution[str(cluster_id)] = len(indices)

    # Equal cluster influence prevents the largest update mode from collapsing
    # this controlled clustered strategy back into ordinary FedAvg.
    aggregated = weighted_state_average(cluster_states, [1] * len(cluster_states))
    return aggregated, distribution


def train_selected_client(client, initial_server, device):
    local_server = copy.deepcopy(initial_server).to(device)
    local_server.train()
    client.client_model.train()
    optimizer = optim.Adam(
        list(client.client_model.parameters()) + list(local_server.parameters()),
        lr=client.learning_rate,
    )
    gradient_sums = OrderedDict(
        (name, torch.zeros_like(parameter))
        for name, parameter in local_server.named_parameters()
    )
    gradient_weight = 0
    loss_sum = 0.0
    examples = 0
    communication = SplitCommunicationMeter()

    for _ in range(client.local_epochs):
        for inputs, targets in client.train_loader:
            inputs = inputs.to(device)
            targets = targets.float().to(device)
            optimizer.zero_grad()
            embeddings = client.client_model(inputs, general_features=None)
            features = local_server(embeddings)
            communication.record_split_batch(embeddings, features)
            outputs = client.client_model(None, general_features=features)
            predictions = policy_action_mean(
                outputs,
                getattr(client, "policy_type", "deterministic"),
            )
            if predictions.shape[-1] == 1:
                predictions = predictions.squeeze(-1)
            loss = compute_policy_loss(
                outputs,
                targets,
                client.task_type,
                getattr(client, "policy_type", "deterministic"),
            )
            loss.backward()

            batch_size = int(inputs.shape[0])
            for name, parameter in local_server.named_parameters():
                if parameter.grad is not None:
                    gradient_sums[name] += parameter.grad.detach() * batch_size
            gradient_weight += batch_size
            loss_sum += loss.item() * batch_size
            examples += batch_size
            optimizer.step()

    mean_gradients = OrderedDict(
        (name, gradient / max(gradient_weight, 1))
        for name, gradient in gradient_sums.items()
    )
    return {
        "server_state": clone_state_dict(local_server.state_dict()),
        "client_state": clone_state_dict(client.client_model.state_dict()),
        "gradients": mean_gradients,
        "sample_count": client.num_samples,
        "loss": loss_sum / max(examples, 1),
        "communication": communication.summary(),
    }


def task_evaluation_clients(clients, task_states):
    task_clients = {}
    for task_id in sorted({int(client.task_id) for client in clients}):
        matching = [client for client in clients if int(client.task_id) == task_id]
        representative = copy.copy(matching[0])
        representative.client_model = copy.deepcopy(matching[0].client_model)
        representative.client_model.load_state_dict(task_states[task_id])
        task_clients[task_id] = representative
    return list(task_clients.values())


def evaluate_controlled(
    clients,
    task_states,
    server_model,
    test_loaders,
    tasks,
    device,
    measure_latency=False,
):
    evaluation_clients = task_evaluation_clients(clients, task_states)
    server = SimpleNamespace(server_model=server_model)
    return evaluate_fedbone_multitask(
        evaluation_clients,
        server,
        test_loaders,
        device,
        tasks,
        measure_latency=measure_latency,
    ), evaluation_clients


def apply_gradients(server_model, optimizer, gradients):
    optimizer.zero_grad()
    for name, parameter in server_model.named_parameters():
        parameter.grad = gradients[name].detach().clone()
    optimizer.step()


def run_controlled_method(
    method,
    clients,
    server_model,
    schedule,
    test_loaders,
    tasks,
    device,
    learning_rate,
    num_clusters=3,
    seed=42,
    progress_callback=None,
):
    if method not in METHODS:
        raise ValueError(f"Unknown controlled method: {method}")

    clients_by_id = {int(client.client_id): client for client in clients}
    task_states = initialize_task_states(clients)
    synchronize_clients_to_task_states(clients, task_states)
    server_model = server_model.to(device)
    server_optimizer = optim.Adam(server_model.parameters(), lr=learning_rate)
    gp_aggregator = GPAggregation(
        gradient_dim=sum(parameter.numel() for parameter in server_model.parameters())
    )
    history = {
        "rounds": [],
        "loss": [],
        "mae": [],
        "rmse": [],
        "r2": [],
        "task_success_rate": [],
        "per_task_metrics": [],
        "conflict_scores": [],
        "communication_bytes_per_round": [],
        "communication_mb_per_round": [],
        "cluster_distribution": [],
        "selected_client_ids": [],
    }

    for round_index, selected_ids in enumerate(schedule):
        round_number = round_index + 1
        initial_state = clone_state_dict(server_model.state_dict())
        updates = []
        for client_id in selected_ids:
            client = clients_by_id[client_id]
            client.client_model.load_state_dict(task_states[int(client.task_id)])
            updates.append(train_selected_client(
                client, server_model, device
            ))

        new_task_states = aggregate_selected_task_states(
            clients_by_id, selected_ids
        )
        task_states.update(new_task_states)

        states = [update["server_state"] for update in updates]
        gradients = [update["gradients"] for update in updates]
        sample_counts = [update["sample_count"] for update in updates]
        cluster_distribution = {}

        if method == "shared_backbone_fedavg":
            server_model.load_state_dict(weighted_state_average(states, sample_counts))
        elif method == "shared_backbone_clustered":
            aggregated, cluster_distribution = clustered_state_aggregation(
                initial_state,
                states,
                sample_counts,
                num_clusters,
                seed + round_number,
            )
            server_model.load_state_dict(aggregated)
        elif method == "fedbone_simple":
            aggregated = simple_average_aggregation(gradients, sample_counts)
            apply_gradients(server_model, server_optimizer, aggregated)
        else:
            aggregated = gp_aggregator.aggregate(gradients, sample_counts)
            apply_gradients(server_model, server_optimizer, aggregated)

        metrics, _ = evaluate_controlled(
            clients,
            task_states,
            server_model,
            test_loaders,
            tasks,
            device,
            measure_latency=round_number == len(schedule),
        )
        split_bytes = sum(
            update["communication"]["total_bytes"] for update in updates
        )
        client_model_bytes = sum(
            2 * state_dict_nbytes(update["client_state"]) for update in updates
        )
        total_bytes = split_bytes + client_model_bytes
        conflict = compute_gradient_conflict_score(gradients)
        history["rounds"].append(round_number)
        history["loss"].append(float(np.mean([update["loss"] for update in updates])))
        for key in ("mae", "rmse", "r2", "task_success_rate"):
            history[key].append(metrics[key])
        history["per_task_metrics"].append(metrics["per_task_metrics"])
        history["conflict_scores"].append(conflict)
        history["communication_bytes_per_round"].append(int(total_bytes))
        history["communication_mb_per_round"].append(total_bytes / (1024 ** 2))
        history["cluster_distribution"].append(cluster_distribution)
        history["selected_client_ids"].append(list(selected_ids))

        print(
            f"[{method}] Round {round_number}/{len(schedule)} "
            f"RMSE={metrics['rmse']:.4f} "
            f"TSR={metrics['task_success_rate']:.4f}"
        )
        if progress_callback is not None:
            progress_callback(history)

    history["rounds_to_convergence"] = rounds_to_convergence(
        history["rmse"], mode="min"
    )
    _, checkpoint_clients = evaluate_controlled(
        clients, task_states, server_model, test_loaders, tasks, device
    )
    return history, server_model, checkpoint_clients
