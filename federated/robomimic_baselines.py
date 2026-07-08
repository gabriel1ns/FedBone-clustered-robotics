import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans

from data.datasets import get_dataloader
from federated.fedbone_fl import (
    calculate_regression_metrics,
    compute_task_success,
    denormalize_regression_values,
)
from utils.robotics_metrics import measure_inference_latency, rounds_to_convergence, state_dict_nbytes


def _parameters(model):
    return [value.detach().cpu().numpy() for value in model.state_dict().values()]


def _set_parameters(model, parameters):
    state = OrderedDict(
        (key, torch.as_tensor(value))
        for key, value in zip(model.state_dict().keys(), parameters)
    )
    model.load_state_dict(state, strict=True)


def _average_parameters(parameter_sets, sample_counts):
    total = float(sum(sample_counts))
    return [
        sum(params[index] * (count / total) for params, count in zip(parameter_sets, sample_counts))
        for index in range(len(parameter_sets[0]))
    ]


class RoboMimicRegressionClient:
    def __init__(self, client_id, model, dataset, device, batch_size, learning_rate, local_epochs):
        self.client_id = client_id
        self.model = model.to(device)
        self.device = device
        self.loader = get_dataloader(dataset, batch_size, shuffle=True)
        self.num_samples = len(dataset)
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.cluster_id = None
        sequences = dataset.sequences.detach().cpu().numpy().reshape(len(dataset), -1)
        actions = dataset.labels.detach().cpu().numpy().reshape(len(dataset), -1)
        self.data_signature = np.concatenate([
            sequences.mean(axis=0),
            sequences.std(axis=0),
            actions.mean(axis=0),
            actions.std(axis=0),
        ])

    def fit(self, parameters):
        _set_parameters(self.model, parameters)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        for _ in range(self.local_epochs):
            for sequences, actions in self.loader:
                sequences = sequences.to(self.device)
                actions = actions.float().to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(sequences), actions)
                loss.backward()
                optimizer.step()
        return _parameters(self.model)


def evaluate_regression_model(model, test_loader, device, task_info, measure_latency=False):
    model.eval()
    predictions = []
    targets = []
    normalized_loss = 0.0
    batches = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for sequences, actions in test_loader:
            sequences = sequences.to(device)
            actions = actions.float().to(device)
            outputs = model(sequences)
            normalized_loss += criterion(outputs, actions).item()
            batches += 1
            predictions.extend(outputs.cpu().numpy())
            targets.extend(actions.cpu().numpy())

    predictions = denormalize_regression_values(predictions, task_info)
    targets = denormalize_regression_values(targets, task_info)
    metrics = calculate_regression_metrics(targets, predictions)
    metrics["loss"] = normalized_loss / max(batches, 1)
    metrics["task_success_rate"] = compute_task_success(targets, predictions, task_info)
    if measure_latency:
        metrics["inference_latency_ms"] = measure_inference_latency(
            test_loader, device, lambda inputs: model(inputs)
        )
    return metrics


def _empty_history():
    return {
        "rounds": [],
        "loss": [],
        "mae": [],
        "rmse": [],
        "r2": [],
        "task_success_rate": [],
        "per_task_metrics": [],
        "communication_bytes_per_round": [],
        "communication_mb_per_round": [],
    }


def _append_metrics(history, round_number, metrics, communication_bytes, task_name):
    history["rounds"].append(round_number)
    for key in ("loss", "mae", "rmse", "r2", "task_success_rate"):
        history[key].append(metrics[key])
    history["per_task_metrics"].append({task_name: metrics})
    history["communication_bytes_per_round"].append(int(communication_bytes))
    history["communication_mb_per_round"].append(communication_bytes / (1024 ** 2))


def run_fedavg_regression(
    clients,
    global_model,
    test_loader,
    device,
    task_info,
    num_rounds,
    clients_per_round,
    seed=42,
):
    history = _empty_history()
    rng = np.random.default_rng(seed)
    model_bytes = state_dict_nbytes(global_model.state_dict())
    selected_count = min(clients_per_round, len(clients))

    for round_number in range(1, num_rounds + 1):
        selected_indices = rng.choice(len(clients), size=selected_count, replace=False)
        selected_clients = [clients[index] for index in selected_indices]
        global_parameters = _parameters(global_model)
        updates = [client.fit(global_parameters) for client in selected_clients]
        aggregated = _average_parameters(updates, [client.num_samples for client in selected_clients])
        _set_parameters(global_model, aggregated)

        metrics = evaluate_regression_model(
            global_model,
            test_loader,
            device,
            task_info,
            measure_latency=round_number == num_rounds,
        )
        communication_bytes = 2 * model_bytes * selected_count
        _append_metrics(
            history, round_number, metrics, communication_bytes, task_info["name"]
        )
        print(
            f"  Round {round_number}/{num_rounds}: "
            f"RMSE={metrics['rmse']:.4f} TSR={metrics['task_success_rate']:.4f}"
        )

    history["rounds_to_convergence"] = rounds_to_convergence(history["rmse"], mode="min")
    return history


def _cluster_clients(clients, num_clusters, method, seed):
    # Distribution signatures give the initial clustering a real non-IID
    # signal without granting Clustered FL an unreported extra training round.
    representations = np.asarray([client.data_signature for client in clients])
    cluster_count = min(num_clusters, len(clients))
    if cluster_count == 1:
        labels = np.zeros(len(clients), dtype=int)
    elif method == "kmeans":
        labels = KMeans(n_clusters=cluster_count, random_state=seed, n_init=10).fit_predict(representations)
    elif method == "hierarchical":
        labels = fcluster(linkage(representations, method="ward"), cluster_count, criterion="maxclust") - 1
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    clusters = {}
    for client, label in zip(clients, labels):
        client.cluster_id = int(label)
        clusters.setdefault(int(label), []).append(client)
    return clusters


def run_clustered_regression(
    clients,
    initial_model,
    test_loader,
    device,
    task_info,
    num_rounds,
    num_clusters,
    clustering_method,
    reclustering_interval,
    seed=42,
):
    history = _empty_history()
    history.update({
        "cluster_distribution": [],
        "per_cluster_metrics": [],
        "cluster_rmse_std": [],
    })

    clusters = _cluster_clients(clients, num_clusters, clustering_method, seed)
    cluster_models = {
        cluster_id: copy.deepcopy(initial_model).to(device)
        for cluster_id in clusters
    }

    for round_number in range(1, num_rounds + 1):
        if (
            reclustering_interval > 0
            and round_number > 1
            and (round_number - 1) % reclustering_interval == 0
        ):
            clusters = _cluster_clients(
                clients, num_clusters, clustering_method, seed + round_number
            )
            for cluster_id in clusters:
                if cluster_id not in cluster_models:
                    cluster_models[cluster_id] = copy.deepcopy(initial_model).to(device)

        communication_bytes = 0
        for cluster_id, members in clusters.items():
            cluster_parameters = _parameters(cluster_models[cluster_id])
            updates = [client.fit(cluster_parameters) for client in members]
            _set_parameters(
                cluster_models[cluster_id],
                _average_parameters(updates, [client.num_samples for client in members]),
            )
            communication_bytes += (
                2 * state_dict_nbytes(cluster_models[cluster_id].state_dict()) * len(members)
            )

        cluster_metrics = {
            str(cluster_id): evaluate_regression_model(
                cluster_models[cluster_id], test_loader, device, task_info
            )
            for cluster_id in clusters
        }
        largest_cluster_id = max(clusters, key=lambda cluster_id: len(clusters[cluster_id]))
        primary_metrics = cluster_metrics[str(largest_cluster_id)]
        if round_number == num_rounds:
            primary_metrics = evaluate_regression_model(
                cluster_models[largest_cluster_id],
                test_loader,
                device,
                task_info,
                measure_latency=True,
            )

        _append_metrics(
            history, round_number, primary_metrics, communication_bytes, task_info["name"]
        )
        history["cluster_distribution"].append({
            str(cluster_id): len(members) for cluster_id, members in clusters.items()
        })
        history["per_cluster_metrics"].append(cluster_metrics)
        history["cluster_rmse_std"].append(float(np.std([
            metrics["rmse"] for metrics in cluster_metrics.values()
        ])))
        print(
            f"  Round {round_number}/{num_rounds}: "
            f"RMSE={primary_metrics['rmse']:.4f} "
            f"TSR={primary_metrics['task_success_rate']:.4f}"
        )

    history["rounds_to_convergence"] = rounds_to_convergence(history["rmse"], mode="min")
    return history, cluster_models
