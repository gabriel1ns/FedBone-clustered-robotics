import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from typing import List, Dict, Tuple
import copy
import math
import numpy as np
import sys
sys.path.append('..')

from models.fedbone_model import create_fedbone_client, create_fedbone_server
from federated.gp_aggregation import GPAggregation, compute_gradient_conflict_score
from utils.utils import calculate_metrics
from utils.robotics_metrics import (
    SplitCommunicationMeter,
    measure_inference_latency,
    rounds_to_convergence,
    task_success_rate,
)
from data.datasets import get_dataloader


def get_task_criterion(task_type):
    if task_type == 'regression':
        return nn.MSELoss()
    return nn.CrossEntropyLoss()


def split_gaussian_outputs(outputs):
    mean, log_std = torch.chunk(outputs, 2, dim=-1)
    log_std = torch.clamp(log_std, min=-5.0, max=2.0)
    return mean, log_std


def gaussian_nll_loss(outputs, targets):
    mean, log_std = split_gaussian_outputs(outputs)
    variance = torch.exp(2.0 * log_std)
    loss = 0.5 * (((targets - mean) ** 2) / variance + 2.0 * log_std + math.log(2.0 * math.pi))
    return loss.mean()


def policy_action_mean(outputs, policy_type="deterministic"):
    if policy_type == "gaussian":
        return split_gaussian_outputs(outputs)[0]
    return outputs


def compute_policy_loss(outputs, targets, task_type, policy_type="deterministic"):
    if task_type == "regression" and policy_type == "gaussian":
        return gaussian_nll_loss(outputs, targets.float())
    criterion = get_task_criterion(task_type)
    return criterion(outputs, targets)


def compute_task_success(y_true, y_pred, task_info):
    task_type = task_info.get('type', 'classification') if task_info else 'classification'

    if len(y_true) == 0:
        return 0.0

    if task_type == 'regression':
        threshold = task_info.get('success_threshold', 0.15) if task_info else 0.15
        true = np.asarray(y_true, dtype=np.float64)
        pred = np.asarray(y_pred, dtype=np.float64)
        errors = np.linalg.norm(true - pred, axis=-1) if true.ndim > 1 else np.abs(true - pred)
        return float(np.mean(errors <= threshold))

    success_mapping = task_info.get('success_mapping') if task_info else None
    if success_mapping:
        successes = [
            success_mapping.get(int(true), int(true)) == success_mapping.get(int(pred), int(pred))
            for true, pred in zip(y_true, y_pred)
        ]
        return sum(successes) / len(successes)

    return task_success_rate(y_true, y_pred)


def calculate_regression_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0}

    true = torch.tensor(np.asarray(y_true), dtype=torch.float32)
    pred = torch.tensor(np.asarray(y_pred), dtype=torch.float32)
    errors = pred - true
    mae = torch.mean(torch.abs(errors)).item()
    rmse = torch.sqrt(torch.mean(errors ** 2)).item()

    total_var = torch.sum((true - torch.mean(true)) ** 2).item()
    residual = torch.sum(errors ** 2).item()
    r2 = 1.0 - residual / total_var if total_var > 0 else 0.0

    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def denormalize_regression_values(values, task_info):
    normalization = task_info.get('normalization') if task_info else None
    if not normalization:
        return values

    array = np.asarray(values, dtype=np.float32)
    action_mean = np.asarray(normalization['action_mean'], dtype=np.float32)
    action_std = np.asarray(normalization['action_std'], dtype=np.float32)
    return array * action_std + action_mean


class FedBoneClientTrainer:
    """Handles client-side training in split learning setup"""
    def __init__(self, client_id, client_model, train_dataset, device, 
                 batch_size, learning_rate, local_epochs, task_type='classification',
                 policy_type='deterministic'):
        self.client_id = client_id
        self.client_model = client_model.to(device)
        self.device = device
        self.train_loader = get_dataloader(train_dataset, batch_size, shuffle=True)
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.num_samples = len(train_dataset)
        self.task_type = task_type
        self.policy_type = policy_type
        
        self.optimizer = optim.Adam(
            self.client_model.get_client_parameters(),
            lr=learning_rate
        )
        
    def compute_embeddings(self, batch_x):
        """Step 1: Compute embeddings to send to server"""
        self.client_model.eval()
        with torch.no_grad():
            embeddings = self.client_model(batch_x, general_features=None)
        return embeddings
    
    def backward_from_server(self, general_features, labels, criterion):
        """
        Step 3: Receive features from server, complete forward pass,
        compute loss, and backward
        
        Returns: gradients to send back to server
        """
        self.client_model.train()
    
        self.client_model.train()
        
        general_features = general_features.detach().requires_grad_(True)
        outputs = self.client_model(None, general_features=general_features)
        loss = compute_policy_loss(outputs, labels, self.task_type, self.policy_type)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        grad_general_features = general_features.grad.clone() if general_features.grad is not None else torch.zeros_like(general_features)
        self.optimizer.step()
        
        return grad_general_features, loss.item(), outputs
    
    def get_client_state(self):
        """Get client model state dict"""
        return self.client_model.state_dict()
    
    def set_client_state(self, state_dict):
        """Set client model state dict"""
        self.client_model.load_state_dict(state_dict)


class FedBoneServer:
    """Handles server-side operations in split learning setup"""
    
    def __init__(self, server_model, device, num_clients, embed_dim):
        self.server_model = server_model.to(device)
        self.device = device
        self.num_clients = num_clients
        self.stored_features = {}
        
        # Maintain separate general model for each client during mini-batch
        self.client_general_models = {}
        for i in range(num_clients):
            self.client_general_models[i] = copy.deepcopy(server_model)
        
        # GP Aggregation
        # Calculate gradient dimension (approximate)
        total_params = sum(p.numel() for p in server_model.parameters())
        self.gp_aggregator = GPAggregation(gradient_dim=total_params)
        
        self.optimizer = optim.Adam(server_model.parameters(), lr=0.001)
        
    def forward_for_client(self, client_id, embeddings):
        model = self.client_general_models[client_id]
        model.train()
        model.zero_grad()
        
        stored_emb = embeddings.detach().clone().requires_grad_(True)
        general_features = model(stored_emb)
        self.stored_features[client_id] = general_features  # guarda com grafo
        
        return general_features.detach()  # envia detachado ao cliente
    
    def backward_from_client(self, client_id, grad_general_features):
        """
        Step 4: Receive gradients from client and compute gradients for general model
        
        Returns: gradients of general model parameters
        """
        model = self.client_general_models[client_id]
        
        if grad_general_features is None:
            return OrderedDict()
        
        model = self.client_general_models[client_id]
        stored_features = self.stored_features[client_id]
        
        model.zero_grad()
        stored_features.backward(grad_general_features)
        
        client_grads = OrderedDict()
        for name, param in model.named_parameters():
            if param.grad is not None:
                client_grads[name] = param.grad.clone()
        
        return client_grads
        
    def aggregate_and_update(self, client_gradients, client_sizes, use_gp=True):
        """
        Step 5: Aggregate gradients from all clients and update general model
        
        Args:
            client_gradients: List of gradient dicts from each client
            client_sizes: List of dataset sizes
            use_gp: Whether to use GP aggregation or simple average
        """
        if use_gp:
            aggregated_grads = self.gp_aggregator.aggregate(client_gradients, client_sizes)
            conflict_score = compute_gradient_conflict_score(client_gradients)
        else:
            from federated.gp_aggregation import simple_average_aggregation
            aggregated_grads = simple_average_aggregation(client_gradients, client_sizes)
            conflict_score = 0.0
        
        # Apply aggregated gradients to server model
        self.optimizer.zero_grad()
        for name, param in self.server_model.named_parameters():
            if name in aggregated_grads and aggregated_grads[name] is not None:
                param.grad = aggregated_grads[name]
        
        self.optimizer.step()
        
        # Update all client general models with new server model
        for i in range(self.num_clients):
            self.client_general_models[i].load_state_dict(
                self.server_model.state_dict()
            )
        
        return conflict_score
    
    def get_server_state(self):
        """Get server model state dict"""
        return self.server_model.state_dict()


def run_fedbone(clients, server, test_loader, device, num_rounds,
                clients_per_round, use_gp_aggregation=True,
                test_loaders_by_task=None, tasks_info=None):

    history = {
        'rounds': [], 'accuracy': [], 'f1': [],
        'loss': [], 'conflict_scores': [],
        'task_success_rate': [],
        'mae': [], 'rmse': [], 'r2': [],
        'per_task_metrics': [],
        'inference_latency_ms': [],
        'communication_bytes_per_round': [],
        'communication_mb_per_round': [],
        'communication_breakdown': [],
    }

    print("\n" + "="*80)
    print("FEDBONE TRAINING")
    print("="*80)
    print(f"GP Aggregation: {'ENABLED' if use_gp_aggregation else 'DISABLED'}")

    for round_num in range(num_rounds):
        print(f"\n{'='*60}")
        print(f"Round {round_num + 1}/{num_rounds}")
        print(f"{'='*60}")

        import random
        selected_clients = random.sample(clients, clients_per_round)

        round_gradients = []
        round_sizes = []
        round_losses = []
        communication_meter = SplitCommunicationMeter()

        for client in selected_clients:
            # Resetar gradientes do modelo do servidor para este cliente
            client_server_model = server.client_general_models[client.client_id]
            client_server_model.train()
            client.client_model.train()

            client_optimizer = optim.Adam(
                list(client.client_model.parameters()) +
                list(client_server_model.parameters()),
                lr=client.learning_rate
            )

            client_loss_sum = 0
            num_batches = 0

            for epoch in range(client.local_epochs):
                for batch_x, batch_y in client.train_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    criterion = get_task_criterion(client.task_type)

                    client_optimizer.zero_grad()

                    # Forward end-to-end (simulação de split learning)
                    embeddings = client.client_model(batch_x, general_features=None)
                    general_features = client_server_model(embeddings)
                    communication_meter.record_split_batch(embeddings, general_features)
                    outputs = client.client_model(None, general_features=general_features)
                    outputs_for_loss = outputs
                    outputs = policy_action_mean(outputs, getattr(client, "policy_type", "deterministic"))

                    if client.task_type == 'regression' and outputs.shape[-1] == 1:
                        outputs = outputs.squeeze(-1)
                        batch_y = batch_y.float()

                    loss = compute_policy_loss(
                        outputs_for_loss,
                        batch_y.float() if client.task_type == "regression" else batch_y,
                        client.task_type,
                        getattr(client, "policy_type", "deterministic"),
                    )
                    loss.backward()
                    client_optimizer.step()

                    client_loss_sum += loss.item()
                    num_batches += 1

            # Coletar gradientes do modelo do servidor para este cliente
            client_grads = OrderedDict()
            for name, param in client_server_model.named_parameters():
                if param.grad is not None:
                    client_grads[name] = param.grad.clone()
                else:
                    client_grads[name] = torch.zeros_like(param)

            round_gradients.append(client_grads)
            round_sizes.append(client.num_samples)
            round_losses.append(client_loss_sum / max(num_batches, 1))

        # Agregar gradientes no servidor global
        conflict_score = server.aggregate_and_update(
            round_gradients, round_sizes, use_gp=use_gp_aggregation
        )

        avg_loss = sum(round_losses) / len(round_losses)
        if test_loaders_by_task is not None:
            metrics = evaluate_fedbone_multitask(
                clients,
                server,
                test_loaders_by_task,
                device,
                tasks_info,
                measure_latency=True,
            )
        else:
            metrics = evaluate_fedbone(
                clients[0],
                server,
                test_loader,
                device,
                get_task_criterion(clients[0].task_type),
                measure_latency=True,
                task_info=tasks_info[0] if tasks_info else None,
            )
        communication_summary = communication_meter.summary()

        history['rounds'].append(round_num + 1)
        history['accuracy'].append(metrics['accuracy'])
        history['f1'].append(metrics['f1'])
        history['loss'].append(avg_loss)
        history['conflict_scores'].append(conflict_score)
        history['task_success_rate'].append(metrics['task_success_rate'])
        history['mae'].append(metrics.get('mae'))
        history['rmse'].append(metrics.get('rmse'))
        history['r2'].append(metrics.get('r2'))
        history['per_task_metrics'].append(metrics.get('per_task_metrics', {}))
        history['inference_latency_ms'].append(metrics['inference_latency_ms'])
        history['communication_bytes_per_round'].append(communication_summary['total_bytes'])
        history['communication_mb_per_round'].append(communication_summary['total_mb'])
        history['communication_breakdown'].append(communication_summary)

        print(f"\nRound {round_num + 1} Results:")
        if metrics.get('accuracy') is not None:
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1-Score: {metrics['f1']:.4f}")
        if metrics.get('mae') is not None:
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  R2: {metrics['r2']:.4f}")
        print(f"  Task Success Rate: {metrics['task_success_rate']:.4f}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Gradient Conflict: {conflict_score:.4f}")
        print(f"  Inference Latency: {metrics['inference_latency_ms']['mean_ms']:.2f} ms")
        print(f"  Communication: {communication_summary['total_mb']:.2f} MB")

    accuracy_values = [value for value in history['accuracy'] if value is not None]
    rmse_values = [value for value in history['rmse'] if value is not None]
    if accuracy_values:
        history['rounds_to_convergence'] = rounds_to_convergence(history['accuracy'])
    elif rmse_values:
        history['rounds_to_convergence'] = rounds_to_convergence(history['rmse'], mode="min")
    else:
        history['rounds_to_convergence'] = None
    return history


def evaluate_fedbone(sample_client, server, test_loader, device, criterion,
                     measure_latency=False, task_info=None):
    sample_client.client_model.eval()
    server.server_model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0
    num_batches = 0
    task_type = sample_client.task_type

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            embeddings = sample_client.client_model(sequences, general_features=None)
            general_features = server.server_model(embeddings)
            outputs = sample_client.client_model(None, general_features=general_features)
            outputs = policy_action_mean(outputs, getattr(sample_client, "policy_type", "deterministic"))

            try:
                if task_type == 'regression' and outputs.shape[-1] == 1:
                    outputs = outputs.squeeze(-1)
                if task_type == 'regression':
                    labels = labels.float()
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            except Exception:
                pass

            num_batches += 1
            if task_type == 'regression':
                preds = outputs
            else:
                _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if task_type == 'regression':
        metric_labels = denormalize_regression_values(all_labels, task_info or {})
        metric_preds = denormalize_regression_values(all_preds, task_info or {})
        metrics = calculate_regression_metrics(metric_labels, metric_preds)
        metrics['accuracy'] = 0.0
        metrics['f1'] = 0.0
    else:
        metric_labels = all_labels
        metric_preds = all_preds
        metrics = calculate_metrics(all_labels, all_preds)

    metrics['loss'] = total_loss / max(num_batches, 1)
    metrics['task_success_rate'] = compute_task_success(metric_labels, metric_preds, task_info or {})
    if measure_latency:
        metrics['inference_latency_ms'] = measure_inference_latency(
            test_loader,
            device,
            lambda inputs: sample_client.client_model(
                None,
                general_features=server.server_model(
                    sample_client.client_model(inputs, general_features=None)
                ),
            ),
        )
    else:
        metrics['inference_latency_ms'] = {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}
    return metrics


def evaluate_fedbone_multitask(clients, server, test_loaders_by_task, device,
                               tasks_info=None, measure_latency=False):
    per_task_metrics = {}
    classification_accuracy = []
    classification_f1 = []
    regression_mae = []
    regression_rmse = []
    regression_r2 = []
    task_success_values = []
    latency_values = []

    for task_id, test_loader in test_loaders_by_task.items():
        task_info = tasks_info[task_id] if tasks_info else {}
        matching_clients = [client for client in clients if getattr(client, 'task_id', None) == task_id]
        if not matching_clients:
            continue

        task_metrics = evaluate_fedbone(
            matching_clients[0],
            server,
            test_loader,
            device,
            get_task_criterion(task_info.get('type', 'classification')),
            measure_latency=measure_latency,
            task_info=task_info,
        )

        task_name = task_info.get('name', str(task_id))
        per_task_metrics[task_name] = task_metrics
        task_success_values.append(task_metrics['task_success_rate'])
        latency_values.append(task_metrics['inference_latency_ms']['mean_ms'])

        if task_info.get('type', 'classification') == 'classification':
            classification_accuracy.append(task_metrics['accuracy'])
            classification_f1.append(task_metrics['f1'])
        elif task_info.get('type') == 'regression':
            regression_mae.append(task_metrics['mae'])
            regression_rmse.append(task_metrics['rmse'])
            regression_r2.append(task_metrics['r2'])

    mean_latency = sum(latency_values) / len(latency_values) if latency_values else 0.0

    return {
        'accuracy': sum(classification_accuracy) / len(classification_accuracy) if classification_accuracy else None,
        'f1': sum(classification_f1) / len(classification_f1) if classification_f1 else None,
        'mae': sum(regression_mae) / len(regression_mae) if regression_mae else None,
        'rmse': sum(regression_rmse) / len(regression_rmse) if regression_rmse else None,
        'r2': sum(regression_r2) / len(regression_r2) if regression_r2 else None,
        'task_success_rate': sum(task_success_values) / len(task_success_values) if task_success_values else 0.0,
        'inference_latency_ms': {
            'mean_ms': mean_latency,
            'median_ms': mean_latency,
            'p95_ms': max(latency_values) if latency_values else 0.0,
        },
        'per_task_metrics': per_task_metrics,
    }
