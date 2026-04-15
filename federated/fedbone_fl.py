import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from typing import List, Dict, Tuple
import copy
import sys
sys.path.append('..')

from models.fedbone_model import create_fedbone_client, create_fedbone_server
from federated.gp_aggregation import GPAggregation, compute_gradient_conflict_score
from utils.utils import calculate_metrics
from data.dataset_loader import get_dataloader


class FedBoneClientTrainer:
    """Handles client-side training in split learning setup"""
    def __init__(self, client_id, client_model, train_dataset, device, 
                 batch_size, learning_rate, local_epochs, task_type='classification'):
        self.client_id = client_id
        self.client_model = client_model.to(device)
        self.device = device
        self.train_loader = get_dataloader(train_dataset, batch_size, shuffle=True)
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.num_samples = len(train_dataset)
        self.task_type = task_type
        
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
        loss = criterion(outputs, labels)
        
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
                clients_per_round, use_gp_aggregation=True):

    history = {
        'rounds': [], 'accuracy': [], 'f1': [],
        'loss': [], 'conflict_scores': []
    }

    criterion = nn.CrossEntropyLoss()

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

                    client_optimizer.zero_grad()

                    # Forward end-to-end (simulação de split learning)
                    embeddings = client.client_model(batch_x, general_features=None)
                    general_features = client_server_model(embeddings)
                    outputs = client.client_model(None, general_features=general_features)

                    loss = criterion(outputs, batch_y)
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
        metrics = evaluate_fedbone(clients[0], server, test_loader, device, criterion)

        history['rounds'].append(round_num + 1)
        history['accuracy'].append(metrics['accuracy'])
        history['f1'].append(metrics['f1'])
        history['loss'].append(avg_loss)
        history['conflict_scores'].append(conflict_score)

        print(f"\nRound {round_num + 1} Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Gradient Conflict: {conflict_score:.4f}")

    return history


def evaluate_fedbone(sample_client, server, test_loader, device, criterion):
    sample_client.client_model.eval()
    server.server_model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            embeddings = sample_client.client_model(sequences, general_features=None)
            general_features = server.server_model(embeddings)
            outputs = sample_client.client_model(None, general_features=general_features)

            try:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            except Exception:
                pass

            num_batches += 1
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = calculate_metrics(all_labels, all_preds)
    return metrics