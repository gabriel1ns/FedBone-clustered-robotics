import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from flwr.common import Metrics
from typing import List, Tuple, Dict
from collections import OrderedDict
import numpy as np
import sys
sys.path.append('..')

from utils.utils import calculate_metrics
from data.dataset_loader import get_dataloader


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, model, train_dataset, device, batch_size, learning_rate, local_epochs):
        self.client_id = client_id
        self.model = model.to(device)
        self.device = device
        self.train_loader = get_dataloader(train_dataset, batch_size, shuffle=True)
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.num_samples = len(train_dataset)
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        for epoch in range(self.local_epochs):
            for sequences, labels in self.train_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        return self.get_parameters(config={}), self.num_samples, {}
    
    def evaluate(self, parameters, config):
        return 0.0, self.num_samples, {}


def get_evaluate_fn(model, test_loader, device):
    
    def evaluate(server_round: int, parameters, config):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = calculate_metrics(all_labels, all_preds)
        avg_loss = total_loss / len(test_loader)
        
        print(f"\nRound {server_round} - Server Evaluation:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  Loss: {avg_loss:.4f}")
        
        return avg_loss, {"accuracy": metrics['accuracy'], "f1": metrics['f1']}
    
    return evaluate


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def run_fedavg(clients, model, test_loader, device, num_rounds, clients_per_round):
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=clients_per_round / len(clients),
        fraction_evaluate=0.0,
        min_fit_clients=clients_per_round,
        min_evaluate_clients=0,
        min_available_clients=len(clients),
        evaluate_fn=get_evaluate_fn(model, test_loader, device),
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    def client_fn(cid: str):
        return clients[int(cid)]
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(clients),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0}
    )
    
    return history