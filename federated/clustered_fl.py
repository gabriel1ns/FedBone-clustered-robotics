import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from flwr.common import Metrics
from typing import List, Tuple, Dict
from collections import OrderedDict
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
import sys
sys.path.append('..')

from utils.utils import calculate_metrics, aggregate_weights
from data.dataset_loader import get_dataloader


class ClusteredFlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, model, train_dataset, device, batch_size, learning_rate, local_epochs):
        self.client_id = client_id
        self.model = model.to(device)
        self.device = device
        self.train_loader = get_dataloader(train_dataset, batch_size, shuffle=True)
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.num_samples = len(train_dataset)
        self.cluster_id = None  # Será atribuído durante clustering
    
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
        self.set_parameters(parameters)
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for sequences, labels in self.train_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(sequences)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = calculate_metrics(all_labels, all_preds)
        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss, self.num_samples, {"accuracy": metrics['accuracy']}


def cluster_clients(clients, num_clusters, method="kmeans"):
    client_weights = []
    for client in clients:
        params = client.get_parameters(config={})
        flat_params = np.concatenate([p.flatten() for p in params])
        client_weights.append(flat_params)
    
    client_weights = np.array(client_weights)
    if method == "kmeans":
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(client_weights)
    elif method == "hierarchical":
        linkage_matrix = linkage(client_weights, method='ward')
        cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
        cluster_labels = cluster_labels - 1
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    clusters = {}
    for i, client in enumerate(clients):
        cluster_id = int(cluster_labels[i])
        client.cluster_id = cluster_id
        
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(client)
    
    return clusters


def aggregate_cluster(clients):
    weights_list = [client.get_parameters(config={}) for client in clients]
    sizes = [client.num_samples for client in clients]
    
    state_dicts = []
    for params in weights_list:
        state_dict = OrderedDict()
        for i, param in enumerate(params):
            state_dict[f"param_{i}"] = torch.tensor(param)
        state_dicts.append(state_dict)
    
    aggregated = aggregate_weights(state_dicts, sizes)
    
    return [aggregated[f"param_{i}"].numpy() for i in range(len(params))]


def run_clustered_fl(clients, cluster_models, test_loader, device, num_rounds, 
                     num_clusters, clustering_method, reclustering_interval):
    history = {
        'rounds': [],
        'accuracy': [],
        'f1': [],
        'loss': [],
        'cluster_distribution': []
    }
    
    # Clustering inicial
    print("\n[Initial Clustering]")
    clusters = cluster_clients(clients, num_clusters, clustering_method)
    print(f"Cluster distribution: {[len(c) for c in clusters.values()]}")
    
    for round_num in range(num_rounds):
        print(f"\n{'='*60}")
        print(f"Round {round_num + 1}/{num_rounds}")
        print(f"{'='*60}")
        
        # Reclusterizar se necessário
        if reclustering_interval > 0 and round_num > 0 and round_num % reclustering_interval == 0:
            print(f"\n[Re-clustering at round {round_num + 1}]")
            clusters = cluster_clients(clients, num_clusters, clustering_method)
            print(f"New cluster distribution: {[len(c) for c in clusters.values()]}")
        
        for cluster_id, cluster_clients in clusters.items():
            cluster_model_params = cluster_models[cluster_id].get_parameters(config={})
            
            for client in cluster_clients:
                client.fit(cluster_model_params, config={})
            
            aggregated_params = aggregate_cluster(cluster_clients)
            cluster_models[cluster_id].set_parameters(aggregated_params)
        
        all_preds = []
        all_labels = []
        total_loss = 0
        
        criterion = nn.CrossEntropyLoss()
        
        for cluster_id, model_client in cluster_models.items():
            model = model_client.model
            model.eval()
            
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
        
        metrics = calculate_metrics(all_labels, all_labels) #change afterwards
        avg_loss = total_loss / (len(test_loader) * len(cluster_models))
        
        
        largest_cluster = max(clusters.items(), key=lambda x: len(x[1]))
        best_model = cluster_models[largest_cluster[0]].model
        best_model.eval()
        
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                outputs = best_model(sequences)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = calculate_metrics(all_labels, all_preds)
        
    
        history['rounds'].append(round_num + 1)
        history['accuracy'].append(metrics['accuracy'])
        history['f1'].append(metrics['f1'])
        history['loss'].append(avg_loss)
        history['cluster_distribution'].append([len(c) for c in clusters.values()])
        
        print(f"\nRound {round_num + 1} Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  Loss: {avg_loss:.4f}")
    
    return history