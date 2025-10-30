import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split


class HARDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def load_har_data(data_dir):
    data_dir = Path(data_dir)
    
    X_train = np.loadtxt(data_dir / 'train' / 'X_train.txt')
    y_train = np.loadtxt(data_dir / 'train' / 'y_train.txt')
    
    X_test = np.loadtxt(data_dir / 'test' / 'X_test.txt')
    y_test = np.loadtxt(data_dir / 'test' / 'y_test.txt')
    
    y_train = y_train - 1
    y_test = y_test - 1
    
    #(n_samples, sequence_length, num_features)
    
    sequence_length = 128
    num_features = X_train.shape[1] // sequence_length
    
    if X_train.shape[1] % sequence_length != 0:
        X_train = X_train.reshape(-1, 1, X_train.shape[1])
        X_test = X_test.reshape(-1, 1, X_test.shape[1])
    else:
        X_train = X_train.reshape(-1, sequence_length, num_features)
        X_test = X_test.reshape(-1, sequence_length, num_features)
    
    return X_train, y_train.astype(int), X_test, y_test.astype(int)


def create_non_iid_split(data, labels, num_clients, alpha=0.5):
#dirichlet for non-iid dist
    num_classes = len(np.unique(labels))
    label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
    
    client_data = [[] for _ in range(num_clients)]
    client_labels = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)
        
        proportions = label_distribution[c]
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        idx_splits = np.split(idx_c, proportions)
        
        for i, idx in enumerate(idx_splits):
            if len(idx) > 0:
                client_data[i].extend(data[idx])
                client_labels[i].extend(labels[idx])
    
    #array conversion
    client_datasets = []
    for i in range(num_clients):
        if len(client_data[i]) > 0:
            data_i = np.array(client_data[i])
            labels_i = np.array(client_labels[i])
            client_datasets.append((data_i, labels_i))
    
    return client_datasets


def load_data_for_clients(data_dir, num_clients, non_iid=True, alpha=0.5):
    X_train, y_train, X_test, y_test = load_har_data(data_dir)
    
    if non_iid:
        #non-iid distribution
        client_data = create_non_iid_split(X_train, y_train, num_clients, alpha)
    else:
        samples_per_client = len(X_train) // num_clients
        client_data = []
        for i in range(num_clients):
            start = i * samples_per_client
            end = start + samples_per_client
            client_data.append((X_train[start:end], y_train[start:end]))
    
    #training for each client
    train_datasets = []
    for data, labels in client_data:
        dataset = HARDataset(data, labels)
        train_datasets.append(dataset)
    
    test_dataset = HARDataset(X_test, y_test)
    
    return train_datasets, test_dataset


def get_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)