import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Tuple
import random


class MultiTaskDataset(Dataset):
    """Dataset that supports multiple tasks with different label spaces"""
    
    def __init__(self, sequences, labels, task_type='classification', task_id=0):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels) if task_type == 'classification' else torch.FloatTensor(labels)
        self.task_type = task_type
        self.task_id = task_id
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def create_multitask_split(X_train, y_train, num_tasks=3, task_configs=None):
    """
    Create multiple tasks from single dataset by:
    1. Different class subsets (e.g., binary classification on different class pairs)
    2. Different granularities (fine-grained vs coarse classification)
    3. Regression tasks (predict activity intensity, duration, etc.)
    
    Args:
        X_train: Training sequences
        y_train: Training labels
        num_tasks: Number of different tasks to create
        task_configs: List of task configurations
        
    Returns:
        List of (task_data, task_labels, task_type, num_classes) tuples
    """
    
    if task_configs is None:
        # Default: create varied tasks from HAR dataset
        task_configs = [
            {
                'name': 'activity_classification',
                'type': 'classification',
                'classes': [0, 1, 2, 3, 4, 5],  # All 6 activities
                'description': 'Full 6-class activity recognition'
            },
            {
                'name': 'stationary_vs_moving',
                'type': 'classification',
                'classes': [3, 4, 5, 0, 1, 2],  # Remap to binary
                'binary_mapping': {3: 0, 4: 0, 5: 0, 0: 1, 1: 1, 2: 1},  # sitting/standing/laying vs walking activities
                'description': 'Binary: stationary vs moving'
            },
            {
                'name': 'walking_variants',
                'type': 'classification',
                'classes': [0, 1, 2],  # Only walking activities
                'description': '3-class walking type classification'
            },
            {
                'name': 'activity_intensity',
                'type': 'regression',
                'intensity_map': {0: 0.5, 1: 0.8, 2: 0.6, 3: 0.1, 4: 0.15, 5: 0.0},
                'description': 'Regression: predict activity intensity'
            }
        ]
    
    tasks = []
    
    for i, config in enumerate(task_configs[:num_tasks]):
        task_type = config['type']
        
        if task_type == 'classification':
            if 'binary_mapping' in config:
                # Binary classification task
                task_labels = np.array([config['binary_mapping'].get(label, -1) for label in y_train])
                valid_idx = task_labels >= 0
                task_data = X_train[valid_idx]
                task_labels = task_labels[valid_idx]
                num_classes = 2
                
            elif 'classes' in config and len(config['classes']) < 6:
                # Subset classification
                valid_idx = np.isin(y_train, config['classes'])
                task_data = X_train[valid_idx]
                task_labels = y_train[valid_idx]
                
                # Remap labels to 0, 1, 2, ...
                label_mapping = {old: new for new, old in enumerate(config['classes'])}
                task_labels = np.array([label_mapping[label] for label in task_labels])
                num_classes = len(config['classes'])
            else:
                # Full classification
                task_data = X_train
                task_labels = y_train
                num_classes = len(np.unique(y_train))
        
        elif task_type == 'regression':
            # Regression task: map classes to continuous values
            intensity_map = config['intensity_map']
            task_data = X_train
            task_labels = np.array([intensity_map[label] for label in y_train], dtype=np.float32)
            num_classes = 1  # Regression outputs 1 value
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        tasks.append({
            'data': task_data,
            'labels': task_labels,
            'type': task_type,
            'num_classes': num_classes,
            'name': config['name'],
            'description': config.get('description', '')
        })
    
    return tasks


def assign_tasks_to_clients(tasks, num_clients, distribution='uniform'):
    """
    Assign tasks to clients
    
    Args:
        tasks: List of task dicts
        num_clients: Number of clients
        distribution: 'uniform' (all clients do all tasks), 
                     'specialized' (each client has 1-2 tasks),
                     'mixed' (varied distribution)
    
    Returns:
        List of client task assignments
    """
    
    num_tasks = len(tasks)
    client_tasks = [[] for _ in range(num_clients)]
    
    if distribution == 'uniform':
        # All clients do all tasks
        for i in range(num_clients):
            client_tasks[i] = list(range(num_tasks))
    
    elif distribution == 'specialized':
        # Each client specializes in 1-2 tasks
        for i in range(num_clients):
            num_client_tasks = random.choice([1, 2])
            client_tasks[i] = random.sample(range(num_tasks), num_client_tasks)
    
    elif distribution == 'mixed':
        # Mix of specialized and generalist clients
        specialized_clients = num_clients // 2
        
        for i in range(specialized_clients):
            client_tasks[i] = [random.randint(0, num_tasks - 1)]
        
        for i in range(specialized_clients, num_clients):
            num_client_tasks = random.randint(2, num_tasks)
            client_tasks[i] = random.sample(range(num_tasks), num_client_tasks)
    
    return client_tasks


def create_multitask_federated_split(tasks, num_clients, task_distribution='mixed', alpha=0.5):
    """
    Create federated split with multiple tasks per client
    
    Returns:
        List of client datasets, where each client may have multiple tasks
    """
    from data.dataset_loader import create_non_iid_split
    
    client_datasets = []
    client_tasks_assignment = assign_tasks_to_clients(tasks, num_clients, task_distribution)
    
    for client_id in range(num_clients):
        client_task_datasets = []
        
        for task_id in client_tasks_assignment[client_id]:
            task = tasks[task_id]
            
            # Split this task's data among clients who have this task
            clients_with_task = [i for i in range(num_clients) if task_id in client_tasks_assignment[i]]
            num_clients_for_task = len(clients_with_task)
            
            # Create non-IID split for this task
            task_split = create_non_iid_split(
                task['data'],
                task['labels'],
                num_clients_for_task,
                alpha=alpha
            )
            
            # Find this client's index among clients with this task
            client_idx_in_task = clients_with_task.index(client_id)
            
            # Create dataset
            data, labels = task_split[client_idx_in_task]
            dataset = MultiTaskDataset(
                data,
                labels,
                task_type=task['type'],
                task_id=task_id
            )
            
            client_task_datasets.append({
                'dataset': dataset,
                'task_id': task_id,
                'task_type': task['type'],
                'task_name': task['name'],
                'num_classes': task['num_classes']
            })
        
        client_datasets.append(client_task_datasets)
    
    return client_datasets, tasks


def load_multitask_data(data_dir, num_clients, num_tasks=3, task_distribution='mixed', alpha=0.5):
    """
    Main function to load multi-task federated data
    
    Returns:
        client_datasets: List of client task datasets
        test_datasets: Dict of test datasets per task
        tasks_info: Task metadata
    """
    from data.dataset_loader import load_har_data
    
    # Load base dataset
    X_train, y_train, X_test, y_test = load_har_data(data_dir)
    
    # Create multiple tasks
    tasks = create_multitask_split(X_train, y_train, num_tasks=num_tasks)
    
    print("\n" + "="*60)
    print("MULTI-TASK SETUP")
    print("="*60)
    for i, task in enumerate(tasks):
        print(f"\nTask {i}: {task['name']}")
        print(f"  Type: {task['type']}")
        print(f"  Classes: {task['num_classes']}")
        print(f"  Samples: {len(task['data'])}")
        print(f"  Description: {task['description']}")
    
    # Create federated split
    client_datasets, tasks = create_multitask_federated_split(
        tasks,
        num_clients,
        task_distribution=task_distribution,
        alpha=alpha
    )
    
    print("\n" + "="*60)
    print("CLIENT-TASK ASSIGNMENT")
    print("="*60)
    for i, client_data in enumerate(client_datasets):
        task_names = [td['task_name'] for td in client_data]
        print(f"Client {i}: {len(client_data)} tasks - {task_names}")
    
    # Create test datasets for each task
    test_datasets = {}
    for i, task in enumerate(tasks):
        if task['type'] == 'classification':
            # Filter test data for this task
            if 'binary_mapping' in [tc for tc in [
                {'name': 'activity_classification', 'type': 'classification', 'classes': [0, 1, 2, 3, 4, 5]},
                {'name': 'stationary_vs_moving', 'type': 'classification', 'classes': [3, 4, 5, 0, 1, 2], 
                 'binary_mapping': {3: 0, 4: 0, 5: 0, 0: 1, 1: 1, 2: 1}},
            ] if tc['name'] == task['name']][0] if any(tc['name'] == task['name'] for tc in [
                {'name': 'activity_classification', 'type': 'classification', 'classes': [0, 1, 2, 3, 4, 5]},
                {'name': 'stationary_vs_moving', 'type': 'classification', 'classes': [3, 4, 5, 0, 1, 2], 
                 'binary_mapping': {3: 0, 4: 0, 5: 0, 0: 1, 1: 1, 2: 1}},
            ]) else {}:
                # Binary mapping task
                config = [tc for tc in [
                    {'name': 'stationary_vs_moving', 'type': 'classification', 'classes': [3, 4, 5, 0, 1, 2], 
                     'binary_mapping': {3: 0, 4: 0, 5: 0, 0: 1, 1: 1, 2: 1}},
                ] if tc['name'] == task['name']][0]
                test_labels = np.array([config['binary_mapping'].get(label, -1) for label in y_test])
                valid_idx = test_labels >= 0
                test_data = X_test[valid_idx]
                test_labels = test_labels[valid_idx]
            else:
                test_data = X_test
                test_labels = y_test
        else:
            # Regression task
            test_data = X_test
            test_labels = y_test
        
        test_datasets[i] = MultiTaskDataset(
            test_data,
            test_labels,
            task_type=task['type'],
            task_id=i
        )
    
    return client_datasets, test_datasets, tasks