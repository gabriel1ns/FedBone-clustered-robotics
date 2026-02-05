import torch
import numpy as np
import random
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def save_results(results, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)

def calculate_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0)
    }

def aggregate_weights(weights_list, weights_sizes=None):
    if weights_sizes is None:
        weights_sizes = [1] * len(weights_list)
    
    total_size = sum(weights_sizes)
    aggregated = {}
    
    for key in weights_list[0].keys():
        aggregated[key] = sum(
            weights_list[i][key] * (weights_sizes[i] / total_size)
            for i in range(len(weights_list))
        )
    
    return aggregated

def print_metrics(metrics, prefix=""):
    print(f"{prefix}Accuracy: {metrics['accuracy']:.4f}")
    print(f"{prefix}F1-Score: {metrics['f1']:.4f}")
    print(f"{prefix}Precision: {metrics['precision']:.4f}")
    print(f"{prefix}Recall: {metrics['recall']:.4f}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(device_name):
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")