import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config import config
from data.dataset_loader import load_data_for_clients, get_dataloader
from models.model import create_model
from federated.baseline_fl import FlowerClient, run_fedavg
from federated.clustered_fl import ClusteredFlowerClient, run_clustered_fl
from utils.utils import set_seed, get_device, save_results

def run_baseline_experiment(train_datasets, test_dataset, device):
    print("\n" + "="*80)
    print("EXPERIMENT 1: BASELINE FEDERATED LEARNING")
    print("="*80)
    
    test_loader = get_dataloader(test_dataset, config.BATCH_SIZE, shuffle=False)
    
    print("\n[1/3] Creating global model...")
    global_model = create_model(
        num_features=config.NUM_FEATURES,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT
    )
    
    print("[2/3] Creating Flower clients...")
    clients = []
    for i, dataset in enumerate(train_datasets):
        local_model = create_model(
            num_features=config.NUM_FEATURES,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            num_classes=config.NUM_CLASSES,
            dropout=config.DROPOUT
        )
        
        client = FlowerClient(
            client_id=i,
            model=local_model,
            train_dataset=dataset,
            device=device,
            batch_size=config.BATCH_SIZE,
            learning_rate=config.LEARNING_RATE,
            local_epochs=config.LOCAL_EPOCHS
        )
        clients.append(client)
    
    print(f"Created {len(clients)} clients")
    
    print(f"\n[3/3] Running FedAvg...")
    print(f"  Number of rounds: {config.NUM_ROUNDS}")
    print(f"  Clients per round: {config.CLIENTS_PER_ROUND}")
    print(f"  Local epochs: {config.LOCAL_EPOCHS}")
    
    history = run_fedavg(
        clients=clients,
        model=global_model,
        test_loader=test_loader,
        device=device,
        num_rounds=config.NUM_ROUNDS,
        clients_per_round=config.CLIENTS_PER_ROUND
    )
    
    results = {
        "experiment": "baseline_fedavg",
        "config": {
            "num_rounds": config.NUM_ROUNDS,
            "num_clients": config.NUM_ROBOTS,
            "clients_per_round": config.CLIENTS_PER_ROUND,
            "local_epochs": config.LOCAL_EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "alpha": config.ALPHA,
        },
        "metrics": {
            "losses_centralized": history.losses_centralized,
            "metrics_centralized": history.metrics_centralized,
        }
    }
    
    results_path = config.RESULTS_DIR / "baseline_fedavg_results.json"
    save_results(results, results_path)
    print(f"\n✓ Results saved to: {results_path}")
    
    if len(history.metrics_centralized) > 0:
        final_metrics = history.metrics_centralized[-1][1]
        print("\nFinal Results:")
        print(f"  Accuracy: {final_metrics.get('accuracy', 0):.4f}")
        print(f"  F1-Score: {final_metrics.get('f1', 0):.4f}")
    
    return results


def run_clustered_experiment(train_datasets, test_dataset, device):
    print("\n" + "="*80)
    print("EXPERIMENT 2: CLUSTERED FEDERATED LEARNING")
    print("="*80)
    
    
    test_loader = get_dataloader(test_dataset, config.BATCH_SIZE, shuffle=False)
    
    print("\n[1/3] Creating clustered clients...")
    clients = []
    for i, dataset in enumerate(train_datasets):
        local_model = create_model(
            num_features=config.NUM_FEATURES,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            num_classes=config.NUM_CLASSES,
            dropout=config.DROPOUT
        )
        
        client = ClusteredFlowerClient(
            client_id=i,
            model=local_model,
            train_dataset=dataset,
            device=device,
            batch_size=config.BATCH_SIZE,
            learning_rate=config.LEARNING_RATE,
            local_epochs=config.LOCAL_EPOCHS
        )
        clients.append(client)
    
    print(f"Created {len(clients)} clients")
    
    print(f"\n[2/3] Creating cluster models...")
    cluster_models = {}
    for cluster_id in range(config.NUM_CLUSTERS):
        model = create_model(
            num_features=config.NUM_FEATURES,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            num_classes=config.NUM_CLASSES,
            dropout=config.DROPOUT
        )
        cluster_models[cluster_id] = ClusteredFlowerClient(
            client_id=f"cluster_{cluster_id}",
            model=model,
            train_dataset=train_datasets[0],  # Dummy dataset
            device=device,
            batch_size=config.BATCH_SIZE,
            learning_rate=config.LEARNING_RATE,
            local_epochs=config.LOCAL_EPOCHS
        )
    
    print(f"Created {len(cluster_models)} cluster models")
    
    print(f"\n[3/3] Running Clustered FL...")
    print(f"  Number of rounds: {config.NUM_ROUNDS}")
    print(f"  Number of clusters: {config.NUM_CLUSTERS}")
    print(f"  Clustering method: {config.CLUSTERING_METHOD}")
    print(f"  Reclustering interval: {config.RECLUSTERING_INTERVAL}")
    
    history = run_clustered_fl(
        clients=clients,
        cluster_models=cluster_models,
        test_loader=test_loader,
        device=device,
        num_rounds=config.NUM_ROUNDS,
        num_clusters=config.NUM_CLUSTERS,
        clustering_method=config.CLUSTERING_METHOD,
        reclustering_interval=config.RECLUSTERING_INTERVAL
    )
    
    results = {
        "experiment": "clustered_fl",
        "config": {
            "num_rounds": config.NUM_ROUNDS,
            "num_clients": config.NUM_ROBOTS,
            "num_clusters": config.NUM_CLUSTERS,
            "clustering_method": config.CLUSTERING_METHOD,
            "reclustering_interval": config.RECLUSTERING_INTERVAL,
            "local_epochs": config.LOCAL_EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "alpha": config.ALPHA,
        },
        "metrics": history
    }
    
    results_path = config.RESULTS_DIR / "clustered_fl_results.json"
    save_results(results, results_path)
    print(f"\n✓ Results saved to: {results_path}")
    
    if len(history['accuracy']) > 0:
        print("\nFinal Results:")
        print(f"  Accuracy: {history['accuracy'][-1]:.4f}")
        print(f"  F1-Score: {history['f1'][-1]:.4f}")
        print(f"  Loss: {history['loss'][-1]:.4f}")
    
    return results


def main():
    print("="*80)
    print("FEDERATED LEARNING EXPERIMENTS")
    print("="*80)
    
    print("\n[Setup] Setting seed...")
    set_seed(config.SEED)
    
    print("[Setup] Setting device...")
    device = get_device(config.DEVICE)
    print(f"Using device: {device}")
    
    print("\n[Setup] Loading HAR dataset...")
    train_datasets, test_dataset = load_data_for_clients(
        data_dir=config.DATA_DIR,
        num_clients=config.NUM_ROBOTS,
        non_iid=True,
        alpha=config.ALPHA
    )
    
    print(f"Number of clients: {len(train_datasets)}")
    print(f"Test samples: {len(test_dataset)}")
    print("Client data distribution:")
    for i, dataset in enumerate(train_datasets):
        print(f"  Client {i}: {len(dataset)} samples")
    
    baseline_results = run_baseline_experiment(train_datasets, test_dataset, device)
    clustered_results = run_clustered_experiment(train_datasets, test_dataset, device)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80)
    print("\nResults saved in:")
    print(f"  - {config.RESULTS_DIR / 'baseline_fedavg_results.json'}")
    print(f"  - {config.RESULTS_DIR / 'clustered_fl_results.json'}")


if __name__ == "__main__":
    main()