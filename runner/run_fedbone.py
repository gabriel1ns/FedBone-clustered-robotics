import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from data.multitask_loader import load_multitask_data
from data.dataset_loader import get_dataloader
from models.fedbone_model import create_fedbone_client, create_fedbone_server, count_parameters
from federated.fedbone_fl import FedBoneClientTrainer, FedBoneServer, run_fedbone
from utils.utils import set_seed, get_device, save_results


def run_fedbone_experiment(client_datasets, test_datasets, tasks, device):
    """
    Run complete FedBone experiment with multi-task learning
    """
    
    print("\n" + "="*80)
    print("FEDBONE MULTI-TASK FEDERATED LEARNING")
    print("="*80)
    
    # Configuration
    EMBED_DIM = 64
    HIDDEN_SIZE = config.HIDDEN_SIZE
    NUM_LAYERS = config.NUM_LAYERS
    
    print("\n[1/4] Creating server model...")
    server_model = create_fedbone_server(
        embed_dim=EMBED_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=config.DROPOUT
    )
    
    server_params = count_parameters(server_model)
    print(f"  Server (General Model) parameters: {server_params:,}")
    
    print("\n[2/4] Creating client models...")
    clients = []
    
    for client_id, client_task_datasets in enumerate(client_datasets):
        # For simplicity, use the first task of each client for training
        # In full implementation, would handle multiple tasks per client
        
        if len(client_task_datasets) == 0:
            continue
        
        # Use primary task (first one)
        primary_task = client_task_datasets[0]
        
        client_model = create_fedbone_client(
            num_features=config.NUM_FEATURES,
            embed_dim=EMBED_DIM,
            hidden_size=HIDDEN_SIZE,
            num_classes=primary_task['num_classes'],
            task_type=primary_task['task_type']
        )
        
        client = FedBoneClientTrainer(
            client_id=client_id,
            client_model=client_model,
            train_dataset=primary_task['dataset'],
            device=device,
            batch_size=config.BATCH_SIZE,
            learning_rate=config.LEARNING_RATE,
            local_epochs=config.LOCAL_EPOCHS,
            task_type=primary_task['task_type']
        )
        
        clients.append(client)
    
    client_params = count_parameters(clients[0].client_model)
    print(f"  Client model parameters (each): {client_params:,}")
    print(f"  Created {len(clients)} clients with heterogeneous tasks")
    
    print("\n[3/4] Initializing FedBone server...")
    server = FedBoneServer(
        server_model=server_model,
        device=device,
        num_clients=len(clients),
        embed_dim=EMBED_DIM
    )
    
    # Use first task's test set for evaluation
    primary_test_task = list(test_datasets.values())[0]
    test_loader = get_dataloader(primary_test_task, config.BATCH_SIZE, shuffle=False)
    
    print("\n[4/4] Running FedBone training...")
    print(f"  Number of rounds: {config.NUM_ROUNDS}")
    print(f"  Clients per round: {config.CLIENTS_PER_ROUND}")
    print(f"  Local epochs: {config.LOCAL_EPOCHS}")
    print(f"  GP Aggregation: ENABLED")
    
    # Run with GP Aggregation
    history_gp = run_fedbone(
        clients=clients,
        server=server,
        test_loader=test_loader,
        device=device,
        num_rounds=config.NUM_ROUNDS,
        clients_per_round=min(config.CLIENTS_PER_ROUND, len(clients)),
        use_gp_aggregation=True
    )
    
    # Save results
    results = {
        "experiment": "fedbone_multitask",
        "config": {
            "num_rounds": config.NUM_ROUNDS,
            "num_clients": len(clients),
            "clients_per_round": config.CLIENTS_PER_ROUND,
            "local_epochs": config.LOCAL_EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "embed_dim": EMBED_DIM,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "server_params": server_params,
            "client_params": client_params,
            "gp_aggregation": True,
        },
        "tasks": [
            {
                "task_id": i,
                "name": task['name'],
                "type": task['type'],
                "num_classes": task['num_classes']
            }
            for i, task in enumerate(tasks)
        ],
        "metrics": history_gp
    }
    
    results_path = config.RESULTS_DIR / "fedbone_multitask_results.json"
    save_results(results, results_path)
    print(f"\n✓ Results saved to: {results_path}")
    
    if len(history_gp['accuracy']) > 0:
        print("\nFinal Results (with GP Aggregation):")
        print(f"  Accuracy: {history_gp['accuracy'][-1]:.4f}")
        print(f"  F1-Score: {history_gp['f1'][-1]:.4f}")
        print(f"  Loss: {history_gp['loss'][-1]:.4f}")
        print(f"  Avg Gradient Conflict: {sum(history_gp['conflict_scores'])/len(history_gp['conflict_scores']):.4f}")
    
    return results


def compare_gp_vs_baseline(client_datasets, test_datasets, tasks, device):
    """
    Compare FedBone with GP Aggregation vs Simple Averaging
    """
    
    print("\n" + "="*80)
    print("COMPARISON: GP AGGREGATION vs BASELINE")
    print("="*80)
    
    EMBED_DIM = 64
    HIDDEN_SIZE = config.HIDDEN_SIZE
    
    results_comparison = {}
    
    for use_gp, exp_name in [(False, "baseline"), (True, "gp_aggregation")]:
        print(f"\n{'='*60}")
        print(f"Running: {exp_name.upper()}")
        print(f"{'='*60}")
        
        # Create fresh models
        server_model = create_fedbone_server(EMBED_DIM, HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT)
        
        clients = []
        for client_id, client_task_datasets in enumerate(client_datasets):
            if len(client_task_datasets) == 0:
                continue
            
            primary_task = client_task_datasets[0]
            client_model = create_fedbone_client(
                config.NUM_FEATURES, EMBED_DIM, HIDDEN_SIZE,
                primary_task['num_classes'], primary_task['task_type']
            )
            
            client = FedBoneClientTrainer(
                client_id, client_model, primary_task['dataset'],
                device, config.BATCH_SIZE, config.LEARNING_RATE,
                config.LOCAL_EPOCHS, primary_task['task_type']
            )
            clients.append(client)
        
        server = FedBoneServer(server_model, device, len(clients), EMBED_DIM)
        
        primary_test_task = list(test_datasets.values())[0]
        test_loader = get_dataloader(primary_test_task, config.BATCH_SIZE, shuffle=False)
        
        history = run_fedbone(
            clients, server, test_loader, device,
            config.NUM_ROUNDS, min(config.CLIENTS_PER_ROUND, len(clients)),
            use_gp_aggregation=use_gp
        )
        
        results_comparison[exp_name] = history
    
    # Save comparison
    comparison_results = {
        "experiment": "fedbone_comparison",
        "baseline": results_comparison["baseline"],
        "gp_aggregation": results_comparison["gp_aggregation"]
    }
    
    results_path = config.RESULTS_DIR / "fedbone_comparison_results.json"
    save_results(comparison_results, results_path)
    
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print("\nBaseline (Simple Averaging):")
    print(f"  Final Accuracy: {results_comparison['baseline']['accuracy'][-1]:.4f}")
    print(f"  Final F1: {results_comparison['baseline']['f1'][-1]:.4f}")
    
    print("\nGP Aggregation:")
    print(f"  Final Accuracy: {results_comparison['gp_aggregation']['accuracy'][-1]:.4f}")
    print(f"  Final F1: {results_comparison['gp_aggregation']['f1'][-1]:.4f}")
    print(f"  Avg Conflict Score: {sum(results_comparison['gp_aggregation']['conflict_scores'])/len(results_comparison['gp_aggregation']['conflict_scores']):.4f}")
    
    improvement = (results_comparison['gp_aggregation']['accuracy'][-1] - 
                   results_comparison['baseline']['accuracy'][-1]) * 100
    print(f"\n✓ Improvement: {improvement:+.2f}%")
    
    return comparison_results


def main():
    print("="*80)
    print("FEDBONE MULTI-TASK EXPERIMENTS")
    print("="*80)
    
    set_seed(config.SEED)
    device = get_device(config.DEVICE)
    print(f"Using device: {device}")
    
    # Load multi-task data
    print("\nLoading multi-task HAR dataset...")
    client_datasets, test_datasets, tasks = load_multitask_data(
        data_dir=config.DATA_DIR,
        num_clients=config.NUM_ROBOTS,
        num_tasks=3,  # 3 different tasks
        task_distribution='mixed',  # Mix of specialized and generalist clients
        alpha=config.ALPHA
    )
    
    # Run main experiment
    results = run_fedbone_experiment(client_datasets, test_datasets, tasks, device)
    
    # Optional: Run comparison
    print("\n" + "="*80)
    print("Run GP Aggregation comparison? (y/n)")
    # comparison = compare_gp_vs_baseline(client_datasets, test_datasets, tasks, device)
    
    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETED!")
    print("="*80)
    print(f"\nResults saved in: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()