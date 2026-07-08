import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from data.robomimic_loader import load_robomimic_data
from data.datasets import get_dataloader
from models.fedbone_model import create_fedbone_client, create_fedbone_server, count_parameters
from federated.fedbone_fl import FedBoneClientTrainer, FedBoneServer, run_fedbone
from utils.utils import set_seed, get_device, save_results


def build_fedbone_clients(client_datasets, device, embed_dim, hidden_size):
    clients = []
    virtual_client_id = 0
    client_params = 0

    for robot_id, client_task_datasets in enumerate(client_datasets):
        if len(client_task_datasets) == 0:
            continue

        for task_dataset in client_task_datasets:
            num_features = int(task_dataset['dataset'].sequences.shape[-1])
            client_model = create_fedbone_client(
                num_features=num_features,
                embed_dim=embed_dim,
                hidden_size=hidden_size,
                num_classes=task_dataset['num_classes'],
                task_type=task_dataset['task_type']
            )

            client = FedBoneClientTrainer(
                client_id=virtual_client_id,
                client_model=client_model,
                train_dataset=task_dataset['dataset'],
                device=device,
                batch_size=config.BATCH_SIZE,
                learning_rate=config.LEARNING_RATE,
                local_epochs=config.LOCAL_EPOCHS,
                task_type=task_dataset['task_type']
            )
            client.robot_id = task_dataset.get('robot_id', robot_id)
            client.task_id = task_dataset['task_id']
            client.task_name = task_dataset['task_name']

            clients.append(client)
            virtual_client_id += 1
            client_params = count_parameters(client_model)

    return clients, client_params


def build_test_loaders(test_datasets):
    return {
        task_id: get_dataloader(dataset, config.BATCH_SIZE, shuffle=False)
        for task_id, dataset in test_datasets.items()
    }


def save_fedbone_checkpoint(server, clients, tasks, path, embed_dim, hidden_size, num_layers):
    path.parent.mkdir(parents=True, exist_ok=True)
    client_states = {}
    for client in clients:
        task_id = int(client.task_id)
        if task_id in client_states:
            continue
        client_states[task_id] = {
            "client_id": int(client.client_id),
            "robot_id": int(getattr(client, "robot_id", client.client_id)),
            "task_id": task_id,
            "task_name": client.task_name,
            "task_type": client.task_type,
            "state_dict": client.client_model.state_dict(),
        }

    torch.save(
        {
            "server_state_dict": server.server_model.state_dict(),
            "client_states": client_states,
            "tasks": tasks,
            "model_config": {
                "embed_dim": embed_dim,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": config.DROPOUT,
            },
        },
        path,
    )


def load_configured_multitask_data():
    return load_robomimic_data(
        data_dir=config.ROBOMIMIC_DATA_DIR,
        num_clients=config.NUM_ROBOTS,
        task_files=config.ROBOMIMIC_TASK_FILES,
        obs_keys=config.ROBOMIMIC_OBS_KEYS,
        test_ratio=config.ROBOMIMIC_TEST_RATIO,
        max_demos_per_task=config.ROBOMIMIC_MAX_DEMOS_PER_TASK,
        seed=config.SEED,
        success_threshold=config.ROBOMIMIC_SUCCESS_THRESHOLD,
    )


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
    clients, client_params = build_fedbone_clients(
        client_datasets,
        device,
        EMBED_DIM,
        HIDDEN_SIZE,
    )
    print(f"  Client model parameters (each): {client_params:,}")
    print(f"  Created {len(clients)} robot-task clients with heterogeneous tasks")
    
    print("\n[3/4] Initializing FedBone server...")
    server = FedBoneServer(
        server_model=server_model,
        device=device,
        num_clients=len(clients),
        embed_dim=EMBED_DIM
    )
    
    test_loaders_by_task = build_test_loaders(test_datasets)
    first_task_id = sorted(test_datasets.keys())[0]
    test_loader = test_loaders_by_task[first_task_id]
    
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
        use_gp_aggregation=True,
        test_loaders_by_task=test_loaders_by_task,
        tasks_info=tasks
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
                "num_classes": task['num_classes'],
                "success_mapping": task.get('success_mapping'),
                "success_threshold": task.get('success_threshold')
            }
            for i, task in enumerate(tasks)
        ],
        "metrics": history_gp
    }
    
    results_path = config.RESULTS_DIR / "fedbone_multitask_results.json"
    save_results(results, results_path)
    print(f"\nOK Results saved to: {results_path}")

    checkpoint_path = config.RESULTS_DIR / "models" / "fedbone_multitask_final.pth"
    save_fedbone_checkpoint(
        server,
        clients,
        tasks,
        checkpoint_path,
        EMBED_DIM,
        HIDDEN_SIZE,
        NUM_LAYERS,
    )
    print(f"OK Checkpoint saved to: {checkpoint_path}")
    
    if len(history_gp['rounds']) > 0:
        print("\nFinal Results (with GP Aggregation):")
        if history_gp['accuracy'][-1] is not None:
            print(f"  Macro Classification Accuracy: {history_gp['accuracy'][-1]:.4f}")
            print(f"  Macro Classification F1-Score: {history_gp['f1'][-1]:.4f}")
        if history_gp['mae'][-1] is not None:
            print(f"  Macro MAE: {history_gp['mae'][-1]:.4f}")
            print(f"  Macro RMSE: {history_gp['rmse'][-1]:.4f}")
            print(f"  Macro R2: {history_gp['r2'][-1]:.4f}")
        print(f"  Macro Task Success Rate: {history_gp['task_success_rate'][-1]:.4f}")
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
        
        clients, _ = build_fedbone_clients(
            client_datasets,
            device,
            EMBED_DIM,
            HIDDEN_SIZE,
        )
        
        server = FedBoneServer(server_model, device, len(clients), EMBED_DIM)
        
        test_loaders_by_task = build_test_loaders(test_datasets)
        first_task_id = sorted(test_datasets.keys())[0]
        test_loader = test_loaders_by_task[first_task_id]
        
        history = run_fedbone(
            clients, server, test_loader, device,
            config.NUM_ROUNDS, min(config.CLIENTS_PER_ROUND, len(clients)),
            use_gp_aggregation=use_gp,
            test_loaders_by_task=test_loaders_by_task,
            tasks_info=tasks
        )
        
        results_comparison[exp_name] = history

        checkpoint_path = config.RESULTS_DIR / "models" / f"fedbone_{exp_name}_final.pth"
        save_fedbone_checkpoint(
            server,
            clients,
            tasks,
            checkpoint_path,
            EMBED_DIM,
            HIDDEN_SIZE,
            config.NUM_LAYERS,
        )
        print(f"OK {exp_name} checkpoint saved to: {checkpoint_path}")
    
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
    if results_comparison['baseline']['accuracy'][-1] is not None:
        print(f"  Final Macro Accuracy: {results_comparison['baseline']['accuracy'][-1]:.4f}")
        print(f"  Final Macro F1: {results_comparison['baseline']['f1'][-1]:.4f}")
    if results_comparison['baseline']['mae'][-1] is not None:
        print(f"  Final Macro MAE: {results_comparison['baseline']['mae'][-1]:.4f}")
        print(f"  Final Macro RMSE: {results_comparison['baseline']['rmse'][-1]:.4f}")
    print(f"  Final Macro TSR: {results_comparison['baseline']['task_success_rate'][-1]:.4f}")
    
    print("\nGP Aggregation:")
    if results_comparison['gp_aggregation']['accuracy'][-1] is not None:
        print(f"  Final Macro Accuracy: {results_comparison['gp_aggregation']['accuracy'][-1]:.4f}")
        print(f"  Final Macro F1: {results_comparison['gp_aggregation']['f1'][-1]:.4f}")
    if results_comparison['gp_aggregation']['mae'][-1] is not None:
        print(f"  Final Macro MAE: {results_comparison['gp_aggregation']['mae'][-1]:.4f}")
        print(f"  Final Macro RMSE: {results_comparison['gp_aggregation']['rmse'][-1]:.4f}")
    print(f"  Final Macro TSR: {results_comparison['gp_aggregation']['task_success_rate'][-1]:.4f}")
    print(f"  Avg Conflict Score: {sum(results_comparison['gp_aggregation']['conflict_scores'])/len(results_comparison['gp_aggregation']['conflict_scores']):.4f}")
    
    if results_comparison['gp_aggregation']['accuracy'][-1] is not None:
        improvement = (results_comparison['gp_aggregation']['accuracy'][-1] -
                       results_comparison['baseline']['accuracy'][-1]) * 100
        print(f"\nOK Accuracy Improvement: {improvement:+.2f}%")
    elif results_comparison['gp_aggregation']['rmse'][-1] is not None:
        improvement = (results_comparison['baseline']['rmse'][-1] -
                       results_comparison['gp_aggregation']['rmse'][-1])
        print(f"\nOK RMSE Reduction: {improvement:+.4f}")
    
    return comparison_results


def main():
    print("="*80)
    print("FEDBONE MULTI-TASK EXPERIMENTS")
    print("="*80)
    
    set_seed(config.SEED)
    device = get_device(config.DEVICE)
    print(f"Using device: {device}")
    
    # Load multi-task data
    print("\nLoading ROBOMIMIC multi-task dataset...")
    client_datasets, test_datasets, tasks = load_configured_multitask_data()
    
    # Run main experiment
    results = run_fedbone_experiment(client_datasets, test_datasets, tasks, device)
    
    if config.RUN_FEDBONE_ABLATIONS:
        compare_gp_vs_baseline(client_datasets, test_datasets, tasks, device)
    
    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETED!")
    print("="*80)
    print(f"\nResults saved in: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
