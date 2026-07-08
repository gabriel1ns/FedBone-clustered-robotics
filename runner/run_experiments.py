import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from data.datasets import get_dataloader
from data.robomimic_loader import load_robomimic_data
from federated.robomimic_baselines import (
    RoboMimicRegressionClient,
    run_clustered_regression,
    run_fedavg_regression,
)
from models.model import create_model
from utils.utils import get_device, save_results, set_seed


def load_data():
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


def task_client_datasets(client_datasets, task_id):
    datasets = []
    for robot_tasks in client_datasets:
        match = next(
            (entry["dataset"] for entry in robot_tasks if entry["task_id"] == task_id),
            None,
        )
        if match is not None:
            datasets.append(match)
    return datasets


def build_model(task_dataset, task_info):
    return create_model(
        num_features=int(task_dataset.sequences.shape[-1]),
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_classes=int(task_info["num_classes"]),
        dropout=config.DROPOUT,
    )


def build_clients(datasets, model, device):
    return [
        RoboMimicRegressionClient(
            client_id=index,
            model=copy.deepcopy(model),
            dataset=dataset,
            device=device,
            batch_size=config.BATCH_SIZE,
            learning_rate=config.LEARNING_RATE,
            local_epochs=config.LOCAL_EPOCHS,
        )
        for index, dataset in enumerate(datasets)
    ]


def macro_history(per_task):
    metric_names = ("loss", "mae", "rmse", "r2", "task_success_rate")
    rounds = next(iter(per_task.values()))["rounds"] if per_task else []
    history = {"rounds": rounds, "per_task": per_task}
    for metric in metric_names:
        history[metric] = [
            float(np.mean([task_history[metric][index] for task_history in per_task.values()]))
            for index in range(len(rounds))
        ]
    history["communication_bytes_per_round"] = [
        int(sum(
            task_history["communication_bytes_per_round"][index]
            for task_history in per_task.values()
        ))
        for index in range(len(rounds))
    ]
    history["communication_mb_per_round"] = [
        value / (1024 ** 2) for value in history["communication_bytes_per_round"]
    ]
    return history


def save_baseline_checkpoint(path, algorithm, task_models, tasks):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "algorithm": algorithm,
        "tasks": tasks,
        "task_models": {
            task_name: {
                "model_config": model_config,
                "state_dict": model.state_dict(),
            }
            for task_name, (model, model_config) in task_models.items()
        },
    }, path)


def experiment_config():
    return {
        "dataset": "robomimic",
        "num_rounds": config.NUM_ROUNDS,
        "num_robots": config.NUM_ROBOTS,
        "clients_per_round": config.CLIENTS_PER_ROUND,
        "local_epochs": config.LOCAL_EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
    }


def run_fedavg(client_datasets, test_datasets, tasks, device):
    print("\n" + "=" * 80)
    print("FEDAVG ON ROBOMIMIC")
    print("=" * 80)
    histories = {}
    task_models = {}

    for task_info in tasks:
        task_id = task_info["task_id"]
        datasets = task_client_datasets(client_datasets, task_id)
        model = build_model(datasets[0], task_info).to(device)
        clients = build_clients(datasets, model, device)
        test_loader = get_dataloader(
            test_datasets[task_id], config.BATCH_SIZE, shuffle=False
        )
        print(f"\nTask {task_info['name']} ({len(clients)} robot clients)")
        histories[task_info["name"]] = run_fedavg_regression(
            clients=clients,
            global_model=model,
            test_loader=test_loader,
            device=device,
            task_info=task_info,
            num_rounds=config.NUM_ROUNDS,
            clients_per_round=config.CLIENTS_PER_ROUND,
            seed=config.SEED + task_id,
        )
        task_models[task_info["name"]] = (
            model,
            {
                "num_features": int(datasets[0].sequences.shape[-1]),
                "hidden_size": config.HIDDEN_SIZE,
                "num_layers": config.NUM_LAYERS,
                "output_dim": int(task_info["num_classes"]),
                "dropout": config.DROPOUT,
            },
        )

    results = {
        "experiment": "robomimic_fedavg",
        "config": experiment_config(),
        "tasks": tasks,
        "metrics": macro_history(histories),
    }
    save_results(results, config.RESULTS_DIR / "baseline_fedavg_results.json")
    save_baseline_checkpoint(
        config.RESULTS_DIR / "models" / "robomimic_fedavg_final.pth",
        "fedavg",
        task_models,
        tasks,
    )
    return results


def run_clustered(client_datasets, test_datasets, tasks, device):
    print("\n" + "=" * 80)
    print("CLUSTERED FL ON ROBOMIMIC")
    print("=" * 80)
    histories = {}
    checkpoint_models = {}

    for task_info in tasks:
        task_id = task_info["task_id"]
        datasets = task_client_datasets(client_datasets, task_id)
        model = build_model(datasets[0], task_info).to(device)
        clients = build_clients(datasets, model, device)
        test_loader = get_dataloader(
            test_datasets[task_id], config.BATCH_SIZE, shuffle=False
        )
        print(f"\nTask {task_info['name']} ({len(clients)} robot clients)")
        history, cluster_models = run_clustered_regression(
            clients=clients,
            initial_model=model,
            test_loader=test_loader,
            device=device,
            task_info=task_info,
            num_rounds=config.NUM_ROUNDS,
            num_clusters=config.NUM_CLUSTERS,
            clustering_method=config.CLUSTERING_METHOD,
            reclustering_interval=config.RECLUSTERING_INTERVAL,
            seed=config.SEED + task_id,
        )
        histories[task_info["name"]] = history
        checkpoint_models[task_info["name"]] = {
            "model_config": {
                "num_features": int(datasets[0].sequences.shape[-1]),
                "hidden_size": config.HIDDEN_SIZE,
                "num_layers": config.NUM_LAYERS,
                "output_dim": int(task_info["num_classes"]),
                "dropout": config.DROPOUT,
            },
            "cluster_state_dicts": {
                str(cluster_id): cluster_model.state_dict()
                for cluster_id, cluster_model in cluster_models.items()
            },
        }

    results = {
        "experiment": "robomimic_clustered_fl",
        "config": {
            **experiment_config(),
            "num_clusters": config.NUM_CLUSTERS,
            "clustering_method": config.CLUSTERING_METHOD,
            "reclustering_interval": config.RECLUSTERING_INTERVAL,
        },
        "tasks": tasks,
        "metrics": macro_history(histories),
    }
    save_results(results, config.RESULTS_DIR / "clustered_fl_results.json")
    checkpoint_path = config.RESULTS_DIR / "models" / "robomimic_clustered_final.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "algorithm": "clustered_fl",
        "tasks": tasks,
        "task_models": checkpoint_models,
    }, checkpoint_path)
    return results


def main():
    print("=" * 80)
    print("ROBOMIMIC FEDERATED BASELINES")
    print("=" * 80)
    set_seed(config.SEED)
    device = get_device(config.DEVICE)
    print(f"Using device: {device}")
    client_datasets, test_datasets, tasks = load_data()
    print(f"Loaded {len(tasks)} tasks: {', '.join(task['name'] for task in tasks)}")

    fedavg_results = run_fedavg(client_datasets, test_datasets, tasks, device)
    clustered_results = run_clustered(client_datasets, test_datasets, tasks, device)
    print("\nExperiments completed.")
    print(
        f"FedAvg final macro RMSE: {fedavg_results['metrics']['rmse'][-1]:.4f}"
    )
    print(
        f"Clustered FL final macro RMSE: {clustered_results['metrics']['rmse'][-1]:.4f}"
    )


if __name__ == "__main__":
    main()
