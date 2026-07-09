import argparse
import gc
import hashlib
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from data.datasets import get_dataloader
from data.robomimic_loader import load_robomimic_data
from federated.controlled_comparison import (
    METHODS,
    build_stratified_schedule,
    clone_state_dict,
    initialize_task_states,
    run_controlled_method,
    synchronize_clients_to_task_states,
)
from models.fedbone_model import create_fedbone_server, count_parameters
from runner.run_fedbone import build_fedbone_clients
from utils.utils import get_device, save_results, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a controlled shared-backbone comparison on RoboMimic."
    )
    parser.add_argument("--rounds", type=int, default=config.NUM_ROUNDS)
    parser.add_argument(
        "--clients-per-round", type=int, default=config.CLIENTS_PER_ROUND
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )
    parser.add_argument(
        "--max-demos-per-task",
        type=int,
        default=config.ROBOMIMIC_MAX_DEMOS_PER_TASK,
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument(
        "--output",
        default=str(config.RESULTS_DIR / "controlled_comparison_results.json"),
    )
    return parser.parse_args()


def state_hash(server_state, client_states):
    digest = hashlib.sha256()
    for state in [server_state, *client_states]:
        for key, value in state.items():
            digest.update(key.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def restore_clients(client_datasets, initial_states, device):
    clients, _ = build_fedbone_clients(
        client_datasets,
        device,
        config.EMBED_DIM,
        config.HIDDEN_SIZE,
    )
    for client, state in zip(clients, initial_states):
        client.client_model.load_state_dict(state)
    return clients


def save_controlled_checkpoint(path, method, server, clients, tasks):
    client_states = {}
    for client in clients:
        task_id = int(client.task_id)
        client_states[task_id] = {
            "client_id": int(client.client_id),
            "robot_id": int(getattr(client, "robot_id", client.client_id)),
            "task_id": task_id,
            "task_name": client.task_name,
            "task_type": client.task_type,
            "state_dict": clone_state_dict(client.client_model.state_dict()),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "algorithm": method,
        "server_state_dict": clone_state_dict(server.state_dict()),
        "client_states": client_states,
        "tasks": tasks,
        "model_config": {
            "embed_dim": config.EMBED_DIM,
            "hidden_size": config.HIDDEN_SIZE,
            "num_layers": config.NUM_LAYERS,
            "dropout": config.DROPOUT,
        },
    }, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    if sys.platform == "win32" and device.type == "cuda":
        # Repeatedly cloning / destroying cuDNN LSTMs can abort the Python
        # process during DLL teardown on Windows. The native CUDA LSTM path is
        # stable and preserves GPU execution for this controlled protocol.
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled for stable repeated LSTM trials on Windows.")
    print(f"Using device: {device}")

    client_datasets, test_datasets, tasks = load_robomimic_data(
        data_dir=config.ROBOMIMIC_DATA_DIR,
        num_clients=config.NUM_ROBOTS,
        task_files=config.ROBOMIMIC_TASK_FILES,
        obs_keys=config.ROBOMIMIC_OBS_KEYS,
        test_ratio=config.ROBOMIMIC_TEST_RATIO,
        max_demos_per_task=args.max_demos_per_task,
        seed=args.seed,
        success_threshold=config.ROBOMIMIC_SUCCESS_THRESHOLD,
    )
    test_loaders = {
        task_id: get_dataloader(dataset, config.BATCH_SIZE, shuffle=False)
        for task_id, dataset in test_datasets.items()
    }

    set_seed(args.seed)
    initial_clients, client_params = build_fedbone_clients(
        client_datasets,
        device,
        config.EMBED_DIM,
        config.HIDDEN_SIZE,
    )
    initial_task_states = initialize_task_states(initial_clients)
    synchronize_clients_to_task_states(initial_clients, initial_task_states)
    initial_server = create_fedbone_server(
        config.EMBED_DIM,
        config.HIDDEN_SIZE,
        config.NUM_LAYERS,
        config.DROPOUT,
    ).to(device)
    initial_server_state = clone_state_dict(initial_server.state_dict(), "cpu")
    server_params = count_parameters(initial_server)
    initial_client_states = [
        clone_state_dict(client.client_model.state_dict(), "cpu")
        for client in initial_clients
    ]
    initialization_hash = state_hash(initial_server_state, initial_client_states)
    num_virtual_clients = len(initial_clients)
    schedule = build_stratified_schedule(
        initial_clients,
        args.rounds,
        args.clients_per_round,
        args.seed,
    )
    task_counts = {
        str(task["task_id"]): sum(
            int(next(
                client for client in initial_clients if client.client_id == client_id
            ).task_id) == int(task["task_id"])
            for client_id in schedule[0]
        )
        for task in tasks
    }
    print(f"Virtual clients: {num_virtual_clients}")
    print(f"Clients per round by task: {task_counts}")
    print(f"Initialization SHA-256: {initialization_hash}")
    del initial_clients, initial_server
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    output_path = Path(args.output)
    result = {
        "experiment": "controlled_shared_backbone_comparison",
        "protocol": {
            "methods": args.methods,
            "num_rounds": args.rounds,
            "num_virtual_clients": num_virtual_clients,
            "num_robots": config.NUM_ROBOTS,
            "num_tasks": len(tasks),
            "clients_per_round": args.clients_per_round,
            "selection": "stratified_by_task",
            "clients_per_round_by_task": task_counts,
            "local_epochs": config.LOCAL_EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "seed": args.seed,
            "initialization_hash": initialization_hash,
            "server_params": server_params,
            "client_params": client_params,
            "max_demos_per_task": args.max_demos_per_task,
            "schedule": schedule,
        },
        "tasks": tasks,
        "methods": {},
    }
    save_results(result, output_path)

    for method in args.methods:
        print("\n" + "=" * 80)
        print(f"CONTROLLED METHOD: {method}")
        print("=" * 80)
        set_seed(args.seed)
        clients = restore_clients(client_datasets, initial_client_states, device)
        server = create_fedbone_server(
            config.EMBED_DIM,
            config.HIDDEN_SIZE,
            config.NUM_LAYERS,
            config.DROPOUT,
        ).to(device)
        server.load_state_dict(initial_server_state)

        def save_progress(partial_history):
            result["methods"][method] = partial_history
            save_results(result, output_path)

        history, trained_server, checkpoint_clients = run_controlled_method(
            method=method,
            clients=clients,
            server_model=server,
            schedule=schedule,
            test_loaders=test_loaders,
            tasks=tasks,
            device=device,
            learning_rate=config.LEARNING_RATE,
            num_clusters=config.NUM_CLUSTERS,
            seed=args.seed,
            progress_callback=save_progress,
        )
        result["methods"][method] = history
        save_results(result, output_path)
        checkpoint_path = (
            config.RESULTS_DIR / "models" / f"controlled_{method}_final.pth"
        )
        save_controlled_checkpoint(
            checkpoint_path,
            method,
            trained_server,
            checkpoint_clients,
            tasks,
        )
        print(f"Saved {method} checkpoint to {checkpoint_path}")
        del clients, server, trained_server, checkpoint_clients
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    print(f"\nControlled comparison saved to {output_path}")


if __name__ == "__main__":
    main()
