import argparse
import json
import sys
from collections import deque
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from models.fedbone_model import create_fedbone_client, create_fedbone_server
from federated.fedbone_fl import policy_action_mean
from models.model import create_model
from utils.utils import get_device, save_results, set_seed


ROBOSUITE_TASKS = {
    "lift": ("Lift", "Panda"),
    "can": ("PickPlaceCan", "Panda"),
    "square": ("NutAssemblySquare", "Panda"),
    "tool_hang": ("ToolHang", "Panda"),
    "transport": ("TwoArmTransport", ["Panda", "Panda"]),
}
DEMO_VARIANTS = {"ph", "mh", "mg"}
DEFAULT_TASK_HORIZONS = {
    "lift": 400,
    "can": 400,
    "square": 400,
    "tool_hang": 800,
    "transport": 800,
}

OBS_KEY_ALIASES = {
    "object": "object-state",
}


def flatten_obs_value(value):
    array = np.asarray(value, dtype=np.float32)
    return array.reshape(-1)


def build_low_dim_observation(obs, obs_keys):
    resolved_keys = [key if key in obs else OBS_KEY_ALIASES.get(key, key) for key in obs_keys]
    missing = [key for key in resolved_keys if key not in obs]
    if missing:
        available = ", ".join(sorted(obs.keys()))
        raise KeyError(f"Missing obs keys for online evaluation: {missing}. Available: {available}")
    return np.concatenate([flatten_obs_value(obs[key]) for key in resolved_keys], axis=0)


def load_policy(checkpoint_path, task_name, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    tasks = {task["name"]: task for task in checkpoint["tasks"]}
    if task_name not in tasks:
        raise KeyError(f"Task {task_name!r} not found in checkpoint. Available: {sorted(tasks)}")

    task = dict(tasks[task_name])
    algorithm = checkpoint.get("algorithm", "fedbone")

    if algorithm == "fedavg":
        task_state = checkpoint["task_models"][task_name]
        model_config = task_state["model_config"]
        model = create_model(
            num_features=int(model_config["num_features"]),
            hidden_size=int(model_config["hidden_size"]),
            num_layers=int(model_config["num_layers"]),
            num_classes=int(model_config["output_dim"]),
            dropout=float(model_config["dropout"]),
        ).to(device)
        model.load_state_dict(task_state["state_dict"])
        model.eval()
        task.setdefault("sequence_length", int(model_config.get("sequence_length", 1)))
        obs_keys = task.get("obs_keys")
        if not obs_keys:
            raise ValueError(f"Checkpoint task {task_name!r} does not include obs_keys.")
        return model, None, obs_keys, task, algorithm

    controlled_algorithms = {
        "shared_backbone_fedavg",
        "shared_backbone_clustered",
        "fedbone_simple",
        "fedbone_gp",
    }
    if algorithm != "fedbone" and algorithm not in controlled_algorithms:
        raise ValueError(
            f"Online evaluation does not support checkpoint algorithm {algorithm!r}. "
            "Supported algorithms: fedbone, fedavg, and controlled shared-backbone methods."
        )

    task_id = int(task["task_id"])
    client_state = checkpoint["client_states"][task_id]
    model_config = checkpoint["model_config"]
    task.setdefault("sequence_length", int(model_config.get("sequence_length", 1)))
    task.setdefault("policy_type", model_config.get("policy_type", "deterministic"))
    obs_keys = task.get("obs_keys")
    if not obs_keys:
        raise ValueError(f"Checkpoint task {task_name!r} does not include obs_keys.")

    # Infer feature size from the client embedding layer, which is the exact
    # shape used during training.
    embedding_weight = client_state["state_dict"]["embedding.projection.weight"]
    num_features = int(embedding_weight.shape[1])

    client_model = create_fedbone_client(
        num_features=num_features,
        embed_dim=model_config["embed_dim"],
        hidden_size=model_config["hidden_size"],
        num_classes=int(task["num_classes"]),
        task_type=task["type"],
        policy_type=task.get("policy_type", "deterministic"),
    ).to(device)
    server_model = create_fedbone_server(
        embed_dim=model_config["embed_dim"],
        hidden_size=model_config["hidden_size"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
    ).to(device)

    client_model.load_state_dict(client_state["state_dict"])
    server_model.load_state_dict(checkpoint["server_state_dict"])
    client_model.eval()
    server_model.eval()
    return client_model, server_model, obs_keys, task, algorithm


def normalize_observation(obs_vector, task):
    normalization = task.get("normalization")
    if not normalization:
        return obs_vector
    mean = np.asarray(normalization["obs_mean"], dtype=np.float32)
    std = np.asarray(normalization["obs_std"], dtype=np.float32)
    return (obs_vector - mean) / std


def denormalize_action(action, task):
    normalization = task.get("normalization")
    if not normalization:
        return action
    mean = np.asarray(normalization["action_mean"], dtype=np.float32)
    std = np.asarray(normalization["action_std"], dtype=np.float32)
    return action * std + mean


def base_task_name(task_name):
    """Remove the RoboMimic demonstration suffix without breaking tool_hang."""
    parts = task_name.lower().split("_")
    if parts and parts[-1] in DEMO_VARIANTS:
        parts.pop()
    return "_".join(parts)


def resolve_task_horizon(task_name, horizon_override=None):
    if horizon_override is not None:
        return int(horizon_override)
    return DEFAULT_TASK_HORIZONS.get(base_task_name(task_name), 400)


def available_checkpoint_tasks(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return [task["name"] for task in checkpoint["tasks"]]


def predict_action(client_model, server_model, obs_vector, task, device):
    model_input = normalize_observation(obs_vector, task)
    if model_input.ndim == 1:
        model_input = model_input[None, :]
    inputs = torch.tensor(model_input, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        if server_model is None:
            action = client_model(inputs)
        else:
            embeddings = client_model(inputs, general_features=None)
            general_features = server_model(embeddings)
            action = client_model(None, general_features=general_features)
        action = policy_action_mean(action, task.get("policy_type", "deterministic"))
    action = action.squeeze(0).cpu().numpy()
    return denormalize_action(action, task)


def policy_sequence_length(task):
    return int(task.get("sequence_length", 1))


def load_robomimic_env_args(task):
    source_file = task.get("source_file")
    if not source_file:
        return None
    path = Path(source_file)
    if not path.exists():
        return None
    with h5py.File(path, "r") as handle:
        env_args = handle["data"].attrs.get("env_args")
    return json.loads(env_args) if env_args else None


def sorted_demo_keys(handle):
    def demo_index(key):
        try:
            return int(key.split("_")[-1])
        except ValueError:
            return 0

    return sorted(handle["data"].keys(), key=demo_index)


def load_demo_initial_state(task, episode):
    source_file = task.get("source_file")
    if not source_file:
        return None
    path = Path(source_file)
    if not path.exists():
        return None

    with h5py.File(path, "r") as handle:
        demos = sorted_demo_keys(handle)
        if not demos:
            return None
        demo_key = demos[episode % len(demos)]
        demo = handle["data"][demo_key]
        if "states" not in demo:
            return None
        return {
            "demo_key": demo_key,
            "model_file": demo.attrs.get("model_file"),
            "state": np.asarray(demo["states"][0], dtype=np.float64),
        }


def reset_to_demo_initial_state(env, initial_state):
    model_file = initial_state.get("model_file")
    state = initial_state["state"]
    env.reset()
    if model_file:
        edit_model_xml = getattr(env, "edit_model_xml", None)
        xml = edit_model_xml(model_file) if callable(edit_model_xml) else model_file
        env.reset_from_xml_string(xml)
        env.sim.reset()
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    return env._get_observations(force_update=True)


def reset_episode(env, task, episode, seed, use_dataset_initial_states):
    set_seed(seed)
    seed_environment(env, seed)
    initial_state = load_demo_initial_state(task, episode) if use_dataset_initial_states else None
    if initial_state is None:
        return env.reset(), None
    return reset_to_demo_initial_state(env, initial_state), initial_state["demo_key"]


def make_env(suite, args, task):
    horizon = resolve_task_horizon(args.task, args.horizon)
    env_args = load_robomimic_env_args(task) if args.use_dataset_env_args else None
    if env_args:
        env_name = env_args["env_name"]
        env_kwargs = dict(env_args.get("env_kwargs", {}))
        env_kwargs["has_renderer"] = args.render
        env_kwargs["has_offscreen_renderer"] = args.record_video is not None
        env_kwargs["use_camera_obs"] = False
        env_kwargs["use_object_obs"] = True
        if args.reward_shaping is not None:
            env_kwargs["reward_shaping"] = args.reward_shaping
        env_kwargs["hard_reset"] = False
        env_kwargs["horizon"] = horizon
        return env_name, env_kwargs.get("robots", args.robots), suite.make(env_name=env_name, **env_kwargs)

    env_name, robots = ROBOSUITE_TASKS.get(base_task_name(args.task), (args.env_name, args.robots))
    if env_name is None:
        raise ValueError(f"No RoboSuite mapping for {args.task!r}. Pass --env-name and --robots.")

    return env_name, robots, suite.make(
        env_name=env_name,
        robots=robots,
        has_renderer=args.render,
        has_offscreen_renderer=args.record_video is not None,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=bool(args.reward_shaping),
        hard_reset=False,
        horizon=horizon,
        control_freq=args.control_freq,
    )


def seed_environment(env, seed):
    seed_fn = getattr(env, "seed", None)
    if callable(seed_fn):
        try:
            seed_fn(seed)
        except TypeError:
            seed_fn(seed=seed)


def check_success(env, info):
    success = bool(info.get("success", False))
    if hasattr(env, "_check_success"):
        success = success or bool(env._check_success())
    return success


def evaluate_task(args):
    import robosuite as suite
    import imageio.v2 as imageio

    set_seed(args.seed)
    device = get_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    horizon = resolve_task_horizon(args.task, args.horizon)

    client_model, server_model, obs_keys, task, algorithm = load_policy(
        checkpoint_path, args.task, device
    )
    env_name, robots, env = make_env(suite, args, task)
    sequence_length = policy_sequence_length(task)

    low, high = env.action_spec
    episode_results = []
    raw_actions = []
    all_actions = []
    all_obs_norms = []
    saturation_masks = []
    video_writer = None
    if args.record_video:
        video_path = Path(args.record_video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=args.video_fps)

    try:
        for episode in range(args.episodes):
            episode_seed = int(args.seed + episode)
            obs, demo_key = reset_episode(
                env,
                task,
                episode,
                episode_seed,
                args.use_dataset_initial_states,
            )
            total_reward = 0.0
            success = False
            steps = 0
            episode_saturation = []
            initial_obs_vector = build_low_dim_observation(obs, obs_keys)
            obs_history = deque(
                [initial_obs_vector.copy() for _ in range(sequence_length)],
                maxlen=sequence_length,
            )

            for step in range(horizon):
                obs_vector = build_low_dim_observation(obs, obs_keys)
                obs_history.append(obs_vector)
                obs_sequence = np.stack(list(obs_history), axis=0)
                raw_action = predict_action(client_model, server_model, obs_sequence, task, device)
                saturated = (raw_action < low) | (raw_action > high)
                action = np.clip(raw_action, low, high)
                raw_actions.append(raw_action)
                all_actions.append(action)
                saturation_masks.append(saturated)
                episode_saturation.append(saturated)
                all_obs_norms.append(float(np.linalg.norm(obs_vector)))
                obs, reward, done, info = env.step(action)
                total_reward += float(reward)
                steps = step + 1

                if video_writer and step % args.video_every == 0:
                    frame = env.sim.render(
                        camera_name=args.video_camera,
                        width=args.video_width,
                        height=args.video_height,
                    )
                    video_writer.append_data(np.flipud(frame))

                if args.render:
                    env.render()

                success = check_success(env, info)
                if done or (success and args.stop_on_success):
                    break

            episode_results.append({
                "episode": episode,
                "seed": episode_seed,
                "demo_key": demo_key,
                "success": success,
                "return": total_reward,
                "steps": steps,
                "saturation_fraction": float(np.mean(episode_saturation)) if episode_saturation else 0.0,
            })
            print(f"Episode {episode + 1}/{args.episodes}: success={success} return={total_reward:.3f} steps={steps}")
    finally:
        if video_writer:
            video_writer.close()
            print(f"Video saved to: {args.record_video}")
        env.close()

    success_rate = float(np.mean([item["success"] for item in episode_results])) if episode_results else 0.0
    mean_return = float(np.mean([item["return"] for item in episode_results])) if episode_results else 0.0
    mean_steps = float(np.mean([item["steps"] for item in episode_results])) if episode_results else 0.0
    actions = np.asarray(all_actions, dtype=np.float32) if all_actions else np.zeros((0, len(low)), dtype=np.float32)
    raw_actions = np.asarray(raw_actions, dtype=np.float32) if raw_actions else np.zeros((0, len(low)), dtype=np.float32)
    saturation_masks = np.asarray(saturation_masks, dtype=bool) if saturation_masks else np.zeros((0, len(low)), dtype=bool)
    action_stats = {
        "mean": actions.mean(axis=0).tolist() if len(actions) else [],
        "std": actions.std(axis=0).tolist() if len(actions) else [],
        "min": actions.min(axis=0).tolist() if len(actions) else [],
        "max": actions.max(axis=0).tolist() if len(actions) else [],
        "raw_mean": raw_actions.mean(axis=0).tolist() if len(raw_actions) else [],
        "raw_std": raw_actions.std(axis=0).tolist() if len(raw_actions) else [],
        "raw_min": raw_actions.min(axis=0).tolist() if len(raw_actions) else [],
        "raw_max": raw_actions.max(axis=0).tolist() if len(raw_actions) else [],
        "saturation_fraction": float(np.mean(saturation_masks)) if len(saturation_masks) else 0.0,
        "clip_fraction": float(np.mean(saturation_masks)) if len(saturation_masks) else 0.0,
    }
    results = {
        "task": args.task,
        "algorithm": algorithm,
        "env_name": env_name,
        "robots": robots,
        "checkpoint": str(checkpoint_path),
        "seed": args.seed,
        "episodes": args.episodes,
        "horizon": horizon,
        "sequence_length": sequence_length,
        "stop_on_success": args.stop_on_success,
        "use_dataset_initial_states": args.use_dataset_initial_states,
        "success_rate": success_rate,
        "mean_return": mean_return,
        "mean_steps": mean_steps,
        "action_stats": action_stats,
        "obs_norm_mean": float(np.mean(all_obs_norms)) if all_obs_norms else 0.0,
        "obs_norm_std": float(np.std(all_obs_norms)) if all_obs_norms else 0.0,
        "record_video": args.record_video,
        "task_info": task,
        "episodes_detail": episode_results,
    }

    return results


def summarize_online_results(checkpoint_path, task_results):
    completed = {
        task_name: result
        for task_name, result in task_results.items()
        if "success_rate" in result
    }
    results = {
        "checkpoint": str(checkpoint_path),
        "tasks": task_results,
    }
    if completed:
        results.update({
            "macro_success_rate": float(np.mean([
                result["success_rate"] for result in completed.values()
            ])),
            "macro_mean_return": float(np.mean([
                result["mean_return"] for result in completed.values()
            ])),
            "macro_mean_steps": float(np.mean([
                result["mean_steps"] for result in completed.values()
            ])),
            "macro_saturation_fraction": float(np.mean([
                result["action_stats"]["saturation_fraction"] for result in completed.values()
            ])),
            "completed_tasks": sorted(completed),
        })
    return results


def evaluate_online(args):
    checkpoint_path = Path(args.checkpoint)
    task_names = (
        available_checkpoint_tasks(checkpoint_path)
        if args.task.lower() == "all"
        else [name.strip() for name in args.task.split(",") if name.strip()]
    )
    if not task_names:
        raise ValueError("No online-evaluation tasks were selected.")

    task_results = {}
    output_path = Path(args.output)
    for task_name in task_names:
        print(f"\nEvaluating RoboMimic task: {task_name}")
        task_args = argparse.Namespace(**vars(args))
        task_args.task = task_name
        task_args.record_video = video_path_for_task(args.record_video, task_name, len(task_names))
        try:
            task_results[task_name] = evaluate_task(task_args)
        except Exception as exc:
            if args.fail_fast:
                raise
            task_results[task_name] = {
                "task": task_name,
                "checkpoint": str(checkpoint_path),
                "seed": args.seed,
                "episodes": args.episodes,
                "horizon": resolve_task_horizon(task_name, args.horizon),
                "error": repr(exc),
            }
            print(f"Task {task_name} failed: {exc!r}")

        if len(task_results) > 1 or len(task_names) > 1:
            partial_results = summarize_online_results(checkpoint_path, task_results)
            save_results(partial_results, output_path)
            print(f"Partial online results saved to: {output_path}")

    if len(task_results) == 1:
        results = next(iter(task_results.values()))
    else:
        results = summarize_online_results(checkpoint_path, task_results)

    save_results(results, output_path)
    print(json.dumps(results if len(task_results) > 1 else {
        key: results[key]
        for key in ["task", "success_rate", "mean_return", "mean_steps", "horizon", "obs_norm_mean", "action_stats"]
        if key in results
    }, indent=2))
    print(f"Online results saved to: {output_path}")
    return results


def video_path_for_task(path, task_name, multiple_tasks):
    if not path or not multiple_tasks:
        return path
    video_path = Path(path)
    return str(video_path.with_name(f"{video_path.stem}_{task_name}{video_path.suffix}"))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained FedBone policy online in RoboSuite.")
    parser.add_argument("--checkpoint", default=str(config.RESULTS_DIR / "models" / "fedbone_multitask_final.pth"))
    parser.add_argument(
        "--task",
        default="all",
        help="Checkpoint task name, comma-separated names, or 'all' (default).",
    )
    parser.add_argument("--env-name", default=None)
    parser.add_argument("--robots", default="Panda")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Override the per-task default horizon. Defaults: lift/can/square=400, tool_hang/transport=800.",
    )
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--reward-shaping",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override dataset reward_shaping. By default, keep the value stored in env_args.",
    )
    parser.add_argument("--use-dataset-env-args", action="store_true", default=True)
    parser.add_argument(
        "--use-dataset-initial-states",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset each episode to the initial simulator state of a dataset demo when available.",
    )
    parser.add_argument("--record-video", default=None)
    parser.add_argument("--video-camera", default="frontview")
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-every", type=int, default=1)
    parser.add_argument("--stop-on-success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", default=str(config.RESULTS_DIR / "online_eval_all.json"))
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_online(parse_args())
