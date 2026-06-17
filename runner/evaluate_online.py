import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from models.fedbone_model import create_fedbone_client, create_fedbone_server
from utils.utils import get_device, save_results, set_seed


ROBOSUITE_TASKS = {
    "lift_ph": ("Lift", "Panda"),
    "can_ph": ("PickPlaceCan", "Panda"),
    "square_ph": ("NutAssemblySquare", "Panda"),
    "tool_hang_ph": ("ToolHang", "Panda"),
    "transport_ph": ("TwoArmTransport", "Panda"),
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

    task = tasks[task_name]
    task_id = int(task["task_id"])
    client_state = checkpoint["client_states"][task_id]
    model_config = checkpoint["model_config"]
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
    return client_model, server_model, obs_keys, task


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


def predict_action(client_model, server_model, obs_vector, task, device):
    model_input = normalize_observation(obs_vector, task)
    inputs = torch.tensor(model_input, dtype=torch.float32, device=device).view(1, 1, -1)
    with torch.no_grad():
        embeddings = client_model(inputs, general_features=None)
        general_features = server_model(embeddings)
        action = client_model(None, general_features=general_features)
    action = action.squeeze(0).cpu().numpy()
    return denormalize_action(action, task)


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


def make_env(suite, args, task):
    env_args = load_robomimic_env_args(task) if args.use_dataset_env_args else None
    if env_args:
        env_name = env_args["env_name"]
        env_kwargs = dict(env_args.get("env_kwargs", {}))
        env_kwargs["has_renderer"] = args.render
        env_kwargs["has_offscreen_renderer"] = args.record_video is not None
        env_kwargs["use_camera_obs"] = False
        env_kwargs["use_object_obs"] = True
        env_kwargs["reward_shaping"] = args.reward_shaping
        env_kwargs["hard_reset"] = False
        env_kwargs["horizon"] = args.horizon
        return env_name, env_kwargs.get("robots", args.robots), suite.make(env_name=env_name, **env_kwargs)

    env_name, robots = ROBOSUITE_TASKS.get(args.task, (args.env_name, args.robots))
    if env_name is None:
        raise ValueError(f"No RoboSuite mapping for {args.task!r}. Pass --env-name and --robots.")

    return env_name, robots, suite.make(
        env_name=env_name,
        robots=robots,
        has_renderer=args.render,
        has_offscreen_renderer=args.record_video is not None,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=args.reward_shaping,
        hard_reset=False,
        horizon=args.horizon,
        control_freq=args.control_freq,
    )


def evaluate_online(args):
    import robosuite as suite
    import imageio.v2 as imageio

    set_seed(args.seed)
    device = get_device(args.device)
    checkpoint_path = Path(args.checkpoint)

    client_model, server_model, obs_keys, task = load_policy(checkpoint_path, args.task, device)
    env_name, robots, env = make_env(suite, args, task)

    low, high = env.action_spec
    episode_results = []
    all_actions = []
    all_obs_norms = []
    video_writer = None
    if args.record_video:
        video_path = Path(args.record_video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=args.video_fps)

    try:
        for episode in range(args.episodes):
            obs = env.reset()
            total_reward = 0.0
            success = False
            steps = 0

            for step in range(args.horizon):
                obs_vector = build_low_dim_observation(obs, obs_keys)
                action = predict_action(client_model, server_model, obs_vector, task, device)
                action = np.clip(action, low, high)
                all_actions.append(action)
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

                if hasattr(env, "_check_success") and env._check_success():
                    success = True
                if bool(info.get("success", False)):
                    success = True
                if done:
                    break

            episode_results.append({
                "episode": episode,
                "success": success,
                "return": total_reward,
                "steps": steps,
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
    action_stats = {
        "mean": actions.mean(axis=0).tolist() if len(actions) else [],
        "std": actions.std(axis=0).tolist() if len(actions) else [],
        "min": actions.min(axis=0).tolist() if len(actions) else [],
        "max": actions.max(axis=0).tolist() if len(actions) else [],
        "clip_fraction": float(np.mean((actions <= low) | (actions >= high))) if len(actions) else 0.0,
    }
    results = {
        "task": args.task,
        "env_name": env_name,
        "robots": robots,
        "checkpoint": str(checkpoint_path),
        "seed": args.seed,
        "episodes": args.episodes,
        "horizon": args.horizon,
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

    output_path = Path(args.output)
    save_results(results, output_path)
    print(json.dumps({
        k: results[k]
        for k in ["task", "success_rate", "mean_return", "mean_steps", "obs_norm_mean", "action_stats"]
    }, indent=2))
    print(f"Online results saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained FedBone policy online in RoboSuite.")
    parser.add_argument("--checkpoint", default=str(config.RESULTS_DIR / "models" / "fedbone_multitask_final.pth"))
    parser.add_argument("--task", default="lift_ph")
    parser.add_argument("--env-name", default=None)
    parser.add_argument("--robots", default="Panda")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--reward-shaping", action="store_true", default=True)
    parser.add_argument("--use-dataset-env-args", action="store_true", default=True)
    parser.add_argument("--record-video", default=None)
    parser.add_argument("--video-camera", default="frontview")
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-every", type=int, default=1)
    parser.add_argument("--output", default=str(config.RESULTS_DIR / "online_eval_lift_ph.json"))
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_online(parse_args())
