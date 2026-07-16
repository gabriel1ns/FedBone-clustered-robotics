from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np

from data.datasets import RoboMimicTaskDataset


IMAGE_LIKE_SUFFIXES = ("image", "rgb", "depth", "segmentation")


def _discover_hdf5_files(data_dir: Path, task_files: Sequence[str]) -> List[Path]:
    if task_files:
        files = [Path(path) for path in task_files]
        return [path if path.is_absolute() else data_dir / path for path in files]

    files = [
        path for pattern in ("*.hdf5", "*.h5")
        for path in data_dir.rglob(pattern)
        if ".cache" not in path.parts
    ]
    files = sorted(files)
    if not files:
        raise FileNotFoundError(
            f"No RoboMimic HDF5 files found under {data_dir}. "
            "Place low_dim datasets there or set ROBOMIMIC_TASK_FILES."
        )
    return files


def _task_name_from_path(path: Path) -> str:
    # Common RoboMimic paths look like task/demo_type/low_dim.hdf5.
    if path.parent.name in {"ph", "mh", "mg"} and path.parent.parent.name:
        return f"{path.parent.parent.name}_{path.parent.name}"
    if path.stem.startswith("low_dim") and path.parent.name:
        return path.parent.name
    return path.stem


def _demo_keys(handle, max_demos: int) -> List[str]:
    keys = sorted(handle["data"].keys(), key=lambda value: int(value.split("_")[-1]))
    if max_demos > 0:
        keys = keys[:max_demos]
    return keys


def _is_low_dim_obs(name: str, dataset) -> bool:
    lowered = name.lower()
    if any(suffix in lowered for suffix in IMAGE_LIKE_SUFFIXES):
        return False
    if not np.issubdtype(dataset.dtype, np.number):
        return False
    return dataset.ndim <= 2


def _infer_obs_keys(handle, demo_key: str, requested_keys: Sequence[str]) -> List[str]:
    obs_group = handle["data"][demo_key]["obs"]
    if requested_keys:
        missing = [key for key in requested_keys if key not in obs_group]
        if missing:
            raise KeyError(f"Missing RoboMimic obs keys in {demo_key}: {missing}")
        return list(requested_keys)

    keys = [
        key for key, value in obs_group.items()
        if _is_low_dim_obs(key, value)
    ]
    if not keys:
        available = ", ".join(obs_group.keys())
        raise ValueError(
            "Could not infer low-dimensional RoboMimic obs keys. "
            f"Available keys: {available}. Set ROBOMIMIC_OBS_KEYS explicitly."
        )
    return sorted(keys)


def _flatten_obs(obs_group, obs_keys: Sequence[str]) -> np.ndarray:
    parts = []
    for key in obs_keys:
        values = np.asarray(obs_group[key], dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        parts.append(values.reshape(values.shape[0], -1))
    return np.concatenate(parts, axis=-1)


def _load_demo(handle, demo_key: str, obs_keys: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    demo = handle["data"][demo_key]
    observations = _flatten_obs(demo["obs"], obs_keys)
    actions = np.asarray(demo["actions"], dtype=np.float32)
    length = min(len(observations), len(actions))
    return observations[:length], actions[:length]


def _window_demo(
    observations: np.ndarray,
    actions: np.ndarray,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1.")
    if len(observations) == 0:
        return (
            np.zeros((0, sequence_length, observations.shape[-1]), dtype=np.float32),
            np.zeros((0, actions.shape[-1]), dtype=np.float32),
        )

    windows = []
    targets = []
    for end_index in range(len(observations)):
        start_index = end_index - sequence_length + 1
        if start_index < 0:
            padding = np.repeat(
                observations[0:1],
                repeats=-start_index,
                axis=0,
            )
            window = np.concatenate([padding, observations[:end_index + 1]], axis=0)
        else:
            window = observations[start_index:end_index + 1]
        windows.append(window)
        targets.append(actions[end_index])
    return np.asarray(windows, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def _stack_demos(
    items: Sequence[Tuple[np.ndarray, np.ndarray]],
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    windowed = [
        _window_demo(observations, actions, sequence_length)
        for observations, actions in items
    ]
    windowed = [
        (observations, actions)
        for observations, actions in windowed
        if len(observations) > 0
    ]
    if not windowed:
        raise ValueError(
            "No demonstrations are long enough for "
            f"sequence_length={sequence_length}."
        )
    observations = np.concatenate([item[0] for item in windowed], axis=0)
    actions = np.concatenate([item[1] for item in windowed], axis=0)
    return observations, actions


def _split_demo_keys(keys: Sequence[str], test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    shuffled = list(keys)
    rng.shuffle(shuffled)
    split = max(1, int(len(shuffled) * (1.0 - test_ratio)))
    train_keys = shuffled[:split]
    test_keys = shuffled[split:] if split < len(shuffled) else shuffled[-1:]
    return train_keys, test_keys


def _load_task_file(path: Path, obs_keys, test_ratio, max_demos, seed, sequence_length):
    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise KeyError(f"{path} is not a RoboMimic-style HDF5 file: missing /data.")
        keys = _demo_keys(handle, max_demos)
        if not keys:
            raise ValueError(f"{path} has no demonstrations under /data.")

        selected_obs_keys = _infer_obs_keys(handle, keys[0], obs_keys)
        train_keys, test_keys = _split_demo_keys(keys, test_ratio, seed)
        train_items = [_load_demo(handle, key, selected_obs_keys) for key in train_keys]
        test_items = [_load_demo(handle, key, selected_obs_keys) for key in test_keys]

    train_x, train_y = _stack_demos(train_items, sequence_length)
    test_x, test_y = _stack_demos(test_items, sequence_length)
    return train_x, train_y, test_x, test_y, selected_obs_keys


def _safe_std(values: np.ndarray, axis=0) -> np.ndarray:
    std = values.std(axis=axis).astype(np.float32)
    return np.where(std < 1e-6, 1.0, std)


def _normalize_task_arrays(train_x, train_y, test_x, test_y):
    obs_mean = train_x.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    obs_std = _safe_std(train_x, axis=(0, 1))[None, None, :]
    action_mean = train_y.mean(axis=0, keepdims=True).astype(np.float32)
    action_std = _safe_std(train_y, axis=0)[None, :]

    train_x_norm = (train_x - obs_mean) / obs_std
    test_x_norm = (test_x - obs_mean) / obs_std
    train_y_norm = (train_y - action_mean) / action_std
    test_y_norm = (test_y - action_mean) / action_std

    stats = {
        "obs_mean": obs_mean.reshape(-1).tolist(),
        "obs_std": obs_std.reshape(-1).tolist(),
        "action_mean": action_mean.reshape(-1).tolist(),
        "action_std": action_std.reshape(-1).tolist(),
    }
    return train_x_norm, train_y_norm, test_x_norm, test_y_norm, stats


def load_robomimic_data(
    data_dir,
    num_clients,
    task_files=None,
    obs_keys=None,
    test_ratio=0.2,
    max_demos_per_task=0,
    seed=42,
    success_threshold=0.05,
    sequence_length=1,
):
    """
    Load RoboMimic low-dimensional HDF5 datasets for FedBone.

    Each HDF5 file is treated as one manipulation task. The loader reads
    /data/demo_*/obs/<low_dim_keys> as inputs and /data/demo_*/actions as
    action-regression targets.
    """
    data_dir = Path(data_dir)
    files = _discover_hdf5_files(data_dir, task_files or [])

    client_datasets = [[] for _ in range(num_clients)]
    test_datasets: Dict[int, MultiTaskDataset] = {}
    tasks = []

    for task_id, path in enumerate(files):
        task_name = _task_name_from_path(path)
        train_x, train_y, test_x, test_y, selected_obs_keys = _load_task_file(
            path,
            obs_keys or [],
            test_ratio,
            max_demos_per_task,
            seed + task_id,
            sequence_length,
        )
        train_x, train_y, test_x, test_y, normalization_stats = _normalize_task_arrays(
            train_x,
            train_y,
            test_x,
            test_y,
        )

        action_dim = 1 if train_y.ndim == 1 else train_y.shape[-1]
        tasks.append({
            "task_id": task_id,
            "name": task_name,
            "description": f"RoboMimic imitation task from {path.name}",
            "type": "regression",
            "num_classes": int(action_dim),
            "success_mapping": None,
            "success_threshold": float(success_threshold),
            "source_file": str(path),
            "obs_keys": selected_obs_keys,
            "sequence_length": int(sequence_length),
            "normalization": normalization_stats,
        })

        test_datasets[task_id] = RoboMimicTaskDataset(
            test_x,
            test_y,
            task_id=task_id,
        )

        shards = np.array_split(np.arange(len(train_x)), num_clients)
        for client_id, shard in enumerate(shards):
            if len(shard) == 0:
                continue
            dataset = RoboMimicTaskDataset(
                train_x[shard],
                train_y[shard],
                task_id=task_id,
            )
            client_datasets[client_id].append({
                "dataset": dataset,
                "task_id": task_id,
                "task_type": "regression",
                "task_name": task_name,
                "num_classes": int(action_dim),
                "robot_id": client_id,
                "environment_id": path.parent.name,
            })

    return client_datasets, test_datasets, tasks
