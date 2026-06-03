import time
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch


def task_success_rate(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """
    Classification proxy for robotic task success.

    For HAR this is equivalent to accuracy. For manipulation datasets, success
    should be passed as binary success/failure labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def positioning_error_mm(y_true_positions, y_pred_positions) -> Dict[str, float]:
    """
    Euclidean position error in millimeters for grasping/pick-and-place outputs.
    Inputs can be shaped as (n, 2), (n, 3), or any final coordinate dimension.
    """
    y_true = np.asarray(y_true_positions, dtype=np.float64)
    y_pred = np.asarray(y_pred_positions, dtype=np.float64)
    if y_true.size == 0 or y_pred.size == 0:
        return {"mean_mm": 0.0, "median_mm": 0.0, "p95_mm": 0.0}

    errors = np.linalg.norm(y_true - y_pred, axis=-1)
    return {
        "mean_mm": float(np.mean(errors)),
        "median_mm": float(np.median(errors)),
        "p95_mm": float(np.percentile(errors, 95)),
    }


def rounds_to_convergence(
    values: Sequence[float],
    *,
    mode: str = "max",
    min_delta: float = 1e-3,
    patience: int = 5,
    target: Optional[float] = None,
) -> Optional[int]:
    """
    Return the first 1-indexed round that reaches a target or plateaus.

    mode='max' is used for metrics like accuracy/TSR. mode='min' is used for
    loss, RMSE, latency, or positioning error.
    """
    if not values:
        return None

    if target is not None:
        for idx, value in enumerate(values, start=1):
            if (mode == "max" and value >= target) or (mode == "min" and value <= target):
                return idx

    best = values[0]
    stale_rounds = 0
    for idx, value in enumerate(values[1:], start=2):
        improved = (value - best) > min_delta if mode == "max" else (best - value) > min_delta
        if improved:
            best = value
            stale_rounds = 0
        else:
            stale_rounds += 1
            if stale_rounds >= patience:
                return idx

    return None


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def state_dict_nbytes(state_dict: Dict[str, torch.Tensor]) -> int:
    total = 0
    for value in state_dict.values():
        if torch.is_tensor(value):
            total += tensor_nbytes(value)
        else:
            total += np.asarray(value).nbytes
    return int(total)


class SplitCommunicationMeter:
    """
    Tracks simulated split-learning communication volume.

    Counts the four tensors exchanged by FedBone per mini-batch:
    embeddings up, general features down, feature gradients up, and embedding
    gradients down.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_bytes = 0
        self.by_direction = defaultdict(int)
        self.by_payload = defaultdict(int)

    def record(self, payload: str, direction: str, tensor: torch.Tensor):
        nbytes = tensor_nbytes(tensor.detach())
        self.total_bytes += nbytes
        self.by_direction[direction] += nbytes
        self.by_payload[payload] += nbytes

    def record_split_batch(self, embeddings: torch.Tensor, general_features: torch.Tensor):
        self.record("embeddings", "client_to_server", embeddings)
        self.record("general_features", "server_to_client", general_features)
        self.record("feature_gradients", "client_to_server", general_features)
        self.record("embedding_gradients", "server_to_client", embeddings)

    def summary(self) -> Dict[str, object]:
        return {
            "total_bytes": int(self.total_bytes),
            "total_mb": self.total_bytes / (1024**2),
            "by_direction": {key: int(value) for key, value in self.by_direction.items()},
            "by_payload": {key: int(value) for key, value in self.by_payload.items()},
        }


def measure_inference_latency(
    dataloader: Iterable,
    device: torch.device,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    warmup_batches: int = 2,
    max_batches: int = 30,
) -> Dict[str, float]:
    latencies_ms: List[float] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch[0].to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            forward_fn(inputs)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed_ms = (time.perf_counter() - start) * 1000
            if batch_idx >= warmup_batches:
                latencies_ms.append(elapsed_ms)

            if batch_idx + 1 >= warmup_batches + max_batches:
                break

    if not latencies_ms:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}

    values = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def per_group_accuracy(records: Sequence[Dict[str, object]], group_key: str) -> Dict[str, float]:
    grouped = defaultdict(lambda: {"correct": 0, "total": 0})
    for record in records:
        group = str(record[group_key])
        grouped[group]["correct"] += int(record["y_true"] == record["y_pred"])
        grouped[group]["total"] += 1

    return {
        group: values["correct"] / values["total"] if values["total"] else 0.0
        for group, values in grouped.items()
    }


def cross_group_accuracy(records: Sequence[Dict[str, object]], train_group_key: str, eval_group_key: str) -> float:
    cross_records = [
        record for record in records
        if record.get(train_group_key) is not None
        and record.get(eval_group_key) is not None
        and record[train_group_key] != record[eval_group_key]
    ]
    if not cross_records:
        return 0.0

    correct = sum(int(record["y_true"] == record["y_pred"]) for record in cross_records)
    return correct / len(cross_records)
