from typing import Dict, Optional

import numpy as np


def mean_iou(y_true, y_pred, num_classes: int, ignore_index: Optional[int] = None) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ious = []

    for class_id in range(num_classes):
        if ignore_index is not None and class_id == ignore_index:
            continue

        true_mask = y_true == class_id
        pred_mask = y_pred == class_id
        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()
        if union > 0:
            ious.append(intersection / union)

    return float(np.mean(ious)) if ious else 0.0


def rmse(y_true, y_pred, mask=None) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if y_true.size == 0:
        return 0.0

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_angular_error_deg(y_true_normals, y_pred_normals, mask=None, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true_normals, dtype=np.float64)
    y_pred = np.asarray(y_pred_normals, dtype=np.float64)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if y_true.size == 0:
        return 0.0

    true_norm = y_true / np.maximum(np.linalg.norm(y_true, axis=-1, keepdims=True), eps)
    pred_norm = y_pred / np.maximum(np.linalg.norm(y_pred, axis=-1, keepdims=True), eps)
    cosine = np.sum(true_norm * pred_norm, axis=-1)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)).mean())


def f_measure(y_true_binary, y_score, threshold: float = 0.5, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true_binary).astype(bool)
    y_pred = np.asarray(y_score) >= threshold

    tp = np.logical_and(y_true, y_pred).sum()
    fp = np.logical_and(~y_true, y_pred).sum()
    fn = np.logical_and(y_true, ~y_pred).sum()

    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    return float((2 * precision * recall) / max(precision + recall, eps))


def max_f_measure(y_true_binary, y_score, thresholds=None) -> float:
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    return max(f_measure(y_true_binary, y_score, float(threshold)) for threshold in thresholds)


def dense_prediction_metrics(outputs: Dict[str, object], targets: Dict[str, object]) -> Dict[str, float]:
    """
    Convenience wrapper for NYUv2/PASCAL-style multi-task evaluation.
    Expected keys are optional: segmentation, depth, normals, boundary, saliency.
    """
    metrics = {}

    if "segmentation" in outputs and "segmentation" in targets:
        num_classes = int(targets.get("num_segmentation_classes", np.max(targets["segmentation"]) + 1))
        metrics["mIoU"] = mean_iou(targets["segmentation"], outputs["segmentation"], num_classes)

    if "depth" in outputs and "depth" in targets:
        metrics["RMSE"] = rmse(targets["depth"], outputs["depth"], targets.get("depth_mask"))

    if "normals" in outputs and "normals" in targets:
        metrics["mErr"] = mean_angular_error_deg(
            targets["normals"],
            outputs["normals"],
            targets.get("normal_mask"),
        )

    if "boundary" in outputs and "boundary" in targets:
        metrics["odsF"] = max_f_measure(targets["boundary"], outputs["boundary"])

    if "saliency" in outputs and "saliency" in targets:
        metrics["maxF"] = max_f_measure(targets["saliency"], outputs["saliency"])

    return metrics
