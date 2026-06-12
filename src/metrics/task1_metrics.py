#!/usr/bin/env python3
"""
Metrics for MVAA Task 1.

Task 1:
    Mitral Valve Segmentation and Landmark Localization in CT scans.

Official challenge-relevant metrics:
    - Dice Similarity Coefficient (DSC)
    - Hausdorff Distance (HD)
    - Mean Radial Error (MRE) for landmarks
    - Inference time

Additional development metrics:
    - IoU / Jaccard
    - Precision
    - Recall / Sensitivity
    - Specificity
    - HD95
    - ASD / ASSD
    - Relative volume error
    - Surface Dice at tolerance
    - Empty prediction / empty target flags

Why this file matters:
    Dice alone is not enough for a small and ambiguous structure like the
    mitral valve. We need overlap, boundary, surface, volume, and failure
    detection metrics together.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import ndimage


ArrayLike = np.ndarray | torch.Tensor


EPS = 1e-8


@dataclass
class SegmentationMetrics:
    """Container for binary segmentation metrics."""

    dice: float
    jaccard: float
    precision: float
    recall: float
    sensitivity: float
    specificity: float

    hd: float
    hd95: float
    asd: float
    assd: float
    surface_dice: float

    pred_voxels: int
    target_voxels: int
    intersection_voxels: int
    false_positive_voxels: int
    false_negative_voxels: int
    true_negative_voxels: int

    relative_volume_error: float
    absolute_volume_error: int

    pred_empty: bool
    target_empty: bool
    both_empty: bool

    spacing: Tuple[float, float, float]
    surface_dice_tolerance_mm: float


def to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert torch tensor or numpy array to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def squeeze_binary_mask(mask: ArrayLike) -> np.ndarray:
    """
    Convert input into binary numpy mask.

    Accepted shapes:
        [D, H, W]
        [1, D, H, W]
        [B, 1, D, H, W] only if B == 1
    """
    arr = to_numpy(mask)

    if arr.ndim == 5:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for single mask, got shape {arr.shape}")
        arr = arr[0]

    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected channel size 1 for binary mask, got shape {arr.shape}")
        arr = arr[0]

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {arr.shape}")

    return arr.astype(bool)


def logits_to_binary_mask(logits: ArrayLike) -> np.ndarray:
    """
    Convert model logits/probabilities to binary foreground mask.

    Accepted shapes:
        [C, D, H, W]
        [B, C, D, H, W] only if B == 1

    For C == 2:
        argmax over channels.

    For C == 1:
        sigmoid threshold at 0.5.
    """
    arr = to_numpy(logits)

    if arr.ndim == 5:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for single logits, got shape {arr.shape}")
        arr = arr[0]

    if arr.ndim != 4:
        raise ValueError(f"Expected logits shape [C, D, H, W], got {arr.shape}")

    channels = arr.shape[0]

    if channels == 1:
        prob = 1.0 / (1.0 + np.exp(-arr[0]))
        return prob >= 0.5

    return np.argmax(arr, axis=0).astype(np.uint8) == 1


def normalize_spacing(spacing: Optional[Sequence[float]]) -> Tuple[float, float, float]:
    """Normalize spacing to a 3-value tuple."""
    if spacing is None:
        return (1.0, 1.0, 1.0)

    if len(spacing) != 3:
        raise ValueError(f"Spacing must contain 3 values, got: {spacing}")

    return (float(spacing[0]), float(spacing[1]), float(spacing[2]))


def confusion_counts(pred: np.ndarray, target: np.ndarray) -> Dict[str, int]:
    """Compute binary confusion counts."""
    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, np.logical_not(target)).sum())
    fn = int(np.logical_and(np.logical_not(pred), target).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(target)).sum())

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def dice_score_from_counts(tp: int, fp: int, fn: int) -> float:
    """Compute Dice score from confusion counts."""
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0
    return float((2 * tp) / (denom + EPS))


def jaccard_score_from_counts(tp: int, fp: int, fn: int) -> float:
    """Compute IoU / Jaccard score from confusion counts."""
    denom = tp + fp + fn
    if denom == 0:
        return 1.0
    return float(tp / (denom + EPS))


def precision_from_counts(tp: int, fp: int) -> float:
    """Compute precision."""
    denom = tp + fp
    if denom == 0:
        return 1.0
    return float(tp / (denom + EPS))


def recall_from_counts(tp: int, fn: int) -> float:
    """Compute recall / sensitivity."""
    denom = tp + fn
    if denom == 0:
        return 1.0
    return float(tp / (denom + EPS))


def specificity_from_counts(tn: int, fp: int) -> float:
    """Compute specificity."""
    denom = tn + fp
    if denom == 0:
        return 1.0
    return float(tn / (denom + EPS))


def relative_volume_error(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute relative volume error.

    Formula:
        (pred_volume - target_volume) / target_volume

    If target is empty:
        - both empty: 0.0
        - target empty but prediction non-empty: inf
    """
    pred_vol = int(pred.sum())
    target_vol = int(target.sum())

    if target_vol == 0:
        return 0.0 if pred_vol == 0 else math.inf

    return float((pred_vol - target_vol) / (target_vol + EPS))


def get_binary_surface(mask: np.ndarray) -> np.ndarray:
    """Extract binary surface voxels from a 3D binary mask."""
    mask = mask.astype(bool)

    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)

    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    surface = np.logical_xor(mask, eroded)

    return surface


def surface_distances(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute bidirectional surface distances.

    Returns:
        pred_to_target_distances
        target_to_pred_distances
    """
    spacing = normalize_spacing(spacing)

    pred_surface = get_binary_surface(pred)
    target_surface = get_binary_surface(target)

    if pred_surface.sum() == 0 or target_surface.sum() == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    # Distance transform of the complement of the target surface gives distance
    # from every voxel to the nearest target surface voxel.
    target_distance_map = ndimage.distance_transform_edt(
        np.logical_not(target_surface),
        sampling=spacing,
    )
    pred_distance_map = ndimage.distance_transform_edt(
        np.logical_not(pred_surface),
        sampling=spacing,
    )

    pred_to_target = target_distance_map[pred_surface]
    target_to_pred = pred_distance_map[target_surface]

    return pred_to_target.astype(np.float64), target_to_pred.astype(np.float64)


def hausdorff_distance(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    percentile: Optional[float] = None,
) -> float:
    """
    Compute symmetric Hausdorff distance.

    Args:
        percentile:
            None for full HD.
            95.0 for HD95.
    """
    pred_empty = pred.sum() == 0
    target_empty = target.sum() == 0

    if pred_empty and target_empty:
        return 0.0

    if pred_empty != target_empty:
        return math.inf

    d1, d2 = surface_distances(pred, target, spacing)
    if d1.size == 0 or d2.size == 0:
        return math.inf

    distances = np.concatenate([d1, d2])

    if percentile is None:
        return float(np.max(distances))

    return float(np.percentile(distances, percentile))


def average_surface_distance(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    symmetric: bool = True,
) -> float:
    """
    Compute ASD or ASSD.

    symmetric=False:
        average pred-to-target surface distance.

    symmetric=True:
        average symmetric surface distance.
    """
    pred_empty = pred.sum() == 0
    target_empty = target.sum() == 0

    if pred_empty and target_empty:
        return 0.0

    if pred_empty != target_empty:
        return math.inf

    d1, d2 = surface_distances(pred, target, spacing)
    if d1.size == 0 or d2.size == 0:
        return math.inf

    if symmetric:
        return float(np.mean(np.concatenate([d1, d2])))

    return float(np.mean(d1))


def surface_dice_at_tolerance(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    tolerance_mm: float = 1.0,
) -> float:
    """
    Compute surface Dice at a distance tolerance.

    A surface point is counted as correct if it lies within tolerance_mm of the
    opposite surface.

    This is useful when boundaries are ambiguous and small millimeter-level
    deviations should not be treated the same as large outliers.
    """
    pred_empty = pred.sum() == 0
    target_empty = target.sum() == 0

    if pred_empty and target_empty:
        return 1.0

    if pred_empty != target_empty:
        return 0.0

    d1, d2 = surface_distances(pred, target, spacing)
    if d1.size == 0 or d2.size == 0:
        return 0.0

    pred_ok = np.sum(d1 <= tolerance_mm)
    target_ok = np.sum(d2 <= tolerance_mm)

    denom = d1.size + d2.size
    if denom == 0:
        return 1.0

    return float((pred_ok + target_ok) / (denom + EPS))


def compute_binary_segmentation_metrics(
    pred: ArrayLike,
    target: ArrayLike,
    *,
    spacing: Optional[Sequence[float]] = None,
    surface_dice_tolerance_mm: float = 1.0,
) -> SegmentationMetrics:
    """
    Compute full binary segmentation metrics.

    Args:
        pred:
            Binary mask or mask-like array.
        target:
            Binary ground truth mask.
        spacing:
            Physical voxel spacing in mm. Use real NIfTI spacing when available.
        surface_dice_tolerance_mm:
            Tolerance for surface Dice.
    """
    pred_mask = squeeze_binary_mask(pred)
    target_mask = squeeze_binary_mask(target)

    if pred_mask.shape != target_mask.shape:
        raise ValueError(
            f"Prediction and target shapes must match. "
            f"Got pred={pred_mask.shape}, target={target_mask.shape}"
        )

    spacing_tuple = normalize_spacing(spacing)

    counts = confusion_counts(pred_mask, target_mask)
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]

    pred_voxels = int(pred_mask.sum())
    target_voxels = int(target_mask.sum())

    pred_empty = pred_voxels == 0
    target_empty = target_voxels == 0
    both_empty = pred_empty and target_empty

    hd = hausdorff_distance(pred_mask, target_mask, spacing_tuple, percentile=None)
    hd95 = hausdorff_distance(pred_mask, target_mask, spacing_tuple, percentile=95.0)
    asd = average_surface_distance(pred_mask, target_mask, spacing_tuple, symmetric=False)
    assd = average_surface_distance(pred_mask, target_mask, spacing_tuple, symmetric=True)
    surf_dice = surface_dice_at_tolerance(
        pred_mask,
        target_mask,
        spacing_tuple,
        tolerance_mm=surface_dice_tolerance_mm,
    )

    return SegmentationMetrics(
        dice=dice_score_from_counts(tp, fp, fn),
        jaccard=jaccard_score_from_counts(tp, fp, fn),
        precision=precision_from_counts(tp, fp),
        recall=recall_from_counts(tp, fn),
        sensitivity=recall_from_counts(tp, fn),
        specificity=specificity_from_counts(tn, fp),
        hd=hd,
        hd95=hd95,
        asd=asd,
        assd=assd,
        surface_dice=surf_dice,
        pred_voxels=pred_voxels,
        target_voxels=target_voxels,
        intersection_voxels=tp,
        false_positive_voxels=fp,
        false_negative_voxels=fn,
        true_negative_voxels=tn,
        relative_volume_error=relative_volume_error(pred_mask, target_mask),
        absolute_volume_error=int(abs(pred_voxels - target_voxels)),
        pred_empty=pred_empty,
        target_empty=target_empty,
        both_empty=both_empty,
        spacing=spacing_tuple,
        surface_dice_tolerance_mm=float(surface_dice_tolerance_mm),
    )


def compute_metrics_from_logits(
    logits: ArrayLike,
    target: ArrayLike,
    *,
    spacing: Optional[Sequence[float]] = None,
    surface_dice_tolerance_mm: float = 1.0,
) -> SegmentationMetrics:
    """Convert logits to binary mask and compute metrics."""
    pred = logits_to_binary_mask(logits)
    return compute_binary_segmentation_metrics(
        pred,
        target,
        spacing=spacing,
        surface_dice_tolerance_mm=surface_dice_tolerance_mm,
    )


def mean_radial_error(
    pred_points: ArrayLike,
    target_points: ArrayLike,
    *,
    spacing: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Compute Mean Radial Error for landmark localization.

    Args:
        pred_points:
            Array of shape [N, 3].
        target_points:
            Array of shape [N, 3].
        spacing:
            If points are voxel coordinates, pass spacing to convert to mm.
            If points are already in mm/world coordinates, keep spacing=None.

    Returns:
        mean, median, max, per_point_errors
    """
    pred = to_numpy(pred_points).astype(np.float64)
    target = to_numpy(target_points).astype(np.float64)

    if pred.shape != target.shape:
        raise ValueError(f"Point shapes must match. Got pred={pred.shape}, target={target.shape}")

    if pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError(f"Expected points shape [N, 3], got {pred.shape}")

    if spacing is not None:
        spacing_arr = np.asarray(normalize_spacing(spacing), dtype=np.float64)
        pred = pred * spacing_arr
        target = target * spacing_arr

    errors = np.linalg.norm(pred - target, axis=1)

    return {
        "mre": float(np.mean(errors)) if errors.size else math.nan,
        "median_re": float(np.median(errors)) if errors.size else math.nan,
        "max_re": float(np.max(errors)) if errors.size else math.nan,
        "num_points": int(errors.size),
        "per_point_errors": errors.tolist(),
    }


def summarize_metric_dicts(metric_dicts: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Summarize per-case metric dictionaries.

    Ignores non-numeric fields and non-finite values for mean/std/median.
    """
    numeric: Dict[str, List[float]] = {}

    for item in metric_dicts:
        for key, value in item.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                numeric.setdefault(key, []).append(float(value))

    summary: Dict[str, Dict[str, float]] = {}

    for key, values in numeric.items():
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]

        if finite.size == 0:
            summary[key] = {
                "mean": math.nan,
                "std": math.nan,
                "median": math.nan,
                "min": math.nan,
                "max": math.nan,
            }
            continue

        summary[key] = {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "median": float(np.median(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
        }

    return summary


class InferenceTimer:
    """Simple context manager for inference timing."""

    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.elapsed_seconds: Optional[float] = None

    def __enter__(self) -> "InferenceTimer":
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        self.elapsed_seconds = end_time - float(self.start_time)


def _json_safe(obj: Any) -> Any:
    """Convert dataclass / inf / numpy values to JSON-safe object."""
    if isinstance(obj, SegmentationMetrics):
        return _json_safe(asdict(obj))

    if isinstance(obj, dict):
        return {key: _json_safe(value) for key, value in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        obj = float(obj)

    if isinstance(obj, float):
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        if math.isnan(obj):
            return "nan"
        return obj

    return obj


def parse_args() -> argparse.Namespace:
    """CLI args for metric sanity test."""
    parser = argparse.ArgumentParser(description="Test MVAA Task 1 metrics.")
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument("--surface-dice-tolerance-mm", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    """Run a synthetic sanity check."""
    args = parse_args()

    target = np.zeros((32, 32, 32), dtype=np.uint8)
    pred = np.zeros((32, 32, 32), dtype=np.uint8)

    target[10:20, 10:20, 10:20] = 1
    pred[11:21, 10:20, 10:20] = 1

    metrics = compute_binary_segmentation_metrics(
        pred,
        target,
        spacing=args.spacing,
        surface_dice_tolerance_mm=args.surface_dice_tolerance_mm,
    )

    print(json.dumps(_json_safe(metrics), indent=2))

    pred_points = np.asarray([[10, 10, 10], [20, 20, 20]], dtype=np.float64)
    target_points = np.asarray([[11, 10, 10], [20, 22, 20]], dtype=np.float64)
    mre = mean_radial_error(pred_points, target_points, spacing=args.spacing)

    print(json.dumps(_json_safe({"landmark_metrics": mre}), indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
