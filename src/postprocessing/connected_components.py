#!/usr/bin/env python3
"""
Connected-component post-processing for MVAA Task 1.

Task:
    Mitral valve segmentation from 3D cardiac CT.

Why this matters:
    The mitral valve is a small target structure. Segmentation models may
    produce small false-positive islands far away from the true valve. These
    islands can severely hurt boundary metrics such as HD / HD95.

Design philosophy:
    - Do not apply aggressive post-processing by default.
    - Make every operation optional and configurable.
    - Preserve the ability to inspect component statistics.
    - Never silently erase all foreground unless explicitly allowed.

Supported operations:
    - connected-component labeling
    - keep largest component
    - keep top-k components
    - remove small components by voxel count or physical volume
    - fill binary holes
    - binary opening / closing
    - full config-driven post-processing pipeline
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import ndimage


ArrayLike = np.ndarray | torch.Tensor


@dataclass
class ComponentStats:
    """Statistics for one connected component."""

    label_id: int
    voxel_count: int
    volume_mm3: float
    bbox_min: Tuple[int, int, int]
    bbox_max: Tuple[int, int, int]
    centroid_voxel: Tuple[float, float, float]


def to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert torch tensor or numpy array to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def squeeze_binary_mask(mask: ArrayLike) -> np.ndarray:
    """
    Convert input to a 3D binary numpy mask.

    Accepted shapes:
        [D, H, W]
        [1, D, H, W]
        [B, 1, D, H, W] only if B == 1
    """
    arr = to_numpy(mask)

    if arr.ndim == 5:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got shape {arr.shape}")
        arr = arr[0]

    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected channel size 1, got shape {arr.shape}")
        arr = arr[0]

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {arr.shape}")

    return arr.astype(bool)


def normalize_spacing(spacing: Optional[Sequence[float]]) -> Tuple[float, float, float]:
    """Normalize spacing to 3-value tuple."""
    if spacing is None:
        return (1.0, 1.0, 1.0)

    if len(spacing) != 3:
        raise ValueError(f"spacing must contain 3 values, got: {spacing}")

    return (float(spacing[0]), float(spacing[1]), float(spacing[2]))


def voxel_volume_mm3(spacing: Optional[Sequence[float]]) -> float:
    """Return physical voxel volume in mm^3."""
    sx, sy, sz = normalize_spacing(spacing)
    return float(sx * sy * sz)


def get_connectivity_structure(connectivity: int = 26) -> np.ndarray:
    """
    Return a 3D connectivity structure.

    Args:
        connectivity:
            6, 18, or 26.

    Notes:
        6-connectivity:
            face-neighbor connectivity.
        18-connectivity:
            face + edge.
        26-connectivity:
            face + edge + corner.

    For small 3D anatomical structures, 26-connectivity is usually the most
    permissive and avoids splitting diagonally touching voxels.
    """
    if connectivity == 6:
        return ndimage.generate_binary_structure(rank=3, connectivity=1)

    if connectivity == 18:
        return ndimage.generate_binary_structure(rank=3, connectivity=2)

    if connectivity == 26:
        return ndimage.generate_binary_structure(rank=3, connectivity=3)

    raise ValueError("connectivity must be one of: 6, 18, 26")


def label_connected_components(
    mask: ArrayLike,
    *,
    connectivity: int = 26,
) -> Tuple[np.ndarray, int]:
    """
    Label connected foreground components in a binary mask.

    Returns:
        labeled_mask:
            int32 array with 0 as background and 1..N as component labels.
        num_components:
            Number of foreground connected components.
    """
    binary = squeeze_binary_mask(mask)
    structure = get_connectivity_structure(connectivity)

    labeled, num_components = ndimage.label(binary, structure=structure)
    return labeled.astype(np.int32), int(num_components)


def component_stats(
    mask: ArrayLike,
    *,
    spacing: Optional[Sequence[float]] = None,
    connectivity: int = 26,
) -> List[ComponentStats]:
    """Return component statistics sorted by descending voxel count."""
    labeled, num_components = label_connected_components(mask, connectivity=connectivity)
    vv = voxel_volume_mm3(spacing)

    stats: List[ComponentStats] = []

    if num_components == 0:
        return stats

    objects = ndimage.find_objects(labeled)

    for label_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue

        component = labeled == label_id
        coords = np.argwhere(component)

        if coords.size == 0:
            continue

        voxel_count = int(coords.shape[0])
        bbox_min = tuple(int(v) for v in coords.min(axis=0))
        bbox_max = tuple(int(v) for v in coords.max(axis=0))
        centroid = tuple(float(v) for v in coords.mean(axis=0))

        stats.append(
            ComponentStats(
                label_id=label_id,
                voxel_count=voxel_count,
                volume_mm3=float(voxel_count * vv),
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                centroid_voxel=centroid,
            )
        )

    stats.sort(key=lambda item: item.voxel_count, reverse=True)
    return stats


def count_components(mask: ArrayLike, *, connectivity: int = 26) -> int:
    """Count foreground connected components."""
    _, num_components = label_connected_components(mask, connectivity=connectivity)
    return int(num_components)


def keep_components_by_label_ids(
    mask: ArrayLike,
    label_ids: Iterable[int],
    *,
    connectivity: int = 26,
) -> np.ndarray:
    """Keep only selected connected-component label ids."""
    binary = squeeze_binary_mask(mask)
    labeled, _ = label_connected_components(binary, connectivity=connectivity)

    label_ids = set(int(x) for x in label_ids)
    if not label_ids:
        return np.zeros_like(binary, dtype=np.uint8)

    kept = np.isin(labeled, list(label_ids))
    return kept.astype(np.uint8)


def keep_largest_component(
    mask: ArrayLike,
    *,
    connectivity: int = 26,
    min_voxels_to_apply: int = 1,
) -> np.ndarray:
    """
    Keep only the largest foreground component.

    Args:
        min_voxels_to_apply:
            If total foreground voxels are below this number, return mask as-is.
            This prevents over-processing tiny/empty predictions.
    """
    binary = squeeze_binary_mask(mask)

    if int(binary.sum()) < int(min_voxels_to_apply):
        return binary.astype(np.uint8)

    stats = component_stats(binary, connectivity=connectivity)
    if not stats:
        return binary.astype(np.uint8)

    largest_label = stats[0].label_id
    return keep_components_by_label_ids(binary, [largest_label], connectivity=connectivity)


def keep_top_k_components(
    mask: ArrayLike,
    *,
    k: int = 1,
    connectivity: int = 26,
) -> np.ndarray:
    """Keep top-k largest foreground components."""
    if k <= 0:
        raise ValueError("k must be positive.")

    binary = squeeze_binary_mask(mask)
    stats = component_stats(binary, connectivity=connectivity)

    if not stats:
        return binary.astype(np.uint8)

    label_ids = [item.label_id for item in stats[:k]]
    return keep_components_by_label_ids(binary, label_ids, connectivity=connectivity)


def remove_small_components(
    mask: ArrayLike,
    *,
    min_voxels: Optional[int] = None,
    min_volume_mm3: Optional[float] = None,
    spacing: Optional[Sequence[float]] = None,
    connectivity: int = 26,
    keep_at_least_largest: bool = True,
) -> np.ndarray:
    """
    Remove connected components smaller than thresholds.

    Args:
        min_voxels:
            Minimum component size in voxels.
        min_volume_mm3:
            Minimum component size in physical volume.
        spacing:
            Used when min_volume_mm3 is provided.
        keep_at_least_largest:
            If all components are removed, restore the largest component.
            This is safer for small target structures.
    """
    binary = squeeze_binary_mask(mask)

    if min_voxels is None and min_volume_mm3 is None:
        return binary.astype(np.uint8)

    stats = component_stats(binary, spacing=spacing, connectivity=connectivity)
    if not stats:
        return binary.astype(np.uint8)

    vv = voxel_volume_mm3(spacing)
    min_voxels_from_volume = None
    if min_volume_mm3 is not None:
        min_voxels_from_volume = int(math.ceil(float(min_volume_mm3) / max(vv, 1e-8)))

    thresholds = []
    if min_voxels is not None:
        thresholds.append(int(min_voxels))
    if min_voxels_from_volume is not None:
        thresholds.append(int(min_voxels_from_volume))

    effective_min_voxels = max(thresholds) if thresholds else 0

    kept_label_ids = [
        item.label_id
        for item in stats
        if item.voxel_count >= effective_min_voxels
    ]

    if not kept_label_ids and keep_at_least_largest:
        kept_label_ids = [stats[0].label_id]

    return keep_components_by_label_ids(binary, kept_label_ids, connectivity=connectivity)


def fill_binary_holes(mask: ArrayLike) -> np.ndarray:
    """
    Fill holes inside binary foreground mask.

    Use carefully:
        Hole filling can increase predicted volume.
    """
    binary = squeeze_binary_mask(mask)
    filled = ndimage.binary_fill_holes(binary)
    return filled.astype(np.uint8)


def binary_opening(
    mask: ArrayLike,
    *,
    iterations: int = 1,
    connectivity: int = 26,
) -> np.ndarray:
    """
    Binary opening: erosion followed by dilation.

    This can remove small protrusions/noise but may also shrink thin structures.
    Use carefully for mitral valve segmentation.
    """
    binary = squeeze_binary_mask(mask)

    if iterations <= 0:
        return binary.astype(np.uint8)

    structure = get_connectivity_structure(connectivity)
    opened = ndimage.binary_opening(binary, structure=structure, iterations=iterations)
    return opened.astype(np.uint8)


def binary_closing(
    mask: ArrayLike,
    *,
    iterations: int = 1,
    connectivity: int = 26,
) -> np.ndarray:
    """
    Binary closing: dilation followed by erosion.

    This can close tiny gaps and smooth small holes but may thicken structures.
    """
    binary = squeeze_binary_mask(mask)

    if iterations <= 0:
        return binary.astype(np.uint8)

    structure = get_connectivity_structure(connectivity)
    closed = ndimage.binary_closing(binary, structure=structure, iterations=iterations)
    return closed.astype(np.uint8)


def postprocess_binary_mask(
    mask: ArrayLike,
    *,
    spacing: Optional[Sequence[float]] = None,
    connectivity: int = 26,
    keep_largest: bool = False,
    keep_top_k: Optional[int] = None,
    remove_small: bool = False,
    min_voxels: Optional[int] = None,
    min_volume_mm3: Optional[float] = None,
    fill_holes: bool = False,
    opening_iterations: int = 0,
    closing_iterations: int = 0,
    keep_at_least_largest: bool = True,
    return_report: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
    """
    Configurable binary post-processing pipeline.

    Recommended first experiments:
        A) no postprocess:
            all options false
        B) remove tiny false-positive islands:
            remove_small=True, min_voxels=100
        C) largest component only:
            keep_largest=True
        D) safer than largest:
            keep_top_k=2 or 3

    Operation order:
        1. optional opening
        2. optional closing
        3. optional fill holes
        4. optional remove small components
        5. optional keep largest / top-k

    Returns:
        processed mask as uint8 array, optionally with report.
    """
    original = squeeze_binary_mask(mask)
    processed = original.astype(np.uint8)

    before_stats = component_stats(
        processed,
        spacing=spacing,
        connectivity=connectivity,
    )

    if opening_iterations > 0:
        processed = binary_opening(
            processed,
            iterations=opening_iterations,
            connectivity=connectivity,
        )

    if closing_iterations > 0:
        processed = binary_closing(
            processed,
            iterations=closing_iterations,
            connectivity=connectivity,
        )

    if fill_holes:
        processed = fill_binary_holes(processed)

    if remove_small:
        processed = remove_small_components(
            processed,
            min_voxels=min_voxels,
            min_volume_mm3=min_volume_mm3,
            spacing=spacing,
            connectivity=connectivity,
            keep_at_least_largest=keep_at_least_largest,
        )

    if keep_largest:
        processed = keep_largest_component(
            processed,
            connectivity=connectivity,
        )

    if keep_top_k is not None:
        processed = keep_top_k_components(
            processed,
            k=int(keep_top_k),
            connectivity=connectivity,
        )

    processed = processed.astype(np.uint8)

    after_stats = component_stats(
        processed,
        spacing=spacing,
        connectivity=connectivity,
    )

    report = {
        "original_voxels": int(original.sum()),
        "processed_voxels": int(processed.sum()),
        "removed_voxels": int(original.sum()) - int(processed.sum()),
        "before_num_components": len(before_stats),
        "after_num_components": len(after_stats),
        "before_components": [asdict(item) for item in before_stats[:20]],
        "after_components": [asdict(item) for item in after_stats[:20]],
        "settings": {
            "spacing": normalize_spacing(spacing),
            "connectivity": connectivity,
            "keep_largest": keep_largest,
            "keep_top_k": keep_top_k,
            "remove_small": remove_small,
            "min_voxels": min_voxels,
            "min_volume_mm3": min_volume_mm3,
            "fill_holes": fill_holes,
            "opening_iterations": opening_iterations,
            "closing_iterations": closing_iterations,
            "keep_at_least_largest": keep_at_least_largest,
        },
    }

    if return_report:
        return processed, report

    return processed


def postprocess_from_config(
    mask: ArrayLike,
    config: Mapping[str, Any],
    *,
    spacing: Optional[Sequence[float]] = None,
    return_report: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run post-processing from config.

    Expected config section:

        postprocessing:
          enabled: true
          connectivity: 26
          keep_largest_component: false
          keep_top_k_components: null
          remove_small_objects: true
          min_object_size: 100
          min_object_volume_mm3: null
          fill_holes: false
          opening_iterations: 0
          closing_iterations: 0
          keep_at_least_largest: true
    """
    cfg = dict(config.get("postprocessing", {}))

    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        binary = squeeze_binary_mask(mask).astype(np.uint8)
        if return_report:
            return binary, {
                "enabled": False,
                "original_voxels": int(binary.sum()),
                "processed_voxels": int(binary.sum()),
            }
        return binary

    keep_top_k = cfg.get("keep_top_k_components", None)
    if keep_top_k in {"null", "none", "None", ""}:
        keep_top_k = None

    return postprocess_binary_mask(
        mask,
        spacing=spacing,
        connectivity=int(cfg.get("connectivity", 26)),
        keep_largest=bool(cfg.get("keep_largest_component", False)),
        keep_top_k=int(keep_top_k) if keep_top_k is not None else None,
        remove_small=bool(cfg.get("remove_small_objects", False)),
        min_voxels=cfg.get("min_object_size", None),
        min_volume_mm3=cfg.get("min_object_volume_mm3", None),
        fill_holes=bool(cfg.get("fill_holes", False)),
        opening_iterations=int(cfg.get("opening_iterations", 0)),
        closing_iterations=int(cfg.get("closing_iterations", 0)),
        keep_at_least_largest=bool(cfg.get("keep_at_least_largest", True)),
        return_report=return_report,
    )


def _json_safe(obj: Any) -> Any:
    """Convert numpy/dataclass objects into JSON-safe values."""
    if isinstance(obj, dict):
        return {key: _json_safe(value) for key, value in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, float):
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        if math.isnan(obj):
            return "nan"

    return obj


def parse_args() -> argparse.Namespace:
    """CLI args for synthetic sanity test."""
    parser = argparse.ArgumentParser(description="Test connected-component post-processing.")
    parser.add_argument("--connectivity", type=int, default=26, choices=[6, 18, 26])
    parser.add_argument("--keep-largest", action="store_true")
    parser.add_argument("--keep-top-k", type=int, default=None)
    parser.add_argument("--remove-small", action="store_true")
    parser.add_argument("--min-voxels", type=int, default=50)
    parser.add_argument("--fill-holes", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run synthetic sanity test."""
    args = parse_args()

    mask = np.zeros((64, 64, 64), dtype=np.uint8)

    # Main component.
    mask[20:35, 20:35, 20:35] = 1

    # Small false-positive islands.
    mask[5:7, 5:7, 5:7] = 1
    mask[50:52, 50:52, 50:52] = 1

    # Small hole inside main component.
    mask[25:27, 25:27, 25:27] = 0

    processed, report = postprocess_binary_mask(
        mask,
        connectivity=args.connectivity,
        keep_largest=args.keep_largest,
        keep_top_k=args.keep_top_k,
        remove_small=args.remove_small,
        min_voxels=args.min_voxels,
        fill_holes=args.fill_holes,
        return_report=True,
    )

    print(json.dumps(_json_safe(report), indent=2))
    print(f"processed shape: {processed.shape}")
    print(f"processed dtype: {processed.dtype}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
