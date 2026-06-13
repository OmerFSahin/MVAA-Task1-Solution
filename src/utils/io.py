#!/usr/bin/env python3
"""
I/O utilities for MVAA Task 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage


def load_nifti(path: str | Path) -> nib.spatialimages.SpatialImage:
    return nib.load(str(path))


def get_nifti_spacing(path_or_img: str | Path | nib.spatialimages.SpatialImage) -> Tuple[float, float, float]:
    img = load_nifti(path_or_img) if not isinstance(path_or_img, nib.spatialimages.SpatialImage) else path_or_img
    zooms = img.header.get_zooms()
    if len(zooms) < 3:
        return (1.0, 1.0, 1.0)
    return (float(zooms[0]), float(zooms[1]), float(zooms[2]))


def resize_mask_nearest(mask: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    target_shape = tuple(int(x) for x in target_shape)

    if tuple(mask.shape) == target_shape:
        return mask.astype(np.uint8)

    zoom = [target_shape[i] / float(mask.shape[i]) for i in range(3)]
    resized = ndimage.zoom(mask.astype(np.uint8), zoom=zoom, order=0).astype(np.uint8)

    if resized.shape != target_shape:
        fixed = np.zeros(target_shape, dtype=np.uint8)
        common = tuple(min(resized.shape[i], target_shape[i]) for i in range(3))
        fixed[: common[0], : common[1], : common[2]] = resized[: common[0], : common[1], : common[2]]
        resized = fixed

    return resized.astype(np.uint8)


def save_mask_like_source(
    mask: np.ndarray,
    source_image_path: str | Path,
    output_path: str | Path,
) -> Dict[str, Any]:
    source_img = nib.load(str(source_image_path))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_shape = tuple(int(x) for x in source_img.shape[:3])
    mask = resize_mask_nearest(mask, original_shape)
    mask = (mask > 0).astype(np.uint8)

    header = source_img.header.copy()
    header.set_data_dtype(np.uint8)
    header["cal_min"] = 0
    header["cal_max"] = 1

    out_img = nib.Nifti1Image(mask, affine=source_img.affine, header=header)

    try:
        qform, qcode = source_img.get_qform(coded=True)
        if qform is not None:
            out_img.set_qform(qform, int(qcode))
    except Exception:
        pass

    try:
        sform, scode = source_img.get_sform(coded=True)
        if sform is not None:
            out_img.set_sform(sform, int(scode))
    except Exception:
        pass

    nib.save(out_img, str(output_path))

    return {
        "output_path": str(output_path),
        "shape": list(mask.shape),
        "spacing": list(get_nifti_spacing(source_img)),
        "foreground_voxels": int(mask.sum()),
    }
