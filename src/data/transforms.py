#!/usr/bin/env python3
"""
MONAI transforms for MVAA Task 1.

This file defines transform builders for:

    - labeled supervised training
    - unlabeled training
    - validation / inference

Design goals:
    - keep all hyperparameters configurable,
    - use safe CT preprocessing,
    - support patch-based 3D training,
    - keep transforms compatible with MONAI Dataset/DataLoader.

Recommended first baseline:
    - CT intensity clipping: [-1000, 1000]
    - intensity scaling: [0, 1]
    - ROI size: [128, 128, 128]
    - positive/negative sampling: 1:1
    - simple 3D augmentation: flips, rotate90, intensity scale/shift, light noise

Important:
    RandCropByPosNegLabeld returns multiple samples per volume.
    In dataloaders.py, use MONAI list_data_collate.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)


Config = Mapping[str, Any]


def _section(config: Optional[Config], name: str) -> Dict[str, Any]:
    """Safely read a nested config section."""
    if config is None:
        return {}
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Config section '{name}' must be a mapping, got {type(value)}")
    return dict(value)


def _as_tuple3(value: Sequence[int | float], name: str) -> Tuple[int | float, int | float, int | float]:
    """Validate and convert a 3-value list/tuple into a tuple."""
    if len(value) != 3:
        raise ValueError(f"{name} must have exactly 3 values, got: {value}")
    return tuple(value)  # type: ignore[return-value]


def _bool_value(value: Any) -> bool:
    """Convert config-like values to bool safely."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _preprocessing_params(config: Optional[Config]) -> Dict[str, Any]:
    """Return preprocessing parameters with safe defaults."""
    prep = _section(config, "preprocessing")

    return {
        "intensity_min": float(prep.get("intensity_min", -1000.0)),
        "intensity_max": float(prep.get("intensity_max", 1000.0)),
        "intensity_out_min": float(prep.get("intensity_out_min", 0.0)),
        "intensity_out_max": float(prep.get("intensity_out_max", 1.0)),
        "enable_orientation": _bool_value(prep.get("enable_orientation", False)),
        "orientation_axcodes": str(prep.get("orientation_axcodes", "RAS")),
        "enable_spacing_resample": _bool_value(prep.get("enable_spacing_resample", False)),
        "target_spacing": _as_tuple3(prep.get("target_spacing", [0.5, 0.5, 0.5]), "target_spacing"),
    }


def _training_params(config: Optional[Config]) -> Dict[str, Any]:
    """Return training transform parameters with safe defaults."""
    train = _section(config, "training")

    return {
        "roi_size": _as_tuple3(train.get("roi_size", [128, 128, 128]), "roi_size"),
        "train_crops_per_volume": int(train.get("train_crops_per_volume", 2)),
    }


def _inference_params(config: Optional[Config]) -> Dict[str, Any]:
    """Return inference transform parameters with safe defaults."""
    infer = _section(config, "inference")

    return {
        "roi_size": _as_tuple3(infer.get("roi_size", [128, 128, 128]), "roi_size"),
    }


def _augmentation_params(config: Optional[Config]) -> Dict[str, Any]:
    """
    Return augmentation parameters.

    These are intentionally conservative for the first baseline.
    We can make them stronger after the first end-to-end run works.
    """
    aug = _section(config, "augmentation")

    return {
        "flip_prob": float(aug.get("flip_prob", 0.5)),
        "rotate90_prob": float(aug.get("rotate90_prob", 0.2)),
        "intensity_scale_prob": float(aug.get("intensity_scale_prob", 0.15)),
        "intensity_scale_factors": float(aug.get("intensity_scale_factors", 0.10)),
        "intensity_shift_prob": float(aug.get("intensity_shift_prob", 0.15)),
        "intensity_shift_offsets": float(aug.get("intensity_shift_offsets", 0.10)),
        "gaussian_noise_prob": float(aug.get("gaussian_noise_prob", 0.10)),
        "gaussian_noise_std": float(aug.get("gaussian_noise_std", 0.01)),
    }


def _load_channel_transforms(keys: Sequence[str]) -> List[Any]:
    """Load NIfTI files and ensure channel-first shape."""
    return [
        LoadImaged(keys=list(keys)),
        EnsureChannelFirstd(keys=list(keys)),
    ]


def _optional_orientation_transform(
    keys: Sequence[str],
    preprocessing: Mapping[str, Any],
) -> List[Any]:
    """Optionally orient images/labels to a common axis convention."""
    if not preprocessing["enable_orientation"]:
        return []

    return [
        Orientationd(
            keys=list(keys),
            axcodes=preprocessing["orientation_axcodes"],
        )
    ]


def _optional_spacing_transform(
    keys: Sequence[str],
    modes: Sequence[str],
    preprocessing: Mapping[str, Any],
) -> List[Any]:
    """Optionally resample image/label spacing."""
    if not preprocessing["enable_spacing_resample"]:
        return []

    return [
        Spacingd(
            keys=list(keys),
            pixdim=preprocessing["target_spacing"],
            mode=tuple(modes),
        )
    ]


def _intensity_transform(preprocessing: Mapping[str, Any]) -> List[Any]:
    """CT intensity clipping and min-max scaling."""
    return [
        ScaleIntensityRanged(
            keys=["image"],
            a_min=preprocessing["intensity_min"],
            a_max=preprocessing["intensity_max"],
            b_min=preprocessing["intensity_out_min"],
            b_max=preprocessing["intensity_out_max"],
            clip=True,
        )
    ]


def _basic_spatial_augmentation(keys: Sequence[str], augmentation: Mapping[str, Any]) -> List[Any]:
    """Basic 3D spatial augmentations for image and label together."""
    return [
        RandFlipd(keys=list(keys), spatial_axis=0, prob=augmentation["flip_prob"]),
        RandFlipd(keys=list(keys), spatial_axis=1, prob=augmentation["flip_prob"]),
        RandFlipd(keys=list(keys), spatial_axis=2, prob=augmentation["flip_prob"]),
        RandRotate90d(
            keys=list(keys),
            prob=augmentation["rotate90_prob"],
            max_k=3,
            spatial_axes=(0, 1),
        ),
    ]


def _basic_intensity_augmentation(augmentation: Mapping[str, Any]) -> List[Any]:
    """Light CT intensity augmentations applied only to image."""
    return [
        RandScaleIntensityd(
            keys=["image"],
            factors=augmentation["intensity_scale_factors"],
            prob=augmentation["intensity_scale_prob"],
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=augmentation["intensity_shift_offsets"],
            prob=augmentation["intensity_shift_prob"],
        ),
        RandGaussianNoised(
            keys=["image"],
            prob=augmentation["gaussian_noise_prob"],
            mean=0.0,
            std=augmentation["gaussian_noise_std"],
        ),
    ]


def build_labeled_train_transforms(config: Optional[Config] = None) -> Compose:
    """
    Build transforms for supervised labeled training.

    Input record must contain:
        - image
        - label

    Output:
        Random cropped 3D patches with image and label.
    """
    preprocessing = _preprocessing_params(config)
    training = _training_params(config)
    augmentation = _augmentation_params(config)

    keys = ["image", "label"]
    roi_size = training["roi_size"]

    transforms: List[Any] = []

    transforms += _load_channel_transforms(keys)
    transforms += _optional_orientation_transform(keys, preprocessing)
    transforms += _optional_spacing_transform(keys, modes=["bilinear", "nearest"], preprocessing=preprocessing)
    transforms += _intensity_transform(preprocessing)

    # Make sure volumes are at least as large as the training patch.
    transforms.append(
        SpatialPadd(
            keys=keys,
            spatial_size=roi_size,
        )
    )

    # Positive/negative patch sampling is important for small target structures.
    transforms.append(
        RandCropByPosNegLabeld(
            keys=keys,
            label_key="label",
            spatial_size=roi_size,
            pos=1.0,
            neg=1.0,
            num_samples=training["train_crops_per_volume"],
            image_key="image",
            image_threshold=0.0,
        )
    )

    transforms += _basic_spatial_augmentation(keys, augmentation)
    transforms += _basic_intensity_augmentation(augmentation)

    transforms.append(EnsureTyped(keys=keys, track_meta=True))

    return Compose(transforms)


def build_unlabeled_train_transforms(
    config: Optional[Config] = None,
    *,
    strong: bool = False,
) -> Compose:
    """
    Build transforms for unlabeled images.

    Input record must contain:
        - image

    For the initial supervised baseline, this transform may not be used.
    It is included now so semi-supervised training can be added cleanly later.

    Args:
        strong:
            If True, applies stronger intensity augmentation for student branch.
    """
    preprocessing = _preprocessing_params(config)
    training = _training_params(config)
    augmentation = _augmentation_params(config)

    keys = ["image"]
    roi_size = training["roi_size"]

    transforms: List[Any] = []

    transforms += _load_channel_transforms(keys)
    transforms += _optional_orientation_transform(keys, preprocessing)
    transforms += _optional_spacing_transform(keys, modes=["bilinear"], preprocessing=preprocessing)
    transforms += _intensity_transform(preprocessing)

    transforms.append(
        SpatialPadd(
            keys=keys,
            spatial_size=roi_size,
        )
    )

    transforms.append(
        RandSpatialCropSamplesd(
            keys=keys,
            roi_size=roi_size,
            num_samples=training["train_crops_per_volume"],
            random_size=False,
        )
    )

    transforms += _basic_spatial_augmentation(keys, augmentation)

    if strong:
        # Slightly stronger intensity perturbation for future semi-supervised training.
        strong_aug = dict(augmentation)
        strong_aug["intensity_scale_prob"] = max(augmentation["intensity_scale_prob"], 0.30)
        strong_aug["intensity_shift_prob"] = max(augmentation["intensity_shift_prob"], 0.30)
        strong_aug["gaussian_noise_prob"] = max(augmentation["gaussian_noise_prob"], 0.20)
        transforms += _basic_intensity_augmentation(strong_aug)
    else:
        transforms += _basic_intensity_augmentation(augmentation)

    transforms.append(EnsureTyped(keys=keys, track_meta=True))

    return Compose(transforms)


def build_validation_transforms(
    config: Optional[Config] = None,
    *,
    has_labels: bool = False,
) -> Compose:
    """
    Build deterministic validation transforms.

    If labels exist locally, set has_labels=True.
    Challenge validation/test images usually do not include labels.
    """
    preprocessing = _preprocessing_params(config)

    keys = ["image", "label"] if has_labels else ["image"]
    modes = ["bilinear", "nearest"] if has_labels else ["bilinear"]

    transforms: List[Any] = []

    transforms += _load_channel_transforms(keys)
    transforms += _optional_orientation_transform(keys, preprocessing)
    transforms += _optional_spacing_transform(keys, modes=modes, preprocessing=preprocessing)
    transforms += _intensity_transform(preprocessing)
    transforms.append(EnsureTyped(keys=keys, track_meta=True))

    return Compose(transforms)


def build_inference_transforms(config: Optional[Config] = None) -> Compose:
    """
    Build deterministic inference transforms.

    Input record must contain:
        - image

    Prediction saving should preserve original affine/header separately
    in the inference code.
    """
    return build_validation_transforms(config=config, has_labels=False)


def build_task1_transforms(
    split: str,
    config: Optional[Config] = None,
    *,
    has_labels: bool = False,
    strong: bool = False,
) -> Compose:
    """
    Generic transform factory.

    Args:
        split:
            One of:
                - train_labeled
                - train_unlabeled
                - val
                - inference
        config:
            Loaded YAML config.
        has_labels:
            Used for validation if label files are available.
        strong:
            Used for unlabeled strong augmentation branch.
    """
    split = split.lower()

    if split in {"train", "train_labeled", "labeled"}:
        return build_labeled_train_transforms(config)

    if split in {"train_unlabeled", "unlabeled"}:
        return build_unlabeled_train_transforms(config, strong=strong)

    if split in {"val", "validation"}:
        return build_validation_transforms(config, has_labels=has_labels)

    if split in {"infer", "inference", "test"}:
        return build_inference_transforms(config)

    raise ValueError(
        f"Unknown split '{split}'. Expected one of: "
        "train_labeled, train_unlabeled, val, inference"
    )


def describe_transform(transform: Compose) -> str:
    """Return a readable transform summary."""
    lines = []
    for idx, item in enumerate(transform.transforms, start=1):
        lines.append(f"{idx:02d}. {item.__class__.__name__}: {item}")
    return "\n".join(lines)


def _load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)

    return loaded or {}


def _main() -> int:
    """Small CLI for inspecting transform composition."""
    import argparse

    parser = argparse.ArgumentParser(description="Inspect MVAA Task 1 MONAI transforms.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/task1_train.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train_labeled",
        help="Split: train_labeled, train_unlabeled, val, inference.",
    )
    parser.add_argument(
        "--has-labels",
        action="store_true",
        help="Use labels for validation transform.",
    )
    parser.add_argument(
        "--strong",
        action="store_true",
        help="Use strong unlabeled augmentation.",
    )
    args = parser.parse_args()

    config = _load_yaml(args.config)
    transform = build_task1_transforms(
        split=args.split,
        config=config,
        has_labels=args.has_labels,
        strong=args.strong,
    )

    print(describe_transform(transform))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
