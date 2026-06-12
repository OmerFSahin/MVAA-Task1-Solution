#!/usr/bin/env python3
"""
DataLoader builders for MVAA Task 1.

This module connects:

    - src/data/task1_dataset.py
    - src/data/transforms.py

and returns MONAI/PyTorch DataLoaders for:

    - labeled supervised training
    - optional unlabeled training
    - validation / inference

Important:
    The labeled train transform uses RandCropByPosNegLabeld with num_samples > 1.
    Therefore, DataLoader should use MONAI's list_data_collate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import yaml
from monai.data import DataLoader, Dataset, list_data_collate

try:
    from src.data.task1_dataset import (
        Task1Paths,
        discover_inference_cases,
        discover_labeled_cases,
        discover_unlabeled_cases,
        summarize_records,
    )
    from src.data.transforms import (
        build_inference_transforms,
        build_labeled_train_transforms,
        build_unlabeled_train_transforms,
        build_validation_transforms,
    )
except ModuleNotFoundError:
    # Allows running this file directly as:
    # python src/data/dataloaders.py
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))

    from src.data.task1_dataset import (
        Task1Paths,
        discover_inference_cases,
        discover_labeled_cases,
        discover_unlabeled_cases,
        summarize_records,
    )
    from src.data.transforms import (
        build_inference_transforms,
        build_labeled_train_transforms,
        build_unlabeled_train_transforms,
        build_validation_transforms,
    )


Config = Mapping[str, Any]


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def _section(config: Config, name: str) -> Dict[str, Any]:
    """Safely get a config section."""
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Config section '{name}' must be a mapping, got {type(value)}")
    return dict(value)


def _resolve_path(path: str | Path, project_root: str | Path = ".") -> Path:
    """
    Resolve a path relative to project root.

    Absolute paths are preserved.
    Relative paths are interpreted from project root.
    """
    path = Path(path)

    if path.is_absolute():
        return path

    return Path(project_root) / path


def _device_is_cuda(config: Config) -> bool:
    """Return True if config requests CUDA."""
    experiment = _section(config, "experiment")
    device = str(experiment.get("device", "cuda")).lower()
    return device.startswith("cuda") and torch.cuda.is_available()


def get_num_workers(config: Config) -> int:
    """Read number of DataLoader workers."""
    training = _section(config, "training")
    return int(training.get("num_workers", 0))


def get_train_batch_size(config: Config) -> int:
    """Read train batch size."""
    training = _section(config, "training")
    return int(training.get("batch_size", 1))


def get_val_batch_size(config: Config) -> int:
    """
    Read validation batch size.

    For 3D sliding-window inference, batch size 1 is usually safest.
    """
    validation = _section(config, "validation")
    return int(validation.get("batch_size", 1))


def get_project_root(config: Config) -> Path:
    """Read project root from config if available."""
    project = _section(config, "project")
    return Path(project.get("root", "."))


def task1_paths_from_config(config: Config) -> Task1Paths:
    """Create Task1Paths from config data section."""
    project_root = get_project_root(config)
    data = _section(config, "data")

    return Task1Paths(
        train_images_dir=_resolve_path(data.get("train_images_dir", "data/t1_ct/train/images"), project_root),
        train_labels_dir=_resolve_path(data.get("train_labels_dir", "data/t1_ct/train/labels"), project_root),
        unlabeled_images_dir=_resolve_path(
            data.get("unlabeled_images_dir", "data/t1_ct/unlabeled/images"),
            project_root,
        ),
        val_images_dir=_resolve_path(data.get("val_images_dir", data.get("images_dir", "data/t1_ct/val/images")), project_root),
        val_labels_dir=_resolve_path(data.get("val_labels_dir", "data/t1_ct/val/labels"), project_root)
        if data.get("val_labels_dir", None) is not None
        else None,
    )


def build_monai_loader(
    records,
    transform,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
) -> DataLoader:
    """Build a MONAI DataLoader from records and transform."""
    dataset = Dataset(data=records, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=list_data_collate,
    )


def build_labeled_train_loader(config: Config) -> DataLoader:
    """Build labeled supervised training DataLoader."""
    paths = task1_paths_from_config(config)

    records = discover_labeled_cases(
        images_dir=paths.train_images_dir,
        labels_dir=paths.train_labels_dir,
        split="train_labeled",
    )

    transform = build_labeled_train_transforms(config)

    return build_monai_loader(
        records,
        transform,
        batch_size=get_train_batch_size(config),
        shuffle=True,
        num_workers=get_num_workers(config),
        pin_memory=_device_is_cuda(config),
        drop_last=False,
    )


def build_unlabeled_train_loader(
    config: Config,
    *,
    strong: bool = False,
) -> DataLoader:
    """Build unlabeled training DataLoader for future semi-supervised training."""
    paths = task1_paths_from_config(config)

    records = discover_unlabeled_cases(
        images_dir=paths.unlabeled_images_dir,
        split="train_unlabeled",
    )

    transform = build_unlabeled_train_transforms(config, strong=strong)

    return build_monai_loader(
        records,
        transform,
        batch_size=get_train_batch_size(config),
        shuffle=True,
        num_workers=get_num_workers(config),
        pin_memory=_device_is_cuda(config),
        drop_last=False,
    )


def build_validation_loader(
    config: Config,
    *,
    has_labels: bool = False,
) -> DataLoader:
    """Build validation DataLoader."""
    paths = task1_paths_from_config(config)

    records = discover_inference_cases(
        images_dir=paths.val_images_dir,
        split="val",
    )

    transform = build_validation_transforms(config, has_labels=has_labels)

    return build_monai_loader(
        records,
        transform,
        batch_size=get_val_batch_size(config),
        shuffle=False,
        num_workers=get_num_workers(config),
        pin_memory=_device_is_cuda(config),
        drop_last=False,
    )


def build_inference_loader(config: Config) -> DataLoader:
    """Build inference DataLoader from config."""
    project_root = get_project_root(config)
    data = _section(config, "data")
    images_dir = _resolve_path(data.get("images_dir", "data/t1_ct/val/images"), project_root)

    records = discover_inference_cases(
        images_dir=images_dir,
        split="inference",
    )

    transform = build_inference_transforms(config)

    return build_monai_loader(
        records,
        transform,
        batch_size=1,
        shuffle=False,
        num_workers=get_num_workers(config),
        pin_memory=_device_is_cuda(config),
        drop_last=False,
    )


def discover_records_from_config(config: Config) -> Dict[str, Any]:
    """Discover and summarize records without creating DataLoaders."""
    paths = task1_paths_from_config(config)

    labeled = discover_labeled_cases(
        images_dir=paths.train_images_dir,
        labels_dir=paths.train_labels_dir,
        split="train_labeled",
    )

    unlabeled = discover_unlabeled_cases(
        images_dir=paths.unlabeled_images_dir,
        split="train_unlabeled",
    )

    val = discover_inference_cases(
        images_dir=paths.val_images_dir,
        split="val",
    )

    return {
        "train_labeled": labeled,
        "train_unlabeled": unlabeled,
        "val": val,
    }


def build_task1_dataloaders(
    config: Config,
    *,
    include_unlabeled: bool = False,
) -> Dict[str, DataLoader]:
    """
    Build standard Task 1 DataLoaders.

    Returns:
        {
            "train_labeled": DataLoader,
            "val": DataLoader,
            optionally "train_unlabeled": DataLoader
        }
    """
    loaders: Dict[str, DataLoader] = {
        "train_labeled": build_labeled_train_loader(config),
        "val": build_validation_loader(config, has_labels=False),
    }

    if include_unlabeled:
        loaders["train_unlabeled"] = build_unlabeled_train_loader(config, strong=False)

    return loaders


def _tensor_shape(value: Any) -> Optional[list[int]]:
    """Return tensor shape as list if value is a tensor-like object."""
    if hasattr(value, "shape"):
        return list(value.shape)
    return None


def inspect_batch(batch: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a compact batch summary."""
    summary: Dict[str, Any] = {}

    for key, value in batch.items():
        if key.endswith("_meta_dict"):
            continue

        shape = _tensor_shape(value)
        if shape is not None:
            summary[key] = {
                "type": type(value).__name__,
                "shape": shape,
                "dtype": str(getattr(value, "dtype", "unknown")),
            }
        else:
            summary[key] = {
                "type": type(value).__name__,
                "value": value if isinstance(value, (str, int, float, bool)) else str(value)[:120],
            }

    return summary


def parse_args() -> argparse.Namespace:
    """CLI arguments."""
    parser = argparse.ArgumentParser(description="Inspect MVAA Task 1 DataLoaders.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/task1_train.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--print-records",
        action="store_true",
        help="Print discovered record summaries.",
    )
    parser.add_argument(
        "--check-batch",
        action="store_true",
        help="Actually load one training batch and print tensor shapes.",
    )
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Also build unlabeled loader.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    config = load_yaml_config(args.config)

    records = discover_records_from_config(config)
    record_summary = {name: summarize_records(items) for name, items in records.items()}

    print(json.dumps(record_summary, indent=2))

    if args.print_records:
        for name, items in records.items():
            print(f"\n[{name}]")
            for item in items[:3]:
                print(json.dumps(item, indent=2))

    if args.check_batch:
        loaders = build_task1_dataloaders(
            config,
            include_unlabeled=args.include_unlabeled,
        )

        print("\nChecking one labeled training batch...")
        batch = next(iter(loaders["train_labeled"]))
        print(json.dumps(inspect_batch(batch), indent=2))

        if args.include_unlabeled:
            print("\nChecking one unlabeled training batch...")
            batch = next(iter(loaders["train_unlabeled"]))
            print(json.dumps(inspect_batch(batch), indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
