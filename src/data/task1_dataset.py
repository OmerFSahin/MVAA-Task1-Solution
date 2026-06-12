#!/usr/bin/env python3
"""
Dataset discovery utilities for MVAA Task 1.

Task 1:
    Mitral Valve Segmentation and Landmark Localization in CT scans.

This file is intentionally responsible only for:
    - discovering Task 1 CT image files,
    - matching labeled images with segmentation masks,
    - listing unlabeled training images,
    - listing validation/test images,
    - returning clean dictionaries compatible with MONAI Dataset.

Transform and DataLoader logic should stay in:
    - src/data/transforms.py
    - src/data/dataloaders.py

Expected project data layout:

    data/t1_ct/
      train/
        images/
          0001.nii.gz
          ...
        labels/
          0001-seg.nii.gz
          ...
      unlabeled/
        images/
          0222.nii.gz
          ...
      val/
        images/
          0001.nii.gz
          ...

The returned records use MONAI-friendly keys:

    labeled case:
        {
            "case_id": "0001",
            "image": ".../0001.nii.gz",
            "label": ".../0001-seg.nii.gz",
            "image_path": ".../0001.nii.gz",
            "label_path": ".../0001-seg.nii.gz",
            "split": "train_labeled"
        }

    unlabeled / inference case:
        {
            "case_id": "0222",
            "image": ".../0222.nii.gz",
            "image_path": ".../0222.nii.gz",
            "split": "train_unlabeled"
        }
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


NIFTI_GZ_SUFFIX = ".nii.gz"
SEGMENTATION_SUFFIX = "-seg.nii.gz"


@dataclass(frozen=True)
class Task1Paths:
    """Container for Task 1 data directories."""

    train_images_dir: Path
    train_labels_dir: Path
    unlabeled_images_dir: Path
    val_images_dir: Path
    val_labels_dir: Optional[Path] = None

    @classmethod
    def from_task_root(cls, task_root: str | Path = "data/t1_ct") -> "Task1Paths":
        """Create default paths from the Task 1 root directory."""
        root = Path(task_root)
        return cls(
            train_images_dir=root / "train" / "images",
            train_labels_dir=root / "train" / "labels",
            unlabeled_images_dir=root / "unlabeled" / "images",
            val_images_dir=root / "val" / "images",
            val_labels_dir=root / "val" / "labels",
        )


def _as_path(path: str | Path) -> Path:
    """Convert input path to pathlib.Path."""
    return path if isinstance(path, Path) else Path(path)


def _check_directory(path: str | Path, description: str) -> Path:
    """Validate that a directory exists."""
    path = _as_path(path)

    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")

    if not path.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {path}")

    return path


def is_nifti_gz(path: str | Path) -> bool:
    """Return True if file name ends with .nii.gz."""
    return _as_path(path).name.endswith(NIFTI_GZ_SUFFIX)


def case_id_from_image_path(path: str | Path) -> str:
    """
    Extract case_id from a Task 1 image path.

    Example:
        0001.nii.gz -> 0001
    """
    path = _as_path(path)
    name = path.name

    if not name.endswith(NIFTI_GZ_SUFFIX):
        raise ValueError(f"Expected a .nii.gz file, got: {name}")

    return name[: -len(NIFTI_GZ_SUFFIX)]


def case_id_from_label_path(path: str | Path) -> str:
    """
    Extract case_id from a Task 1 label path.

    Example:
        0001-seg.nii.gz -> 0001
    """
    path = _as_path(path)
    name = path.name

    if not name.endswith(SEGMENTATION_SUFFIX):
        raise ValueError(f"Expected a *-seg.nii.gz file, got: {name}")

    return name[: -len(SEGMENTATION_SUFFIX)]


def expected_label_name(case_id: str) -> str:
    """Return expected segmentation label file name for a case."""
    return f"{case_id}{SEGMENTATION_SUFFIX}"


def expected_label_path(labels_dir: str | Path, case_id: str) -> Path:
    """Return expected label path for a given case_id."""
    return _as_path(labels_dir) / expected_label_name(case_id)


def collect_nii_gz_files(folder: str | Path, recursive: bool = False) -> List[Path]:
    """
    Collect .nii.gz files from a directory.

    Args:
        folder:
            Directory containing NIfTI files.
        recursive:
            If True, search recursively.

    Returns:
        Sorted list of .nii.gz file paths.
    """
    folder = _check_directory(folder, "NIfTI folder")

    pattern = "**/*.nii.gz" if recursive else "*.nii.gz"
    files = sorted(p for p in folder.glob(pattern) if p.is_file() and is_nifti_gz(p))

    return files


def discover_labeled_cases(
    images_dir: str | Path,
    labels_dir: str | Path,
    *,
    split: str = "train_labeled",
    require_labels: bool = True,
) -> List[Dict[str, str]]:
    """
    Discover labeled Task 1 image-label pairs.

    Expected naming:
        image: 0001.nii.gz
        label: 0001-seg.nii.gz

    Args:
        images_dir:
            Directory containing labeled CT images.
        labels_dir:
            Directory containing segmentation masks.
        split:
            Split name to store inside each record.
        require_labels:
            If True, raise an error when any label is missing.
            If False, skip cases with missing labels.

    Returns:
        List of MONAI-friendly dictionaries.
    """
    images_dir = _check_directory(images_dir, "Labeled image directory")
    labels_dir = _check_directory(labels_dir, "Label directory")

    image_files = collect_nii_gz_files(images_dir)
    records: List[Dict[str, str]] = []
    missing: List[Tuple[str, Path]] = []

    for image_path in image_files:
        case_id = case_id_from_image_path(image_path)
        label_path = expected_label_path(labels_dir, case_id)

        if not label_path.exists():
            missing.append((case_id, label_path))
            if require_labels:
                continue
            continue

        records.append(
            {
                "case_id": case_id,
                "image": str(image_path),
                "label": str(label_path),
                "image_path": str(image_path),
                "label_path": str(label_path),
                "split": split,
            }
        )

    if missing and require_labels:
        preview = "\n".join(f"  {case_id}: {path}" for case_id, path in missing[:10])
        raise FileNotFoundError(
            "Missing segmentation labels for labeled cases.\n"
            f"Missing count: {len(missing)}\n"
            f"Examples:\n{preview}"
        )

    _assert_unique_case_ids(records, context=f"labeled split '{split}'")
    return records


def discover_unlabeled_cases(
    images_dir: str | Path,
    *,
    split: str = "train_unlabeled",
) -> List[Dict[str, str]]:
    """
    Discover unlabeled Task 1 CT images.

    Returns records without a label key.
    """
    images_dir = _check_directory(images_dir, "Unlabeled image directory")

    records: List[Dict[str, str]] = []
    for image_path in collect_nii_gz_files(images_dir):
        case_id = case_id_from_image_path(image_path)
        records.append(
            {
                "case_id": case_id,
                "image": str(image_path),
                "image_path": str(image_path),
                "split": split,
            }
        )

    _assert_unique_case_ids(records, context=f"unlabeled split '{split}'")
    return records


def discover_inference_cases(
    images_dir: str | Path,
    *,
    split: str = "val",
) -> List[Dict[str, str]]:
    """
    Discover inference/validation/test images.

    This is used for validation-only submission generation,
    hidden test inference, or any image-only prediction phase.
    """
    images_dir = _check_directory(images_dir, "Inference image directory")

    records: List[Dict[str, str]] = []
    for image_path in collect_nii_gz_files(images_dir):
        case_id = case_id_from_image_path(image_path)
        records.append(
            {
                "case_id": case_id,
                "image": str(image_path),
                "image_path": str(image_path),
                "split": split,
            }
        )

    _assert_unique_case_ids(records, context=f"inference split '{split}'")
    return records


def discover_optional_labeled_inference_cases(
    images_dir: str | Path,
    labels_dir: Optional[str | Path],
    *,
    split: str = "val",
) -> List[Dict[str, str]]:
    """
    Discover inference cases and attach labels if labels exist.

    Useful for local validation if labels are available.
    For challenge validation-only images, labels may not be provided.
    """
    records = discover_inference_cases(images_dir, split=split)

    if labels_dir is None:
        return records

    labels_dir = _as_path(labels_dir)
    if not labels_dir.exists() or not labels_dir.is_dir():
        return records

    for record in records:
        case_id = record["case_id"]
        label_path = expected_label_path(labels_dir, case_id)
        if label_path.exists():
            record["label"] = str(label_path)
            record["label_path"] = str(label_path)

    return records


def discover_task1_data(paths: Task1Paths) -> Dict[str, List[Dict[str, str]]]:
    """
    Discover all standard Task 1 splits from Task1Paths.

    Returns:
        {
            "train_labeled": [...],
            "train_unlabeled": [...],
            "val": [...]
        }
    """
    train_labeled = discover_labeled_cases(
        images_dir=paths.train_images_dir,
        labels_dir=paths.train_labels_dir,
        split="train_labeled",
    )

    train_unlabeled = discover_unlabeled_cases(
        images_dir=paths.unlabeled_images_dir,
        split="train_unlabeled",
    )

    val = discover_optional_labeled_inference_cases(
        images_dir=paths.val_images_dir,
        labels_dir=paths.val_labels_dir,
        split="val",
    )

    return {
        "train_labeled": train_labeled,
        "train_unlabeled": train_unlabeled,
        "val": val,
    }


def summarize_records(records: Sequence[Dict[str, str]]) -> Dict[str, object]:
    """Return a compact summary for a list of records."""
    case_ids = [r["case_id"] for r in records]

    return {
        "count": len(records),
        "first_case_ids": case_ids[:5],
        "last_case_ids": case_ids[-5:] if case_ids else [],
        "has_labels": all("label" in r for r in records) if records else False,
    }


def summarize_task1_data(splits: Dict[str, Sequence[Dict[str, str]]]) -> Dict[str, Dict[str, object]]:
    """Summarize all discovered Task 1 splits."""
    return {name: summarize_records(records) for name, records in splits.items()}


def _assert_unique_case_ids(records: Sequence[Dict[str, str]], *, context: str) -> None:
    """Raise an error if duplicate case_ids are found."""
    seen = set()
    duplicates = set()

    for record in records:
        case_id = record["case_id"]
        if case_id in seen:
            duplicates.add(case_id)
        seen.add(case_id)

    if duplicates:
        dup_preview = ", ".join(sorted(duplicates)[:20])
        raise ValueError(f"Duplicate case_id values found in {context}: {dup_preview}")


def print_split_examples(splits: Dict[str, Sequence[Dict[str, str]]], max_examples: int = 3) -> None:
    """Print a few examples from each split."""
    for split_name, records in splits.items():
        print(f"\n[{split_name}] count={len(records)}")
        for record in list(records)[:max_examples]:
            print(json.dumps(record, indent=2))


def parse_args() -> argparse.Namespace:
    """CLI args for quick dataset inspection."""
    parser = argparse.ArgumentParser(description="Inspect MVAA Task 1 dataset structure.")
    parser.add_argument(
        "--task-root",
        type=str,
        default="data/t1_ct",
        help="Task 1 root directory. Default: data/t1_ct",
    )
    parser.add_argument(
        "--print-examples",
        action="store_true",
        help="Print example records from each split.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint for sanity checking dataset discovery."""
    args = parse_args()

    paths = Task1Paths.from_task_root(args.task_root)
    splits = discover_task1_data(paths)
    summary = summarize_task1_data(splits)

    print(json.dumps(summary, indent=2))

    if args.print_examples:
        print_split_examples(splits)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
