#!/usr/bin/env python3
"""
Check MVAA Task 1 submission folder.

Expected structure:
    submission/t1_ct/
    ├── task1_predictions.json
    ├── 0001-pred.nii.gz
    ├── 0002-pred.nii.gz
    └── ...

Expected JSON:
    {
      "cases": [
        {
          "case_id": "0001",
          "segmentation": "0001-pred.nii.gz"
        }
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import nibabel as nib
import numpy as np


def json_safe(obj: Any) -> Any:
    """Convert numpy values into JSON-serializable Python values."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value

    if isinstance(obj, float):
        if math.isnan(obj):
            return "nan"
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj

    return obj


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_nifti_mask(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction mask not found: {path}")

    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)

    unique_values = np.unique(data)
    unique_preview = unique_values[:20].tolist()

    if data.ndim != 3:
        raise ValueError(f"Mask must be 3D, got shape {data.shape}: {path}")

    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError(
            f"Mask must be binary 0/1. Got values {unique_preview}: {path}"
        )

    return {
        "path": str(path),
        "shape": [int(x) for x in data.shape],
        "dtype": str(data.dtype),
        "foreground_voxels": int((data > 0).sum()),
        "unique_values": [int(x) for x in unique_values[:20]],
        "zooms": [float(x) for x in img.header.get_zooms()[:3]],
    }


def check_submission(submission_dir: Path, expected_cases: int | None = None) -> Dict[str, Any]:
    submission_dir = Path(submission_dir)

    if not submission_dir.exists():
        raise FileNotFoundError(f"Submission dir not found: {submission_dir}")

    json_path = submission_dir / "task1_predictions.json"
    payload = load_json(json_path)

    if "cases" not in payload:
        raise ValueError("task1_predictions.json must contain top-level key: cases")

    cases = payload["cases"]

    if not isinstance(cases, list):
        raise TypeError("'cases' must be a list")

    if expected_cases is not None and len(cases) != expected_cases:
        raise ValueError(f"Expected {expected_cases} cases, found {len(cases)}")

    seen_case_ids = set()
    mask_reports: List[Dict[str, Any]] = []

    for idx, item in enumerate(cases, start=1):
        if not isinstance(item, dict):
            raise TypeError(f"Case entry #{idx} must be a dict")

        if "case_id" not in item:
            raise ValueError(f"Case entry #{idx} missing 'case_id'")

        if "segmentation" not in item:
            raise ValueError(f"Case entry #{idx} missing 'segmentation'")

        case_id = str(item["case_id"])
        segmentation = str(item["segmentation"])

        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate case_id: {case_id}")
        seen_case_ids.add(case_id)

        if "/" in segmentation or "\\" in segmentation:
            raise ValueError(
                f"Task 1 segmentation should be relative filename only, got: {segmentation}"
            )

        expected_name = f"{case_id}-pred.nii.gz"
        if segmentation != expected_name:
            print(
                f"Warning: segmentation name '{segmentation}' does not match "
                f"expected '{expected_name}'"
            )

        mask_path = submission_dir / segmentation
        mask_reports.append(check_nifti_mask(mask_path))

    report = {
        "submission_dir": str(submission_dir),
        "json_path": str(json_path),
        "num_cases": int(len(cases)),
        "case_ids": sorted(seen_case_ids),
        "masks": mask_reports,
    }

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check MVAA Task 1 submission folder.")
    parser.add_argument("--submission-dir", type=str, default="submission/t1_ct")
    parser.add_argument("--expected-cases", type=int, default=None)
    parser.add_argument("--save-report", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    report = check_submission(
        submission_dir=Path(args.submission_dir),
        expected_cases=args.expected_cases,
    )

    safe_report = json_safe(report)
    print(json.dumps(safe_report, indent=2))

    if args.save_report:
        out = Path(args.save_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(safe_report, f, indent=2)

    print("Task 1 submission check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
