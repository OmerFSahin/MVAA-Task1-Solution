#!/usr/bin/env python3
"""
Submission utilities for MVAA Task 1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def make_task1_case_record(case_id: str, prediction_suffix: str = "-pred.nii.gz") -> Dict[str, str]:
    return {
        "case_id": str(case_id),
        "segmentation": f"{case_id}{prediction_suffix}",
    }


def write_task1_predictions_json(
    records: Sequence[Mapping[str, str]],
    output_json: str | Path,
) -> None:
    seen = set()
    cases = []

    for record in records:
        case_id = str(record["case_id"])
        segmentation = str(record["segmentation"])

        if case_id in seen:
            raise ValueError(f"Duplicate case_id: {case_id}")
        seen.add(case_id)

        cases.append(
            {
                "case_id": case_id,
                "segmentation": segmentation,
            }
        )

    save_json({"cases": cases}, output_json)


def load_task1_predictions_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if "cases" not in payload:
        raise ValueError("Task 1 predictions JSON must contain top-level key 'cases'.")

    return payload
