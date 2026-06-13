#!/usr/bin/env python3
"""
Generate MVAA Task 1 prediction masks and task1_predictions.json.

Usage:
    python scripts/generate_task1_predictions.py \
      --config configs/task1_inference.yaml

Smoke test:
    python scripts/generate_task1_predictions.py \
      --config configs/task1_inference.yaml \
      --checkpoint runs/task1_baseline/checkpoints/best_model.pt \
      --images-dir data/t1_ct/val/images \
      --output-dir submission/t1_ct \
      --device cpu \
      --max-cases 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.infer import load_yaml_config, run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MVAA Task 1 predictions.")
    parser.add_argument("--config", type=str, default="configs/task1_inference.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--images-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml_config(args.config)

    run_inference(
        config,
        max_cases=args.max_cases,
        override_checkpoint=args.checkpoint,
        override_images_dir=args.images_dir,
        override_output_dir=args.output_dir,
        override_device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
