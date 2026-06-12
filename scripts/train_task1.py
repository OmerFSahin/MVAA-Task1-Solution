#!/usr/bin/env python3
"""
Command-line entrypoint for MVAA Task 1 training.

Usage:
    python scripts/train_task1.py --config configs/task1_train.yaml

Optional smoke test:
    python scripts/train_task1.py --config configs/task1_train.yaml --epochs 1 --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.train import apply_cli_overrides, load_yaml_config, train


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train MVAA Task 1 segmentation model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/task1_train.yaml",
        help="Path to Task 1 training config.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for number of epochs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override: cpu, cuda, cuda:0.",
    )
    return parser.parse_args()


def main() -> int:
    """Main entrypoint."""
    args = parse_args()

    config = load_yaml_config(args.config)
    config = apply_cli_overrides(config, args)

    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
