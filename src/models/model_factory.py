#!/usr/bin/env python3
"""
Model factory for MVAA Task 1.

This module builds segmentation models from YAML config files.

Current supported model:
    - unet3d

Future supported models:
    - segresnet
    - swin_unetr
    - multitask_unet3d
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn
import yaml

try:
    from src.models.unet3d import build_unet3d, model_summary
except ModuleNotFoundError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))

    from src.models.unet3d import build_unet3d, model_summary


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
    """Safely read a config section."""
    value = config.get(name, {})
    if value is None:
        return {}

    if not isinstance(value, Mapping):
        raise TypeError(f"Config section '{name}' must be a mapping, got {type(value)}")

    return dict(value)


def build_model_from_config(config: Config) -> nn.Module:
    """
    Build a model from config.

    Expected config:

        model:
          name: unet3d
          in_channels: 1
          out_channels: 2
          model_size: base
          num_res_units: 2
          dropout: 0.0
          norm: INSTANCE

    Returns:
        torch.nn.Module
    """
    model_cfg = _section(config, "model")

    name = str(model_cfg.get("name", "unet3d")).lower()

    in_channels = int(model_cfg.get("in_channels", 1))
    out_channels = int(model_cfg.get("out_channels", model_cfg.get("num_classes", 2)))
    model_size = str(model_cfg.get("model_size", "base"))

    num_res_units = int(model_cfg.get("num_res_units", 2))
    dropout = float(model_cfg.get("dropout", 0.0))
    norm = str(model_cfg.get("norm", "INSTANCE"))

    if name in {"unet3d", "unet", "monai_unet"}:
        return build_unet3d(
            in_channels=in_channels,
            out_channels=out_channels,
            model_size=model_size,
            num_res_units=num_res_units,
            dropout=dropout,
            norm=norm,
        )

    raise ValueError(
        f"Unknown model name '{name}'. "
        "Currently supported models: unet3d."
    )


def get_device_from_config(config: Config, override_device: str | None = None) -> torch.device:
    """
    Resolve torch device from config.

    If config asks for CUDA but CUDA is unavailable, CPU is used safely.
    """
    if override_device is not None:
        requested = override_device
    else:
        experiment_cfg = _section(config, "experiment")
        requested = str(experiment_cfg.get("device", "cuda"))

    requested = requested.lower()

    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested)
        return torch.device("cpu")

    return torch.device(requested)


def build_model_and_device(
    config: Config,
    *,
    override_device: str | None = None,
) -> Tuple[nn.Module, torch.device]:
    """
    Build model and move it to device.

    Returns:
        model, device
    """
    model = build_model_from_config(config)
    device = get_device_from_config(config, override_device=override_device)
    model = model.to(device)

    return model, device


def freeze_model(model: nn.Module) -> nn.Module:
    """Freeze all model parameters."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def unfreeze_model(model: nn.Module) -> nn.Module:
    """Unfreeze all model parameters."""
    for parameter in model.parameters():
        parameter.requires_grad = True
    return model


def check_forward_pass(
    model: nn.Module,
    *,
    device: torch.device,
    in_channels: int,
    out_channels: int,
    patch_size: int,
) -> Dict[str, Any]:
    """Run a small forward pass and validate output shape."""
    model.eval()

    x = torch.randn(
        1,
        in_channels,
        patch_size,
        patch_size,
        patch_size,
        device=device,
    )

    with torch.no_grad():
        y = model(x)

    if y.shape[1] != out_channels:
        raise RuntimeError(
            f"Unexpected output channel count: got {y.shape[1]}, expected {out_channels}"
        )

    return {
        "input_shape": list(x.shape),
        "output_shape": list(y.shape),
    }


def parse_args() -> argparse.Namespace:
    """CLI args for model factory inspection."""
    parser = argparse.ArgumentParser(description="Build MVAA Task 1 model from config.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/task1_train.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override: cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--check-forward",
        action="store_true",
        help="Run a small forward pass.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=64,
        help="Patch size for forward-pass test.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()

    config = load_yaml_config(args.config)
    model, device = build_model_and_device(config, override_device=args.device)

    model_cfg = _section(config, "model")
    in_channels = int(model_cfg.get("in_channels", 1))
    out_channels = int(model_cfg.get("out_channels", model_cfg.get("num_classes", 2)))

    summary = model_summary(model)
    summary["device"] = str(device)
    summary["config"] = args.config
    summary["model_config"] = {
        "name": str(model_cfg.get("name", "unet3d")),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "model_size": str(model_cfg.get("model_size", "base")),
        "num_res_units": int(model_cfg.get("num_res_units", 2)),
        "dropout": float(model_cfg.get("dropout", 0.0)),
        "norm": str(model_cfg.get("norm", "INSTANCE")),
    }

    print(json.dumps(summary, indent=2))

    if args.check_forward:
        shapes = check_forward_pass(
            model,
            device=device,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=args.patch_size,
        )
        print(json.dumps(shapes, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
