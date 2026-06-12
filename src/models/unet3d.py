#!/usr/bin/env python3
"""
3D U-Net model for MVAA Task 1.

Task:
    Mitral Valve Segmentation from 3D cardiac CT volumes.

Why this model first?
    A well-configured 3D U-Net is the safest first baseline for medical
    volumetric segmentation. It is simple, robust, memory-controllable,
    and compatible with patch-based training and sliding-window inference.

Input:
    image tensor: [B, 1, D, H, W]

Output:
    logits tensor: [B, C, D, H, W]

Default Task 1 setup:
    in_channels: 1
    out_channels: 2
        class 0 = background
        class 1 = mitral valve
"""

from __future__ import annotations

import argparse
import json
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from monai.networks.nets import UNet


ChannelTuple = Tuple[int, ...]
StrideTuple = Tuple[int, ...]


def get_unet3d_channels(model_size: str = "base") -> ChannelTuple:
    """
    Return channel configuration for 3D U-Net.

    The goal is to make model capacity easy to control on HPC/GPU.

    tiny:
        Debugging / very low memory.
    small:
        Safer on limited GPU memory.
    base:
        Main first baseline.
    large:
        Later experiment if GPU memory allows.
    """
    model_size = model_size.lower()

    if model_size == "tiny":
        return (8, 16, 32, 64)

    if model_size == "small":
        return (16, 32, 64, 128)

    if model_size == "base":
        return (16, 32, 64, 128, 256)

    if model_size == "large":
        return (32, 64, 128, 256, 512)

    raise ValueError(
        f"Unknown model_size='{model_size}'. "
        "Expected one of: tiny, small, base, large."
    )


def get_unet3d_strides(channels: Sequence[int]) -> StrideTuple:
    """
    Return stride configuration for MONAI UNet.

    MONAI UNet expects:
        len(strides) == len(channels) - 1

    For ROI size [128, 128, 128], repeated stride-2 downsampling is safe.
    """
    if len(channels) < 2:
        raise ValueError("UNet requires at least two channel levels.")

    return tuple(2 for _ in range(len(channels) - 1))


def parse_int_tuple(value: Optional[str]) -> Optional[Tuple[int, ...]]:
    """
    Parse comma-separated int tuple.

    Example:
        "16,32,64,128,256" -> (16, 32, 64, 128, 256)
    """
    if value is None:
        return None

    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        return None

    return tuple(int(item) for item in items)


def validate_channels_and_strides(
    channels: Sequence[int],
    strides: Sequence[int],
) -> None:
    """Validate MONAI UNet channel/stride relationship."""
    if len(channels) < 2:
        raise ValueError(f"channels must have at least two levels, got: {channels}")

    if len(strides) != len(channels) - 1:
        raise ValueError(
            "Invalid UNet configuration: "
            f"len(strides) must equal len(channels)-1. "
            f"Got channels={tuple(channels)}, strides={tuple(strides)}"
        )

    if any(c <= 0 for c in channels):
        raise ValueError(f"All channels must be positive, got: {channels}")

    if any(s <= 0 for s in strides):
        raise ValueError(f"All strides must be positive, got: {strides}")


def build_unet3d(
    *,
    in_channels: int = 1,
    out_channels: int = 2,
    model_size: str = "base",
    channels: Optional[Sequence[int]] = None,
    strides: Optional[Sequence[int]] = None,
    num_res_units: int = 2,
    dropout: float = 0.0,
    norm: str = "INSTANCE",
) -> nn.Module:
    """
    Build MONAI 3D U-Net.

    Args:
        in_channels:
            Number of input channels. For CT, this is 1.
        out_channels:
            Number of output classes. For binary segmentation, this is 2.
        model_size:
            One of tiny, small, base, large. Ignored if channels is provided.
        channels:
            Optional explicit channel tuple.
        strides:
            Optional explicit stride tuple.
        num_res_units:
            Number of residual units per level.
        dropout:
            Dropout probability.
        norm:
            Normalization type. INSTANCE is a strong default for 3D medical
            segmentation with small batch sizes.

    Returns:
        torch.nn.Module
    """
    if channels is None:
        channels = get_unet3d_channels(model_size)
    else:
        channels = tuple(int(c) for c in channels)

    if strides is None:
        strides = get_unet3d_strides(channels)
    else:
        strides = tuple(int(s) for s in strides)

    validate_channels_and_strides(channels, strides)

    model = UNet(
        spatial_dims=3,
        in_channels=int(in_channels),
        out_channels=int(out_channels),
        channels=tuple(channels),
        strides=tuple(strides),
        num_res_units=int(num_res_units),
        dropout=float(dropout),
        norm=norm,
    )

    return model


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    return sum(p.numel() for p in model.parameters())


def model_summary(model: nn.Module) -> dict:
    """Return compact model summary."""
    return {
        "model_class": model.__class__.__name__,
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "total_parameters": count_parameters(model, trainable_only=False),
    }


def forward_shape_test(
    model: nn.Module,
    *,
    in_channels: int,
    out_channels: int,
    patch_size: int,
    device: torch.device,
) -> dict:
    """Run a small forward-pass test and return input/output shapes."""
    model = model.to(device)
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
            f"Unexpected output channels. Got {y.shape[1]}, expected {out_channels}."
        )

    return {
        "input_shape": list(x.shape),
        "output_shape": list(y.shape),
    }


def parse_args() -> argparse.Namespace:
    """CLI arguments for quick model testing."""
    parser = argparse.ArgumentParser(description="Build and test MVAA Task 1 3D U-Net.")
    parser.add_argument("--model-size", type=str, default="base")
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=2)
    parser.add_argument("--channels", type=str, default=None)
    parser.add_argument("--strides", type=str, default=None)
    parser.add_argument("--num-res-units", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--norm", type=str, default="INSTANCE")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--check-forward", action="store_true")
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()

    channels = parse_int_tuple(args.channels)
    strides = parse_int_tuple(args.strides)

    model = build_unet3d(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        model_size=args.model_size,
        channels=channels,
        strides=strides,
        num_res_units=args.num_res_units,
        dropout=args.dropout,
        norm=args.norm,
    )

    summary = model_summary(model)
    summary["model_size"] = args.model_size
    summary["channels"] = list(channels or get_unet3d_channels(args.model_size))
    summary["strides"] = list(strides or get_unet3d_strides(channels or get_unet3d_channels(args.model_size)))
    summary["num_res_units"] = args.num_res_units
    summary["dropout"] = args.dropout
    summary["norm"] = args.norm

    print(json.dumps(summary, indent=2))

    if args.check_forward:
        device = torch.device(args.device)
        shapes = forward_shape_test(
            model,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            patch_size=args.patch_size,
            device=device,
        )
        print(json.dumps(shapes, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
