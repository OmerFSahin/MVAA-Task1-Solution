#!/usr/bin/env python3
"""
Segmentation losses for MVAA Task 1.

Task 1 target:
    Mitral valve segmentation in 3D cardiac CT.

Why careful loss design?
    The mitral valve target is small relative to the full CT volume.
    Pure Cross Entropy can be dominated by background voxels.
    Pure Dice can be unstable early in training.
    Boundary/Hausdorff-aware losses can help reduce surface outliers but are
    usually better introduced after a stable baseline exists.

Supported losses:
    - dice
    - dice_ce
    - generalized_dice
    - dice_focal
    - tversky
    - focal_tversky
    - hausdorff_dt          if available in installed MONAI
    - dice_ce_focal_tversky
    - dice_ce_hausdorff     if HausdorffDTLoss is available

Default recommended first baseline:
    dice_ce

Recommended experiment order:
    1. dice_ce
    2. dice_focal
    3. focal_tversky
    4. dice_ce_focal_tversky
    5. dice_ce_hausdorff
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from monai.losses import DiceCELoss, DiceLoss, GeneralizedDiceLoss

try:
    from monai.losses import DiceFocalLoss
except ImportError:  # pragma: no cover
    DiceFocalLoss = None  # type: ignore

try:
    from monai.losses import HausdorffDTLoss
except ImportError:  # pragma: no cover
    HausdorffDTLoss = None  # type: ignore


Config = Mapping[str, Any]


def _section(config: Optional[Config], name: str) -> Dict[str, Any]:
    """Safely read config section."""
    if config is None:
        return {}

    value = config.get(name, {})
    if value is None:
        return {}

    if not isinstance(value, Mapping):
        raise TypeError(f"Config section '{name}' must be a mapping, got {type(value)}")

    return dict(value)


def _bool_value(value: Any) -> bool:
    """Convert config-like value to bool."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}

    return bool(value)


def _prepare_target_indices(target: torch.Tensor) -> torch.Tensor:
    """
    Convert target to class-index tensor.

    Accepts:
        [B, 1, D, H, W]
        [B, D, H, W]

    Returns:
        [B, D, H, W] long tensor.
    """
    if target.ndim == 5 and target.shape[1] == 1:
        target = target[:, 0]

    if target.ndim != 4:
        raise ValueError(
            "Expected target shape [B, 1, D, H, W] or [B, D, H, W], "
            f"got {tuple(target.shape)}"
        )

    return target.long()


def _one_hot_target(
    target: torch.Tensor,
    num_classes: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Convert target to one-hot format [B, C, D, H, W].
    """
    target_indices = _prepare_target_indices(target)
    one_hot = F.one_hot(target_indices, num_classes=num_classes)
    one_hot = one_hot.permute(0, 4, 1, 2, 3).contiguous()
    return one_hot.to(dtype=dtype)


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss for small, imbalanced segmentation targets.

    Important naming:
        alpha_fp weights false positives.
        beta_fn weights false negatives.

    For small structures where missing the valve is harmful, a common choice is:
        alpha_fp = 0.3
        beta_fn = 0.7

    This emphasizes recall by penalizing false negatives more strongly.
    """

    def __init__(
        self,
        *,
        alpha_fp: float = 0.3,
        beta_fn: float = 0.7,
        gamma: float = 4.0 / 3.0,
        smooth: float = 1e-6,
        include_background: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if alpha_fp < 0 or beta_fn < 0:
            raise ValueError("alpha_fp and beta_fn must be non-negative.")

        if gamma <= 0:
            raise ValueError("gamma must be positive.")

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: mean, sum, none.")

        self.alpha_fp = float(alpha_fp)
        self.beta_fn = float(beta_fn)
        self.gamma = float(gamma)
        self.smooth = float(smooth)
        self.include_background = bool(include_background)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:
                [B, C, D, H, W]
            target:
                [B, 1, D, H, W] or [B, D, H, W]

        Returns:
            Scalar loss by default.
        """
        if logits.ndim != 5:
            raise ValueError(f"Expected logits shape [B, C, D, H, W], got {tuple(logits.shape)}")

        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        target_oh = _one_hot_target(target, num_classes=num_classes, dtype=probs.dtype)

        if not self.include_background and num_classes > 1:
            probs = probs[:, 1:]
            target_oh = target_oh[:, 1:]

        reduce_dims = tuple(range(2, probs.ndim))

        true_pos = torch.sum(probs * target_oh, dim=reduce_dims)
        false_pos = torch.sum(probs * (1.0 - target_oh), dim=reduce_dims)
        false_neg = torch.sum((1.0 - probs) * target_oh, dim=reduce_dims)

        tversky = (true_pos + self.smooth) / (
            true_pos
            + self.alpha_fp * false_pos
            + self.beta_fn * false_neg
            + self.smooth
        )

        loss = torch.pow(1.0 - tversky, self.gamma)

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss


class TverskyLoss(FocalTverskyLoss):
    """
    Tversky loss.

    Equivalent to FocalTverskyLoss with gamma=1.0.
    """

    def __init__(
        self,
        *,
        alpha_fp: float = 0.3,
        beta_fn: float = 0.7,
        smooth: float = 1e-6,
        include_background: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__(
            alpha_fp=alpha_fp,
            beta_fn=beta_fn,
            gamma=1.0,
            smooth=smooth,
            include_background=include_background,
            reduction=reduction,
        )


class WeightedSumLoss(nn.Module):
    """
    Weighted sum of multiple loss functions.

    Example:
        loss = 1.0 * DiceCE + 0.5 * FocalTversky
    """

    def __init__(self, losses: Mapping[str, nn.Module], weights: Mapping[str, float]) -> None:
        super().__init__()

        if not losses:
            raise ValueError("WeightedSumLoss requires at least one loss.")

        self.losses = nn.ModuleDict(dict(losses))
        self.weights = {name: float(weights.get(name, 1.0)) for name in self.losses.keys()}

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = logits.new_tensor(0.0)

        for name, loss_fn in self.losses.items():
            total = total + self.weights[name] * loss_fn(logits, target)

        return total


def build_dice_loss(loss_cfg: Mapping[str, Any]) -> nn.Module:
    """Build foreground-focused Dice loss."""
    return DiceLoss(
        include_background=_bool_value(loss_cfg.get("include_background", False)),
        to_onehot_y=True,
        softmax=True,
        smooth_nr=float(loss_cfg.get("smooth_nr", 1e-5)),
        smooth_dr=float(loss_cfg.get("smooth_dr", 1e-5)),
    )


def build_dice_ce_loss(loss_cfg: Mapping[str, Any]) -> nn.Module:
    """Build Dice + Cross Entropy loss."""
    return DiceCELoss(
        include_background=_bool_value(loss_cfg.get("include_background", False)),
        to_onehot_y=True,
        softmax=True,
        lambda_dice=float(loss_cfg.get("dice_weight", 1.0)),
        lambda_ce=float(loss_cfg.get("ce_weight", 1.0)),
        smooth_nr=float(loss_cfg.get("smooth_nr", 1e-5)),
        smooth_dr=float(loss_cfg.get("smooth_dr", 1e-5)),
    )


def build_generalized_dice_loss(loss_cfg: Mapping[str, Any]) -> nn.Module:
    """Build Generalized Dice loss for severe class imbalance."""
    return GeneralizedDiceLoss(
        include_background=_bool_value(loss_cfg.get("include_background", False)),
        to_onehot_y=True,
        softmax=True,
        w_type=str(loss_cfg.get("w_type", "square")),
        smooth_nr=float(loss_cfg.get("smooth_nr", 1e-5)),
        smooth_dr=float(loss_cfg.get("smooth_dr", 1e-5)),
    )


def build_dice_focal_loss(loss_cfg: Mapping[str, Any]) -> nn.Module:
    """Build Dice + Focal loss."""
    if DiceFocalLoss is None:
        raise ImportError(
            "DiceFocalLoss is not available in your installed MONAI version. "
            "Upgrade MONAI or use focal_tversky instead."
        )

    return DiceFocalLoss(
        include_background=_bool_value(loss_cfg.get("include_background", False)),
        to_onehot_y=True,
        softmax=True,
        lambda_dice=float(loss_cfg.get("dice_weight", 1.0)),
        lambda_focal=float(loss_cfg.get("focal_weight", 1.0)),
        gamma=float(loss_cfg.get("focal_gamma", 2.0)),
        smooth_nr=float(loss_cfg.get("smooth_nr", 1e-5)),
        smooth_dr=float(loss_cfg.get("smooth_dr", 1e-5)),
    )


def build_tversky_loss(loss_cfg: Mapping[str, Any]) -> nn.Module:
    """Build Tversky loss."""
    return TverskyLoss(
        alpha_fp=float(loss_cfg.get("alpha_fp", 0.3)),
        beta_fn=float(loss_cfg.get("beta_fn", 0.7)),
        smooth=float(loss_cfg.get("smooth", 1e-6)),
        include_background=_bool_value(loss_cfg.get("include_background", False)),
    )


def build_focal_tversky_loss(loss_cfg: Mapping[str, Any]) -> nn.Module:
    """Build Focal Tversky loss."""
    return FocalTverskyLoss(
        alpha_fp=float(loss_cfg.get("alpha_fp", 0.3)),
        beta_fn=float(loss_cfg.get("beta_fn", 0.7)),
        gamma=float(loss_cfg.get("tversky_gamma", 4.0 / 3.0)),
        smooth=float(loss_cfg.get("smooth", 1e-6)),
        include_background=_bool_value(loss_cfg.get("include_background", False)),
    )


def build_hausdorff_dt_loss(loss_cfg: Mapping[str, Any]) -> nn.Module:
    """
    Build Hausdorff Distance Transform loss if available.

    This is useful for HD-sensitive fine-tuning.
    It can be computationally heavier than DiceCE.
    """
    if HausdorffDTLoss is None:
        raise ImportError(
            "HausdorffDTLoss is not available in your installed MONAI version. "
            "Use dice_ce / focal_tversky for now, or upgrade MONAI."
        )

    return HausdorffDTLoss(
        include_background=_bool_value(loss_cfg.get("include_background", False)),
        to_onehot_y=True,
        softmax=True,
        alpha=float(loss_cfg.get("hausdorff_alpha", 2.0)),
    )


def build_segmentation_loss_from_config(config: Optional[Config] = None) -> nn.Module:
    """
    Build segmentation loss from config.

    Example config:

        loss:
          name: dice_ce
          include_background: false
          dice_weight: 1.0
          ce_weight: 1.0
    """
    loss_cfg = _section(config, "loss")
    name = str(loss_cfg.get("name", "dice_ce")).lower()

    if name == "dice":
        return build_dice_loss(loss_cfg)

    if name in {"dice_ce", "dicece"}:
        return build_dice_ce_loss(loss_cfg)

    if name in {"generalized_dice", "gdl"}:
        return build_generalized_dice_loss(loss_cfg)

    if name in {"dice_focal", "dicefocal"}:
        return build_dice_focal_loss(loss_cfg)

    if name == "tversky":
        return build_tversky_loss(loss_cfg)

    if name in {"focal_tversky", "focaltversky"}:
        return build_focal_tversky_loss(loss_cfg)

    if name in {"hausdorff", "hausdorff_dt", "hd_dt"}:
        return build_hausdorff_dt_loss(loss_cfg)

    if name in {"dice_ce_focal_tversky", "dicece_focaltversky"}:
        dice_ce = build_dice_ce_loss(loss_cfg)
        focal_tversky = build_focal_tversky_loss(loss_cfg)

        return WeightedSumLoss(
            losses={
                "dice_ce": dice_ce,
                "focal_tversky": focal_tversky,
            },
            weights={
                "dice_ce": float(loss_cfg.get("dice_ce_total_weight", 1.0)),
                "focal_tversky": float(loss_cfg.get("focal_tversky_weight", 0.5)),
            },
        )

    if name in {"dice_ce_hausdorff", "dicece_hausdorff"}:
        dice_ce = build_dice_ce_loss(loss_cfg)
        hausdorff = build_hausdorff_dt_loss(loss_cfg)

        return WeightedSumLoss(
            losses={
                "dice_ce": dice_ce,
                "hausdorff_dt": hausdorff,
            },
            weights={
                "dice_ce": float(loss_cfg.get("dice_ce_total_weight", 1.0)),
                "hausdorff_dt": float(loss_cfg.get("hausdorff_weight", 0.1)),
            },
        )

    raise ValueError(
        f"Unknown loss name '{name}'. Supported losses: "
        "dice, dice_ce, generalized_dice, dice_focal, tversky, "
        "focal_tversky, hausdorff_dt, dice_ce_focal_tversky, dice_ce_hausdorff."
    )


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML config."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def parse_args() -> argparse.Namespace:
    """CLI args."""
    parser = argparse.ArgumentParser(description="Inspect/test MVAA Task 1 segmentation loss.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/task1_train.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--loss-name",
        type=str,
        default=None,
        help="Override loss.name for quick testing.",
    )
    parser.add_argument(
        "--check-forward",
        action="store_true",
        help="Run a dummy forward test.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for dummy forward test.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()

    config = load_yaml_config(args.config)

    if args.loss_name is not None:
        config = dict(config)
        config["loss"] = dict(config.get("loss", {}))
        config["loss"]["name"] = args.loss_name

    loss_fn = build_segmentation_loss_from_config(config)

    print(
        json.dumps(
            {
                "loss_class": loss_fn.__class__.__name__,
                "loss_config": dict(config.get("loss", {})),
            },
            indent=2,
        )
    )

    if args.check_forward:
        device = torch.device(args.device)
        logits = torch.randn(2, 2, 32, 32, 32, device=device)

        # Tiny synthetic foreground cube.
        target = torch.zeros(2, 1, 32, 32, 32, device=device, dtype=torch.long)
        target[:, :, 12:20, 12:20, 12:20] = 1

        loss_value = loss_fn(logits, target)

        print(
            json.dumps(
                {
                    "dummy_logits_shape": list(logits.shape),
                    "dummy_target_shape": list(target.shape),
                    "loss_value": float(loss_value.detach().cpu()),
                },
                indent=2,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
