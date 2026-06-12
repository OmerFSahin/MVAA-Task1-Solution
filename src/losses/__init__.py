"""
Loss functions for MVAA Task 1.
"""

from src.losses.segmentation_losses import (
    FocalTverskyLoss,
    TverskyLoss,
    WeightedSumLoss,
    build_segmentation_loss_from_config,
)

__all__ = [
    "FocalTverskyLoss",
    "TverskyLoss",
    "WeightedSumLoss",
    "build_segmentation_loss_from_config",
]
