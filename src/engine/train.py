#!/usr/bin/env python3
"""
Training engine for MVAA Task 1.

Task:
    Mitral valve segmentation from 3D cardiac CT volumes.

Strategy:
    - Use labeled Task 1 data only for first supervised baseline.
    - Split the 27 labeled cases into internal train/validation subsets.
    - Train with patch-based sampling.
    - Validate with full-volume sliding-window inference.
    - Save best checkpoint according to internal validation Dice.

Why internal validation?
    Challenge validation images do not include labels. Therefore, to track
    model quality during development, we split the labeled training set.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from monai.data import DataLoader, Dataset, list_data_collate
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

try:
    from src.data.task1_dataset import discover_labeled_cases, summarize_records
    from src.data.transforms import build_labeled_train_transforms, build_validation_transforms
    from src.losses import build_segmentation_loss_from_config
    from src.models.model_factory import build_model_and_device
except ModuleNotFoundError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))

    from src.data.task1_dataset import discover_labeled_cases, summarize_records
    from src.data.transforms import build_labeled_train_transforms, build_validation_transforms
    from src.losses import build_segmentation_loss_from_config
    from src.models.model_factory import build_model_and_device


Config = Mapping[str, Any]


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML config file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def save_json(data: Any, path: str | Path) -> None:
    """Save data as pretty JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _section(config: Config, name: str) -> Dict[str, Any]:
    """Safely read a config section."""
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


def resolve_path(path: str | Path, project_root: str | Path = ".") -> Path:
    """Resolve relative path from project root."""
    path = Path(path)

    if path.is_absolute():
        return path

    return Path(project_root) / path


def get_project_root(config: Config) -> Path:
    """Read project root from config."""
    project = _section(config, "project")
    return Path(project.get("root", "."))


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


def get_output_dirs(config: Config) -> Dict[str, Path]:
    """Create and return output directories."""
    experiment = _section(config, "experiment")
    output_dir = Path(experiment.get("output_dir", "runs/task1_baseline"))

    dirs = {
        "output": output_dir,
        "checkpoints": output_dir / "checkpoints",
        "logs": output_dir / "logs",
        "splits": output_dir / "splits",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def get_training_params(config: Config) -> Dict[str, Any]:
    """Read training params with safe defaults."""
    training = _section(config, "training")

    return {
        "epochs": int(training.get("epochs", 200)),
        "batch_size": int(training.get("batch_size", 1)),
        "num_workers": int(training.get("num_workers", 0)),
        "learning_rate": float(training.get("learning_rate", 1e-3)),
        "weight_decay": float(training.get("weight_decay", 1e-5)),
        "use_amp": _bool_value(training.get("use_amp", True)),
        "grad_clip_norm": float(training.get("grad_clip_norm", 12.0)),
    }


def get_validation_params(config: Config) -> Dict[str, Any]:
    """Read validation params with safe defaults."""
    validation = _section(config, "validation")
    inference = _section(config, "inference")
    training = _section(config, "training")

    return {
        "internal_val_fraction": float(validation.get("internal_val_fraction", 0.2)),
        "internal_val_min_cases": int(validation.get("internal_val_min_cases", 5)),
        "val_interval": int(validation.get("val_interval", 2)),
        "roi_size": tuple(validation.get("roi_size", inference.get("roi_size", training.get("roi_size", [128, 128, 128])))),
        "sw_batch_size": int(validation.get("sw_batch_size", inference.get("sw_batch_size", 4))),
        "sliding_window_overlap": float(validation.get("sliding_window_overlap", inference.get("sliding_window_overlap", 0.25))),
        "compute_full_surface_metrics": _bool_value(validation.get("compute_full_surface_metrics", False)),
    }


def get_checkpoint_params(config: Config) -> Dict[str, Any]:
    """Read checkpoint params with safe defaults."""
    checkpoint = _section(config, "checkpoint")

    return {
        "save_best": _bool_value(checkpoint.get("save_best", True)),
        "save_last": _bool_value(checkpoint.get("save_last", True)),
        "monitor_metric": str(checkpoint.get("monitor_metric", "dice")),
        "mode": str(checkpoint.get("mode", "max")).lower(),
        "resume_path": checkpoint.get("resume_path", None),
    }


def discover_labeled_records_from_config(config: Config) -> List[Dict[str, str]]:
    """Discover labeled Task 1 records from config."""
    project_root = get_project_root(config)
    data = _section(config, "data")

    train_images_dir = resolve_path(data.get("train_images_dir", "data/t1_ct/train/images"), project_root)
    train_labels_dir = resolve_path(data.get("train_labels_dir", "data/t1_ct/train/labels"), project_root)

    return discover_labeled_cases(
        images_dir=train_images_dir,
        labels_dir=train_labels_dir,
        split="train_labeled",
    )


def split_labeled_records(
    records: Sequence[Dict[str, str]],
    *,
    val_fraction: float,
    min_val_cases: int,
    seed: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Split labeled records into train and internal validation subsets.

    For 27 labeled cases and val_fraction=0.2:
        val_count = max(5, round(27 * 0.2)) = 5
        train_count = 22
    """
    if not records:
        raise ValueError("No labeled records found.")

    records = list(records)
    rng = random.Random(seed)
    rng.shuffle(records)

    val_count = max(min_val_cases, int(round(len(records) * val_fraction)))
    val_count = min(val_count, len(records) - 1)

    val_records = sorted(records[:val_count], key=lambda r: r["case_id"])
    train_records = sorted(records[val_count:], key=lambda r: r["case_id"])

    if not train_records:
        raise ValueError("Internal split produced empty training set.")

    if not val_records:
        raise ValueError("Internal split produced empty validation set.")

    return train_records, val_records


def build_train_loader(
    records: Sequence[Dict[str, str]],
    config: Config,
    *,
    device: torch.device,
) -> DataLoader:
    """Build supervised training loader."""
    params = get_training_params(config)
    transform = build_labeled_train_transforms(config)

    dataset = Dataset(data=list(records), transform=transform)

    return DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=list_data_collate,
    )


def build_internal_val_loader(
    records: Sequence[Dict[str, str]],
    config: Config,
    *,
    device: torch.device,
) -> DataLoader:
    """Build internal validation loader with labels."""
    params = get_training_params(config)
    transform = build_validation_transforms(config, has_labels=True)

    dataset = Dataset(data=list(records), transform=transform)

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=list_data_collate,
    )


def build_optimizer(model: nn.Module, config: Config) -> torch.optim.Optimizer:
    """Build optimizer."""
    params = get_training_params(config)
    training = _section(config, "training")

    optimizer_name = str(training.get("optimizer", "adamw")).lower()

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=params["learning_rate"],
            momentum=float(training.get("momentum", 0.99)),
            weight_decay=params["weight_decay"],
            nesterov=True,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Config,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Build optional scheduler."""
    training = _section(config, "training")

    scheduler_name = str(training.get("scheduler", "cosine")).lower()
    epochs = int(training.get("epochs", 200))

    if scheduler_name in {"none", "null", "off"}:
        return None

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(training.get("min_learning_rate", 1e-6)),
        )

    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(training.get("lr_step_size", 50)),
            gamma=float(training.get("lr_gamma", 0.5)),
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def dice_from_logits(logits: torch.Tensor, labels: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Compute foreground Dice from logits and labels.

    Args:
        logits: [B, C, D, H, W]
        labels: [B, 1, D, H, W] or [B, D, H, W]
    """
    preds = torch.argmax(logits, dim=1)

    if labels.ndim == 5 and labels.shape[1] == 1:
        labels = labels[:, 0]

    labels = labels.long()

    pred_fg = preds == 1
    label_fg = labels == 1

    tp = torch.logical_and(pred_fg, label_fg).sum().float()
    fp = torch.logical_and(pred_fg, torch.logical_not(label_fg)).sum().float()
    fn = torch.logical_and(torch.logical_not(pred_fg), label_fg).sum().float()

    denom = 2.0 * tp + fp + fn

    if denom.item() == 0:
        return 1.0

    return float(((2.0 * tp) / (denom + eps)).detach().cpu())


def prepare_batch(
    batch: Mapping[str, Any],
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Move image/label batch to device."""
    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    # MONAI losses with to_onehot_y=True expect class labels, not one-hot labels.
    labels = labels.long()

    return images, labels


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool,
    grad_clip_norm: float,
) -> Dict[str, float]:
    """Train one epoch."""
    model.train()

    running_loss = 0.0
    running_dice = 0.0
    num_batches = 0

    start_time = time.perf_counter()

    for batch_idx, batch in enumerate(loader, start=1):
        images, labels = prepare_batch(batch, device=device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()

            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()

            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()

        batch_loss = float(loss.detach().cpu())
        batch_dice = dice_from_logits(logits.detach(), labels.detach())

        running_loss += batch_loss
        running_dice += batch_dice
        num_batches += 1

        if batch_idx == 1 or batch_idx % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"batch {batch_idx:04d}/{len(loader):04d} | "
                f"loss={batch_loss:.5f} | "
                f"dice={batch_dice:.5f}"
            )

    elapsed = time.perf_counter() - start_time

    return {
        "train_loss": running_loss / max(num_batches, 1),
        "train_dice": running_dice / max(num_batches, 1),
        "train_time_sec": elapsed,
    }


@torch.no_grad()
def validate(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    roi_size: Sequence[int],
    sw_batch_size: int,
    overlap: float,
    use_amp: bool,
) -> Dict[str, float]:
    """Validate using full-volume sliding-window inference."""
    model.eval()

    case_metrics: List[Dict[str, Any]] = []
    start_time = time.perf_counter()

    for case_idx, batch in enumerate(loader, start=1):
        images, labels = prepare_batch(batch, device=device)

        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = sliding_window_inference(
                    inputs=images,
                    roi_size=tuple(roi_size),
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode="gaussian",
                )
        else:
            logits = sliding_window_inference(
                inputs=images,
                roi_size=tuple(roi_size),
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
                mode="gaussian",
            )

        dice = dice_from_logits(logits, labels)

        case_id = batch.get("case_id", [f"case_{case_idx:04d}"])
        if isinstance(case_id, (list, tuple)):
            case_id = case_id[0]

        case_metrics.append(
            {
                "case_id": str(case_id),
                "dice": float(dice),
            }
        )

        print(f"Validation case {case_idx:03d}/{len(loader):03d} | case_id={case_id} | dice={dice:.5f}")

    elapsed = time.perf_counter() - start_time
    dices = [item["dice"] for item in case_metrics]

    return {
        "val_dice": float(np.mean(dices)) if dices else 0.0,
        "val_dice_std": float(np.std(dices)) if dices else 0.0,
        "val_time_sec": elapsed,
        "val_cases": len(case_metrics),
        "case_metrics": case_metrics,
    }


def is_better_metric(current: float, best: Optional[float], mode: str) -> bool:
    """Return True if current metric is better than best."""
    if best is None:
        return True

    if mode == "max":
        return current > best

    if mode == "min":
        return current < best

    raise ValueError(f"mode must be 'max' or 'min', got: {mode}")


def save_checkpoint(
    *,
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_metric: Optional[float],
    config: Config,
    history: List[Dict[str, Any]],
) -> None:
    """Save training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict(),
        "best_metric": best_metric,
        "config": dict(config),
        "history": history,
    }

    torch.save(checkpoint, path)


def load_checkpoint_if_requested(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler: torch.cuda.amp.GradScaler,
    config: Config,
    device: torch.device,
) -> Tuple[int, Optional[float], List[Dict[str, Any]]]:
    """
    Load checkpoint if config.checkpoint.resume_path is set.

    Returns:
        start_epoch, best_metric, history
    """
    checkpoint_params = get_checkpoint_params(config)
    resume_path = checkpoint_params["resume_path"]

    if not resume_path:
        return 1, None, []

    resume_path = Path(resume_path)

    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")

    ckpt = torch.load(resume_path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    last_epoch = int(ckpt.get("epoch", 0))
    best_metric = ckpt.get("best_metric", None)
    history = ckpt.get("history", [])

    print(f"Resumed from checkpoint: {resume_path}")
    print(f"Resume start epoch: {last_epoch + 1}")
    print(f"Best metric so far: {best_metric}")

    return last_epoch + 1, best_metric, history


def write_split_summary(
    *,
    train_records: Sequence[Dict[str, str]],
    val_records: Sequence[Dict[str, str]],
    output_path: str | Path,
) -> None:
    """Save internal split summary."""
    split_data = {
        "train": {
            "summary": summarize_records(train_records),
            "case_ids": [record["case_id"] for record in train_records],
        },
        "val": {
            "summary": summarize_records(val_records),
            "case_ids": [record["case_id"] for record in val_records],
        },
    }

    save_json(split_data, output_path)


def train(config: Config) -> Dict[str, Any]:
    """Main training routine."""
    experiment = _section(config, "experiment")
    seed = int(experiment.get("seed", 42))

    set_global_seed(seed)

    output_dirs = get_output_dirs(config)
    training_params = get_training_params(config)
    validation_params = get_validation_params(config)
    checkpoint_params = get_checkpoint_params(config)

    print("=" * 80)
    print("MVAA Task 1 training")
    print("=" * 80)

    model, device = build_model_and_device(config)
    loss_fn = build_segmentation_loss_from_config(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    use_amp = training_params["use_amp"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    all_labeled_records = discover_labeled_records_from_config(config)
    train_records, val_records = split_labeled_records(
        all_labeled_records,
        val_fraction=validation_params["internal_val_fraction"],
        min_val_cases=validation_params["internal_val_min_cases"],
        seed=seed,
    )

    write_split_summary(
        train_records=train_records,
        val_records=val_records,
        output_path=output_dirs["splits"] / "internal_split.json",
    )

    train_loader = build_train_loader(train_records, config, device=device)
    val_loader = build_internal_val_loader(val_records, config, device=device)

    print(f"Device: {device}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Loss: {loss_fn.__class__.__name__}")
    print(f"Train cases: {len(train_records)}")
    print(f"Internal val cases: {len(val_records)}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Validation interval: every {validation_params['val_interval']} epoch(s)")
    print(f"Output dir: {output_dirs['output']}")
    print("=" * 80)

    start_epoch, best_metric, history = load_checkpoint_if_requested(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config=config,
        device=device,
    )

    best_path = output_dirs["checkpoints"] / "best_model.pt"
    last_path = output_dirs["checkpoints"] / "last_model.pt"
    history_path = output_dirs["logs"] / "history.json"

    epochs = training_params["epochs"]
    monitor_metric = checkpoint_params["monitor_metric"]
    mode = checkpoint_params["mode"]

    total_start = time.perf_counter()

    for epoch in range(start_epoch, epochs + 1):
        print("\n" + "-" * 80)
        print(f"Epoch {epoch}/{epochs}")
        print("-" * 80)

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            use_amp=use_amp,
            grad_clip_norm=training_params["grad_clip_norm"],
        )

        if scheduler is not None:
            scheduler.step()

        epoch_record: Dict[str, Any] = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **train_metrics,
        }

        should_validate = epoch == 1 or epoch % validation_params["val_interval"] == 0 or epoch == epochs

        if should_validate:
            val_metrics = validate(
                model=model,
                loader=val_loader,
                device=device,
                roi_size=validation_params["roi_size"],
                sw_batch_size=validation_params["sw_batch_size"],
                overlap=validation_params["sliding_window_overlap"],
                use_amp=use_amp,
            )

            epoch_record.update(
                {
                    key: value
                    for key, value in val_metrics.items()
                    if key != "case_metrics"
                }
            )

            save_json(
                val_metrics,
                output_dirs["logs"] / f"val_epoch_{epoch:03d}.json",
            )

            current_metric = float(epoch_record.get(f"val_{monitor_metric}", epoch_record.get(monitor_metric, 0.0)))

            if is_better_metric(current_metric, best_metric, mode):
                best_metric = current_metric
                epoch_record["is_best"] = True

                if checkpoint_params["save_best"]:
                    save_checkpoint(
                        path=best_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        best_metric=best_metric,
                        config=config,
                        history=history + [epoch_record],
                    )
                    print(f"Saved best checkpoint: {best_path} | {monitor_metric}={best_metric:.5f}")
            else:
                epoch_record["is_best"] = False

        history.append(epoch_record)

        if checkpoint_params["save_last"]:
            save_checkpoint(
                path=last_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                config=config,
                history=history,
            )

        save_json(history, history_path)

        summary_parts = [
            f"epoch={epoch}",
            f"loss={epoch_record['train_loss']:.5f}",
            f"train_dice={epoch_record['train_dice']:.5f}",
        ]

        if "val_dice" in epoch_record:
            summary_parts.append(f"val_dice={epoch_record['val_dice']:.5f}")
            summary_parts.append(f"best={best_metric:.5f}" if best_metric is not None else "best=None")

        print(" | ".join(summary_parts))

    total_elapsed = time.perf_counter() - total_start

    final_summary = {
        "epochs": epochs,
        "best_metric": best_metric,
        "monitor_metric": monitor_metric,
        "mode": mode,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history_path": str(history_path),
        "total_time_sec": total_elapsed,
    }

    save_json(final_summary, output_dirs["logs"] / "final_summary.json")

    print("\n" + "=" * 80)
    print("Training finished")
    print(json.dumps(final_summary, indent=2))
    print("=" * 80)

    return final_summary


def parse_args() -> argparse.Namespace:
    """CLI arguments."""
    parser = argparse.ArgumentParser(description="Train MVAA Task 1 segmentation model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/task1_train.yaml",
        help="Path to training YAML config.",
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
        help="Optional override for experiment.device.",
    )
    return parser.parse_args()


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply CLI overrides to config."""
    config = dict(config)

    if args.epochs is not None:
        config["training"] = dict(config.get("training", {}))
        config["training"]["epochs"] = int(args.epochs)

    if args.device is not None:
        config["experiment"] = dict(config.get("experiment", {}))
        config["experiment"]["device"] = str(args.device)

    return config


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    config = load_yaml_config(args.config)
    config = apply_cli_overrides(config, args)

    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
