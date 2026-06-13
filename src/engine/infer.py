#!/usr/bin/env python3
"""
Inference engine for MVAA Task 1.

Task:
    Mitral Valve Segmentation and Landmark Localization in CT scans.

This file generates Task 1 segmentation predictions from a trained checkpoint.

Main responsibilities:
    - load inference config,
    - build model,
    - load checkpoint,
    - load validation/test CT volumes,
    - run sliding-window inference,
    - convert logits to binary segmentation mask,
    - optionally apply connected-component post-processing,
    - save prediction masks as .nii.gz with original affine/header,
    - create task1_predictions.json.

Why careful design?
    For mitral valve CT segmentation, the target is small, boundaries are
    ambiguous, and false-positive islands can hurt HD/HD95. At the same time,
    masks must be saved in the original CT physical space for fair evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import yaml
from monai.data import DataLoader, Dataset, list_data_collate
from monai.inferers import sliding_window_inference
from scipy import ndimage

try:
    from src.data.task1_dataset import discover_inference_cases
    from src.data.transforms import build_inference_transforms
    from src.models.model_factory import build_model_from_config, get_device_from_config
    from src.postprocessing import postprocess_from_config
except ModuleNotFoundError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))

    from src.data.task1_dataset import discover_inference_cases
    from src.data.transforms import build_inference_transforms
    from src.models.model_factory import build_model_from_config, get_device_from_config
    from src.postprocessing import postprocess_from_config


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
    """Save JSON with indentation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _section(config: Config, name: str) -> Dict[str, Any]:
    """Safely read a nested config section."""
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
    """Get project root from config."""
    project = _section(config, "project")
    return Path(project.get("root", "."))


def get_inference_params(config: Config) -> Dict[str, Any]:
    """Read inference parameters."""
    inference = _section(config, "inference")
    training = _section(config, "training")

    return {
        "roi_size": tuple(inference.get("roi_size", training.get("roi_size", [128, 128, 128]))),
        "sw_batch_size": int(inference.get("sw_batch_size", 4)),
        "sliding_window_overlap": float(inference.get("sliding_window_overlap", 0.25)),
        "use_amp": _bool_value(inference.get("use_amp", True)),
        "num_workers": int(inference.get("num_workers", training.get("num_workers", 0))),
        "sliding_window_mode": str(inference.get("sliding_window_mode", "gaussian")),
    }


def get_checkpoint_path(config: Config) -> Path:
    """Read checkpoint path from config."""
    project_root = get_project_root(config)
    checkpoint = _section(config, "checkpoint")
    path = checkpoint.get("path", "runs/task1_baseline/checkpoints/best_model.pt")
    return resolve_path(path, project_root)


def get_input_images_dir(config: Config) -> Path:
    """Read inference images directory from config."""
    project_root = get_project_root(config)
    data = _section(config, "data")
    path = data.get("images_dir", data.get("val_images_dir", "data/t1_ct/val/images"))
    return resolve_path(path, project_root)


def get_output_paths(config: Config) -> Dict[str, Path | str]:
    """Read prediction output paths from config."""
    project_root = get_project_root(config)
    output = _section(config, "output")

    prediction_dir = resolve_path(output.get("prediction_dir", "submission/t1_ct"), project_root)
    output_json = resolve_path(output.get("output_json", "submission/t1_ct/task1_predictions.json"), project_root)
    prediction_suffix = str(output.get("prediction_suffix", "-pred.nii.gz"))

    prediction_dir.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    return {
        "prediction_dir": prediction_dir,
        "output_json": output_json,
        "prediction_suffix": prediction_suffix,
    }


def safe_torch_load(path: str | Path, map_location: torch.device) -> Any:
    """
    Load checkpoint safely across PyTorch versions.

    Newer PyTorch versions expose weights_only; older versions may not.
    """
    path = Path(path)

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove DataParallel 'module.' prefix if present."""
    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module.") :]] = value
        else:
            cleaned[key] = value

    return cleaned


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load model weights from checkpoint.

    Supports checkpoints saved as:
        - {"model_state_dict": ...}
        - raw state_dict
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    ckpt = safe_torch_load(checkpoint_path, map_location=device)

    if isinstance(ckpt, Mapping) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        checkpoint_meta = {
            "epoch": ckpt.get("epoch", None),
            "best_metric": ckpt.get("best_metric", None),
            "history_length": len(ckpt.get("history", [])) if isinstance(ckpt.get("history", []), list) else None,
        }
    else:
        state_dict = ckpt
        checkpoint_meta = {
            "epoch": None,
            "best_metric": None,
            "history_length": None,
        }

    state_dict = strip_module_prefix(state_dict)
    load_result = model.load_state_dict(state_dict, strict=strict)

    checkpoint_meta["checkpoint_path"] = str(checkpoint_path)
    checkpoint_meta["missing_keys"] = list(getattr(load_result, "missing_keys", []))
    checkpoint_meta["unexpected_keys"] = list(getattr(load_result, "unexpected_keys", []))

    return checkpoint_meta


def build_inference_loader(
    records: Sequence[Dict[str, str]],
    config: Config,
    *,
    device: torch.device,
) -> DataLoader:
    """Build MONAI inference DataLoader."""
    params = get_inference_params(config)
    transform = build_inference_transforms(config)

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


def get_first_from_batch(batch_value: Any) -> Any:
    """Extract first value from MONAI-collated metadata/list."""
    if isinstance(batch_value, (list, tuple)):
        return batch_value[0]

    return batch_value


def logits_to_mask(logits: torch.Tensor) -> np.ndarray:
    """
    Convert model logits to uint8 binary foreground mask.

    Expected logits:
        [1, C, D, H, W]

    Returns:
        [D, H, W] uint8 mask with values 0/1.
    """
    if logits.ndim != 5:
        raise ValueError(f"Expected logits [B, C, D, H, W], got {tuple(logits.shape)}")

    if logits.shape[0] != 1:
        raise ValueError(f"Inference expects batch size 1, got {logits.shape[0]}")

    if logits.shape[1] == 1:
        probs = torch.sigmoid(logits[:, 0])
        pred = probs >= 0.5
        mask = pred[0].detach().cpu().numpy().astype(np.uint8)
        return mask

    pred = torch.argmax(logits, dim=1)
    mask = pred[0].detach().cpu().numpy().astype(np.uint8)

    # Task 1 baseline is binary segmentation.
    # If future multi-class model appears, this keeps foreground as nonzero.
    mask = (mask > 0).astype(np.uint8)
    return mask


def resize_mask_nearest(mask: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """
    Resize binary mask to target_shape using nearest-neighbor interpolation.

    Used as a safety fallback if preprocessing/resampling changes shape.
    """
    target_shape = tuple(int(x) for x in target_shape)

    if tuple(mask.shape) == target_shape:
        return mask.astype(np.uint8)

    zoom_factors = [
        target_shape[i] / float(mask.shape[i])
        for i in range(3)
    ]

    resized = ndimage.zoom(mask.astype(np.uint8), zoom=zoom_factors, order=0)
    resized = resized.astype(np.uint8)

    # ndimage.zoom can be off by one because of rounding.
    if resized.shape != target_shape:
        fixed = np.zeros(target_shape, dtype=np.uint8)
        common_shape = tuple(min(resized.shape[i], target_shape[i]) for i in range(3))
        fixed[
            : common_shape[0],
            : common_shape[1],
            : common_shape[2],
        ] = resized[
            : common_shape[0],
            : common_shape[1],
            : common_shape[2],
        ]
        resized = fixed

    return resized.astype(np.uint8)


def get_nifti_spacing(image: nib.spatialimages.SpatialImage) -> Tuple[float, float, float]:
    """Read 3D voxel spacing from NIfTI header."""
    zooms = image.header.get_zooms()
    if len(zooms) < 3:
        return (1.0, 1.0, 1.0)
    return (float(zooms[0]), float(zooms[1]), float(zooms[2]))


def save_mask_like_source(
    mask: np.ndarray,
    source_image_path: str | Path,
    output_path: str | Path,
) -> Dict[str, Any]:
    """
    Save uint8 NIfTI mask using original image affine/header.

    This preserves spatial metadata needed for fair challenge evaluation.
    """
    source_image_path = Path(source_image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_img = nib.load(str(source_image_path))
    original_shape = tuple(int(x) for x in source_img.shape[:3])

    mask = resize_mask_nearest(mask, original_shape)
    mask = (mask > 0).astype(np.uint8)

    header = source_img.header.copy()
    header.set_data_dtype(np.uint8)
    header["cal_min"] = 0
    header["cal_max"] = 1

    out_img = nib.Nifti1Image(mask, affine=source_img.affine, header=header)

    # Preserve qform/sform codes where possible.
    try:
        qform, qcode = source_img.get_qform(coded=True)
        if qform is not None:
            out_img.set_qform(qform, int(qcode))
    except Exception:
        pass

    try:
        sform, scode = source_img.get_sform(coded=True)
        if sform is not None:
            out_img.set_sform(sform, int(scode))
    except Exception:
        pass

    nib.save(out_img, str(output_path))

    return {
        "source_image": str(source_image_path),
        "output_path": str(output_path),
        "shape": list(mask.shape),
        "spacing": list(get_nifti_spacing(source_img)),
        "foreground_voxels": int(mask.sum()),
        "dtype": "uint8",
    }


def synchronize_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA for accurate timing."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def infer_one_case(
    *,
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    config: Config,
    device: torch.device,
    roi_size: Sequence[int],
    sw_batch_size: int,
    overlap: float,
    mode: str,
    use_amp: bool,
    prediction_dir: Path,
    prediction_suffix: str,
) -> Dict[str, Any]:
    """Run inference for one case and save prediction mask."""
    case_id = str(get_first_from_batch(batch["case_id"]))
    source_image_path = Path(str(get_first_from_batch(batch["image_path"])))

    source_img = nib.load(str(source_image_path))
    source_shape = tuple(int(x) for x in source_img.shape[:3])
    source_spacing = get_nifti_spacing(source_img)

    images = batch["image"].to(device)

    synchronize_if_cuda(device)
    start = time.perf_counter()

    autocast_enabled = bool(use_amp and device.type == "cuda")
    with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
        logits = sliding_window_inference(
            inputs=images,
            roi_size=tuple(int(x) for x in roi_size),
            sw_batch_size=int(sw_batch_size),
            predictor=model,
            overlap=float(overlap),
            mode=mode,
        )

    synchronize_if_cuda(device)
    inference_time_sec = time.perf_counter() - start

    mask = logits_to_mask(logits)
    mask = resize_mask_nearest(mask, source_shape)

    postprocess_report: Optional[Dict[str, Any]] = None
    postprocessed = postprocess_from_config(
        mask,
        config,
        spacing=source_spacing,
        return_report=True,
    )

    if isinstance(postprocessed, tuple):
        mask, postprocess_report = postprocessed
    else:
        mask = postprocessed

    output_name = f"{case_id}{prediction_suffix}"
    output_path = prediction_dir / output_name

    save_report = save_mask_like_source(
        mask,
        source_image_path=source_image_path,
        output_path=output_path,
    )

    result = {
        "case_id": case_id,
        "segmentation": output_name,
        "source_image": str(source_image_path),
        "prediction_path": str(output_path),
        "source_shape": list(source_shape),
        "prediction_shape": save_report["shape"],
        "spacing": list(source_spacing),
        "foreground_voxels": int(save_report["foreground_voxels"]),
        "inference_time_sec": float(inference_time_sec),
        "postprocessing": postprocess_report,
    }

    return result


def write_task1_predictions_json(records: Sequence[Mapping[str, Any]], output_json: str | Path) -> None:
    """
    Write official Task 1 predictions JSON.

    Required format:
        {
          "cases": [
            {"case_id": "0001", "segmentation": "0001-pred.nii.gz"}
          ]
        }
    """
    cases = []

    seen = set()
    for record in records:
        case_id = str(record["case_id"])
        segmentation = str(record["segmentation"])

        if case_id in seen:
            raise ValueError(f"Duplicate case_id in predictions: {case_id}")
        seen.add(case_id)

        cases.append(
            {
                "case_id": case_id,
                "segmentation": segmentation,
            }
        )

    save_json({"cases": cases}, output_json)


def run_inference(
    config: Config,
    *,
    max_cases: Optional[int] = None,
    override_checkpoint: Optional[str] = None,
    override_images_dir: Optional[str] = None,
    override_output_dir: Optional[str] = None,
    override_device: Optional[str] = None,
) -> Dict[str, Any]:
    """Main inference routine."""
    params = get_inference_params(config)
    device = get_device_from_config(config, override_device=override_device)

    model = build_model_from_config(config)
    model = model.to(device)
    model.eval()

    checkpoint_path = Path(override_checkpoint) if override_checkpoint else get_checkpoint_path(config)
    checkpoint_meta = load_model_checkpoint(
        model,
        checkpoint_path,
        device=device,
        strict=True,
    )

    images_dir = Path(override_images_dir) if override_images_dir else get_input_images_dir(config)
    records = discover_inference_cases(images_dir=images_dir, split="inference")

    if max_cases is not None:
        records = records[: int(max_cases)]

    output_paths = get_output_paths(config)
    prediction_dir = Path(override_output_dir) if override_output_dir else Path(output_paths["prediction_dir"])
    prediction_dir.mkdir(parents=True, exist_ok=True)

    output_json = Path(output_paths["output_json"])
    if override_output_dir:
        output_json = prediction_dir / "task1_predictions.json"

    prediction_suffix = str(output_paths["prediction_suffix"])

    loader = build_inference_loader(records, config, device=device)

    print("=" * 80)
    print("MVAA Task 1 inference")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Images dir: {images_dir}")
    print(f"Cases: {len(records)}")
    print(f"Prediction dir: {prediction_dir}")
    print(f"Output JSON: {output_json}")
    print(f"ROI size: {params['roi_size']}")
    print(f"SW batch size: {params['sw_batch_size']}")
    print(f"Overlap: {params['sliding_window_overlap']}")
    print(f"AMP: {params['use_amp'] and device.type == 'cuda'}")
    print("=" * 80)

    case_reports: List[Dict[str, Any]] = []
    total_start = time.perf_counter()

    for idx, batch in enumerate(loader, start=1):
        result = infer_one_case(
            model=model,
            batch=batch,
            config=config,
            device=device,
            roi_size=params["roi_size"],
            sw_batch_size=params["sw_batch_size"],
            overlap=params["sliding_window_overlap"],
            mode=params["sliding_window_mode"],
            use_amp=params["use_amp"],
            prediction_dir=prediction_dir,
            prediction_suffix=prediction_suffix,
        )

        case_reports.append(result)

        print(
            f"[{idx:03d}/{len(loader):03d}] "
            f"case_id={result['case_id']} | "
            f"fg={result['foreground_voxels']} | "
            f"time={result['inference_time_sec']:.3f}s | "
            f"file={result['segmentation']}"
        )

    total_time_sec = time.perf_counter() - total_start

    write_task1_predictions_json(case_reports, output_json)

    report = {
        "num_cases": len(case_reports),
        "total_time_sec": float(total_time_sec),
        "mean_time_sec": float(np.mean([r["inference_time_sec"] for r in case_reports])) if case_reports else math.nan,
        "max_time_sec": float(np.max([r["inference_time_sec"] for r in case_reports])) if case_reports else math.nan,
        "checkpoint": checkpoint_meta,
        "output_json": str(output_json),
        "prediction_dir": str(prediction_dir),
        "cases": case_reports,
    }

    report_path = prediction_dir / "inference_report.json"
    save_json(report, report_path)

    print("=" * 80)
    print("Inference finished")
    print(f"Predictions JSON: {output_json}")
    print(f"Inference report: {report_path}")
    print(f"Mean case time: {report['mean_time_sec']:.3f}s")
    print("=" * 80)

    return report


def parse_args() -> argparse.Namespace:
    """CLI args."""
    parser = argparse.ArgumentParser(description="Generate MVAA Task 1 predictions.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/task1_inference.yaml",
        help="Path to inference config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint override.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Optional inference image directory override.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override: cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional maximum number of cases for smoke testing.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
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
