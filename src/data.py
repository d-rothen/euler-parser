"""Data loading bridge between euler_loading and depth-eval.

Provides utilities to build MultiModalDataset instances from config
and convert loaded tensors to the numpy formats expected by depth-eval
metrics.
"""

from typing import Any, Optional

import numpy as np
import torch

from euler_loading import Modality, MultiModalDataset

from .metrics.utils import convert_planar_to_radial


# ---------------------------------------------------------------------------
# Tensor → numpy conversions
# ---------------------------------------------------------------------------


def to_numpy_depth(data: Any) -> np.ndarray:
    """Convert depth data to ``(H, W)`` float32 numpy array.

    Accepts torch tensors ``(1, H, W)`` or ``(H, W)``, or numpy arrays.
    """
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(np.float32)


def to_numpy_rgb(data: Any) -> np.ndarray:
    """Convert RGB data to ``(H, W, 3)`` float32 numpy array.

    Accepts torch tensors ``(3, H, W)`` or ``(H, W, 3)``, or numpy arrays.
    """
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data, dtype=np.float32)
    # CHW → HWC
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    return arr.astype(np.float32)


def to_numpy_mask(data: Any) -> np.ndarray:
    """Convert mask data to ``(H, W)`` bool numpy array.

    Accepts torch tensors ``(1, H, W)`` or ``(H, W)``, or numpy arrays.
    """
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(bool)


def to_numpy_intrinsics(data: Any) -> np.ndarray:
    """Convert intrinsics to ``(3, 3)`` float32 numpy array."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().astype(np.float32)
    return np.asarray(data, dtype=np.float32)


# ---------------------------------------------------------------------------
# Post-load depth processing
# ---------------------------------------------------------------------------


def process_depth(
    depth: np.ndarray,
    scale_to_meters: float,
    is_radial: bool,
    intrinsics_K: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply scale and optional planar→radial conversion to a depth map.

    Args:
        depth: Raw depth array ``(H, W)`` in loader-native units.
        scale_to_meters: Multiplier to convert to metres.
        is_radial: If True, depth is already radial (Euclidean).
                   If False and ``intrinsics_K`` is provided, convert.
        intrinsics_K: ``(3, 3)`` camera matrix. Required when
                      ``is_radial`` is False.

    Returns:
        Depth in metres, ``(H, W)`` float32.
    """
    depth = depth * scale_to_meters
    if not is_radial and intrinsics_K is not None:
        intrinsics_dict = {
            "fx": float(intrinsics_K[0, 0]),
            "fy": float(intrinsics_K[1, 1]),
            "cx": float(intrinsics_K[0, 2]),
            "cy": float(intrinsics_K[1, 2]),
        }
        depth = convert_planar_to_radial(depth, intrinsics_dict)
    return depth.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def build_depth_eval_dataset(
    gt_depth_path: str,
    pred_depth_path: str,
    calibration_path: Optional[str] = None,
    segmentation_path: Optional[str] = None,
) -> MultiModalDataset:
    """Build a MultiModalDataset for depth evaluation.

    The returned dataset yields samples with keys ``"gt"``, ``"pred"``,
    and optionally ``"calibration"`` and ``"segmentation"``.

    Loaders are resolved automatically from each dataset directory's
    ds-crawler index metadata.

    Args:
        gt_depth_path: Path to GT depth dataset root.
        pred_depth_path: Path to prediction depth dataset root.
        calibration_path: Optional path to calibration dataset.
        segmentation_path: Optional path to GT segmentation dataset.

    Returns:
        A MultiModalDataset instance.
    """
    modalities = {
        "gt": Modality(path=gt_depth_path),
        "pred": Modality(path=pred_depth_path),
    }

    hierarchical = {}
    if calibration_path is not None:
        hierarchical["calibration"] = Modality(path=calibration_path)
    if segmentation_path is not None:
        hierarchical["segmentation"] = Modality(path=segmentation_path)

    return MultiModalDataset(
        modalities=modalities,
        hierarchical_modalities=hierarchical if hierarchical else None,
    )


def build_rgb_eval_dataset(
    gt_rgb_path: str,
    pred_rgb_path: str,
    gt_depth_path: Optional[str] = None,
    calibration_path: Optional[str] = None,
    segmentation_path: Optional[str] = None,
) -> MultiModalDataset:
    """Build a MultiModalDataset for RGB evaluation.

    The returned dataset yields samples with keys ``"gt"``, ``"pred"``,
    and optionally ``"gt_depth"``, ``"calibration"``, ``"segmentation"``.

    Loaders are resolved automatically from each dataset directory's
    ds-crawler index metadata.

    Args:
        gt_rgb_path: Path to GT RGB dataset root.
        pred_rgb_path: Path to prediction RGB dataset root.
        gt_depth_path: Optional GT depth path for depth-binned metrics.
        calibration_path: Optional calibration dataset path.
        segmentation_path: Optional GT segmentation dataset path.

    Returns:
        A MultiModalDataset instance.
    """
    modalities: dict[str, Modality] = {
        "gt": Modality(path=gt_rgb_path),
        "pred": Modality(path=pred_rgb_path),
    }
    if gt_depth_path is not None:
        modalities["gt_depth"] = Modality(path=gt_depth_path)

    hierarchical = {}
    if calibration_path is not None:
        hierarchical["calibration"] = Modality(path=calibration_path)
    if segmentation_path is not None:
        hierarchical["segmentation"] = Modality(path=segmentation_path)

    return MultiModalDataset(
        modalities=modalities,
        hierarchical_modalities=hierarchical if hierarchical else None,
    )


def get_depth_metadata(dataset: MultiModalDataset) -> dict[str, Any]:
    """Extract depth modality metadata from a dataset.

    Returns:
        Dict with ``scale_to_meters`` (float) and ``radial_depth`` (bool).
    """
    meta = dataset.get_modality_metadata("gt")
    return {
        "scale_to_meters": float(meta.get("scale_to_meters", 1.0)),
        "radial_depth": bool(meta.get("radial_depth", True)),
    }


def get_rgb_metadata(dataset: MultiModalDataset) -> dict[str, Any]:
    """Extract RGB modality metadata from a dataset.

    Returns:
        Dict with ``rgb_range`` ([min, max]).
    """
    meta = dataset.get_modality_metadata("gt")
    return {
        "rgb_range": meta.get("rgb_range", [0.0, 1.0]),
    }
