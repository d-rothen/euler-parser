"""Data loading bridge between euler_loading and depth-eval.

Provides utilities to build MultiModalDataset instances from config
and convert loaded tensors to the numpy formats expected by depth-eval
metrics.
"""

from typing import Any, Callable, Optional

import numpy as np
import torch

from ds_crawler import index_dataset_from_path
from euler_loading import Modality, MultiModalDataset, resolve_loader_module

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
# Dimension alignment
# ---------------------------------------------------------------------------


def align_to_prediction(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Align GT array dimensions to match prediction.

    Detects the common VAE multiple-of-8 crop pattern (prediction dims are
    multiples of 8 and differ from GT by fewer than 8 pixels per axis) and
    applies the same top-left crop to the GT array.  Falls back to resizing
    for any other mismatch.

    Works for ``(H, W)`` depth/mask and ``(H, W, 3)`` RGB arrays.

    Args:
        gt: Ground-truth array.
        pred: Prediction array whose spatial dimensions are the target.

    Returns:
        GT array with spatial dimensions matching *pred*.
    """
    gt_h, gt_w = gt.shape[:2]
    pred_h, pred_w = pred.shape[:2]

    if gt_h == pred_h and gt_w == pred_w:
        return gt

    dh = gt_h - pred_h
    dw = gt_w - pred_w

    # Detect VAE multiple-of-8 crop: pred dims are multiples of 8 and
    # the GT is at most 7 pixels larger on each axis.
    if (pred_h % 8 == 0 and pred_w % 8 == 0
            and 0 <= dh < 8 and 0 <= dw < 8):
        return gt[:pred_h, :pred_w]

    # Fallback: resize GT to match prediction dimensions.
    import cv2

    is_bool = gt.dtype == bool
    src = gt.astype(np.uint8) if is_bool else gt
    interp = cv2.INTER_NEAREST if src.ndim == 2 else cv2.INTER_LINEAR
    result = cv2.resize(src, (pred_w, pred_h), interpolation=interp)
    return result.astype(bool) if is_bool else result


def compute_scale_and_shift(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float, float]:
    """Align predicted depth to GT via least-squares scale and shift.

    Solves ``min_{s,t}  Σ (s·pred + t − gt)²`` over valid pixels and
    returns the aligned prediction together with the fitted parameters.

    Args:
        pred: Predicted depth ``(H, W)``.
        gt: Ground-truth depth in metres ``(H, W)``.
        valid_mask: Optional ``(H, W)`` bool mask.  When *None*, all
                    finite positive pixels in both arrays are used.

    Returns:
        ``(aligned, scale, shift)`` where *aligned* is
        ``(s·pred + t).astype(float32)`` and *scale*/*shift* are the
        fitted affine parameters.
    """
    if valid_mask is None:
        valid_mask = (gt > 0) & np.isfinite(gt) & np.isfinite(pred)

    n_valid = int(valid_mask.sum())
    if n_valid < 2:
        return pred.copy(), 1.0, 0.0

    pred_valid = pred[valid_mask].astype(np.float64)
    gt_valid = gt[valid_mask].astype(np.float64)

    # Solve  [pred, 1] @ [s, t]^T = gt
    A = np.stack([pred_valid, np.ones(n_valid, dtype=np.float64)], axis=-1)
    params, _, _, _ = np.linalg.lstsq(A, gt_valid, rcond=None)
    s, t = float(params[0]), float(params[1])

    aligned = (s * pred.astype(np.float64) + t).astype(np.float32)
    return aligned, s, t


# ---------------------------------------------------------------------------
# Loader resolution helpers
# ---------------------------------------------------------------------------


def _resolve_sky_mask_loader(segmentation_path: str) -> Callable[..., Any]:
    """Resolve the ``sky_mask`` loader for a segmentation dataset.

    The segmentation dataset's ds-crawler index declares a loader module
    (e.g. ``"vkitti2"``).  This function imports that module and returns
    its ``sky_mask`` function so that raw class-segmentation data is
    converted to a boolean sky mask at load time.
    """
    index = index_dataset_from_path(segmentation_path)
    euler_meta = index.get("euler_loading", {})
    loader_name = euler_meta.get("loader")
    if loader_name is None:
        raise ValueError(
            f"Segmentation dataset at {segmentation_path!r} does not declare "
            f"an 'euler_loading.loader' in its ds-crawler index."
        )
    module = resolve_loader_module(loader_name)
    sky_fn = getattr(module, "sky_mask", None)
    if sky_fn is None or not callable(sky_fn):
        raise ValueError(
            f"Loader module {loader_name!r} (resolved from {segmentation_path!r}) "
            f"does not expose a 'sky_mask' function."
        )
    return sky_fn


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
        sky_fn = _resolve_sky_mask_loader(segmentation_path)
        hierarchical["segmentation"] = Modality(
            path=segmentation_path, loader=sky_fn
        )

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
        sky_fn = _resolve_sky_mask_loader(segmentation_path)
        hierarchical["segmentation"] = Modality(
            path=segmentation_path, loader=sky_fn
        )

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
