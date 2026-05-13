"""Data loading bridge between euler_loading and depth-eval.

Provides utilities to build MultiModalDataset instances from config
and convert loaded tensors to the numpy formats expected by depth-eval
metrics.
"""

from collections.abc import Mapping
from typing import Any, Callable, Optional

import numpy as np
import torch
from ds_crawler import get_dataset_contract, index_dataset_from_path
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

    if arr.ndim == 2:
        return arr.astype(np.float32)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(
                f"Unsupported depth shape {arr.shape}. Expected (H,W), (1,H,W), or (H,W,1)."
            )
        return arr.astype(np.float32)

    if arr.ndim == 4:
        if arr.shape[0] == 1 and arr.shape[1] == 1:
            arr = arr[0, 0]
        elif arr.shape[0] == 1 and arr.shape[-1] == 1:
            arr = arr[0, ..., 0]
        else:
            raise ValueError(
                f"Unsupported depth shape {arr.shape}. Expected a single-sample/single-channel tensor."
            )
        return arr.astype(np.float32)

    raise ValueError(f"Unsupported depth rank {arr.ndim} for shape {arr.shape}.")


def to_numpy_rgb(data: Any) -> np.ndarray:
    """Convert RGB data to ``(H, W, 3)`` float32 numpy array.

    Accepts torch tensors ``(3, H, W)`` or ``(H, W, 3)``, or numpy arrays.
    """
    tensor_input = isinstance(data, torch.Tensor)
    if tensor_input:
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data, dtype=np.float32)

    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 3:
        raise ValueError(
            f"Unsupported RGB shape {arr.shape}. Expected (H,W,3), (3,H,W), or single-sample variants."
        )

    # Tensor RGB from loaders is typically CHW.
    if tensor_input and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.shape[-1] == 3 and arr.shape[0] != 3:
        pass  # HWC
    elif arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))  # CHW
    elif arr.shape[0] == 3 and arr.shape[-1] == 3:
        raise ValueError(
            f"Ambiguous RGB layout for shape {arr.shape}. "
            "Please provide explicit HWC data or tensor CHW data."
        )
    else:
        raise ValueError(
            f"Unsupported RGB shape {arr.shape}. Expected channel dimension of size 3."
        )

    return arr.astype(np.float32)


def to_numpy_mask(data: Any) -> np.ndarray:
    """Convert mask data to ``(H, W)`` bool numpy array.

    Accepts torch tensors ``(1, H, W)`` or ``(H, W)``, or numpy arrays.
    """
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data)

    if arr.ndim == 2:
        return arr.astype(bool)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(
                f"Unsupported mask shape {arr.shape}. Expected (H,W), (1,H,W), or (H,W,1)."
            )
        return arr.astype(bool)

    if arr.ndim == 4:
        if arr.shape[0] == 1 and arr.shape[1] == 1:
            arr = arr[0, 0]
        elif arr.shape[0] == 1 and arr.shape[-1] == 1:
            arr = arr[0, ..., 0]
        else:
            raise ValueError(
                f"Unsupported mask shape {arr.shape}. Expected a single-sample/single-channel tensor."
            )
        return arr.astype(bool)

    raise ValueError(f"Unsupported mask rank {arr.ndim} for shape {arr.shape}.")


def to_numpy_intrinsics(data: Any) -> np.ndarray:
    """Convert intrinsics to ``(3, 3)`` float32 numpy array."""
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy().astype(np.float32)
    else:
        arr = np.asarray(data, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape != (3, 3):
        raise ValueError(f"Unsupported intrinsics shape {arr.shape}. Expected (3,3).")
    return arr


def to_numpy_extrinsics(data: Any) -> np.ndarray:
    """Convert camera extrinsics to a ``(4, 4)`` float32 transform matrix."""
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy().astype(np.float32)
    else:
        arr = np.asarray(data, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape == (3, 4):
        arr = np.vstack([arr, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)])
    if arr.shape != (4, 4):
        raise ValueError(
            f"Unsupported camera extrinsics shape {arr.shape}. Expected (4,4) or (3,4)."
        )
    return arr.astype(np.float32, copy=False)


def to_numpy_point_cloud(data: Any) -> np.ndarray:
    """Convert point-cloud data to ``(N, C)`` float32 numpy array.

    The first three columns are interpreted as ``x, y, z`` in metres. Extra
    columns such as intensity/ring/timestamp are preserved for callers that
    need them, but projection utilities use only ``xyz``.
    """
    if isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy().astype(np.float32)
    else:
        arr = np.asarray(data, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(
            f"Unsupported point-cloud shape {arr.shape}. Expected (N,C) with C >= 3."
        )
    return arr.astype(np.float32, copy=False)


def to_numpy_directions(data: Any) -> np.ndarray:
    """Convert direction map data to ``(H, W, 3)`` float32 numpy array.

    Accepts torch tensors ``(3, H, W)`` or ``(H, W, 3)``, or numpy
    arrays.  The returned vectors are **not** re-normalized; callers
    should normalize if needed.
    """
    tensor_input = isinstance(data, torch.Tensor)
    if tensor_input:
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data, dtype=np.float32)

    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 3:
        raise ValueError(
            f"Unsupported direction map shape {arr.shape}. "
            "Expected (H,W,3), (3,H,W), or single-sample variants."
        )

    # Tensor from loaders is typically CHW.
    if tensor_input and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.shape[-1] == 3 and arr.shape[0] != 3:
        pass  # HWC
    elif arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))  # CHW
    elif arr.shape[0] == 3 and arr.shape[-1] == 3:
        raise ValueError(
            f"Ambiguous direction map layout for shape {arr.shape}. "
            "Please provide explicit HWC data or tensor CHW data."
        )
    else:
        raise ValueError(
            f"Unsupported direction map shape {arr.shape}. "
            "Expected channel dimension of size 3."
        )

    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Post-load depth processing
# ---------------------------------------------------------------------------


def process_depth(
    depth: np.ndarray,
    scale_to_meters: float,
    is_radial: bool,
    intrinsics_K: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply optional scaling and planar→radial conversion to a depth map.

    Args:
        depth: Depth array ``(H, W)``.
        scale_to_meters: Legacy multiplier to convert to metres.
            For euler_loading-provided depth, this should remain ``1.0``.
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


def project_point_cloud_to_depth_map(
    point_cloud: np.ndarray,
    intrinsics_K: np.ndarray,
    camera_extrinsics: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Project a sparse point cloud into the camera image plane.

    Args:
        point_cloud: ``(N, C)`` array whose first three columns are source-frame
            ``x, y, z`` coordinates in metres.
        intrinsics_K: Camera intrinsics matrix.
        camera_extrinsics: Source-to-camera transform. For MUSES this is the
            ``lidar2rgb`` transform returned by the ``camera_extrinsics`` loader.
        image_shape: Target dense prediction shape as ``(height, width)``.

    Returns:
        ``(depth_map, valid_mask, metadata)``. ``depth_map`` stores radial
        camera depth in metres at projected pixels, choosing the nearest point
        by camera ``z`` when multiple points land on the same pixel.
    """
    height, width = image_shape
    depth = np.zeros((height, width), dtype=np.float32)
    valid_mask = np.zeros((height, width), dtype=bool)

    xyz = np.asarray(point_cloud[:, :3], dtype=np.float32)
    finite_xyz = np.isfinite(xyz).all(axis=1)
    if not finite_xyz.any():
        return depth, valid_mask, {
            "input_points": int(point_cloud.shape[0]),
            "finite_points": 0,
            "in_front_points": 0,
            "in_image_points": 0,
            "projected_pixels": 0,
        }

    xyz = xyz[finite_xyz]
    ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
    points_h = np.concatenate([xyz, ones], axis=1)
    camera_points = (camera_extrinsics @ points_h.T).T[:, :3]

    finite_camera = np.isfinite(camera_points).all(axis=1)
    in_front = finite_camera & (camera_points[:, 2] > 1e-8)
    if not in_front.any():
        return depth, valid_mask, {
            "input_points": int(point_cloud.shape[0]),
            "finite_points": int(finite_xyz.sum()),
            "in_front_points": 0,
            "in_image_points": 0,
            "projected_pixels": 0,
        }

    camera_points = camera_points[in_front]
    z = camera_points[:, 2]
    x = camera_points[:, 0]
    y = camera_points[:, 1]

    u = intrinsics_K[0, 0] * (x / z) + intrinsics_K[0, 2]
    v = intrinsics_K[1, 1] * (y / z) + intrinsics_K[1, 2]
    u_px = np.rint(u).astype(np.int64)
    v_px = np.rint(v).astype(np.int64)
    in_image = (u_px >= 0) & (u_px < width) & (v_px >= 0) & (v_px < height)
    if not in_image.any():
        return depth, valid_mask, {
            "input_points": int(point_cloud.shape[0]),
            "finite_points": int(finite_xyz.sum()),
            "in_front_points": int(in_front.sum()),
            "in_image_points": 0,
            "projected_pixels": 0,
        }

    camera_points = camera_points[in_image]
    z = z[in_image]
    u_px = u_px[in_image]
    v_px = v_px[in_image]
    radial_depth = np.linalg.norm(camera_points, axis=1).astype(np.float32)
    flat = v_px * width + u_px

    order = np.lexsort((z, flat))
    sorted_flat = flat[order]
    keep = np.concatenate([[True], sorted_flat[1:] != sorted_flat[:-1]])
    chosen = order[keep]
    chosen_flat = flat[chosen]

    depth.reshape(-1)[chosen_flat] = radial_depth[chosen]
    valid_mask.reshape(-1)[chosen_flat] = True

    return depth, valid_mask, {
        "input_points": int(point_cloud.shape[0]),
        "finite_points": int(finite_xyz.sum()),
        "in_front_points": int(in_front.sum()),
        "in_image_points": int(in_image.sum()),
        "projected_pixels": int(chosen_flat.size),
    }


# ---------------------------------------------------------------------------
# Dimension alignment
# ---------------------------------------------------------------------------


def classify_spatial_alignment(
    gt_h: int, gt_w: int, pred_h: int, pred_w: int
) -> str:
    """Classify how ``align_to_prediction`` would align the given shapes.

    Returns one of ``"none"``, ``"vae_crop"``, or ``"resize"``.
    """
    if gt_h == pred_h and gt_w == pred_w:
        return "none"
    dh = gt_h - pred_h
    dw = gt_w - pred_w
    if pred_h % 8 == 0 and pred_w % 8 == 0 and 0 <= dh < 8 and 0 <= dw < 8:
        return "vae_crop"
    return "resize"


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
    if pred_h % 8 == 0 and pred_w % 8 == 0 and 0 <= dh < 8 and 0 <= dw < 8:
        return gt[:pred_h, :pred_w]

    # Fallback: resize GT to match prediction dimensions.
    import cv2
    import warnings

    # Warn when aspect ratio distortion is significant (>5%).
    gt_ar = gt_w / gt_h
    pred_ar = pred_w / pred_h
    if abs(gt_ar - pred_ar) / max(gt_ar, pred_ar) > 0.05:
        warnings.warn(
            f"Significant aspect-ratio mismatch: GT {gt_w}x{gt_h} "
            f"({gt_ar:.3f}) vs pred {pred_w}x{pred_h} ({pred_ar:.3f}). "
            f"Metrics may not be directly comparable to native-resolution "
            f"evaluations.",
            stacklevel=2,
        )

    is_bool = gt.dtype == bool
    is_downscale = (pred_h * pred_w) < (gt_h * gt_w)
    src = gt.astype(np.uint8) if is_bool else gt

    # INTER_AREA properly averages source pixels when down-sampling and avoids
    # aliasing artefacts that INTER_NEAREST/INTER_LINEAR produce at large
    # scale ratios.  For up-sampling we keep INTER_NEAREST (depth/masks) or
    # INTER_LINEAR (RGB) to preserve values / smoothness respectively.
    if is_bool:
        interp = cv2.INTER_NEAREST
    elif is_downscale:
        interp = cv2.INTER_AREA
    elif src.ndim == 2:
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_LINEAR

    result = cv2.resize(src, (pred_w, pred_h), interpolation=interp)
    return result.astype(bool) if is_bool else result


def compute_scale_and_shift(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    max_gt_percentile: Optional[float] = None,
) -> tuple[np.ndarray, float, float]:
    """Align predicted depth to GT via least-squares scale and shift.

    Solves ``min_{s,t}  Σ (s·pred + t − gt)²`` over valid pixels and
    returns the aligned prediction together with the fitted parameters.

    Args:
        pred: Predicted depth ``(H, W)``.
        gt: Ground-truth depth in metres ``(H, W)``.
        valid_mask: Optional ``(H, W)`` bool mask.  When *None*, all
                    finite positive pixels in both arrays are used.
        max_gt_percentile: Optional upper GT-depth percentile for the fit.
            Pixels above this percentile are ignored when enough samples
            remain, which is useful for suppressing residual sky-depth
            outliers during masked evaluation.

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

    if max_gt_percentile is not None:
        if not 0.0 < max_gt_percentile <= 100.0:
            raise ValueError(
                f"max_gt_percentile must be in (0, 100], got {max_gt_percentile}."
            )

        gt_valid = gt[valid_mask].astype(np.float64)
        gt_cap = float(np.percentile(gt_valid, max_gt_percentile))
        trimmed_mask = valid_mask & (gt <= gt_cap)
        trimmed_count = int(trimmed_mask.sum())
        if trimmed_count >= 2:
            valid_mask = trimmed_mask
            n_valid = trimmed_count

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
    modality = _modality(path=segmentation_path, modality_key="segmentation")
    try:
        index = index_dataset_from_path(
            modality.path,
            metadata_scope=modality.metadata_scope,
        )
    except FileNotFoundError:
        index = index_dataset_from_path(modality.path)
    # Resolve euler_loading metadata the same way euler-loading does:
    # try the contract addon API first (handles addons.euler_loading),
    # then fall back to top-level euler_loading for legacy indices.
    euler_meta = None
    try:
        contract = get_dataset_contract(dict(index))
        euler_meta = contract.get_addon("euler_loading")
    except Exception:
        pass
    if not isinstance(euler_meta, Mapping):
        euler_meta = index.get("euler_loading", {})
    loader_name = euler_meta.get("loader")
    if loader_name is None:
        raise ValueError(
            f"Segmentation dataset at {modality.path!r} does not declare "
            f"an 'euler_loading.loader' in its ds-crawler index."
        )
    module = resolve_loader_module(loader_name)
    sky_fn = getattr(module, "sky_mask", None)
    if sky_fn is None or not callable(sky_fn):
        raise ValueError(
            f"Loader module {loader_name!r} (resolved from {modality.path!r}) "
            f"does not expose a 'sky_mask' function."
        )
    return sky_fn


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def _modality(
    *,
    path: str,
    modality_key: str,
    split: Optional[str] = None,
    loader: Optional[Callable[..., Any]] = None,
    used_as: Optional[str] = None,
) -> Modality:
    """Create an euler-loading modality with an explicit metadata selector."""
    return Modality(
        path=path,
        loader=loader,
        used_as=used_as,
        modality_type=modality_key,
        metadata_scope=modality_key,
        split=split,
    )


def build_depth_eval_dataset(
    gt_depth_path: str,
    pred_depth_path: str,
    calibration_path: Optional[str] = None,
    segmentation_path: Optional[str] = None,
    gt_depth_split: Optional[str] = None,
    pred_depth_split: Optional[str] = None,
    calibration_split: Optional[str] = None,
    segmentation_split: Optional[str] = None,
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
        gt_depth_split: Optional split name for the GT depth modality.
        pred_depth_split: Optional split name for the prediction depth modality.
        calibration_split: Optional split name for the calibration modality.
        segmentation_split: Optional split name for the segmentation modality.

    Returns:
        A MultiModalDataset instance.
    """
    modalities = {
        "gt": _modality(
            path=gt_depth_path,
            modality_key="depth",
            split=gt_depth_split,
        ),
        "pred": _modality(
            path=pred_depth_path,
            modality_key="depth",
            used_as="output",
            split=pred_depth_split,
        ),
    }

    hierarchical = {}
    if calibration_path is not None:
        hierarchical["calibration"] = _modality(
            path=calibration_path,
            modality_key="calibration",
            split=calibration_split,
        )
    if segmentation_path is not None:
        sky_fn = _resolve_sky_mask_loader(segmentation_path)
        hierarchical["segmentation"] = _modality(
            path=segmentation_path,
            modality_key="segmentation",
            loader=sky_fn,
            split=segmentation_split,
        )

    return MultiModalDataset(
        modalities=modalities,
        hierarchical_modalities=hierarchical if hierarchical else None,
    )


def build_sparse_depth_eval_dataset(
    gt_sparse_depth_path: str,
    pred_depth_path: str,
    intrinsics_path: str,
    camera_extrinsics_path: str,
    segmentation_path: Optional[str] = None,
    gt_sparse_depth_split: Optional[str] = None,
    pred_depth_split: Optional[str] = None,
    intrinsics_split: Optional[str] = None,
    camera_extrinsics_split: Optional[str] = None,
    segmentation_split: Optional[str] = None,
) -> MultiModalDataset:
    """Build a MultiModalDataset for sparse pointcloud depth evaluation.

    The returned dataset yields dense predicted depth under ``"pred"``,
    sparse GT point clouds under ``"gt"``, and hierarchical camera
    calibration under ``"intrinsics"`` and ``"camera_extrinsics"``.
    """
    modalities = {
        "gt": _modality(
            path=gt_sparse_depth_path,
            modality_key="sparse_depth",
            split=gt_sparse_depth_split,
        ),
        "pred": _modality(
            path=pred_depth_path,
            modality_key="depth",
            used_as="output",
            split=pred_depth_split,
        ),
    }

    hierarchical = {
        "intrinsics": _modality(
            path=intrinsics_path,
            modality_key="intrinsics",
            split=intrinsics_split,
        ),
        "camera_extrinsics": _modality(
            path=camera_extrinsics_path,
            modality_key="camera_extrinsics",
            split=camera_extrinsics_split,
        ),
    }
    if segmentation_path is not None:
        sky_fn = _resolve_sky_mask_loader(segmentation_path)
        hierarchical["segmentation"] = _modality(
            path=segmentation_path,
            modality_key="segmentation",
            loader=sky_fn,
            split=segmentation_split,
        )

    return MultiModalDataset(
        modalities=modalities,
        hierarchical_modalities=hierarchical,
    )


def build_rgb_eval_dataset(
    gt_rgb_path: str,
    pred_rgb_path: str,
    gt_depth_path: Optional[str] = None,
    calibration_path: Optional[str] = None,
    segmentation_path: Optional[str] = None,
    gt_rgb_split: Optional[str] = None,
    pred_rgb_split: Optional[str] = None,
    gt_depth_split: Optional[str] = None,
    calibration_split: Optional[str] = None,
    segmentation_split: Optional[str] = None,
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
        gt_rgb_split: Optional split name for the GT RGB modality.
        pred_rgb_split: Optional split name for the prediction RGB modality.
        gt_depth_split: Optional split name for the GT depth modality.
        calibration_split: Optional split name for the calibration modality.
        segmentation_split: Optional split name for the segmentation modality.

    Returns:
        A MultiModalDataset instance.
    """
    modalities: dict[str, Modality] = {
        "gt": _modality(
            path=gt_rgb_path,
            modality_key="rgb",
            split=gt_rgb_split,
        ),
        "pred": _modality(
            path=pred_rgb_path,
            modality_key="rgb",
            used_as="output",
            split=pred_rgb_split,
        ),
    }
    if gt_depth_path is not None:
        modalities["gt_depth"] = _modality(
            path=gt_depth_path,
            modality_key="depth",
            split=gt_depth_split,
        )

    hierarchical = {}
    if calibration_path is not None:
        hierarchical["calibration"] = _modality(
            path=calibration_path,
            modality_key="calibration",
            split=calibration_split,
        )
    if segmentation_path is not None:
        sky_fn = _resolve_sky_mask_loader(segmentation_path)
        hierarchical["segmentation"] = _modality(
            path=segmentation_path,
            modality_key="segmentation",
            loader=sky_fn,
            split=segmentation_split,
        )

    return MultiModalDataset(
        modalities=modalities,
        hierarchical_modalities=hierarchical if hierarchical else None,
    )


def get_depth_metadata(dataset: MultiModalDataset) -> dict[str, Any]:
    """Extract depth modality metadata from a dataset.

    Returns:
        Dict with ``scale_to_meters`` (float, always 1.0) and
        ``radial_depth`` (bool).
    """
    meta = dataset.get_modality_metadata("gt")

    return {
        # euler_loading is expected to return depth in metres.
        "scale_to_meters": 1.0,
        "radial_depth": bool(meta.get("radial_depth", True)),
    }


def get_sparse_depth_metadata(dataset: MultiModalDataset) -> dict[str, Any]:
    """Extract sparse GT and dense prediction metadata for sparse-depth eval."""
    sparse_meta = dataset.get_modality_metadata("gt")
    pred_meta = dataset.get_modality_metadata("pred")

    return {
        "scale_to_meters": 1.0,
        "representation": sparse_meta.get("representation", "point_cloud"),
        "coordinate_unit": sparse_meta.get("coordinate_unit", "meters"),
        "point_columns": sparse_meta.get("columns"),
        "pred_radial_depth": bool(pred_meta.get("radial_depth", True)),
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


# ---------------------------------------------------------------------------
# Rays (spherical direction map) dataset construction
# ---------------------------------------------------------------------------


def build_rays_eval_dataset(
    gt_rays_path: str,
    pred_rays_path: str,
    calibration_path: Optional[str] = None,
    gt_rays_split: Optional[str] = None,
    pred_rays_split: Optional[str] = None,
    calibration_split: Optional[str] = None,
) -> MultiModalDataset:
    """Build a MultiModalDataset for rays (spherical direction map) evaluation.

    The returned dataset yields samples with keys ``"gt"``, ``"pred"``,
    and optionally ``"calibration"``.

    Args:
        gt_rays_path: Path to GT ray direction map dataset root.
        pred_rays_path: Path to predicted ray direction map dataset root.
        calibration_path: Optional path to calibration dataset (used for
            FoV domain classification).
        gt_rays_split: Optional split name for the GT rays modality.
        pred_rays_split: Optional split name for the prediction rays modality.
        calibration_split: Optional split name for the calibration modality.

    Returns:
        A MultiModalDataset instance.
    """
    modalities = {
        "gt": _modality(
            path=gt_rays_path,
            modality_key="rays",
            split=gt_rays_split,
        ),
        "pred": _modality(
            path=pred_rays_path,
            modality_key="rays",
            used_as="output",
            split=pred_rays_split,
        ),
    }

    hierarchical = {}
    if calibration_path is not None:
        hierarchical["calibration"] = _modality(
            path=calibration_path,
            modality_key="calibration",
            split=calibration_split,
        )

    return MultiModalDataset(
        modalities=modalities,
        hierarchical_modalities=hierarchical if hierarchical else None,
    )


def get_rays_metadata(dataset: MultiModalDataset) -> dict[str, Any]:
    """Extract rays modality metadata from a dataset.

    Returns:
        Dict with ``fov_domain`` (``"sfov"``, ``"lfov"``, or ``"pano"``).
        Falls back to ``"lfov"`` when metadata is not available.
    """
    meta = dataset.get_modality_metadata("gt")
    return {
        "fov_domain": meta.get("fov_domain", None),
    }
