"""Peak Signal-to-Noise Ratio (PSNR) metric for depth maps."""

import numpy as np
from typing import Optional, Union


def compute_psnr(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    max_val: Optional[float] = None,
    valid_mask: Optional[np.ndarray] = None,
    return_metadata: bool = False,
) -> Union[float, tuple[float, dict]]:
    """Compute PSNR between predicted and ground truth depth maps.

    Args:
        depth_pred: Predicted depth map in meters.
        depth_gt: Ground truth depth map in meters.
        max_val: Maximum depth value for PSNR calculation.
                 If None, uses the max of ground truth valid pixels.
        valid_mask: Optional mask of valid pixels to consider.
        return_metadata: If True, return (psnr, metadata) tuple for sanity checking.

    Returns:
        PSNR value in dB. Higher is better.
        If return_metadata is True, returns (psnr, metadata_dict).
    """
    if valid_mask is None:
        valid_mask = (depth_gt > 0) & (depth_pred > 0)
        valid_mask = valid_mask & np.isfinite(depth_gt) & np.isfinite(depth_pred)

    metadata = {
        "max_val_used": None,
        "valid_pixel_count": int(np.sum(valid_mask)),
        "depth_gt_max": None,
        "depth_gt_min": None,
    }

    if not valid_mask.any():
        if return_metadata:
            return 0.0, metadata
        return 0.0

    pred_valid = depth_pred[valid_mask]
    gt_valid = depth_gt[valid_mask]

    metadata["depth_gt_max"] = float(gt_valid.max())
    metadata["depth_gt_min"] = float(gt_valid.min())

    if max_val is None:
        max_val = gt_valid.max()

    metadata["max_val_used"] = float(max_val)

    mse = np.mean((pred_valid - gt_valid) ** 2)

    if mse < 1e-10:
        psnr = float("inf")
    else:
        psnr = 10 * np.log10((max_val**2) / mse)
        psnr = float(psnr)

    if return_metadata:
        return psnr, metadata
    return psnr


def compute_psnr_batch(
    depths_pred: list[np.ndarray],
    depths_gt: list[np.ndarray],
    max_val: Optional[float] = None,
) -> list[float]:
    """Compute PSNR for a batch of depth map pairs.

    Args:
        depths_pred: List of predicted depth maps.
        depths_gt: List of ground truth depth maps.
        max_val: Maximum depth value for PSNR calculation.

    Returns:
        List of PSNR values.
    """
    return [
        compute_psnr(pred, gt, max_val) for pred, gt in zip(depths_pred, depths_gt)
    ]
