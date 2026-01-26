"""Structural Similarity Index (SSIM) metric for depth maps."""

import numpy as np
from typing import Optional, Union
from scipy import ndimage


def compute_ssim(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
    valid_mask: Optional[np.ndarray] = None,
    return_metadata: bool = False,
) -> Union[float, tuple[float, dict]]:
    """Compute SSIM between predicted and ground truth depth maps.

    Args:
        depth_pred: Predicted depth map in meters.
        depth_gt: Ground truth depth map in meters.
        window_size: Size of the Gaussian window.
        k1: SSIM constant for luminance.
        k2: SSIM constant for contrast.
        valid_mask: Optional mask of valid pixels to consider.
        return_metadata: If True, return (ssim, metadata) tuple for sanity checking.

    Returns:
        SSIM value in [0, 1]. Higher is better.
        If return_metadata is True, returns (ssim, metadata_dict).
    """
    if valid_mask is None:
        valid_mask = (depth_gt > 0) & (depth_pred > 0)
        valid_mask = valid_mask & np.isfinite(depth_gt) & np.isfinite(depth_pred)

    metadata = {
        "valid_pixel_count": int(np.sum(valid_mask)),
        "depth_range": None,
        "depth_min": None,
        "depth_max": None,
    }

    if not valid_mask.any():
        if return_metadata:
            return 0.0, metadata
        return 0.0

    # Normalize depths to [0, 1] for SSIM computation
    valid_min = min(depth_pred[valid_mask].min(), depth_gt[valid_mask].min())
    valid_max = max(depth_pred[valid_mask].max(), depth_gt[valid_mask].max())

    metadata["depth_min"] = float(valid_min)
    metadata["depth_max"] = float(valid_max)
    metadata["depth_range"] = float(valid_max - valid_min)

    if valid_max - valid_min < 1e-8:
        if return_metadata:
            return 1.0, metadata  # Constant images are identical
        return 1.0  # Constant images are identical

    pred_norm = (depth_pred - valid_min) / (valid_max - valid_min)
    gt_norm = (depth_gt - valid_min) / (valid_max - valid_min)

    # Apply mask by setting invalid pixels to 0
    pred_norm = np.where(valid_mask, pred_norm, 0)
    gt_norm = np.where(valid_mask, gt_norm, 0)

    # SSIM constants
    L = 1.0  # Dynamic range
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    # Create Gaussian window
    sigma = window_size / 6.0
    truncate = (window_size - 1) / 2 / sigma

    # Compute means using Gaussian filter
    mu1 = ndimage.gaussian_filter(pred_norm, sigma=sigma, truncate=truncate)
    mu2 = ndimage.gaussian_filter(gt_norm, sigma=sigma, truncate=truncate)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = (
        ndimage.gaussian_filter(pred_norm**2, sigma=sigma, truncate=truncate) - mu1_sq
    )
    sigma2_sq = (
        ndimage.gaussian_filter(gt_norm**2, sigma=sigma, truncate=truncate) - mu2_sq
    )
    sigma12 = (
        ndimage.gaussian_filter(pred_norm * gt_norm, sigma=sigma, truncate=truncate)
        - mu1_mu2
    )

    # SSIM formula
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / denominator

    # Average SSIM over valid regions
    # Create a weight mask based on Gaussian filter of valid mask
    weight_mask = ndimage.gaussian_filter(
        valid_mask.astype(np.float32), sigma=sigma, truncate=truncate
    )
    weight_mask = weight_mask > 0.5  # Threshold to get regions with enough valid pixels

    if not weight_mask.any():
        if return_metadata:
            return 0.0, metadata
        return 0.0

    ssim_value = float(np.mean(ssim_map[weight_mask]))
    if return_metadata:
        return ssim_value, metadata
    return ssim_value


def compute_ssim_batch(
    depths_pred: list[np.ndarray],
    depths_gt: list[np.ndarray],
    window_size: int = 11,
) -> list[float]:
    """Compute SSIM for a batch of depth map pairs.

    Args:
        depths_pred: List of predicted depth maps.
        depths_gt: List of ground truth depth maps.
        window_size: Size of the Gaussian window.

    Returns:
        List of SSIM values.
    """
    return [
        compute_ssim(pred, gt, window_size)
        for pred, gt in zip(depths_pred, depths_gt)
    ]
