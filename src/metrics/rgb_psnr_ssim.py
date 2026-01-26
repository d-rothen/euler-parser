"""PSNR and SSIM metrics for RGB images."""

import numpy as np
from typing import Optional, Union
from scipy import ndimage


def compute_rgb_psnr(
    img_pred: np.ndarray,
    img_gt: np.ndarray,
    max_val: float = 1.0,
    return_metadata: bool = False,
) -> Union[float, tuple[float, dict]]:
    """Compute PSNR between predicted and ground truth RGB images.

    Args:
        img_pred: Predicted RGB image in [0, 1] range, shape (H, W, 3).
        img_gt: Ground truth RGB image in [0, 1] range, shape (H, W, 3).
        max_val: Maximum pixel value (default 1.0 for normalized images).
        return_metadata: If True, return (psnr, metadata) tuple for sanity checking.

    Returns:
        PSNR value in dB. Higher is better.
        If return_metadata is True, returns (psnr, metadata_dict).
    """
    metadata = {
        "rgb_pred_min": float(img_pred.min()),
        "rgb_pred_max": float(img_pred.max()),
        "rgb_gt_min": float(img_gt.min()),
        "rgb_gt_max": float(img_gt.max()),
        "max_val_used": max_val,
    }

    mse = np.mean((img_pred - img_gt) ** 2)

    if mse < 1e-10:
        psnr = float("inf")
    else:
        psnr = 10 * np.log10((max_val**2) / mse)
        psnr = float(psnr)

    if return_metadata:
        return psnr, metadata
    return psnr


def compute_rgb_ssim(
    img_pred: np.ndarray,
    img_gt: np.ndarray,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
    return_metadata: bool = False,
) -> Union[float, tuple[float, dict]]:
    """Compute SSIM between predicted and ground truth RGB images.

    Computes SSIM per channel and averages.

    Args:
        img_pred: Predicted RGB image in [0, 1] range, shape (H, W, 3).
        img_gt: Ground truth RGB image in [0, 1] range, shape (H, W, 3).
        window_size: Size of the Gaussian window.
        k1: SSIM constant for luminance.
        k2: SSIM constant for contrast.
        return_metadata: If True, return (ssim, metadata) tuple for sanity checking.

    Returns:
        SSIM value in [0, 1]. Higher is better.
        If return_metadata is True, returns (ssim, metadata_dict).
    """
    metadata = {
        "rgb_pred_min": float(img_pred.min()),
        "rgb_pred_max": float(img_pred.max()),
        "rgb_gt_min": float(img_gt.min()),
        "rgb_gt_max": float(img_gt.max()),
    }

    # SSIM constants
    L = 1.0  # Dynamic range for [0, 1] images
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    # Gaussian window parameters
    sigma = window_size / 6.0
    truncate = (window_size - 1) / 2 / sigma

    ssim_per_channel = []

    for c in range(3):
        pred_c = img_pred[:, :, c]
        gt_c = img_gt[:, :, c]

        # Compute means using Gaussian filter
        mu1 = ndimage.gaussian_filter(pred_c, sigma=sigma, truncate=truncate)
        mu2 = ndimage.gaussian_filter(gt_c, sigma=sigma, truncate=truncate)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        # Compute variances and covariance
        sigma1_sq = (
            ndimage.gaussian_filter(pred_c**2, sigma=sigma, truncate=truncate)
            - mu1_sq
        )
        sigma2_sq = (
            ndimage.gaussian_filter(gt_c**2, sigma=sigma, truncate=truncate) - mu2_sq
        )
        sigma12 = (
            ndimage.gaussian_filter(pred_c * gt_c, sigma=sigma, truncate=truncate)
            - mu1_mu2
        )

        # SSIM formula
        numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

        ssim_map = numerator / denominator
        ssim_per_channel.append(np.mean(ssim_map))

    ssim_value = float(np.mean(ssim_per_channel))

    if return_metadata:
        return ssim_value, metadata
    return ssim_value


def compute_rgb_psnr_masked(
    img_pred: np.ndarray,
    img_gt: np.ndarray,
    mask: np.ndarray,
    max_val: float = 1.0,
) -> float:
    """Compute PSNR between RGB images over a masked region.

    Args:
        img_pred: Predicted RGB image in [0, 1] range, shape (H, W, 3).
        img_gt: Ground truth RGB image in [0, 1] range, shape (H, W, 3).
        mask: Boolean mask of pixels to consider, shape (H, W).
        max_val: Maximum pixel value.

    Returns:
        PSNR value in dB, or NaN if mask is empty.
    """
    if not mask.any():
        return float("nan")

    # Expand mask to 3 channels
    mask_3c = np.stack([mask, mask, mask], axis=-1)

    pred_masked = img_pred[mask_3c]
    gt_masked = img_gt[mask_3c]

    mse = np.mean((pred_masked - gt_masked) ** 2)

    if mse < 1e-10:
        return float("inf")

    psnr = 10 * np.log10((max_val**2) / mse)
    return float(psnr)
