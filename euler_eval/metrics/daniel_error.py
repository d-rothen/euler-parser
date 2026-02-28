"""Structural Chromatic Error (SCE) metric for RGB images.

SCE compares chromaticity (relative RGB proportions) and weights the
chromaticity error by structural gradient magnitude.
"""

import numpy as np
from typing import Union
from scipy import ndimage


def compute_sce(
    img_pred: np.ndarray,
    img_gt: np.ndarray,
    eps: float = 1e-6,
    use_scharr_filter: bool = True,
    return_metadata: bool = False,
) -> Union[float, tuple[float, dict]]:
    """Compute Structural Chromatic Error (SCE) between RGB images.

    Args:
        img_pred: Predicted RGB image in [0, 1] range, shape (H, W, 3).
        img_gt: Ground truth RGB image in [0, 1] range, shape (H, W, 3).
        eps: Small epsilon to avoid division by zero.
        use_scharr_filter: If True, use Scharr filter for gradients (else Sobel).
        return_metadata: If True, return (sce, metadata) tuple.

    Returns:
        SCE value (lower is better).
        If return_metadata is True, returns (sce, metadata_dict).
    """
    # Chromaticity (relative RGB proportions)
    pred_sum = np.sum(img_pred, axis=-1, keepdims=True)
    gt_sum = np.sum(img_gt, axis=-1, keepdims=True)
    chroma_pred = img_pred / (pred_sum + eps)
    chroma_gt = img_gt / (gt_sum + eps)

    # Per-pixel chromaticity error (mean absolute difference across channels)
    chroma_error = np.mean(np.abs(chroma_pred - chroma_gt), axis=-1)

    # Structural gradient magnitude on luminance
    pred_gray = 0.299 * img_pred[:, :, 0] + 0.587 * img_pred[:, :, 1] + 0.114 * img_pred[:, :, 2]
    gt_gray = 0.299 * img_gt[:, :, 0] + 0.587 * img_gt[:, :, 1] + 0.114 * img_gt[:, :, 2]

    if use_scharr_filter:
        # Scharr kernels for x and y gradients
        scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]) / 16.0
        scharr_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]]) / 16.0
        dx_pred = ndimage.convolve(pred_gray, scharr_x)
        dy_pred = ndimage.convolve(pred_gray, scharr_y)
        dx_gt = ndimage.convolve(gt_gray, scharr_x)
        dy_gt = ndimage.convolve(gt_gray, scharr_y)
    else:
        dx_pred = ndimage.sobel(pred_gray, axis=1)
        dy_pred = ndimage.sobel(pred_gray, axis=0)
        dx_gt = ndimage.sobel(gt_gray, axis=1)
        dy_gt = ndimage.sobel(gt_gray, axis=0)

    grad_pred = np.sqrt(dx_pred**2 + dy_pred**2)
    grad_gt = np.sqrt(dx_gt**2 + dy_gt**2)

    # Symmetric structural weight
    weight = 0.5 * (grad_pred + grad_gt)

    weight_mean = float(np.mean(weight))
    weighted_error = float(np.mean(weight * chroma_error))

    if weight_mean < eps:
        sce = 0.0
    else:
        sce = weighted_error / (weight_mean + eps)

    if return_metadata:
        metadata = {
            "weight_mean": weight_mean,
            "chroma_error_mean": float(np.mean(chroma_error)),
            "weighted_error_mean": weighted_error,
        }
        return sce, metadata

    return sce
