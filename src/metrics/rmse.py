"""Root Mean Square Error (RMSE) metric for depth maps."""

import numpy as np
from typing import Optional


def compute_rmse(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    """Compute RMSE between predicted and ground truth depth maps.

    RMSE = sqrt(mean((pred - gt)^2))

    Args:
        depth_pred: Predicted depth map in meters.
        depth_gt: Ground truth depth map in meters.
        valid_mask: Optional mask of valid pixels to consider.

    Returns:
        RMSE value in meters. Lower is better.
    """
    if valid_mask is None:
        valid_mask = (depth_gt > 0) & (depth_pred > 0)
        valid_mask = valid_mask & np.isfinite(depth_gt) & np.isfinite(depth_pred)

    if not valid_mask.any():
        return float("nan")

    pred_valid = depth_pred[valid_mask]
    gt_valid = depth_gt[valid_mask]

    rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))
    return float(rmse)


def compute_rmse_per_pixel(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute per-pixel squared error for later aggregation.

    Args:
        depth_pred: Predicted depth map in meters.
        depth_gt: Ground truth depth map in meters.
        valid_mask: Optional mask of valid pixels to consider.

    Returns:
        Array of per-pixel squared errors for valid pixels.
    """
    if valid_mask is None:
        valid_mask = (depth_gt > 0) & (depth_pred > 0)
        valid_mask = valid_mask & np.isfinite(depth_gt) & np.isfinite(depth_pred)

    if not valid_mask.any():
        return np.array([])

    pred_valid = depth_pred[valid_mask]
    gt_valid = depth_gt[valid_mask]

    squared_error = (pred_valid - gt_valid) ** 2
    return squared_error


def compute_rmse_stats(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> dict:
    """Compute RMSE statistics for a depth map pair.

    Args:
        depth_pred: Predicted depth map in meters.
        depth_gt: Ground truth depth map in meters.
        valid_mask: Optional mask of valid pixels to consider.

    Returns:
        Dictionary with mean, median, and p90 RMSE values.
    """
    squared_errors = compute_rmse_per_pixel(depth_pred, depth_gt, valid_mask)

    if len(squared_errors) == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}

    # For per-pixel RMSE stats, we take sqrt of the squared errors
    errors = np.sqrt(squared_errors)

    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p90": float(np.percentile(errors, 90)),
    }


def aggregate_rmse(
    squared_errors: list[np.ndarray],
) -> dict:
    """Aggregate RMSE values from multiple depth map pairs.

    Args:
        squared_errors: List of per-pixel squared error arrays from multiple pairs.

    Returns:
        Dictionary with aggregated median and p90 RMSE values.
    """
    all_values = np.concatenate([v for v in squared_errors if len(v) > 0])

    if len(all_values) == 0:
        return {"median": float("nan"), "p90": float("nan")}

    # Take sqrt to convert squared errors to absolute errors
    errors = np.sqrt(all_values)

    return {
        "median": float(np.median(errors)),
        "p90": float(np.percentile(errors, 90)),
    }
