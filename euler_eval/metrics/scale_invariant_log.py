"""Scale-Invariant Logarithmic Error (SILog) metric for depth maps."""

import numpy as np
from typing import Optional


def compute_scale_invariant_log_error(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    lambda_weight: float = 1.0,
) -> float:
    """Compute Scale-Invariant Logarithmic Error between depth maps.

    SILog = sqrt(1/n * sum(d_i^2) - lambda/n^2 * (sum(d_i))^2)
    where d_i = log(pred_i) - log(gt_i)

    This metric measures the quality of relative depth prediction,
    being invariant to global scale differences.

    Args:
        depth_pred: Predicted depth map in meters.
        depth_gt: Ground truth depth map in meters.
        valid_mask: Optional mask of valid pixels to consider.
        lambda_weight: Weight for the scale-invariant term (default 1.0).

    Returns:
        SILog error value. Lower is better.
    """
    if valid_mask is None:
        valid_mask = (depth_gt > 0) & (depth_pred > 0)
        valid_mask = valid_mask & np.isfinite(depth_gt) & np.isfinite(depth_pred)

    if not valid_mask.any():
        return float("nan")

    pred_valid = depth_pred[valid_mask]
    gt_valid = depth_gt[valid_mask]

    # Log difference
    log_diff = np.log(pred_valid) - np.log(gt_valid)

    n = len(log_diff)

    # SILog formula
    silog = np.sqrt(
        np.mean(log_diff**2) - lambda_weight * (np.mean(log_diff) ** 2)
    )

    return float(silog)


def compute_silog_per_pixel(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute per-pixel log differences for later aggregation.

    Args:
        depth_pred: Predicted depth map in meters.
        depth_gt: Ground truth depth map in meters.
        valid_mask: Optional mask of valid pixels to consider.

    Returns:
        Array of per-pixel absolute log differences for valid pixels.
    """
    if valid_mask is None:
        valid_mask = (depth_gt > 0) & (depth_pred > 0)
        valid_mask = valid_mask & np.isfinite(depth_gt) & np.isfinite(depth_pred)

    if not valid_mask.any():
        return np.array([])

    pred_valid = depth_pred[valid_mask]
    gt_valid = depth_gt[valid_mask]

    # Absolute log difference (for percentile computation)
    log_diff = np.abs(np.log(pred_valid) - np.log(gt_valid))
    return log_diff


def compute_silog_stats(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> dict:
    """Compute SILog statistics for a depth map pair.

    Args:
        depth_pred: Predicted depth map in meters.
        depth_gt: Ground truth depth map in meters.
        valid_mask: Optional mask of valid pixels to consider.

    Returns:
        Dictionary with silog, median, and p90 values.
    """
    silog = compute_scale_invariant_log_error(depth_pred, depth_gt, valid_mask)
    log_diffs = compute_silog_per_pixel(depth_pred, depth_gt, valid_mask)

    if len(log_diffs) == 0:
        return {"silog": float("nan"), "median": float("nan"), "p90": float("nan")}

    return {
        "silog": silog,
        "median": float(np.median(log_diffs)),
        "p90": float(np.percentile(log_diffs, 90)),
    }


def aggregate_silog(
    log_diffs: list[np.ndarray],
) -> dict:
    """Aggregate SILog values from multiple depth map pairs.

    Args:
        log_diffs: List of per-pixel absolute log difference arrays.

    Returns:
        Dictionary with aggregated median and p90 values.
    """
    all_values = np.concatenate([v for v in log_diffs if len(v) > 0])

    if len(all_values) == 0:
        return {"median": float("nan"), "p90": float("nan")}

    return {
        "median": float(np.median(all_values)),
        "p90": float(np.percentile(all_values, 90)),
    }
