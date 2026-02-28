"""Absolute Relative Error (AbsRel) metric for depth maps."""

import numpy as np
from typing import Optional, Union


def compute_absrel(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    return_metadata: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, dict]]:
    """Compute per-pixel Absolute Relative Error between predicted and GT depth.

    AbsRel = |pred - gt| / gt

    Args:
        depth_pred: Predicted depth map in meters.
        depth_gt: Ground truth depth map in meters.
        valid_mask: Optional mask of valid pixels to consider.
        return_metadata: If True, return (absrel, metadata) tuple for sanity checking.

    Returns:
        Array of per-pixel AbsRel values for valid pixels.
        If return_metadata is True, returns (absrel_array, metadata_dict).
    """
    if valid_mask is None:
        valid_mask = (depth_gt > 0) & (depth_pred > 0)
        valid_mask = valid_mask & np.isfinite(depth_gt) & np.isfinite(depth_pred)

    metadata = {
        "valid_pixel_count": int(np.sum(valid_mask)),
        "median": None,
        "p90": None,
    }

    if not valid_mask.any():
        if return_metadata:
            return np.array([]), metadata
        return np.array([])

    pred_valid = depth_pred[valid_mask]
    gt_valid = depth_gt[valid_mask]

    absrel = np.abs(pred_valid - gt_valid) / gt_valid

    metadata["median"] = float(np.median(absrel))
    metadata["p90"] = float(np.percentile(absrel, 90))

    if return_metadata:
        return absrel, metadata
    return absrel


def compute_absrel_stats(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> dict:
    """Compute AbsRel statistics (mean, median, p90) for a depth map pair.

    Args:
        depth_pred: Predicted depth map in meters.
        depth_gt: Ground truth depth map in meters.
        valid_mask: Optional mask of valid pixels to consider.

    Returns:
        Dictionary with mean, median, and p90 AbsRel values.
    """
    absrel = compute_absrel(depth_pred, depth_gt, valid_mask)

    if len(absrel) == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}

    return {
        "mean": float(np.mean(absrel)),
        "median": float(np.median(absrel)),
        "p90": float(np.percentile(absrel, 90)),
    }


def aggregate_absrel(
    absrel_values: list[np.ndarray],
) -> dict:
    """Aggregate AbsRel values from multiple depth map pairs.

    Args:
        absrel_values: List of per-pixel AbsRel arrays from multiple pairs.

    Returns:
        Dictionary with aggregated median and p90 values.
    """
    all_values = np.concatenate([v for v in absrel_values if len(v) > 0])

    if len(all_values) == 0:
        return {"median": float("nan"), "p90": float("nan")}

    return {
        "median": float(np.median(all_values)),
        "p90": float(np.percentile(all_values, 90)),
    }
