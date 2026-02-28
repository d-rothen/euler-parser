"""Tail error metrics (p95/p99) for capturing rare ugly regions in RGB images."""

import numpy as np
from typing import Optional


def compute_tail_errors(
    img_pred: np.ndarray,
    img_gt: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """Compute p95 and p99 of per-pixel absolute error.

    These metrics capture rare but significant errors that might indicate
    ugly regions or artifacts in the prediction.

    Args:
        img_pred: Predicted RGB image in [0, 1] range, shape (H, W, 3).
        img_gt: Ground truth RGB image in [0, 1] range, shape (H, W, 3).
        mask: Optional boolean mask of pixels to consider.

    Returns:
        Dictionary with p95 and p99 absolute error values.
    """
    # Compute per-pixel absolute error (mean over RGB channels)
    abs_error = np.abs(img_pred - img_gt).mean(axis=-1)

    if mask is not None:
        abs_error = abs_error[mask]
    else:
        abs_error = abs_error.flatten()

    if len(abs_error) == 0:
        return {
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }

    return {
        "p95": float(np.percentile(abs_error, 95)),
        "p99": float(np.percentile(abs_error, 99)),
        "max": float(np.max(abs_error)),
    }


def compute_tail_errors_per_channel(
    img_pred: np.ndarray,
    img_gt: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """Compute p95 and p99 per RGB channel.

    Args:
        img_pred: Predicted RGB image in [0, 1] range, shape (H, W, 3).
        img_gt: Ground truth RGB image in [0, 1] range, shape (H, W, 3).
        mask: Optional boolean mask of pixels to consider.

    Returns:
        Dictionary with per-channel tail error values.
    """
    results = {}

    for c, channel_name in enumerate(["r", "g", "b"]):
        abs_error = np.abs(img_pred[:, :, c] - img_gt[:, :, c])

        if mask is not None:
            abs_error = abs_error[mask]
        else:
            abs_error = abs_error.flatten()

        if len(abs_error) == 0:
            results[channel_name] = {
                "p95": float("nan"),
                "p99": float("nan"),
            }
        else:
            results[channel_name] = {
                "p95": float(np.percentile(abs_error, 95)),
                "p99": float(np.percentile(abs_error, 99)),
            }

    return results


def get_tail_error_pixels(
    img_pred: np.ndarray,
    img_gt: np.ndarray,
    percentile: float = 95,
) -> np.ndarray:
    """Get mask of pixels in the error tail (above given percentile).

    Useful for visualizing where the worst errors occur.

    Args:
        img_pred: Predicted RGB image in [0, 1] range, shape (H, W, 3).
        img_gt: Ground truth RGB image in [0, 1] range, shape (H, W, 3).
        percentile: Percentile threshold.

    Returns:
        Boolean mask of tail error pixels.
    """
    abs_error = np.abs(img_pred - img_gt).mean(axis=-1)
    threshold = np.percentile(abs_error, percentile)
    return abs_error >= threshold


def aggregate_tail_errors(
    tail_error_arrays: list[np.ndarray],
) -> dict:
    """Aggregate tail errors from multiple image pairs.

    Args:
        tail_error_arrays: List of per-pixel absolute error arrays (flattened).

    Returns:
        Dictionary with aggregated p95 and p99 values.
    """
    all_errors = np.concatenate([e.flatten() for e in tail_error_arrays if len(e) > 0])

    if len(all_errors) == 0:
        return {
            "p95": float("nan"),
            "p99": float("nan"),
        }

    return {
        "p95": float(np.percentile(all_errors, 95)),
        "p99": float(np.percentile(all_errors, 99)),
    }
