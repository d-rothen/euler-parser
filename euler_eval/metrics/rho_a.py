"""ρ_A metric for spherical direction map (ray) evaluation.

Computes the Area Under the Curve (AUC) of the angular accuracy curve
between predicted and ground-truth camera ray directions.  The AUC is
evaluated up to a FoV-dependent angular threshold:

- 15° for Small Field of View (S.FoV) cameras
- 20° for Large Field of View (L.FoV) cameras
- 30° for Panoramic (Pano) cameras

This avoids parametric evaluations (e.g. focal length or FoV) that lack
generality across diverse camera models.
"""

import numpy as np
from typing import Optional, Union


# FoV domain → angular threshold mapping
FOV_THRESHOLDS = {
    "sfov": 15.0,
    "lfov": 20.0,
    "pano": 30.0,
}

# Diagonal FoV boundaries for automatic domain classification (degrees)
_SFOV_MAX = 90.0
_PANO_MIN = 160.0


def classify_fov_domain(
    intrinsics_K: np.ndarray,
    height: int,
    width: int,
) -> str:
    """Classify camera into FoV domain from intrinsics.

    Computes the diagonal FoV from the intrinsics matrix and image
    dimensions, then maps to one of ``"sfov"``, ``"lfov"``, or ``"pano"``.

    Args:
        intrinsics_K: ``(3, 3)`` camera intrinsics matrix.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        One of ``"sfov"``, ``"lfov"``, ``"pano"``.
    """
    fx = float(intrinsics_K[0, 0])
    fy = float(intrinsics_K[1, 1])

    # Half-angles for each axis
    half_fov_x = np.degrees(np.arctan2(width / 2.0, fx))
    half_fov_y = np.degrees(np.arctan2(height / 2.0, fy))

    # Diagonal FoV (full angle)
    diag_fov = 2.0 * np.degrees(
        np.arctan(np.sqrt(np.tan(np.radians(half_fov_x)) ** 2 + np.tan(np.radians(half_fov_y)) ** 2))
    )

    if diag_fov <= _SFOV_MAX:
        return "sfov"
    elif diag_fov >= _PANO_MIN:
        return "pano"
    return "lfov"


def get_threshold_for_domain(fov_domain: str) -> float:
    """Return the angular threshold (degrees) for a FoV domain.

    Args:
        fov_domain: One of ``"sfov"``, ``"lfov"``, ``"pano"``.

    Returns:
        Threshold in degrees.

    Raises:
        ValueError: If *fov_domain* is not recognized.
    """
    key = fov_domain.lower()
    if key not in FOV_THRESHOLDS:
        raise ValueError(
            f"Unknown FoV domain '{fov_domain}'. "
            f"Expected one of {sorted(FOV_THRESHOLDS)}."
        )
    return FOV_THRESHOLDS[key]


def compute_angular_errors(
    directions_pred: np.ndarray,
    directions_gt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    return_metadata: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, dict]]:
    """Compute per-pixel angular errors between direction maps.

    Args:
        directions_pred: Predicted direction map ``(H, W, 3)`` of unit
            vectors.
        directions_gt: Ground-truth direction map ``(H, W, 3)`` of unit
            vectors.
        valid_mask: Optional ``(H, W)`` boolean mask.  Only pixels where
            the mask is True are included.
        return_metadata: If True, return ``(errors, metadata)`` tuple.

    Returns:
        1-D array of angular errors in degrees for valid pixels.
        If *return_metadata* is True, also returns a metadata dict.
    """
    if valid_mask is None:
        # Valid where both directions have finite, non-zero norm
        norm_pred = np.linalg.norm(directions_pred, axis=-1)
        norm_gt = np.linalg.norm(directions_gt, axis=-1)
        valid_mask = (
            np.isfinite(norm_pred)
            & np.isfinite(norm_gt)
            & (norm_pred > 1e-8)
            & (norm_gt > 1e-8)
        )

    valid_count = int(np.sum(valid_mask))

    metadata = {
        "valid_pixel_count": valid_count,
        "total_pixel_count": int(valid_mask.size),
        "mean_angular_error": None,
        "median_angular_error": None,
    }

    if valid_count == 0:
        if return_metadata:
            return np.array([]), metadata
        return np.array([])

    pred = directions_pred[valid_mask]  # (N, 3)
    gt = directions_gt[valid_mask]  # (N, 3)

    # Normalize to unit vectors
    pred = pred / np.linalg.norm(pred, axis=-1, keepdims=True).clip(min=1e-8)
    gt = gt / np.linalg.norm(gt, axis=-1, keepdims=True).clip(min=1e-8)

    # Dot product → angular error
    dot = np.sum(pred * gt, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    angles = np.degrees(np.arccos(dot))

    metadata["mean_angular_error"] = float(np.mean(angles))
    metadata["median_angular_error"] = float(np.median(angles))

    if return_metadata:
        return angles, metadata
    return angles


def compute_rho_a(
    angular_errors: np.ndarray,
    threshold_deg: float,
    num_bins: int = 100,
) -> float:
    """Compute the ρ_A metric (AUC of angular accuracy curve).

    Builds a cumulative accuracy curve — fraction of pixels with angular
    error ≤ θ — for θ in ``[0, threshold_deg]``, then returns the
    normalized area under the curve.

    Args:
        angular_errors: 1-D array of angular errors in degrees.
        threshold_deg: Maximum angle (degrees) for AUC integration.
        num_bins: Number of evaluation points for the curve.

    Returns:
        ρ_A value in ``[0, 1]``.  1.0 means all pixels have zero error.
    """
    if len(angular_errors) == 0:
        return float("nan")

    thresholds = np.linspace(0.0, threshold_deg, num_bins + 1)
    accuracies = np.array(
        [float(np.mean(angular_errors <= t)) for t in thresholds]
    )
    _trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    auc = float(_trapezoid(accuracies, thresholds) / threshold_deg)
    return auc


def aggregate_rho_a(rho_a_values: list[float]) -> dict:
    """Aggregate per-image ρ_A values across a dataset.

    Args:
        rho_a_values: List of per-image ρ_A scores.

    Returns:
        Dict with ``mean``, ``median``, ``std``.
    """
    valid = [v for v in rho_a_values if np.isfinite(v)]
    if not valid:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
        }
    arr = np.array(valid)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }


def aggregate_angular_errors(angle_arrays: list[np.ndarray]) -> dict:
    """Aggregate angular errors from multiple image pairs.

    Args:
        angle_arrays: List of per-pixel angular error arrays.

    Returns:
        Dict with ``mean_angle``, ``median_angle``, and percentage-below
        thresholds at 5°, 10°, and the common FoV thresholds.
    """
    non_empty = [a for a in angle_arrays if len(a) > 0]
    if not non_empty:
        return {
            "mean_angle": float("nan"),
            "median_angle": float("nan"),
            "percent_below_5": float("nan"),
            "percent_below_10": float("nan"),
            "percent_below_15": float("nan"),
            "percent_below_20": float("nan"),
            "percent_below_30": float("nan"),
        }

    all_angles = np.concatenate(non_empty)
    return {
        "mean_angle": float(np.mean(all_angles)),
        "median_angle": float(np.median(all_angles)),
        "percent_below_5": float(np.mean(all_angles < 5.0) * 100),
        "percent_below_10": float(np.mean(all_angles < 10.0) * 100),
        "percent_below_15": float(np.mean(all_angles < 15.0) * 100),
        "percent_below_20": float(np.mean(all_angles < 20.0) * 100),
        "percent_below_30": float(np.mean(all_angles < 30.0) * 100),
    }
