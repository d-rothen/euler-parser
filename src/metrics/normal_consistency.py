"""Normal Consistency metric for depth maps.

Computes surface normals from depth maps and measures their consistency.
"""

import numpy as np
from typing import Optional, Union
from scipy import ndimage


def depth_to_normals(
    depth: np.ndarray,
    focal_length: float = 1.0,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute surface normals from a depth map.

    Uses finite differences to estimate the surface gradient,
    then computes the normal as the cross product of tangent vectors.

    Args:
        depth: Depth map in meters (H, W).
        focal_length: Focal length for proper 3D reconstruction.
                      If not known, use 1.0 for relative normals.
        valid_mask: Optional mask of valid pixels.

    Returns:
        Normal map of shape (H, W, 3) with unit normals.
    """
    height, width = depth.shape

    if valid_mask is None:
        valid_mask = (depth > 0) & np.isfinite(depth)

    # Create pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Convert to 3D points (assuming principal point at center)
    cx, cy = width / 2, height / 2
    x = (u - cx) * depth / focal_length
    y = (v - cy) * depth / focal_length
    z = depth

    # Compute gradients using Sobel filters for better noise handling
    dz_dx = ndimage.sobel(z, axis=1, mode="constant") / 8.0
    dz_dy = ndimage.sobel(z, axis=0, mode="constant") / 8.0

    # For proper normals, we need gradients in 3D space
    # Using the depth gradient approach: n = (-dz/dx, -dz/dy, 1) normalized
    normals = np.zeros((height, width, 3), dtype=np.float32)
    normals[:, :, 0] = -dz_dx
    normals[:, :, 1] = -dz_dy
    normals[:, :, 2] = 1.0

    # Normalize
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    norm = np.maximum(norm, 1e-8)  # Avoid division by zero
    normals = normals / norm

    # Set invalid pixels to zero
    normals[~valid_mask] = 0

    return normals


def compute_normal_consistency(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    focal_length: float = 1.0,
) -> dict:
    """Compute normal consistency between predicted and GT depth maps.

    Measures how well the surface normals derived from the predicted depth
    match those from the ground truth.

    Args:
        depth_pred: Predicted depth map in meters.
        depth_gt: Ground truth depth map in meters.
        valid_mask: Optional mask of valid pixels to consider.
        focal_length: Focal length for normal computation.

    Returns:
        Dictionary with:
        - mean_angle: Mean angular error in degrees
        - median_angle: Median angular error in degrees
        - consistency: Mean dot product (1 = perfect, 0 = perpendicular)
        - percent_below_11_25: Percentage of pixels with angle < 11.25°
        - percent_below_22_5: Percentage of pixels with angle < 22.5°
        - percent_below_30: Percentage of pixels with angle < 30°
    """
    if valid_mask is None:
        valid_mask = (depth_gt > 0) & (depth_pred > 0)
        valid_mask = valid_mask & np.isfinite(depth_gt) & np.isfinite(depth_pred)

    # Compute normals
    normals_pred = depth_to_normals(depth_pred, focal_length, valid_mask)
    normals_gt = depth_to_normals(depth_gt, focal_length, valid_mask)

    # Erode valid mask to avoid edge artifacts from normal computation
    from scipy.ndimage import binary_erosion

    kernel = np.ones((3, 3), dtype=bool)
    valid_mask = binary_erosion(valid_mask, kernel)

    if not valid_mask.any():
        return {
            "mean_angle": float("nan"),
            "median_angle": float("nan"),
            "consistency": float("nan"),
            "percent_below_11_25": float("nan"),
            "percent_below_22_5": float("nan"),
            "percent_below_30": float("nan"),
        }

    # Compute dot products (cosine of angle between normals)
    dot_products = np.sum(normals_pred * normals_gt, axis=2)
    dot_products = np.clip(dot_products[valid_mask], -1.0, 1.0)

    # Convert to angles in degrees
    angles = np.arccos(dot_products) * 180.0 / np.pi

    return {
        "mean_angle": float(np.mean(angles)),
        "median_angle": float(np.median(angles)),
        "consistency": float(np.mean(dot_products)),
        "percent_below_11_25": float(np.mean(angles < 11.25) * 100),
        "percent_below_22_5": float(np.mean(angles < 22.5) * 100),
        "percent_below_30": float(np.mean(angles < 30.0) * 100),
    }


def compute_normal_angles(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    focal_length: float = 1.0,
    return_metadata: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, dict]]:
    """Compute per-pixel angular errors between normals for aggregation.

    Args:
        depth_pred: Predicted depth map in meters.
        depth_gt: Ground truth depth map in meters.
        valid_mask: Optional mask of valid pixels to consider.
        focal_length: Focal length for normal computation.
        return_metadata: If True, return (angles, metadata) tuple for sanity checking.

    Returns:
        Array of angular errors in degrees for valid pixels.
        If return_metadata is True, returns (angles_array, metadata_dict).
    """
    if valid_mask is None:
        valid_mask = (depth_gt > 0) & (depth_pred > 0)
        valid_mask = valid_mask & np.isfinite(depth_gt) & np.isfinite(depth_pred)

    initial_valid_count = int(np.sum(valid_mask))

    normals_pred = depth_to_normals(depth_pred, focal_length, valid_mask)
    normals_gt = depth_to_normals(depth_gt, focal_length, valid_mask)

    # Erode valid mask
    from scipy.ndimage import binary_erosion

    kernel = np.ones((3, 3), dtype=bool)
    eroded_mask = binary_erosion(valid_mask, kernel)
    valid_mask = eroded_mask if eroded_mask is not None else valid_mask

    valid_after_erosion = int(np.sum(valid_mask))

    metadata = {
        "valid_pixels_before_erosion": initial_valid_count,
        "valid_pixels_after_erosion": valid_after_erosion,
        "focal_length_used": focal_length,
        "mean_angle": None,
    }

    if not valid_mask.any():
        if return_metadata:
            return np.array([]), metadata
        return np.array([])

    dot_products = np.sum(normals_pred * normals_gt, axis=2)
    dot_products = np.clip(dot_products[valid_mask], -1.0, 1.0)
    angles = np.arccos(dot_products) * 180.0 / np.pi

    metadata["mean_angle"] = float(np.mean(angles))

    if return_metadata:
        return angles, metadata
    return angles


def aggregate_normal_consistency(
    angle_arrays: list[np.ndarray],
) -> dict:
    """Aggregate normal consistency from multiple depth map pairs.

    Args:
        angle_arrays: List of per-pixel angular error arrays.

    Returns:
        Dictionary with aggregated statistics.
    """
    all_angles = np.concatenate([a for a in angle_arrays if len(a) > 0])

    if len(all_angles) == 0:
        return {
            "mean_angle": float("nan"),
            "median_angle": float("nan"),
            "percent_below_11_25": float("nan"),
            "percent_below_22_5": float("nan"),
            "percent_below_30": float("nan"),
        }

    return {
        "mean_angle": float(np.mean(all_angles)),
        "median_angle": float(np.median(all_angles)),
        "percent_below_11_25": float(np.mean(all_angles < 11.25) * 100),
        "percent_below_22_5": float(np.mean(all_angles < 22.5) * 100),
        "percent_below_30": float(np.mean(all_angles < 30.0) * 100),
    }
