"""Utility functions for depth and RGB processing."""

import numpy as np
from typing import Optional


def convert_planar_to_radial(depth_planar: np.ndarray, intrinsics: dict) -> np.ndarray:
    """Convert planar depth (Z-depth) to radial depth (Euclidean distance).

    Args:
        depth_planar: Depth map where value is Z distance.
        intrinsics: Dictionary with fx, fy, cx, cy.

    Returns:
        Depth map where value is Euclidean distance from camera.
    """
    height, width = depth_planar.shape

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    # Create grid of pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Normalized ray direction (X/Z, Y/Z, 1)
    x_normalized = (u - cx) / fx
    y_normalized = (v - cy) / fy

    # Correction factor = magnitude of ray
    correction_factor = np.sqrt(x_normalized**2 + y_normalized**2 + 1)

    return depth_planar * correction_factor


def normalize_depth_for_visualization(
    depth: np.ndarray,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
) -> np.ndarray:
    """Normalize depth map to [0, 1] range for visualization.

    Args:
        depth: Depth map in meters.
        min_depth: Minimum depth value (uses array min if None).
        max_depth: Maximum depth value (uses array max if None).

    Returns:
        Normalized depth map in [0, 1] range.
    """
    valid_mask = (depth > 0) & np.isfinite(depth)

    if not valid_mask.any():
        return np.zeros_like(depth)

    if min_depth is None:
        min_depth = depth[valid_mask].min()
    if max_depth is None:
        max_depth = depth[valid_mask].max()

    if max_depth - min_depth < 1e-8:
        return np.zeros_like(depth)

    normalized = (depth - min_depth) / (max_depth - min_depth)
    normalized = np.clip(normalized, 0, 1)
    normalized[~valid_mask] = 0

    return normalized


def get_valid_mask(
    depth1: np.ndarray, depth2: np.ndarray, min_depth: float = 1e-3
) -> np.ndarray:
    """Get mask of pixels valid in both depth maps.

    Args:
        depth1: First depth map.
        depth2: Second depth map.
        min_depth: Minimum valid depth value.

    Returns:
        Boolean mask of valid pixels.
    """
    valid1 = (depth1 > min_depth) & np.isfinite(depth1)
    valid2 = (depth2 > min_depth) & np.isfinite(depth2)
    return valid1 & valid2


def depth_to_3channel(depth: np.ndarray) -> np.ndarray:
    """Convert single-channel depth to 3-channel for metrics requiring RGB input.

    Args:
        depth: Single-channel depth map.

    Returns:
        3-channel depth map (H, W, 3).
    """
    normalized = normalize_depth_for_visualization(depth)
    return np.stack([normalized, normalized, normalized], axis=-1)


def get_depth_bins(
    depth: np.ndarray,
    near_threshold: float = 1.0,
    far_threshold: float = 5.0,
) -> dict:
    """Get masks for near/mid/far depth bins.

    Args:
        depth: Depth map in meters.
        near_threshold: Depth threshold for near bin (0, near_threshold].
        far_threshold: Depth threshold for far bin (far_threshold, inf).

    Returns:
        Dictionary with 'near', 'mid', 'far' boolean masks.
    """
    valid_mask = (depth > 0) & np.isfinite(depth)

    near_mask = valid_mask & (depth <= near_threshold)
    mid_mask = valid_mask & (depth > near_threshold) & (depth <= far_threshold)
    far_mask = valid_mask & (depth > far_threshold)

    return {
        "near": near_mask,
        "mid": mid_mask,
        "far": far_mask,
    }
