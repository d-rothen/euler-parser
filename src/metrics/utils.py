"""Utility functions for depth processing and loading."""

import numpy as np
from pathlib import Path
from typing import Optional, Union
from PIL import Image


def load_depth_file(
    file_path: Union[str, Path],
    depth_scale: float = 1.0,
    intrinsics: Optional[dict] = None,
) -> np.ndarray:
    """Load a depth file and convert to meters.

    Supports .npy and .png files.

    Args:
        file_path: Path to the depth file.
        depth_scale: Multiplier to convert raw values to meters.
        intrinsics: Optional camera intrinsics for planar-to-radial conversion.

    Returns:
        Depth map in meters as a 2D numpy array.

    Raises:
        ValueError: If file format is not supported.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".npy":
        depth = np.load(file_path)
    elif suffix == ".png":
        # Load PNG as 16-bit or 8-bit depth
        img = Image.open(file_path)
        depth = np.array(img).astype(np.float64)
    else:
        raise ValueError(f"Unsupported depth file format: {suffix}")

    # Apply depth scale to convert to meters
    depth = depth * depth_scale

    # Convert planar to radial if intrinsics provided
    if intrinsics is not None:
        depth = convert_planar_to_radial(depth, intrinsics)

    return depth.astype(np.float32)


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
