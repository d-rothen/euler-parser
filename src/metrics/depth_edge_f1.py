"""Depth Edge F1 metric for evaluating depth discontinuity preservation."""

import numpy as np
from typing import Optional, Tuple
from scipy import ndimage
from scipy.ndimage import binary_dilation


def detect_depth_edges(
    depth: np.ndarray,
    threshold: float = 0.1,
    method: str = "relative",
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Detect depth edges (discontinuities) in a depth map.

    Args:
        depth: Depth map in meters.
        threshold: Edge detection threshold.
                   For 'relative' method: fraction of depth (e.g., 0.1 = 10%)
                   For 'absolute' method: meters
        method: Edge detection method ('relative', 'absolute', 'sobel').
        valid_mask: Optional mask of valid pixels.

    Returns:
        Binary edge map.
    """
    if valid_mask is None:
        valid_mask = (depth > 0) & np.isfinite(depth)

    # Compute depth gradients
    if method == "sobel":
        # Use Sobel operators for gradient magnitude
        dx = ndimage.sobel(depth, axis=1)
        dy = ndimage.sobel(depth, axis=0)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)
        edges = gradient_magnitude > threshold

    elif method == "relative":
        # Detect edges where depth changes by more than threshold fraction
        # Using max of 4-neighbor differences
        kernel_h = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
        kernel_v = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])

        diff_h = np.abs(ndimage.convolve(depth, kernel_h, mode="constant"))
        diff_v = np.abs(ndimage.convolve(depth, kernel_v, mode="constant"))

        max_diff = np.maximum(diff_h, diff_v)

        # Relative threshold: edge if difference > threshold * local depth
        edges = max_diff > (threshold * depth)

    else:  # absolute
        # Detect edges where depth changes by more than threshold meters
        kernel_h = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
        kernel_v = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])

        diff_h = np.abs(ndimage.convolve(depth, kernel_h, mode="constant"))
        diff_v = np.abs(ndimage.convolve(depth, kernel_v, mode="constant"))

        max_diff = np.maximum(diff_h, diff_v)
        edges = max_diff > threshold

    # Apply valid mask
    edges = edges & valid_mask

    return edges.astype(bool)


def compute_edge_f1(
    edges_pred: np.ndarray,
    edges_gt: np.ndarray,
    tolerance: int = 1,
) -> Tuple[float, float, float]:
    """Compute F1 score between predicted and ground truth edge maps.

    Args:
        edges_pred: Predicted binary edge map.
        edges_gt: Ground truth binary edge map.
        tolerance: Pixel tolerance for edge matching.

    Returns:
        Tuple of (precision, recall, f1).
    """
    if tolerance > 0:
        # Dilate GT edges for tolerance matching
        struct = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=bool)
        edges_gt_dilated = binary_dilation(edges_gt, structure=struct)
        edges_pred_dilated = binary_dilation(edges_pred, structure=struct)
    else:
        edges_gt_dilated = edges_gt
        edges_pred_dilated = edges_pred

    # True positives: predicted edges that match (dilated) GT edges
    tp_pred = np.sum(edges_pred & edges_gt_dilated)

    # For recall: GT edges that match (dilated) predicted edges
    tp_gt = np.sum(edges_gt & edges_pred_dilated)

    # Total predictions and ground truth
    total_pred = np.sum(edges_pred)
    total_gt = np.sum(edges_gt)

    # Precision: what fraction of predicted edges are correct
    precision = tp_pred / total_pred if total_pred > 0 else 0.0

    # Recall: what fraction of GT edges were found
    recall = tp_gt / total_gt if total_gt > 0 else 0.0

    # F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return float(precision), float(recall), float(f1)


def compute_depth_edge_f1(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    edge_threshold: float = 0.1,
    edge_method: str = "relative",
    pixel_tolerance: int = 1,
    valid_mask: Optional[np.ndarray] = None,
) -> dict:
    """Compute depth edge F1 score between predicted and GT depth maps.

    Args:
        depth_pred: Predicted depth map in meters.
        depth_gt: Ground truth depth map in meters.
        edge_threshold: Threshold for edge detection.
        edge_method: Edge detection method ('relative', 'absolute', 'sobel').
        pixel_tolerance: Pixel tolerance for edge matching.
        valid_mask: Optional mask of valid pixels.

    Returns:
        Dictionary with precision, recall, and f1 scores.
    """
    if valid_mask is None:
        valid_mask = (depth_gt > 0) & (depth_pred > 0)
        valid_mask = valid_mask & np.isfinite(depth_gt) & np.isfinite(depth_pred)

    # Detect edges
    edges_pred = detect_depth_edges(depth_pred, edge_threshold, edge_method, valid_mask)
    edges_gt = detect_depth_edges(depth_gt, edge_threshold, edge_method, valid_mask)

    precision, recall, f1 = compute_edge_f1(edges_pred, edges_gt, pixel_tolerance)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_edge_pixels": int(np.sum(edges_pred)),
        "gt_edge_pixels": int(np.sum(edges_gt)),
    }


def aggregate_edge_f1(
    results: list[dict],
) -> dict:
    """Aggregate edge F1 results from multiple depth map pairs.

    Uses micro-averaging (sum of TP/FP/FN across all images).

    Args:
        results: List of per-image edge F1 result dictionaries.

    Returns:
        Dictionary with aggregated precision, recall, and f1.
    """
    total_precision = []
    total_recall = []
    total_f1 = []

    for r in results:
        if r["gt_edge_pixels"] > 0 or r["pred_edge_pixels"] > 0:
            total_precision.append(r["precision"])
            total_recall.append(r["recall"])
            total_f1.append(r["f1"])

    if not total_f1:
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }

    return {
        "precision": float(np.mean(total_precision)),
        "recall": float(np.mean(total_recall)),
        "f1": float(np.mean(total_f1)),
    }
