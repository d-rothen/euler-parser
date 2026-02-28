"""Edge F1 metric for RGB images."""

import numpy as np
from typing import Tuple
from scipy import ndimage
from scipy.ndimage import binary_dilation


def detect_rgb_edges(
    img: np.ndarray,
    threshold: float = 0.1,
    method: str = "sobel",
) -> np.ndarray:
    """Detect edges in an RGB image.

    Args:
        img: RGB image in [0, 1] range, shape (H, W, 3).
        threshold: Edge detection threshold.
        method: Edge detection method ('sobel', 'canny_approx').

    Returns:
        Binary edge map.
    """
    # Convert to grayscale
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    if method == "sobel":
        # Use Sobel operators for gradient magnitude
        dx = ndimage.sobel(gray, axis=1)
        dy = ndimage.sobel(gray, axis=0)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)
        edges = gradient_magnitude > threshold

    elif method == "canny_approx":
        # Approximate Canny: Sobel + non-maximum suppression
        dx = ndimage.sobel(gray, axis=1)
        dy = ndimage.sobel(gray, axis=0)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)
        gradient_direction = np.arctan2(dy, dx)

        # Simple non-maximum suppression
        edges = np.zeros_like(gradient_magnitude, dtype=bool)
        angle = gradient_direction * 180 / np.pi
        angle[angle < 0] += 180

        for i in range(1, gray.shape[0] - 1):
            for j in range(1, gray.shape[1] - 1):
                # Get neighbor pixels based on gradient direction
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    neighbors = [gradient_magnitude[i, j - 1], gradient_magnitude[i, j + 1]]
                elif 22.5 <= angle[i, j] < 67.5:
                    neighbors = [gradient_magnitude[i - 1, j + 1], gradient_magnitude[i + 1, j - 1]]
                elif 67.5 <= angle[i, j] < 112.5:
                    neighbors = [gradient_magnitude[i - 1, j], gradient_magnitude[i + 1, j]]
                else:
                    neighbors = [gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1]]

                if gradient_magnitude[i, j] >= max(neighbors) and gradient_magnitude[i, j] > threshold:
                    edges[i, j] = True

    else:
        # Default: simple gradient threshold
        dx = ndimage.sobel(gray, axis=1)
        dy = ndimage.sobel(gray, axis=0)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)
        edges = gradient_magnitude > threshold

    return edges.astype(bool)


def compute_edge_precision_recall_f1(
    edges_pred: np.ndarray,
    edges_gt: np.ndarray,
    tolerance: int = 1,
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 between edge maps.

    Args:
        edges_pred: Predicted binary edge map.
        edges_gt: Ground truth binary edge map.
        tolerance: Pixel tolerance for edge matching.

    Returns:
        Tuple of (precision, recall, f1).
    """
    if tolerance > 0:
        struct = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=bool)
        edges_gt_dilated = binary_dilation(edges_gt, structure=struct)
        edges_pred_dilated = binary_dilation(edges_pred, structure=struct)
    else:
        edges_gt_dilated = edges_gt
        edges_pred_dilated = edges_pred

    # True positives
    tp_pred = np.sum(edges_pred & edges_gt_dilated)
    tp_gt = np.sum(edges_gt & edges_pred_dilated)

    total_pred = np.sum(edges_pred)
    total_gt = np.sum(edges_gt)

    precision = tp_pred / total_pred if total_pred > 0 else 0.0
    recall = tp_gt / total_gt if total_gt > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return float(precision), float(recall), float(f1)


def compute_rgb_edge_f1(
    img_pred: np.ndarray,
    img_gt: np.ndarray,
    edge_threshold: float = 0.1,
    edge_method: str = "sobel",
    pixel_tolerance: int = 1,
) -> dict:
    """Compute edge F1 score between predicted and GT RGB images.

    Args:
        img_pred: Predicted RGB image in [0, 1] range, shape (H, W, 3).
        img_gt: Ground truth RGB image in [0, 1] range, shape (H, W, 3).
        edge_threshold: Threshold for edge detection.
        edge_method: Edge detection method.
        pixel_tolerance: Pixel tolerance for edge matching.

    Returns:
        Dictionary with precision, recall, and f1 scores.
    """
    edges_pred = detect_rgb_edges(img_pred, edge_threshold, edge_method)
    edges_gt = detect_rgb_edges(img_gt, edge_threshold, edge_method)

    precision, recall, f1 = compute_edge_precision_recall_f1(
        edges_pred, edges_gt, pixel_tolerance
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_edge_pixels": int(np.sum(edges_pred)),
        "gt_edge_pixels": int(np.sum(edges_gt)),
    }


def aggregate_rgb_edge_f1(
    results: list[dict],
) -> dict:
    """Aggregate edge F1 results from multiple image pairs.

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
