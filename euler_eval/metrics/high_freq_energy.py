"""High-frequency energy ratio metric for RGB images.

Measures the ratio of energy above a frequency cutoff vs total energy.
This metric helps detect over-smoothing or loss of fine details.
"""

import numpy as np
from typing import Optional


def compute_high_freq_energy_ratio(
    img: np.ndarray,
    cutoff_ratio: float = 0.25,
) -> float:
    """Compute ratio of high-frequency energy to total energy.

    Uses 2D FFT to compute the frequency spectrum and measures
    the ratio of energy in high frequencies (above cutoff) to total.

    Args:
        img: RGB image in [0, 1] range, shape (H, W, 3).
        cutoff_ratio: Ratio of frequency cutoff (0.25 = top 25% frequencies).

    Returns:
        High-frequency energy ratio in [0, 1]. Higher means more detail.
    """
    # Convert to grayscale for frequency analysis
    if img.ndim == 3:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img

    h, w = gray.shape

    # Compute 2D FFT
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)

    # Compute power spectrum
    power_spectrum = np.abs(fft_shifted) ** 2

    # Create frequency distance mask from center
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Normalize distance to [0, 1] where 1 is the corner
    max_dist = np.sqrt(cx**2 + cy**2)
    normalized_dist = dist_from_center / max_dist

    # High frequency mask (above cutoff)
    high_freq_mask = normalized_dist > (1 - cutoff_ratio)

    # Compute energy ratio
    total_energy = np.sum(power_spectrum)
    high_freq_energy = np.sum(power_spectrum[high_freq_mask])

    if total_energy < 1e-10:
        return 0.0

    return float(high_freq_energy / total_energy)


def compute_high_freq_energy_comparison(
    img_pred: np.ndarray,
    img_gt: np.ndarray,
    cutoff_ratio: float = 0.25,
) -> dict:
    """Compare high-frequency energy between predicted and GT images.

    Args:
        img_pred: Predicted RGB image in [0, 1] range, shape (H, W, 3).
        img_gt: Ground truth RGB image in [0, 1] range, shape (H, W, 3).
        cutoff_ratio: Ratio of frequency cutoff.

    Returns:
        Dictionary with energy ratios and their comparison.
    """
    pred_ratio = compute_high_freq_energy_ratio(img_pred, cutoff_ratio)
    gt_ratio = compute_high_freq_energy_ratio(img_gt, cutoff_ratio)

    # Relative difference: positive means pred has more HF, negative means less
    if gt_ratio > 1e-10:
        relative_diff = (pred_ratio - gt_ratio) / gt_ratio
    else:
        relative_diff = 0.0 if pred_ratio < 1e-10 else float("inf")

    return {
        "pred_hf_ratio": pred_ratio,
        "gt_hf_ratio": gt_ratio,
        "relative_diff": relative_diff,
        "absolute_diff": pred_ratio - gt_ratio,
    }


def compute_high_freq_preservation(
    img_pred: np.ndarray,
    img_gt: np.ndarray,
    cutoff_ratio: float = 0.25,
) -> float:
    """Compute how well high-frequency content is preserved.

    Returns a score in [0, 1] where 1 means perfect preservation
    and 0 means complete loss of high-frequency content.

    Args:
        img_pred: Predicted RGB image in [0, 1] range, shape (H, W, 3).
        img_gt: Ground truth RGB image in [0, 1] range, shape (H, W, 3).
        cutoff_ratio: Ratio of frequency cutoff.

    Returns:
        High-frequency preservation score.
    """
    comparison = compute_high_freq_energy_comparison(img_pred, img_gt, cutoff_ratio)

    gt_ratio = comparison["gt_hf_ratio"]
    pred_ratio = comparison["pred_hf_ratio"]

    if gt_ratio < 1e-10:
        # GT has no high frequency content, perfect match if pred also has none
        return 1.0 if pred_ratio < 1e-10 else 0.0

    # Preservation = min(pred/gt, 1) - don't reward adding HF that wasn't there
    preservation = min(pred_ratio / gt_ratio, 1.0)
    return float(preservation)


def compute_frequency_spectrum_similarity(
    img_pred: np.ndarray,
    img_gt: np.ndarray,
    num_bands: int = 4,
) -> dict:
    """Compute similarity across multiple frequency bands.

    Divides the frequency spectrum into bands and computes
    energy similarity in each band.

    Args:
        img_pred: Predicted RGB image in [0, 1] range, shape (H, W, 3).
        img_gt: Ground truth RGB image in [0, 1] range, shape (H, W, 3).
        num_bands: Number of frequency bands to analyze.

    Returns:
        Dictionary with per-band energy ratios and similarities.
    """
    # Convert to grayscale
    if img_pred.ndim == 3:
        gray_pred = 0.299 * img_pred[:, :, 0] + 0.587 * img_pred[:, :, 1] + 0.114 * img_pred[:, :, 2]
        gray_gt = 0.299 * img_gt[:, :, 0] + 0.587 * img_gt[:, :, 1] + 0.114 * img_gt[:, :, 2]
    else:
        gray_pred = img_pred
        gray_gt = img_gt

    h, w = gray_pred.shape

    # Compute FFT
    fft_pred = np.fft.fftshift(np.fft.fft2(gray_pred))
    fft_gt = np.fft.fftshift(np.fft.fft2(gray_gt))

    power_pred = np.abs(fft_pred) ** 2
    power_gt = np.abs(fft_gt) ** 2

    # Create frequency distance mask
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = np.sqrt(cx**2 + cy**2)
    normalized_dist = dist_from_center / max_dist

    # Analyze each frequency band
    band_edges = np.linspace(0, 1, num_bands + 1)
    results = {"bands": {}}

    for i in range(num_bands):
        low = band_edges[i]
        high = band_edges[i + 1]
        band_mask = (normalized_dist >= low) & (normalized_dist < high)

        pred_energy = np.sum(power_pred[band_mask])
        gt_energy = np.sum(power_gt[band_mask])

        # Compute similarity (1 - normalized absolute difference)
        if gt_energy > 1e-10:
            ratio = pred_energy / gt_energy
            similarity = 1 - min(abs(ratio - 1), 1)
        else:
            similarity = 1.0 if pred_energy < 1e-10 else 0.0

        band_name = f"band_{i}" if i < num_bands - 1 else "high_freq"
        results["bands"][band_name] = {
            "pred_energy": float(pred_energy),
            "gt_energy": float(gt_energy),
            "similarity": float(similarity),
        }

    # Overall high-frequency preservation (top 25%)
    results["hf_preservation"] = compute_high_freq_preservation(img_pred, img_gt)

    return results


def aggregate_high_freq_metrics(
    results: list[dict],
) -> dict:
    """Aggregate high-frequency metrics from multiple image pairs.

    Args:
        results: List of per-image high-frequency comparison dictionaries.

    Returns:
        Aggregated statistics.
    """
    pred_ratios = []
    gt_ratios = []
    relative_diffs = []

    for r in results:
        pred_ratios.append(r["pred_hf_ratio"])
        gt_ratios.append(r["gt_hf_ratio"])
        if np.isfinite(r["relative_diff"]):
            relative_diffs.append(r["relative_diff"])

    return {
        "pred_hf_ratio_mean": float(np.mean(pred_ratios)),
        "gt_hf_ratio_mean": float(np.mean(gt_ratios)),
        "relative_diff_mean": float(np.mean(relative_diffs)) if relative_diffs else float("nan"),
        "relative_diff_std": float(np.std(relative_diffs)) if relative_diffs else float("nan"),
    }
