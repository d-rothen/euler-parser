"""Dataset evaluation orchestrator.

Runs all metrics over two depth datasets with matching structure.
"""

import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from .metrics import (
    load_depth_file,
    compute_psnr,
    compute_ssim,
    LPIPSMetric,
    FIDKIDMetric,
    compute_absrel,
    aggregate_absrel,
    compute_rmse_per_pixel,
    aggregate_rmse,
    compute_silog_per_pixel,
    aggregate_silog,
    compute_scale_invariant_log_error,
    compute_normal_angles,
    aggregate_normal_consistency,
    compute_depth_edge_f1,
    aggregate_edge_f1,
)


def find_matching_files(
    path1: Path,
    path2: Path,
    extensions: tuple = (".npy", ".png"),
) -> list[tuple[Path, Path]]:
    """Find matching depth files between two datasets.

    Args:
        path1: Root path of first dataset.
        path2: Root path of second dataset.
        extensions: Supported file extensions.

    Returns:
        List of (file1, file2) path tuples with matching structure.
    """
    matches = []

    # Find all depth files in dataset 1
    for ext in extensions:
        for file1 in path1.rglob(f"*{ext}"):
            # Get relative path
            rel_path = file1.relative_to(path1)

            # Look for matching file in dataset 2 (with any supported extension)
            stem = rel_path.with_suffix("")
            for ext2 in extensions:
                file2 = path2 / stem.with_suffix(ext2)
                if file2.exists():
                    matches.append((file1, file2))
                    break

    return matches


def evaluate_datasets(
    dataset1_config: dict,
    dataset2_config: dict,
    device: str = "cuda",
    batch_size: int = 16,
    num_workers: int = 4,
    verbose: bool = False,
) -> dict:
    """Evaluate all metrics between two datasets.

    Args:
        dataset1_config: Configuration for first dataset (ground truth).
        dataset2_config: Configuration for second dataset (predictions).
        device: Device for GPU-accelerated metrics.
        batch_size: Batch size for batched metrics.
        num_workers: Number of data loading workers.
        verbose: Enable verbose output.

    Returns:
        Dictionary containing all computed metrics.
    """
    path1 = Path(dataset1_config["path"])
    path2 = Path(dataset2_config["path"])

    depth_scale1 = dataset1_config.get("depth_scale", 1.0)
    depth_scale2 = dataset2_config.get("depth_scale", 1.0)

    intrinsics1 = dataset1_config.get("intrinsics")
    intrinsics2 = dataset2_config.get("intrinsics")

    # Find matching files
    print("Finding matching depth files...")
    matches = find_matching_files(path1, path2)

    if not matches:
        raise ValueError(f"No matching depth files found between {path1} and {path2}")

    print(f"Found {len(matches)} matching depth file pairs")

    # Initialize GPU-accelerated metrics
    print(f"Initializing metrics (device: {device})...")
    lpips_metric = LPIPSMetric(device=device)
    fid_kid_metric = FIDKIDMetric(device=device)

    # Storage for per-image metrics
    psnr_values = []
    ssim_values = []
    lpips_values = []

    absrel_values = []
    rmse_values = []
    silog_values = []
    silog_full_values = []

    normal_angle_values = []
    edge_f1_results = []

    # Load all depth maps for FID/KID (need full dataset)
    all_depths1 = []
    all_depths2 = []

    # Process each pair
    print("Computing per-image metrics...")
    for file1, file2 in tqdm(matches, desc="Processing depth pairs"):
        # Load depth maps
        try:
            depth1 = load_depth_file(file1, depth_scale1, intrinsics1)
            depth2 = load_depth_file(file2, depth_scale2, intrinsics2)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load {file1} or {file2}: {e}")
            continue

        # Store for FID/KID
        all_depths1.append(depth1)
        all_depths2.append(depth2)

        # Image quality metrics
        psnr_values.append(compute_psnr(depth2, depth1))
        ssim_values.append(compute_ssim(depth2, depth1))
        lpips_values.append(lpips_metric.compute(depth2, depth1))

        # Depth-specific metrics (per-pixel for aggregation)
        absrel_values.append(compute_absrel(depth2, depth1))
        rmse_values.append(compute_rmse_per_pixel(depth2, depth1))
        silog_values.append(compute_silog_per_pixel(depth2, depth1))
        silog_full_values.append(compute_scale_invariant_log_error(depth2, depth1))

        # Geometric metrics
        normal_angle_values.append(compute_normal_angles(depth2, depth1))
        edge_f1_results.append(compute_depth_edge_f1(depth2, depth1))

    # Compute FID and KID (dataset-level metrics)
    print("Computing FID/KID (this may take a while)...")
    fid_value = fid_kid_metric.compute_fid(
        all_depths1, all_depths2, batch_size, num_workers
    )
    kid_mean, kid_std = fid_kid_metric.compute_kid(
        all_depths1, all_depths2, batch_size, num_workers
    )

    # Aggregate results
    print("Aggregating results...")

    absrel_agg = aggregate_absrel(absrel_values)
    rmse_agg = aggregate_rmse(rmse_values)
    silog_agg = aggregate_silog(silog_values)
    normal_agg = aggregate_normal_consistency(normal_angle_values)
    edge_f1_agg = aggregate_edge_f1(edge_f1_results)

    results = {
        "image_quality": {
            "psnr_mean": float(np.mean([v for v in psnr_values if np.isfinite(v)])),
            "ssim_mean": float(np.mean([v for v in ssim_values if np.isfinite(v)])),
            "lpips_mean": float(np.mean(lpips_values)),
            "fid": fid_value,
            "kid_mean": kid_mean,
            "kid_std": kid_std,
        },
        "depth_metrics": {
            "absrel": {
                "median": absrel_agg["median"],
                "p90": absrel_agg["p90"],
            },
            "rmse": {
                "median": rmse_agg["median"],
                "p90": rmse_agg["p90"],
            },
            "silog": {
                "mean": float(
                    np.mean([v for v in silog_full_values if np.isfinite(v)])
                ),
                "median": silog_agg["median"],
                "p90": silog_agg["p90"],
            },
        },
        "geometric_metrics": {
            "normal_consistency": {
                "mean_angle": normal_agg["mean_angle"],
                "median_angle": normal_agg["median_angle"],
                "percent_below_11_25": normal_agg["percent_below_11_25"],
                "percent_below_22_5": normal_agg["percent_below_22_5"],
                "percent_below_30": normal_agg["percent_below_30"],
            },
            "depth_edge_f1": {
                "precision": edge_f1_agg["precision"],
                "recall": edge_f1_agg["recall"],
                "f1": edge_f1_agg["f1"],
            },
        },
        "dataset_info": {
            "num_pairs": len(matches),
            "dataset1_name": dataset1_config["name"],
            "dataset2_name": dataset2_config["name"],
            "dataset1_path": str(path1),
            "dataset2_path": str(path2),
        },
    }

    return results


def evaluate_single_pair(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    device: str = "cuda",
) -> dict:
    """Evaluate all per-image metrics for a single depth map pair.

    Useful for quick evaluation without FID/KID.

    Args:
        depth_pred: Predicted depth map in meters.
        depth_gt: Ground truth depth map in meters.
        device: Device for GPU-accelerated metrics.

    Returns:
        Dictionary containing all per-image metrics.
    """
    lpips_metric = LPIPSMetric(device=device)

    absrel = compute_absrel(depth_pred, depth_gt)
    rmse = compute_rmse_per_pixel(depth_pred, depth_gt)
    silog = compute_silog_per_pixel(depth_pred, depth_gt)

    results = {
        "image_quality": {
            "psnr": compute_psnr(depth_pred, depth_gt),
            "ssim": compute_ssim(depth_pred, depth_gt),
            "lpips": lpips_metric.compute(depth_pred, depth_gt),
        },
        "depth_metrics": {
            "absrel_mean": float(np.mean(absrel)) if len(absrel) > 0 else float("nan"),
            "absrel_median": (
                float(np.median(absrel)) if len(absrel) > 0 else float("nan")
            ),
            "rmse": (
                float(np.sqrt(np.mean(rmse))) if len(rmse) > 0 else float("nan")
            ),
            "silog": compute_scale_invariant_log_error(depth_pred, depth_gt),
        },
        "geometric_metrics": {
            "normal_consistency": aggregate_normal_consistency(
                [compute_normal_angles(depth_pred, depth_gt)]
            ),
            "depth_edge_f1": compute_depth_edge_f1(depth_pred, depth_gt),
        },
    }

    return results
