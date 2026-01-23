"""Dataset evaluation orchestrator.

Runs all metrics over depth and RGB datasets with matching structure.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from .metrics import (
    # Depth utilities and metrics
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
    # RGB utilities and metrics
    load_rgb_file,
    compute_rgb_psnr,
    compute_rgb_ssim,
    RGBLPIPSMetric,
    compute_depth_binned_photometric_error,
    aggregate_depth_binned_errors,
    compute_rgb_edge_f1,
    aggregate_rgb_edge_f1,
    compute_tail_errors,
    aggregate_tail_errors,
    compute_high_freq_energy_comparison,
    aggregate_high_freq_metrics,
)


def load_dataset_manifest(dataset_path: Path) -> dict:
    """Load and validate the output.json manifest from a dataset root.

    Args:
        dataset_path: Root path of the dataset.

    Returns:
        Parsed manifest dictionary containing dataset entries.

    Raises:
        FileNotFoundError: If output.json is missing.
        ValueError: If output.json is invalid.
    """
    manifest_path = dataset_path / "output.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing output.json at dataset root: {dataset_path}. "
            f"Each dataset must have an output.json manifest file."
        )

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    if "dataset" not in manifest:
        raise ValueError(
            f"Invalid output.json at {manifest_path}: missing 'dataset' key."
        )

    return manifest


def build_id_to_entry_map(manifest: dict) -> dict[str, dict]:
    """Build a mapping from entry ID to entry data.

    Args:
        manifest: Parsed output.json manifest.

    Returns:
        Dictionary mapping ID strings to entry dictionaries.
    """
    id_map = {}
    for entry in manifest["dataset"]:
        entry_id = entry.get("id")
        if entry_id is None:
            raise ValueError(f"Entry missing 'id' field: {entry}")
        if entry_id in id_map:
            raise ValueError(f"Duplicate entry ID found: {entry_id}")
        id_map[entry_id] = entry
    return id_map


def find_matching_files_by_id(
    path1: Path,
    path2: Path,
) -> list[tuple[Path, Path, str]]:
    """Find matching files between two datasets using output.json ID matching.

    Args:
        path1: Root path of first dataset (GT).
        path2: Root path of second dataset (predictions).

    Returns:
        List of (gt_file, pred_file, id) tuples for entries with matching IDs.

    Raises:
        FileNotFoundError: If output.json is missing from either dataset.
    """
    manifest1 = load_dataset_manifest(path1)
    manifest2 = load_dataset_manifest(path2)

    id_map1 = build_id_to_entry_map(manifest1)
    id_map2 = build_id_to_entry_map(manifest2)

    matches = []
    for entry_id, entry1 in id_map1.items():
        if entry_id in id_map2:
            entry2 = id_map2[entry_id]
            file1 = path1 / entry1["path"]
            file2 = path2 / entry2["path"]
            matches.append((file1, file2, entry_id))

    return matches


def find_matching_depth_for_rgb_by_id(
    rgb_id: str,
    depth_path: Path,
    depth_id_map: dict[str, dict],
) -> Optional[Path]:
    """Find matching depth file for an RGB entry by ID.

    Args:
        rgb_id: ID of the RGB entry.
        depth_path: Root path of the depth dataset.
        depth_id_map: Pre-built ID to entry map for depth dataset.

    Returns:
        Path to matching depth file, or None if not found.
    """
    if rgb_id in depth_id_map:
        return depth_path / depth_id_map[rgb_id]["path"]
    return None


def evaluate_depth_datasets(
    gt_config: dict,
    pred_config: dict,
    device: str = "cuda",
    batch_size: int = 16,
    num_workers: int = 4,
    verbose: bool = False,
) -> dict:
    """Evaluate all depth metrics between GT and prediction datasets.

    Args:
        gt_config: Configuration for ground truth dataset.
        pred_config: Configuration for prediction dataset.
        device: Device for GPU-accelerated metrics.
        batch_size: Batch size for batched metrics.
        num_workers: Number of data loading workers.
        verbose: Enable verbose output.

    Returns:
        Dictionary containing all computed metrics.
    """
    gt_path = Path(gt_config["path"])
    pred_path = Path(pred_config["path"])

    gt_depth_scale = gt_config.get("depth_scale", 1.0)
    pred_depth_scale = pred_config.get("depth_scale", 1.0)

    gt_intrinsics = gt_config.get("intrinsics")
    pred_intrinsics = pred_config.get("intrinsics")

    # Find matching files by ID from output.json manifests
    print("Finding matching depth files...")
    matches = find_matching_files_by_id(gt_path, pred_path)

    if not matches:
        raise ValueError(f"No matching depth files found between {gt_path} and {pred_path}")

    print(f"Found {len(matches)} matching depth file pairs")

    # Initialize GPU-accelerated metrics
    print(f"Initializing depth metrics (device: {device})...")
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

    # Track entry IDs for per-file metrics
    processed_entry_ids = []

    # Load all depth maps for FID/KID
    all_depths_gt = []
    all_depths_pred = []

    # Process each pair
    print("Computing per-image depth metrics...")
    for gt_file, pred_file, entry_id in tqdm(matches, desc="Processing depth pairs"):
        try:
            depth_gt = load_depth_file(gt_file, gt_depth_scale, gt_intrinsics)
            depth_pred = load_depth_file(pred_file, pred_depth_scale, pred_intrinsics)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load {gt_file} or {pred_file}: {e}")
                print(f"Might be due to incomplete prediction dataset.")
            continue

        all_depths_gt.append(depth_gt)
        all_depths_pred.append(depth_pred)

        # Track successfully processed entry ID
        processed_entry_ids.append(entry_id)

        # Image quality metrics
        psnr_values.append(compute_psnr(depth_pred, depth_gt))
        ssim_values.append(compute_ssim(depth_pred, depth_gt))
        lpips_values.append(lpips_metric.compute(depth_pred, depth_gt))

        # Depth-specific metrics
        absrel_values.append(compute_absrel(depth_pred, depth_gt))
        rmse_values.append(compute_rmse_per_pixel(depth_pred, depth_gt))
        silog_values.append(compute_silog_per_pixel(depth_pred, depth_gt))
        silog_full_values.append(compute_scale_invariant_log_error(depth_pred, depth_gt))

        # Geometric metrics
        normal_angle_values.append(compute_normal_angles(depth_pred, depth_gt))
        edge_f1_results.append(compute_depth_edge_f1(depth_pred, depth_gt))

    # Compute FID and KID
    print("Computing FID/KID (this may take a while)...")
    fid_value = fid_kid_metric.compute_fid(
        all_depths_gt, all_depths_pred, batch_size, num_workers
    )
    kid_mean, kid_std = fid_kid_metric.compute_kid(
        all_depths_gt, all_depths_pred, batch_size, num_workers
    )

    # Aggregate results
    print("Aggregating depth results...")

    absrel_agg = aggregate_absrel(absrel_values)
    rmse_agg = aggregate_rmse(rmse_values)
    silog_agg = aggregate_silog(silog_values)
    normal_agg = aggregate_normal_consistency(normal_angle_values)
    edge_f1_agg = aggregate_edge_f1(edge_f1_results)

    # Build per-file metrics (excluding FID/KID which are distribution metrics)
    per_file_metrics = {}
    for i, entry_id in enumerate(processed_entry_ids):
        absrel_arr = absrel_values[i]
        rmse_arr = rmse_values[i]
        normal_angles = normal_angle_values[i]
        edge_f1 = edge_f1_results[i]

        per_file_metrics[entry_id] = {
            "psnr": float(psnr_values[i]) if np.isfinite(psnr_values[i]) else None,
            "ssim": float(ssim_values[i]) if np.isfinite(ssim_values[i]) else None,
            "lpips": float(lpips_values[i]),
            "absrel_mean": float(np.mean(absrel_arr)) if len(absrel_arr) > 0 else None,
            "rmse": float(np.sqrt(np.mean(rmse_arr))) if len(rmse_arr) > 0 else None,
            "silog": float(silog_full_values[i]) if np.isfinite(silog_full_values[i]) else None,
            "normal_mean_angle": float(np.mean(normal_angles)) if len(normal_angles) > 0 else None,
            "edge_f1_precision": float(edge_f1["precision"]),
            "edge_f1_recall": float(edge_f1["recall"]),
            "edge_f1_f1": float(edge_f1["f1"]),
        }

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
                "mean": float(np.mean([v for v in silog_full_values if np.isfinite(v)])),
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
            "gt_name": gt_config["name"],
            "pred_name": pred_config["name"],
            "gt_path": str(gt_path),
            "pred_path": str(pred_path),
        },
        "per_file_metrics": per_file_metrics,
    }

    return results


def evaluate_rgb_datasets(
    gt_config: dict,
    pred_config: dict,
    depth_gt_config: Optional[dict] = None,
    device: str = "cuda",
    batch_size: int = 16,
    num_workers: int = 4,
    verbose: bool = False,
) -> dict:
    """Evaluate all RGB metrics between GT and prediction datasets.

    Args:
        gt_config: Configuration for ground truth RGB dataset.
        pred_config: Configuration for prediction RGB dataset.
        depth_gt_config: Optional depth GT config for depth-binned metrics.
        device: Device for GPU-accelerated metrics.
        batch_size: Batch size for batched metrics.
        num_workers: Number of data loading workers.
        verbose: Enable verbose output.

    Returns:
        Dictionary containing all computed RGB metrics.
    """
    gt_path = Path(gt_config["path"])
    pred_path = Path(pred_config["path"])

    depth_path = None
    depth_scale = 1.0
    depth_intrinsics = None
    depth_id_map = None
    if depth_gt_config is not None:
        depth_path = Path(depth_gt_config["path"])
        depth_scale = depth_gt_config.get("depth_scale", 1.0)
        depth_intrinsics = depth_gt_config.get("intrinsics")
        # Load depth manifest for ID-based matching
        depth_manifest = load_dataset_manifest(depth_path)
        depth_id_map = build_id_to_entry_map(depth_manifest)

    # Find matching RGB files by ID from output.json manifests
    print("Finding matching RGB files...")
    matches = find_matching_files_by_id(gt_path, pred_path)

    if not matches:
        raise ValueError(f"No matching RGB files found between {gt_path} and {pred_path}")

    print(f"Found {len(matches)} matching RGB file pairs")

    # Initialize GPU-accelerated metrics
    print(f"Initializing RGB metrics (device: {device})...")
    lpips_metric = RGBLPIPSMetric(device=device)

    # Storage for per-image metrics
    psnr_values = []
    ssim_values = []
    lpips_values = []
    edge_f1_results = []
    tail_error_arrays = []
    high_freq_results = []
    depth_binned_results = []

    # Track entry IDs for per-file metrics
    processed_entry_ids = []
    # Track depth-binned results per entry (None if not available)
    depth_binned_per_entry = []

    has_depth = depth_path is not None

    # Process each pair
    print("Computing per-image RGB metrics...")
    for gt_file, pred_file, entry_id in tqdm(matches, desc="Processing RGB pairs"):
        try:
            img_gt = load_rgb_file(gt_file)
            img_pred = load_rgb_file(pred_file)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load {gt_file} or {pred_file}: {e}")
                print(f"Might be due to incomplete prediction dataset.")
            continue

        # Track successfully processed entry ID
        processed_entry_ids.append(entry_id)

        # Basic image quality metrics
        psnr_values.append(compute_rgb_psnr(img_pred, img_gt))
        ssim_values.append(compute_rgb_ssim(img_pred, img_gt))
        lpips_values.append(lpips_metric.compute(img_pred, img_gt))

        # Edge F1
        edge_f1_results.append(compute_rgb_edge_f1(img_pred, img_gt))

        # Tail errors
        abs_error = np.abs(img_pred - img_gt).mean(axis=-1)
        tail_error_arrays.append(abs_error)

        # High-frequency energy
        high_freq_results.append(compute_high_freq_energy_comparison(img_pred, img_gt))

        # Depth-binned photometric error (if depth available)
        depth_binned_entry = None
        if has_depth and depth_id_map is not None:
            depth_file = find_matching_depth_for_rgb_by_id(entry_id, depth_path, depth_id_map)
            if depth_file is not None:
                try:
                    depth = load_depth_file(depth_file, depth_scale, depth_intrinsics)
                    depth_binned_entry = compute_depth_binned_photometric_error(img_pred, img_gt, depth)
                    depth_binned_results.append(depth_binned_entry)
                except Exception as e:
                    if verbose:
                        print(f"Warning: Failed to load depth {depth_file}: {e}")
        depth_binned_per_entry.append(depth_binned_entry)

    # Aggregate results
    print("Aggregating RGB results...")

    edge_f1_agg = aggregate_rgb_edge_f1(edge_f1_results)
    tail_agg = aggregate_tail_errors(tail_error_arrays)
    high_freq_agg = aggregate_high_freq_metrics(high_freq_results)

    # Build per-file metrics
    per_file_metrics = {}
    for i, entry_id in enumerate(processed_entry_ids):
        edge_f1 = edge_f1_results[i]
        tail_error_arr = tail_error_arrays[i]
        high_freq = high_freq_results[i]

        entry_metrics = {
            "psnr": float(psnr_values[i]) if np.isfinite(psnr_values[i]) else None,
            "ssim": float(ssim_values[i]) if np.isfinite(ssim_values[i]) else None,
            "lpips": float(lpips_values[i]),
            "edge_f1_precision": float(edge_f1["precision"]),
            "edge_f1_recall": float(edge_f1["recall"]),
            "edge_f1_f1": float(edge_f1["f1"]),
            "tail_error_p95": float(np.percentile(tail_error_arr, 95)) if len(tail_error_arr) > 0 else None,
            "tail_error_p99": float(np.percentile(tail_error_arr, 99)) if len(tail_error_arr) > 0 else None,
            "high_freq_pred_ratio": float(high_freq["pred_hf_ratio"]),
            "high_freq_gt_ratio": float(high_freq["gt_hf_ratio"]),
            "high_freq_relative_diff": float(high_freq["relative_diff"]) if np.isfinite(high_freq["relative_diff"]) else None,
        }

        # Add depth-binned metrics if available for this entry
        depth_binned = depth_binned_per_entry[i]
        if depth_binned is not None:
            entry_metrics["depth_binned_error"] = depth_binned

        per_file_metrics[entry_id] = entry_metrics

    results = {
        "image_quality": {
            "psnr_mean": float(np.mean([v for v in psnr_values if np.isfinite(v)])),
            "ssim_mean": float(np.mean([v for v in ssim_values if np.isfinite(v)])),
            "lpips_mean": float(np.mean(lpips_values)),
        },
        "edge_f1": {
            "precision": edge_f1_agg["precision"],
            "recall": edge_f1_agg["recall"],
            "f1": edge_f1_agg["f1"],
        },
        "tail_errors": {
            "p95": tail_agg["p95"],
            "p99": tail_agg["p99"],
        },
        "high_frequency": {
            "pred_hf_ratio_mean": high_freq_agg["pred_hf_ratio_mean"],
            "gt_hf_ratio_mean": high_freq_agg["gt_hf_ratio_mean"],
            "relative_diff_mean": high_freq_agg["relative_diff_mean"],
        },
        "dataset_info": {
            "num_pairs": len(matches),
            "gt_name": gt_config["name"],
            "pred_name": pred_config["name"],
            "gt_path": str(gt_path),
            "pred_path": str(pred_path),
        },
        "per_file_metrics": per_file_metrics,
    }

    # Add depth-binned metrics if available
    if depth_binned_results:
        depth_binned_agg = aggregate_depth_binned_errors(depth_binned_results)
        results["depth_binned_photometric"] = depth_binned_agg

    return results


def evaluate_single_depth_pair(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    device: str = "cuda",
) -> dict:
    """Evaluate all per-image metrics for a single depth map pair.

    Args:
        depth_pred: Predicted depth map in meters.
        depth_gt: Ground truth depth map in meters.
        device: Device for GPU-accelerated metrics.

    Returns:
        Dictionary containing all per-image depth metrics.
    """
    lpips_metric = LPIPSMetric(device=device)

    absrel = compute_absrel(depth_pred, depth_gt)
    rmse = compute_rmse_per_pixel(depth_pred, depth_gt)

    results = {
        "image_quality": {
            "psnr": compute_psnr(depth_pred, depth_gt),
            "ssim": compute_ssim(depth_pred, depth_gt),
            "lpips": lpips_metric.compute(depth_pred, depth_gt),
        },
        "depth_metrics": {
            "absrel_mean": float(np.mean(absrel)) if len(absrel) > 0 else float("nan"),
            "absrel_median": float(np.median(absrel)) if len(absrel) > 0 else float("nan"),
            "rmse": float(np.sqrt(np.mean(rmse))) if len(rmse) > 0 else float("nan"),
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


def evaluate_single_rgb_pair(
    img_pred: np.ndarray,
    img_gt: np.ndarray,
    depth: Optional[np.ndarray] = None,
    device: str = "cuda",
) -> dict:
    """Evaluate all per-image metrics for a single RGB image pair.

    Args:
        img_pred: Predicted RGB image in [0, 1] range, shape (H, W, 3).
        img_gt: Ground truth RGB image in [0, 1] range, shape (H, W, 3).
        depth: Optional depth map for depth-binned metrics.
        device: Device for GPU-accelerated metrics.

    Returns:
        Dictionary containing all per-image RGB metrics.
    """
    lpips_metric = RGBLPIPSMetric(device=device)

    results = {
        "image_quality": {
            "psnr": compute_rgb_psnr(img_pred, img_gt),
            "ssim": compute_rgb_ssim(img_pred, img_gt),
            "lpips": lpips_metric.compute(img_pred, img_gt),
        },
        "edge_f1": compute_rgb_edge_f1(img_pred, img_gt),
        "tail_errors": compute_tail_errors(img_pred, img_gt),
        "high_frequency": compute_high_freq_energy_comparison(img_pred, img_gt),
    }

    if depth is not None:
        results["depth_binned_photometric"] = compute_depth_binned_photometric_error(
            img_pred, img_gt, depth
        )

    return results
