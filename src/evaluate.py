"""Dataset evaluation orchestrator.

Runs all metrics over depth and RGB datasets with matching structure.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from PIL import Image

from .utils.hierarchy_parser import get_matches, set_value
from .sanity_checker import SanityChecker

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


def find_matching_files_by_id(
    path1: Path,
    path2: Path,
) -> list[dict]:
    """Find matching files between two datasets using hierarchical output.json matching.

    Args:
        path1: Root path of first dataset (GT).
        path2: Root path of second dataset (predictions).

    Returns:
        List of match dicts containing:
        - hierarchy: List of keys forming the path to this entry
        - id: The matched file id
        - gt_file: Path to ground truth file
        - pred_file: Path to prediction file

    Raises:
        FileNotFoundError: If output.json is missing from either dataset.
    """
    manifest1 = load_dataset_manifest(path1)
    manifest2 = load_dataset_manifest(path2)

    # Use get_matches with the hierarchical dataset properties
    raw_matches = get_matches([manifest1["dataset"], manifest2["dataset"]])

    matches = []
    for match in raw_matches:
        # Only include matches where both paths are present
        if match["paths"][0] is not None and match["paths"][1] is not None:
            matches.append({
                "hierarchy": match["hierarchy"],
                "id": match["id"],
                "gt_file": path1 / match["paths"][0],
                "pred_file": path2 / match["paths"][1],
            })

    return matches


def find_matching_depth_for_rgb(
    hierarchy: list[str],
    file_id: str,
    depth_path: Path,
    depth_map: dict[tuple[tuple[str, ...], str], str],
) -> Optional[Path]:
    """Find matching depth file for an RGB entry by hierarchy and ID.

    Args:
        hierarchy: List of keys forming the path to this entry.
        file_id: ID of the file entry.
        depth_path: Root path of the depth dataset.
        depth_map: Pre-built (hierarchy, id) -> path map for depth dataset.

    Returns:
        Path to matching depth file, or None if not found.
    """
    key = (tuple(hierarchy), file_id)
    if key in depth_map:
        return depth_path / depth_map[key]
    return None


def build_depth_hierarchy_map(manifest: dict) -> dict[tuple[tuple[str, ...], str], str]:
    """Build a mapping from (hierarchy, id) to path for depth dataset.

    Args:
        manifest: Parsed output.json manifest with hierarchical dataset.

    Returns:
        Dictionary mapping (hierarchy_tuple, id) to file path.
    """
    # Use get_matches with single dataset to extract all entries
    matches = get_matches([manifest["dataset"]])
    depth_map = {}
    for match in matches:
        if match["paths"][0] is not None:
            key = (tuple(match["hierarchy"]), match["id"])
            depth_map[key] = match["paths"][0]
    return depth_map


_CROP_THRESHOLD = 0.05


def _validate_target_dim(dim: Optional[list[int]]) -> Optional[tuple[int, int]]:
    if dim is None:
        return None
    if not isinstance(dim, (list, tuple)) or len(dim) != 2:
        raise ValueError("rgb.datasets dim must be a list like [height, width]")
    target_h, target_w = dim
    try:
        target_h = int(target_h)
        target_w = int(target_w)
    except (TypeError, ValueError) as exc:
        raise ValueError("rgb.datasets dim entries must be integers") from exc
    if target_h <= 0 or target_w <= 0:
        raise ValueError("rgb.datasets dim entries must be positive")
    return target_h, target_w


def _should_center_crop(
    height: int,
    width: int,
    target_h: int,
    target_w: int,
    threshold: float = _CROP_THRESHOLD,
) -> bool:
    if target_h > height or target_w > width:
        return False
    if height == 0 or width == 0:
        return False
    diff_h = (height - target_h) / height
    diff_w = (width - target_w) / width
    return max(diff_h, diff_w) <= threshold


def _center_crop(array: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    height, width = array.shape[:2]
    top = (height - target_h) // 2
    left = (width - target_w) // 2
    return array[top : top + target_h, left : left + target_w]


def _resize_rgb_image(
    img: np.ndarray,
    target_h: int,
    target_w: int,
    resample: int,
    pixel_value_max: float = 255.0,
) -> np.ndarray:
    img_uint8 = np.clip(np.round(img * pixel_value_max), 0, 255).astype(np.uint8)
    resized = Image.fromarray(img_uint8, mode="RGB").resize(
        (target_w, target_h), resample=resample
    )
    return np.asarray(resized).astype(np.float32) / pixel_value_max


def _resize_depth_map(
    depth: np.ndarray,
    target_h: int,
    target_w: int,
    resample: int,
) -> np.ndarray:
    depth_img = Image.fromarray(depth.astype(np.float32), mode="F")
    resized = depth_img.resize((target_w, target_h), resample=resample)
    return np.asarray(resized).astype(np.float32)


def _adjust_rgb_gt_dimensions(
    img_gt: np.ndarray,
    target_dim: tuple[int, int],
    pixel_value_max: float = 255.0,
) -> np.ndarray:
    target_h, target_w = target_dim
    height, width = img_gt.shape[:2]
    if (height, width) == (target_h, target_w):
        return img_gt
    if _should_center_crop(height, width, target_h, target_w):
        # Small reductions are cropped to avoid interpolation artifacts.
        return _center_crop(img_gt, target_h, target_w)
    resample = Image.LANCZOS if target_h < height or target_w < width else Image.BICUBIC
    return _resize_rgb_image(img_gt, target_h, target_w, resample, pixel_value_max)


def _adjust_depth_to_rgb_dimensions(
    depth: np.ndarray, target_dim: tuple[int, int]
) -> np.ndarray:
    target_h, target_w = target_dim
    height, width = depth.shape[:2]
    if (height, width) == (target_h, target_w):
        return depth
    if _should_center_crop(height, width, target_h, target_w):
        return _center_crop(depth, target_h, target_w)
    return _resize_depth_map(depth, target_h, target_w, Image.BILINEAR)


def evaluate_depth_datasets(
    gt_config: dict,
    pred_config: dict,
    device: str = "cuda",
    batch_size: int = 16,
    num_workers: int = 4,
    verbose: bool = False,
    sanity_checker: Optional[SanityChecker] = None,
) -> dict:
    """Evaluate all depth metrics between GT and prediction datasets.

    Args:
        gt_config: Configuration for ground truth dataset.
        pred_config: Configuration for prediction dataset.
        device: Device for GPU-accelerated metrics.
        batch_size: Batch size for batched metrics.
        num_workers: Number of data loading workers.
        verbose: Enable verbose output.
        sanity_checker: Optional SanityChecker for post-processing validation.

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

    # Track entries (hierarchy + id) for per-file metrics
    processed_entries = []

    # Load all depth maps for FID/KID
    all_depths_gt = []
    all_depths_pred = []

    # Storage for sanity check metadata
    psnr_metadata_list = []
    ssim_metadata_list = []
    absrel_metadata_list = []
    normal_metadata_list = []

    # Process each pair
    print("Computing per-image depth metrics...")
    for match in tqdm(matches, desc="Processing depth pairs"):
        gt_file = match["gt_file"]
        pred_file = match["pred_file"]
        hierarchy = match["hierarchy"]
        entry_id = match["id"]

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

        # Track successfully processed entry (hierarchy + id)
        processed_entries.append({"hierarchy": hierarchy, "id": entry_id})

        # Validate depth input if sanity checker is enabled
        if sanity_checker is not None:
            sanity_checker.validate_depth_input(depth_gt, depth_pred, entry_id)

        # Image quality metrics (with metadata for sanity checking)
        psnr_val, psnr_meta = compute_psnr(depth_pred, depth_gt, return_metadata=True)
        psnr_values.append(psnr_val)
        psnr_metadata_list.append(psnr_meta)

        ssim_val, ssim_meta = compute_ssim(depth_pred, depth_gt, return_metadata=True)
        ssim_values.append(ssim_val)
        ssim_metadata_list.append(ssim_meta)

        lpips_values.append(lpips_metric.compute(depth_pred, depth_gt))

        # Depth-specific metrics (with metadata for sanity checking)
        absrel_arr, absrel_meta = compute_absrel(depth_pred, depth_gt, return_metadata=True)
        absrel_values.append(absrel_arr)
        absrel_metadata_list.append(absrel_meta)

        rmse_values.append(compute_rmse_per_pixel(depth_pred, depth_gt))
        silog_values.append(compute_silog_per_pixel(depth_pred, depth_gt))
        silog_full_values.append(compute_scale_invariant_log_error(depth_pred, depth_gt))

        # Geometric metrics (with metadata for sanity checking)
        normal_angles, normal_meta = compute_normal_angles(depth_pred, depth_gt, return_metadata=True)
        normal_angle_values.append(normal_angles)
        normal_metadata_list.append(normal_meta)

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

    # Post-processing sanity checks (does not impact GPU computation)
    if sanity_checker is not None:
        print("Running post-processing sanity checks...")
        for i, entry in enumerate(processed_entries):
            entry_id = entry["id"]

            # Validate PSNR
            psnr_meta = psnr_metadata_list[i]
            if psnr_meta["max_val_used"] is not None:
                sanity_checker.validate_depth_psnr(
                    psnr_values[i], psnr_meta["max_val_used"], entry_id
                )

            # Validate SSIM
            ssim_meta = ssim_metadata_list[i]
            if ssim_meta["depth_range"] is not None:
                sanity_checker.validate_depth_ssim(
                    ssim_values[i], ssim_meta["depth_range"], entry_id
                )

            # Validate AbsRel
            absrel_meta = absrel_metadata_list[i]
            if absrel_meta["median"] is not None:
                sanity_checker.validate_depth_absrel(
                    absrel_meta["median"], absrel_meta["p90"], entry_id
                )

            # Validate RMSE
            if ssim_meta["depth_range"] is not None and len(rmse_values[i]) > 0:
                rmse_val = float(np.sqrt(np.mean(rmse_values[i])))
                sanity_checker.validate_depth_rmse(
                    rmse_val, ssim_meta["depth_range"], entry_id
                )

            # Validate SILog
            silog_val = silog_full_values[i]
            if np.isfinite(silog_val):
                sanity_checker.validate_depth_silog(silog_val, entry_id)

            # Validate Normal Consistency
            normal_meta = normal_metadata_list[i]
            if normal_meta["mean_angle"] is not None:
                sanity_checker.validate_normal_consistency(
                    normal_meta["mean_angle"],
                    normal_meta["valid_pixels_after_erosion"],
                    entry_id
                )

            # Validate Depth Edge F1
            edge_f1 = edge_f1_results[i]
            sanity_checker.validate_depth_edge_f1(
                edge_f1["pred_edge_pixels"],
                edge_f1["gt_edge_pixels"],
                edge_f1["total_pixels"],
                edge_f1["f1"],
                entry_id
            )

    # Build per-file metrics (excluding FID/KID which are distribution metrics)
    per_file_metrics = {}
    for i, entry in enumerate(processed_entries):
        hierarchy = entry["hierarchy"]
        entry_id = entry["id"]
        absrel_arr = absrel_values[i]
        rmse_arr = rmse_values[i]
        normal_angles = normal_angle_values[i]
        edge_f1 = edge_f1_results[i]

        depth_metrics_value = {
            "depth": {
                "image_quality": {
                    "psnr": float(psnr_values[i]) if np.isfinite(psnr_values[i]) else None,
                    "ssim": float(ssim_values[i]) if np.isfinite(ssim_values[i]) else None,
                    "lpips": float(lpips_values[i]),
                },
                "depth_metrics": {
                    "absrel": float(np.mean(absrel_arr)) if len(absrel_arr) > 0 else None,
                    "rmse": float(np.sqrt(np.mean(rmse_arr))) if len(rmse_arr) > 0 else None,
                    "silog": float(silog_full_values[i]) if np.isfinite(silog_full_values[i]) else None,
                },
                "geometric_metrics": {
                    "normal_consistency": {
                        "mean_angle": float(np.mean(normal_angles)) if len(normal_angles) > 0 else None,
                    },
                    "depth_edge_f1": {
                        "precision": float(edge_f1["precision"]),
                        "recall": float(edge_f1["recall"]),
                        "f1": float(edge_f1["f1"]),
                    },
                },
            },
        }

        # Use set_value to place metrics in hierarchical structure
        set_value(per_file_metrics, hierarchy, entry_id, {"id": entry_id, "metrics": depth_metrics_value})

    results = {
        "depth": {
            "image_quality": {
                "psnr": float(np.mean([v for v in psnr_values if np.isfinite(v)])),
                "ssim": float(np.mean([v for v in ssim_values if np.isfinite(v)])),
                "lpips": float(np.mean(lpips_values)),
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
    sanity_checker: Optional[SanityChecker] = None,
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
        sanity_checker: Optional SanityChecker for post-processing validation.

    Returns:
        Dictionary containing all computed RGB metrics.
    """
    def _warn_metric_failure(metric_name: str, entry_id: str, exc: Exception) -> None:
        print(f"Warning: Failed to compute {metric_name} for {entry_id}: {exc}")

    def _safe_compute(metric_name: str, entry_id: str, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            _warn_metric_failure(metric_name, entry_id, exc)
            return None

    def _safe_mean(values: list[Optional[float]], metric_name: str) -> Optional[float]:
        valid = [float(v) for v in values if v is not None and np.isfinite(v)]
        if not valid:
            print(f"Warning: No valid values for {metric_name}; setting to None.")
            return None
        return float(np.mean(valid))

    def _none_if_nan(value: Optional[float]) -> Optional[float]:
        if value is None or not np.isfinite(value):
            return None
        return float(value)

    gt_path = Path(gt_config["path"])
    pred_path = Path(pred_config["path"])

    gt_pixel_value_max = gt_config.get("pixel_value_max", 255.0)
    pred_pixel_value_max = pred_config.get("pixel_value_max", 255.0)

    target_dim = _validate_target_dim(pred_config.get("dim"))
    if target_dim is not None:
        print(f"Applying RGB GT preprocessing to target dim: {target_dim[0]}x{target_dim[1]}")

    depth_path = None
    depth_scale = 1.0
    depth_intrinsics = None
    depth_hierarchy_map = None
    if depth_gt_config is not None:
        depth_path = Path(depth_gt_config["path"])
        depth_scale = depth_gt_config.get("depth_scale", 1.0)
        depth_intrinsics = depth_gt_config.get("intrinsics")
        # Load depth manifest for hierarchy-based matching
        depth_manifest = load_dataset_manifest(depth_path)
        depth_hierarchy_map = build_depth_hierarchy_map(depth_manifest)

    # Find matching RGB files by ID from output.json manifests
    print("Finding matching RGB files...")
    matches = find_matching_files_by_id(gt_path, pred_path)

    if not matches:
        raise ValueError(f"No matching RGB files found between {gt_path} and {pred_path}")

    print(f"Found {len(matches)} matching RGB file pairs")

    # Initialize GPU-accelerated metrics
    print(f"Initializing RGB metrics (device: {device})...")
    try:
        lpips_metric = RGBLPIPSMetric(device=device)
    except Exception as exc:
        print(f"Warning: Failed to initialize LPIPS metric: {exc}")
        lpips_metric = None

    # Storage for per-image metrics
    psnr_values = []
    ssim_values = []
    lpips_values = []
    edge_f1_results = []
    tail_error_arrays = []
    high_freq_results = []
    depth_binned_results = []

    # Track entries (hierarchy + id) for per-file metrics
    processed_entries = []
    # Track depth-binned results per entry (None if not available)
    depth_binned_per_entry = []
    depth_binned_attempted = []

    has_depth = depth_path is not None

    # Process each pair
    print("Computing per-image RGB metrics...")
    for match in tqdm(matches, desc="Processing RGB pairs"):
        gt_file = match["gt_file"]
        pred_file = match["pred_file"]
        hierarchy = match["hierarchy"]
        entry_id = match["id"]

        try:
            img_gt = load_rgb_file(gt_file, gt_pixel_value_max)
            img_pred = load_rgb_file(pred_file, pred_pixel_value_max)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load {gt_file} or {pred_file}: {e}")
                print(f"Might be due to incomplete prediction dataset.")
            continue

        if target_dim is not None:
            img_gt = _adjust_rgb_gt_dimensions(img_gt, target_dim, gt_pixel_value_max)

        if img_gt.shape != img_pred.shape:
            if verbose:
                print(
                    f"Warning: Size mismatch for {entry_id}: "
                    f"gt {img_gt.shape} vs pred {img_pred.shape}. Skipping."
                )
            continue

        # Track successfully processed entry (hierarchy + id)
        processed_entries.append({"hierarchy": hierarchy, "id": entry_id})

        # Validate RGB input if sanity checker is enabled
        if sanity_checker is not None:
            sanity_checker.validate_rgb_input(img_gt, img_pred, entry_id)

        # Basic image quality metrics
        psnr_values.append(
            _safe_compute("psnr", entry_id, compute_rgb_psnr, img_pred, img_gt)
        )
        ssim_values.append(
            _safe_compute("ssim", entry_id, compute_rgb_ssim, img_pred, img_gt)
        )
        if lpips_metric is not None:
            lpips_values.append(
                _safe_compute("lpips", entry_id, lpips_metric.compute, img_pred, img_gt)
            )
        else:
            lpips_values.append(None)

        # Edge F1
        edge_f1_results.append(
            _safe_compute("edge_f1", entry_id, compute_rgb_edge_f1, img_pred, img_gt)
        )

        # Tail errors
        tail_error_arrays.append(
            _safe_compute(
                "tail_errors",
                entry_id,
                lambda pred, gt: np.abs(pred - gt).mean(axis=-1),
                img_pred,
                img_gt,
            )
        )

        # High-frequency energy
        high_freq_results.append(
            _safe_compute(
                "high_frequency",
                entry_id,
                compute_high_freq_energy_comparison,
                img_pred,
                img_gt,
            )
        )

        # Depth-binned photometric error (if depth available)
        depth_binned_entry = None
        depth_binned_attempt = False
        if has_depth and depth_hierarchy_map is not None:
            depth_file = find_matching_depth_for_rgb(hierarchy, entry_id, depth_path, depth_hierarchy_map)
            if depth_file is not None:
                depth_binned_attempt = True
                try:
                    depth = load_depth_file(depth_file, depth_scale, depth_intrinsics)
                    if target_dim is not None:
                        depth = _adjust_depth_to_rgb_dimensions(depth, target_dim)
                    elif depth.shape[:2] != img_gt.shape[:2]:
                        depth = _adjust_depth_to_rgb_dimensions(depth, img_gt.shape[:2])
                    if depth.shape[:2] != img_gt.shape[:2]:
                        raise ValueError(
                            f"Depth shape {depth.shape} does not match RGB shape {img_gt.shape}"
                        )
                    depth_binned_entry = compute_depth_binned_photometric_error(img_pred, img_gt, depth)
                    depth_binned_results.append(depth_binned_entry)
                except Exception as e:
                    _warn_metric_failure("depth_binned_photometric", entry_id, e)
        depth_binned_per_entry.append(depth_binned_entry)
        depth_binned_attempted.append(depth_binned_attempt)

    # Aggregate results
    print("Aggregating RGB results...")

    edge_f1_valid = [r for r in edge_f1_results if r is not None]
    if edge_f1_valid:
        edge_f1_agg = aggregate_rgb_edge_f1(edge_f1_valid)
    else:
        print("Warning: No valid edge_f1 results; setting aggregate to None.")
        edge_f1_agg = {"precision": None, "recall": None, "f1": None}

    tail_valid = [r for r in tail_error_arrays if r is not None]
    if tail_valid:
        tail_agg = aggregate_tail_errors(tail_valid)
    else:
        print("Warning: No valid tail_errors results; setting aggregate to None.")
        tail_agg = {"p95": None, "p99": None}

    high_freq_valid = [r for r in high_freq_results if r is not None]
    if high_freq_valid:
        high_freq_agg = aggregate_high_freq_metrics(high_freq_valid)
    else:
        print("Warning: No valid high_frequency results; setting aggregate to None.")
        high_freq_agg = {
            "pred_hf_ratio_mean": None,
            "gt_hf_ratio_mean": None,
            "relative_diff_mean": None,
        }

    # Post-processing sanity checks for RGB (does not impact GPU computation)
    if sanity_checker is not None:
        print("Running post-processing sanity checks for RGB...")
        for i, entry in enumerate(processed_entries):
            entry_id = entry["id"]

            # Validate RGB PSNR
            psnr_val = psnr_values[i]
            if psnr_val is not None and np.isfinite(psnr_val):
                sanity_checker.validate_rgb_psnr(psnr_val, entry_id)

            # Validate RGB SSIM
            ssim_val = ssim_values[i]
            if ssim_val is not None and np.isfinite(ssim_val):
                sanity_checker.validate_rgb_ssim(ssim_val, entry_id)

            # Validate RGB LPIPS
            lpips_val = lpips_values[i]
            if lpips_val is not None and np.isfinite(lpips_val):
                sanity_checker.validate_rgb_lpips(lpips_val, entry_id)

            # Validate tail errors
            tail_arr = tail_error_arrays[i]
            if tail_arr is not None and len(tail_arr) > 0:
                p99 = float(np.percentile(tail_arr, 99))
                sanity_checker.validate_tail_errors(p99, entry_id)

            # Validate high-frequency energy
            hf_result = high_freq_results[i]
            if hf_result is not None and np.isfinite(hf_result.get("relative_diff", float("nan"))):
                sanity_checker.validate_high_freq_energy(hf_result["relative_diff"], entry_id)

            # Validate depth-binned metrics
            depth_binned = depth_binned_per_entry[i]
            if depth_binned is not None:
                sanity_checker.validate_depth_binned(depth_binned, entry_id)

    # Build per-file metrics
    per_file_metrics = {}
    for i, entry in enumerate(processed_entries):
        hierarchy = entry["hierarchy"]
        entry_id = entry["id"]
        edge_f1 = edge_f1_results[i]
        tail_error_arr = tail_error_arrays[i]
        high_freq = high_freq_results[i]

        rgb_metrics = {
            "image_quality": {
                "psnr": _none_if_nan(psnr_values[i]),
                "ssim": _none_if_nan(ssim_values[i]),
                "lpips": _none_if_nan(lpips_values[i]),
            },
            "edge_f1": {
                "precision": _none_if_nan(edge_f1["precision"]) if edge_f1 else None,
                "recall": _none_if_nan(edge_f1["recall"]) if edge_f1 else None,
                "f1": _none_if_nan(edge_f1["f1"]) if edge_f1 else None,
            },
            "tail_errors": {
                "p95": float(np.percentile(tail_error_arr, 95))
                if tail_error_arr is not None and len(tail_error_arr) > 0
                else None,
                "p99": float(np.percentile(tail_error_arr, 99))
                if tail_error_arr is not None and len(tail_error_arr) > 0
                else None,
            },
            "high_frequency": {
                "pred_hf_ratio": _none_if_nan(high_freq["pred_hf_ratio"]) if high_freq else None,
                "gt_hf_ratio": _none_if_nan(high_freq["gt_hf_ratio"]) if high_freq else None,
                "relative_diff": _none_if_nan(high_freq["relative_diff"]) if high_freq else None,
            },
        }

        # Add depth-binned metrics if available for this entry
        depth_binned = depth_binned_per_entry[i]
        if depth_binned is not None or depth_binned_attempted[i]:
            rgb_metrics["depth_binned_photometric"] = depth_binned

        rgb_metrics_value = {"rgb": rgb_metrics}

        # Use set_value to place metrics in hierarchical structure
        set_value(per_file_metrics, hierarchy, entry_id, {"id": entry_id, "metrics": rgb_metrics_value})

    rgb_results = {
        "image_quality": {
            "psnr": _safe_mean(psnr_values, "psnr"),
            "ssim": _safe_mean(ssim_values, "ssim"),
            "lpips": _safe_mean(lpips_values, "lpips"),
        },
        "edge_f1": {
            "precision": _none_if_nan(edge_f1_agg["precision"]),
            "recall": _none_if_nan(edge_f1_agg["recall"]),
            "f1": _none_if_nan(edge_f1_agg["f1"]),
        },
        "tail_errors": {
            "p95": _none_if_nan(tail_agg["p95"]),
            "p99": _none_if_nan(tail_agg["p99"]),
        },
        "high_frequency": {
            "pred_hf_ratio": _none_if_nan(high_freq_agg["pred_hf_ratio_mean"]),
            "gt_hf_ratio": _none_if_nan(high_freq_agg["gt_hf_ratio_mean"]),
            "relative_diff": _none_if_nan(high_freq_agg["relative_diff_mean"]),
        },
        "dataset_info": {
            "num_pairs": len(matches),
            "gt_name": gt_config["name"],
            "pred_name": pred_config["name"],
            "gt_path": str(gt_path),
            "pred_path": str(pred_path),
        },
    }

    # Add depth-binned metrics if available
    if depth_binned_results:
        depth_binned_agg = aggregate_depth_binned_errors(depth_binned_results)
        rgb_results["depth_binned_photometric"] = depth_binned_agg
    elif has_depth:
        print("Warning: No valid depth_binned_photometric results; setting aggregate to None.")
        rgb_results["depth_binned_photometric"] = None

    results = {
        "rgb": rgb_results,
        "per_file_metrics": per_file_metrics,
    }

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
