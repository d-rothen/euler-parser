"""Dataset evaluation orchestrator.

Runs all metrics over depth and RGB datasets loaded via euler_loading.
"""

import numpy as np
from typing import Optional
from tqdm import tqdm

from euler_loading import MultiModalDataset

from .data import (
    align_to_prediction,
    compute_scale_and_shift,
    process_depth,
    to_numpy_depth,
    to_numpy_intrinsics,
    to_numpy_mask,
    to_numpy_rgb,
)
from .sanity_checker import SanityChecker
from .utils.hierarchy_parser import set_value

from .metrics import (
    # Depth utilities and metrics
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
    compute_rgb_psnr,
    compute_rgb_ssim,
    RGBLPIPSMetric,
    compute_sce,
    compute_depth_binned_photometric_error,
    aggregate_depth_binned_errors,
    compute_rgb_edge_f1,
    aggregate_rgb_edge_f1,
    aggregate_tail_errors,
    compute_high_freq_energy_comparison,
    aggregate_high_freq_metrics,
)


# ---------------------------------------------------------------------------
# Sample statistics logging
# ---------------------------------------------------------------------------


def _get_dtype_precision_str(arr: np.ndarray) -> str:
    dtype = arr.dtype
    if dtype == np.float16:
        return "16-bit float"
    elif dtype == np.float32:
        return "32-bit float"
    elif dtype == np.float64:
        return "64-bit float"
    elif dtype == np.uint8:
        return "8-bit unsigned int"
    elif dtype == np.uint16:
        return "16-bit unsigned int"
    else:
        return str(dtype)


def _compute_array_stats(arr: np.ndarray) -> dict:
    valid_mask = np.isfinite(arr)
    valid_values = arr[valid_mask]
    if len(valid_values) == 0:
        return {
            "shape": arr.shape, "dtype": str(arr.dtype),
            "min": float("nan"), "max": float("nan"),
            "mean": float("nan"), "precision": _get_dtype_precision_str(arr),
        }
    return {
        "shape": arr.shape, "dtype": str(arr.dtype),
        "min": float(np.min(valid_values)),
        "max": float(np.max(valid_values)),
        "mean": float(np.mean(valid_values)),
        "precision": _get_dtype_precision_str(arr),
    }


def _log_sample_stats(gt: np.ndarray, pred: np.ndarray, label: str) -> None:
    gt_stats = _compute_array_stats(gt)
    pred_stats = _compute_array_stats(pred)
    print(f"\n{'=' * 77}")
    print(f"  {label} SAMPLE STATISTICS")
    print("=" * 77)
    for tag, stats in [("GT", gt_stats), ("Pred", pred_stats)]:
        print(f"  {tag}: shape={stats['shape']}  dtype={stats['dtype']}  "
              f"range=[{stats['min']:.4f}, {stats['max']:.4f}]  "
              f"mean={stats['mean']:.4f}")
    print("=" * 77 + "\n")


# ---------------------------------------------------------------------------
# Sky mask extraction
# ---------------------------------------------------------------------------


def _get_sky_mask(sample: dict) -> Optional[np.ndarray]:
    """Extract inverted sky mask from sample (True = valid, non-sky pixel)."""
    seg = sample.get("segmentation")
    if seg is None:
        return None
    # segmentation is a dict from hierarchical modality: {file_id: loaded_mask}
    if isinstance(seg, dict):
        if not seg:
            return None
        # Use the deepest (most specific) entry
        mask_data = next(iter(seg.values()))
    else:
        mask_data = seg
    sky = to_numpy_mask(mask_data)
    return ~sky  # invert: True = non-sky = valid


def _get_intrinsics_K(sample: dict) -> Optional[np.ndarray]:
    """Extract (3,3) intrinsics matrix from sample calibration data."""
    cal = sample.get("calibration")
    if cal is None:
        return None
    if isinstance(cal, dict):
        if not cal:
            return None
        K_data = next(iter(cal.values()))
    else:
        K_data = cal
    return to_numpy_intrinsics(K_data)


# ---------------------------------------------------------------------------
# Hierarchy extraction from sample
# ---------------------------------------------------------------------------


def _extract_hierarchy(sample: dict) -> tuple[list[str], str]:
    """Return (hierarchy_path_list, file_id) from a MultiModalDataset sample."""
    file_id = sample["id"]
    full_id = sample.get("full_id", f"/{file_id}")
    parts = [p for p in full_id.strip("/").split("/") if p]
    if len(parts) > 1:
        return parts[:-1], parts[-1]
    return [], file_id


# ---------------------------------------------------------------------------
# Depth evaluation
# ---------------------------------------------------------------------------


def evaluate_depth_samples(
    dataset: MultiModalDataset,
    scale_to_meters: float,
    is_radial: bool,
    gt_name: str = "GT",
    pred_name: str = "Pred",
    device: str = "cuda",
    batch_size: int = 16,
    num_workers: int = 4,
    verbose: bool = False,
    sanity_checker: Optional[SanityChecker] = None,
    sky_mask_enabled: bool = False,
    scale_and_shift: bool = True,
) -> dict:
    """Evaluate all depth metrics from a MultiModalDataset.

    The dataset must yield samples with ``"gt"`` and ``"pred"`` keys
    containing depth data (tensors or arrays).

    Returns:
        Dictionary containing aggregate and per-file metrics.
    """
    num_samples = len(dataset)
    if num_samples == 0:
        raise ValueError("Dataset has no matched samples")

    print(f"Initializing depth metrics (device: {device})...")
    lpips_metric = LPIPSMetric(device=device)
    fid_kid_metric = FIDKIDMetric(device=device)

    # Per-image storage
    psnr_values = []
    ssim_values = []
    lpips_values = []
    absrel_values = []
    rmse_values = []
    silog_values = []
    silog_full_values = []
    normal_angle_values = []
    edge_f1_results = []
    processed_entries = []

    all_depths_gt = []
    all_depths_pred = []

    # Sanity check metadata
    psnr_metadata_list = []
    ssim_metadata_list = []
    absrel_metadata_list = []
    normal_metadata_list = []

    logged_stats = False
    logged_alignment = False
    do_alignment = False

    print("Computing per-image depth metrics...")
    for i in tqdm(range(num_samples), desc="Processing depth pairs"):
        sample = dataset[i]
        hierarchy, entry_id = _extract_hierarchy(sample)

        depth_gt = to_numpy_depth(sample["gt"])
        depth_pred = to_numpy_depth(sample["pred"])

        # Align GT to prediction dimensions (e.g. VAE multiple-of-8 crop)
        if depth_gt.shape[:2] != depth_pred.shape[:2]:
            if not logged_alignment:
                print(
                    f"  Aligning GT {depth_gt.shape[:2]} -> "
                    f"pred {depth_pred.shape[:2]}"
                )
                logged_alignment = True
            depth_gt = align_to_prediction(depth_gt, depth_pred)

        intrinsics_K = _get_intrinsics_K(sample)

        # Extract sky mask early (needed for both SNS fitting and metrics)
        sky_valid = None
        if sky_mask_enabled:
            sky_valid = _get_sky_mask(sample)
            if sky_valid is not None and sky_valid.shape[:2] != depth_pred.shape[:2]:
                sky_valid = align_to_prediction(sky_valid, depth_pred)

        # Detect normalized predictions on first sample
        if scale_and_shift and i == 0:
            pred_min = float(np.nanmin(depth_pred))
            pred_max = float(np.nanmax(depth_pred))
            if pred_max <= 1.0 + 1e-3 and pred_min >= -1.0 - 1e-3:
                do_alignment = True
                print(
                    f"  Scale-and-shift: detected normalized predictions "
                    f"(range [{pred_min:.3f}, {pred_max:.3f}])"
                )
            else:
                print(
                    f"  Scale-and-shift: predictions appear metric "
                    f"(range [{pred_min:.1f}, {pred_max:.1f}]), "
                    f"skipping alignment"
                )

        # Process depth: GT always gets scale/radial conversion;
        # pred is either processed the same way or aligned via LSQ.
        depth_gt = process_depth(depth_gt, scale_to_meters, is_radial, intrinsics_K)

        if do_alignment:
            fit_mask = (depth_gt > 0) & np.isfinite(depth_gt) & np.isfinite(depth_pred)
            if sky_valid is not None:
                fit_mask = fit_mask & sky_valid
            depth_pred, s, t = compute_scale_and_shift(
                depth_pred, depth_gt, fit_mask,
            )
            if verbose and not logged_stats:
                print(f"  Fitted scale={s:.4f}, shift={t:.4f}")
        else:
            depth_pred = process_depth(
                depth_pred, scale_to_meters, is_radial, intrinsics_K,
            )

        # Log statistics for first sample
        if verbose and not logged_stats:
            _log_sample_stats(depth_gt, depth_pred, "DEPTH")
            logged_stats = True

        # Build valid mask (optionally excluding sky)
        valid_mask = None
        if sky_valid is not None:
            valid_mask = (
                (depth_gt > 0) & (depth_pred > 0)
                & np.isfinite(depth_gt) & np.isfinite(depth_pred)
                & sky_valid
            )

        all_depths_gt.append(depth_gt)
        all_depths_pred.append(depth_pred)
        processed_entries.append({"hierarchy": hierarchy, "id": entry_id})

        if sanity_checker is not None:
            sanity_checker.validate_depth_input(depth_gt, depth_pred, entry_id)

        # Image quality metrics
        psnr_val, psnr_meta = compute_psnr(
            depth_pred, depth_gt, valid_mask=valid_mask, return_metadata=True
        )
        psnr_values.append(psnr_val)
        psnr_metadata_list.append(psnr_meta)

        ssim_val, ssim_meta = compute_ssim(depth_pred, depth_gt, return_metadata=True)
        ssim_values.append(ssim_val)
        ssim_metadata_list.append(ssim_meta)

        lpips_values.append(lpips_metric.compute(depth_pred, depth_gt))

        # Depth-specific metrics
        absrel_arr, absrel_meta = compute_absrel(
            depth_pred, depth_gt, valid_mask=valid_mask, return_metadata=True
        )
        absrel_values.append(absrel_arr)
        absrel_metadata_list.append(absrel_meta)

        rmse_values.append(
            compute_rmse_per_pixel(depth_pred, depth_gt, valid_mask=valid_mask)
        )
        silog_values.append(
            compute_silog_per_pixel(depth_pred, depth_gt, valid_mask=valid_mask)
        )
        silog_full_values.append(
            compute_scale_invariant_log_error(depth_pred, depth_gt, valid_mask=valid_mask)
        )

        # Geometric metrics
        normal_angles, normal_meta = compute_normal_angles(
            depth_pred, depth_gt, valid_mask=valid_mask, return_metadata=True
        )
        normal_angle_values.append(normal_angles)
        normal_metadata_list.append(normal_meta)

        edge_f1_results.append(
            compute_depth_edge_f1(depth_pred, depth_gt, valid_mask=valid_mask)
        )

    # FID / KID
    print("Computing FID/KID (this may take a while)...")
    fid_value = fid_kid_metric.compute_fid(
        all_depths_gt, all_depths_pred, batch_size, num_workers
    )
    kid_mean, kid_std = fid_kid_metric.compute_kid(
        all_depths_gt, all_depths_pred, batch_size, num_workers
    )

    # Aggregate
    print("Aggregating depth results...")
    absrel_agg = aggregate_absrel(absrel_values)
    rmse_agg = aggregate_rmse(rmse_values)
    silog_agg = aggregate_silog(silog_values)
    normal_agg = aggregate_normal_consistency(normal_angle_values)
    edge_f1_agg = aggregate_edge_f1(edge_f1_results)

    # Post-processing sanity checks
    if sanity_checker is not None:
        print("Running post-processing sanity checks...")
        for i, entry in enumerate(processed_entries):
            eid = entry["id"]
            pm = psnr_metadata_list[i]
            if pm["max_val_used"] is not None:
                sanity_checker.validate_depth_psnr(psnr_values[i], pm["max_val_used"], eid)
            sm = ssim_metadata_list[i]
            if sm["depth_range"] is not None:
                sanity_checker.validate_depth_ssim(ssim_values[i], sm["depth_range"], eid)
            am = absrel_metadata_list[i]
            if am["median"] is not None:
                sanity_checker.validate_depth_absrel(am["median"], am["p90"], eid)
            if sm["depth_range"] is not None and len(rmse_values[i]) > 0:
                rmse_val = float(np.sqrt(np.mean(rmse_values[i])))
                sanity_checker.validate_depth_rmse(rmse_val, sm["depth_range"], eid)
            sv = silog_full_values[i]
            if np.isfinite(sv):
                sanity_checker.validate_depth_silog(sv, eid)
            nm = normal_metadata_list[i]
            if nm["mean_angle"] is not None:
                sanity_checker.validate_normal_consistency(
                    nm["mean_angle"], nm["valid_pixels_after_erosion"], eid
                )
            ef = edge_f1_results[i]
            sanity_checker.validate_depth_edge_f1(
                ef["pred_edge_pixels"], ef["gt_edge_pixels"],
                ef["total_pixels"], ef["f1"], eid
            )

    # Build per-file metrics
    per_file_metrics = {}
    for i, entry in enumerate(processed_entries):
        hierarchy = entry["hierarchy"]
        eid = entry["id"]
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
        set_value(per_file_metrics, hierarchy, eid, {"id": eid, "metrics": depth_metrics_value})

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
                "absrel": {"median": absrel_agg["median"], "p90": absrel_agg["p90"]},
                "rmse": {"median": rmse_agg["median"], "p90": rmse_agg["p90"]},
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
                "num_pairs": num_samples,
                "gt_name": gt_name,
                "pred_name": pred_name,
            },
        },
        "per_file_metrics": per_file_metrics,
    }
    return results


# ---------------------------------------------------------------------------
# RGB evaluation
# ---------------------------------------------------------------------------


def evaluate_rgb_samples(
    dataset: MultiModalDataset,
    depth_meta: Optional[dict] = None,
    gt_name: str = "GT",
    pred_name: str = "Pred",
    device: str = "cuda",
    batch_size: int = 16,
    num_workers: int = 4,
    verbose: bool = False,
    sanity_checker: Optional[SanityChecker] = None,
    sky_mask_enabled: bool = False,
) -> dict:
    """Evaluate all RGB metrics from a MultiModalDataset.

    The dataset must yield samples with ``"gt"`` and ``"pred"`` keys
    containing RGB data. Optionally ``"gt_depth"`` for depth-binned
    metrics, ``"segmentation"`` for sky masking, and ``"calibration"``.

    Args:
        dataset: The MultiModalDataset to iterate.
        depth_meta: Depth metadata dict (scale_to_meters, radial_depth)
                    for depth-binned metrics. None to skip.
        gt_name: GT dataset display name.
        pred_name: Prediction dataset display name.
        device: Computation device.
        batch_size: Batch size for batched metrics.
        num_workers: Data loading workers.
        verbose: Enable verbose output.
        sanity_checker: Optional SanityChecker.
        sky_mask_enabled: If True, use segmentation for sky masking.

    Returns:
        Dictionary containing aggregate and per-file metrics.
    """
    def _warn_metric_failure(metric_name: str, entry_id: str, exc: Exception) -> None:
        print(f"Warning: Failed to compute {metric_name} for {entry_id}: {exc}")

    def _safe_compute(metric_name: str, entry_id: str, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            _warn_metric_failure(metric_name, entry_id, exc)
            return None

    def _safe_mean(values: list, metric_name: str) -> Optional[float]:
        valid = [float(v) for v in values if v is not None and np.isfinite(v)]
        if not valid:
            print(f"Warning: No valid values for {metric_name}; setting to None.")
            return None
        return float(np.mean(valid))

    def _none_if_nan(value) -> Optional[float]:
        if value is None or not np.isfinite(value):
            return None
        return float(value)

    num_samples = len(dataset)
    if num_samples == 0:
        raise ValueError("Dataset has no matched samples")

    has_depth = "gt_depth" in dataset.modality_paths() and depth_meta is not None

    print(f"Initializing RGB metrics (device: {device})...")
    try:
        lpips_metric = RGBLPIPSMetric(device=device)
    except Exception as exc:
        print(f"Warning: Failed to initialize LPIPS metric: {exc}")
        lpips_metric = None

    # Per-image storage
    psnr_values = []
    ssim_values = []
    lpips_values = []
    sce_values = []
    edge_f1_results = []
    tail_error_arrays = []
    high_freq_results = []
    depth_binned_results = []
    processed_entries = []
    depth_binned_per_entry = []

    logged_stats = False
    logged_alignment = False

    print("Computing per-image RGB metrics...")
    for i in tqdm(range(num_samples), desc="Processing RGB pairs"):
        sample = dataset[i]
        hierarchy, entry_id = _extract_hierarchy(sample)

        img_gt = to_numpy_rgb(sample["gt"])
        img_pred = to_numpy_rgb(sample["pred"])

        # Align GT to prediction dimensions (e.g. VAE multiple-of-8 crop)
        if img_gt.shape[:2] != img_pred.shape[:2]:
            if not logged_alignment:
                print(
                    f"  Aligning GT {img_gt.shape[:2]} -> "
                    f"pred {img_pred.shape[:2]}"
                )
                logged_alignment = True
            img_gt = align_to_prediction(img_gt, img_pred)

        if not logged_stats:
            _log_sample_stats(img_gt, img_pred, "RGB")
            logged_stats = True

        processed_entries.append({"hierarchy": hierarchy, "id": entry_id})

        # Sky mask for RGB: zero out masked regions for metrics that don't accept masks
        sky_valid = None
        if sky_mask_enabled:
            sky_valid = _get_sky_mask(sample)
            if sky_valid is not None and sky_valid.shape[:2] != img_pred.shape[:2]:
                sky_valid = align_to_prediction(sky_valid, img_pred)

        if sanity_checker is not None:
            sanity_checker.validate_rgb_input(img_gt, img_pred, entry_id)

        # Apply sky mask to images for metrics that don't support explicit masks
        gt_masked = img_gt
        pred_masked = img_pred
        if sky_valid is not None:
            mask_3c = np.stack([sky_valid] * 3, axis=-1)
            gt_masked = img_gt * mask_3c
            pred_masked = img_pred * mask_3c

        # Basic quality metrics
        psnr_values.append(
            _safe_compute("psnr", entry_id, compute_rgb_psnr, pred_masked, gt_masked)
        )
        ssim_values.append(
            _safe_compute("ssim", entry_id, compute_rgb_ssim, pred_masked, gt_masked)
        )
        sce_values.append(
            _safe_compute("sce", entry_id, compute_sce, pred_masked, gt_masked)
        )
        if lpips_metric is not None:
            lpips_values.append(
                _safe_compute("lpips", entry_id, lpips_metric.compute, pred_masked, gt_masked)
            )
        else:
            lpips_values.append(None)

        # Edge F1
        edge_f1_results.append(
            _safe_compute("edge_f1", entry_id, compute_rgb_edge_f1, pred_masked, gt_masked)
        )

        # Tail errors
        tail_error_arrays.append(
            _safe_compute(
                "tail_errors", entry_id,
                lambda pred, gt: np.abs(pred - gt).mean(axis=-1),
                pred_masked, gt_masked,
            )
        )

        # High-frequency energy
        high_freq_results.append(
            _safe_compute(
                "high_frequency", entry_id,
                compute_high_freq_energy_comparison, pred_masked, gt_masked,
            )
        )

        # Depth-binned photometric error
        depth_binned_entry = None
        if has_depth:
            try:
                gt_depth_raw = to_numpy_depth(sample["gt_depth"])
                intrinsics_K = _get_intrinsics_K(sample)
                gt_depth = process_depth(
                    gt_depth_raw,
                    depth_meta["scale_to_meters"],
                    depth_meta["radial_depth"],
                    intrinsics_K,
                )
                if gt_depth.shape[:2] != img_pred.shape[:2]:
                    gt_depth = align_to_prediction(gt_depth, img_pred)
                depth_binned_entry = compute_depth_binned_photometric_error(
                    pred_masked, gt_masked, gt_depth
                )
                depth_binned_results.append(depth_binned_entry)
            except Exception as e:
                _warn_metric_failure("depth_binned_photometric", entry_id, e)

        depth_binned_per_entry.append(depth_binned_entry)

    # Aggregate
    print("Aggregating RGB results...")

    edge_f1_valid = [r for r in edge_f1_results if r is not None]
    edge_f1_agg = (
        aggregate_rgb_edge_f1(edge_f1_valid)
        if edge_f1_valid
        else {"precision": None, "recall": None, "f1": None}
    )

    tail_valid = [r for r in tail_error_arrays if r is not None]
    tail_agg = (
        aggregate_tail_errors(tail_valid)
        if tail_valid
        else {"p95": None, "p99": None}
    )

    high_freq_valid = [r for r in high_freq_results if r is not None]
    high_freq_agg = (
        aggregate_high_freq_metrics(high_freq_valid)
        if high_freq_valid
        else {"pred_hf_ratio_mean": None, "gt_hf_ratio_mean": None, "relative_diff_mean": None}
    )

    # Post-processing sanity checks
    if sanity_checker is not None:
        print("Running post-processing sanity checks for RGB...")
        for i, entry in enumerate(processed_entries):
            eid = entry["id"]
            pv = psnr_values[i]
            if pv is not None and np.isfinite(pv):
                sanity_checker.validate_rgb_psnr(pv, eid)
            sv = ssim_values[i]
            if sv is not None and np.isfinite(sv):
                sanity_checker.validate_rgb_ssim(sv, eid)
            lv = lpips_values[i]
            if lv is not None and np.isfinite(lv):
                sanity_checker.validate_rgb_lpips(lv, eid)
            ta = tail_error_arrays[i]
            if ta is not None and len(ta) > 0:
                p99 = float(np.percentile(ta, 99))
                sanity_checker.validate_tail_errors(p99, eid)
            hf = high_freq_results[i]
            if hf is not None and np.isfinite(hf.get("relative_diff", float("nan"))):
                sanity_checker.validate_high_freq_energy(hf["relative_diff"], eid)
            db = depth_binned_per_entry[i]
            if db is not None:
                sanity_checker.validate_depth_binned(db, eid)

    # Build per-file metrics
    per_file_metrics = {}
    for i, entry in enumerate(processed_entries):
        hierarchy = entry["hierarchy"]
        eid = entry["id"]
        edge_f1 = edge_f1_results[i]
        tail_arr = tail_error_arrays[i]
        high_freq = high_freq_results[i]

        rgb_metrics = {
            "image_quality": {
                "psnr": _none_if_nan(psnr_values[i]),
                "ssim": _none_if_nan(ssim_values[i]),
                "sce": _none_if_nan(sce_values[i]),
                "lpips": _none_if_nan(lpips_values[i]),
            },
            "edge_f1": {
                "precision": _none_if_nan(edge_f1["precision"]) if edge_f1 else None,
                "recall": _none_if_nan(edge_f1["recall"]) if edge_f1 else None,
                "f1": _none_if_nan(edge_f1["f1"]) if edge_f1 else None,
            },
            "tail_errors": {
                "p95": float(np.percentile(tail_arr, 95)) if tail_arr is not None and len(tail_arr) > 0 else None,
                "p99": float(np.percentile(tail_arr, 99)) if tail_arr is not None and len(tail_arr) > 0 else None,
            },
            "high_frequency": {
                "pred_hf_ratio": _none_if_nan(high_freq["pred_hf_ratio"]) if high_freq else None,
                "gt_hf_ratio": _none_if_nan(high_freq["gt_hf_ratio"]) if high_freq else None,
                "relative_diff": _none_if_nan(high_freq["relative_diff"]) if high_freq else None,
            },
        }

        db = depth_binned_per_entry[i]
        if db is not None or has_depth:
            rgb_metrics["depth_binned_photometric"] = db

        set_value(
            per_file_metrics, hierarchy, eid,
            {"id": eid, "metrics": {"rgb": rgb_metrics}},
        )

    rgb_results = {
        "image_quality": {
            "psnr": _safe_mean(psnr_values, "psnr"),
            "ssim": _safe_mean(ssim_values, "ssim"),
            "sce": _safe_mean(sce_values, "sce"),
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
            "num_pairs": num_samples,
            "gt_name": gt_name,
            "pred_name": pred_name,
        },
    }

    if depth_binned_results:
        rgb_results["depth_binned_photometric"] = aggregate_depth_binned_errors(
            depth_binned_results
        )
    elif has_depth:
        print("Warning: No valid depth_binned_photometric results.")
        rgb_results["depth_binned_photometric"] = None

    return {
        "rgb": rgb_results,
        "per_file_metrics": per_file_metrics,
    }
