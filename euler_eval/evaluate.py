"""Dataset evaluation orchestrator.

Runs all metrics over depth and RGB datasets loaded via euler_loading.
"""

import copy
import tempfile
from pathlib import Path

import numpy as np
import torch
from typing import Iterator, Optional, Tuple
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from euler_loading import MultiModalDataset

from .data import (
    align_to_prediction,
    classify_spatial_alignment,
    compute_scale_and_shift,
    process_depth,
    to_numpy_depth,
    to_numpy_directions,
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
    compute_clean_fid,
    compute_sce,
    compute_depth_binned_photometric_error,
    aggregate_depth_binned_errors,
    compute_rgb_edge_f1,
    aggregate_rgb_edge_f1,
    compute_high_freq_energy_comparison,
    aggregate_high_freq_metrics,
    # Rays utilities and metrics
    compute_angular_errors,
    compute_rho_a,
    aggregate_rho_a,
    aggregate_angular_errors,
    classify_fov_domain,
    get_threshold_for_domain,
    # Benchmark utilities
    get_benchmark_depth_bins,
    _BENCHMARK_BIN_NAMES,
)

SKY_MASK_ALIGNMENT_MAX_GT_PERCENTILE = 95.0


# ---------------------------------------------------------------------------
# Benchmark depth-range helpers
# ---------------------------------------------------------------------------


def _init_benchmark_bin_store(temp_dir: Path, prefix: str) -> dict:
    """Create streaming stores for a single benchmark depth bin."""
    return {
        "absrel_store": _StreamingValueStore(str(temp_dir / f"{prefix}_absrel.bin")),
        "rmse_store": _StreamingValueStore(str(temp_dir / f"{prefix}_rmse.bin")),
        "silog_store": _StreamingValueStore(str(temp_dir / f"{prefix}_silog.bin")),
        "silog_full_values": [],
        "normal_store": _StreamingValueStore(str(temp_dir / f"{prefix}_normal.bin")),
        "normal_below_11_25": 0,
        "normal_below_22_5": 0,
        "normal_below_30": 0,
    }


def _close_benchmark_stores(stores: dict) -> None:
    """Close all streaming stores in a benchmark store dict."""
    for bin_name in _BENCHMARK_BIN_NAMES:
        s = stores[bin_name]
        s["absrel_store"].close()
        s["rmse_store"].close()
        s["silog_store"].close()
        s["normal_store"].close()


def _safe_mean_values(values: list) -> Optional[float]:
    """Compute mean of finite values, returning None if empty."""
    valid = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(valid)) if valid else None


def _build_benchmark_bin_summary(store: dict) -> dict:
    """Build aggregate metric summary for a single benchmark depth bin."""
    absrel_median, absrel_p90 = store["absrel_store"].quantiles([0.5, 0.9])
    rmse_median, rmse_p90 = store["rmse_store"].quantiles([0.5, 0.9])
    silog_median, silog_p90 = store["silog_store"].quantiles([0.5, 0.9])
    normal_median = store["normal_store"].quantiles([0.5])[0]
    normal_count = store["normal_store"].count

    if normal_count > 0:
        normal_mean = store["normal_store"].mean()
        pct_11_25 = store["normal_below_11_25"] / normal_count * 100.0
        pct_22_5 = store["normal_below_22_5"] / normal_count * 100.0
        pct_30 = store["normal_below_30"] / normal_count * 100.0
    else:
        normal_mean = float("nan")
        pct_11_25 = float("nan")
        pct_22_5 = float("nan")
        pct_30 = float("nan")

    return {
        "depth_metrics": {
            "absrel": {"median": absrel_median, "p90": absrel_p90},
            "rmse": {"median": rmse_median, "p90": rmse_p90},
            "silog": {
                "mean": _safe_mean_values(store["silog_full_values"]),
                "median": silog_median,
                "p90": silog_p90,
            },
        },
        "geometric_metrics": {
            "normal_consistency": {
                "mean_angle": normal_mean,
                "median_angle": normal_median,
                "percent_below_11_25": pct_11_25,
                "percent_below_22_5": pct_22_5,
                "percent_below_30": pct_30,
            },
        },
    }


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
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "precision": _get_dtype_precision_str(arr),
        }
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
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
        print(
            f"  {tag}: shape={stats['shape']}  dtype={stats['dtype']}  "
            f"range=[{stats['min']:.4f}, {stats['max']:.4f}]  "
            f"mean={stats['mean']:.4f}"
        )
    print("=" * 77 + "\n")


class _StreamingValueStore:
    """Append-only float store backed by a temporary binary file."""

    def __init__(self, path: str):
        self.path = path
        self._file = open(path, "wb")
        self.count = 0
        self.total = 0.0

    def append(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return
        arr.tofile(self._file)
        self.count += int(arr.size)
        self.total += float(np.sum(arr, dtype=np.float64))

    def mean(self) -> Optional[float]:
        if self.count == 0:
            return None
        return float(self.total / self.count)

    def quantiles(self, probs: list[float]) -> list[float]:
        if self.count == 0:
            return [float("nan")] * len(probs)

        self._file.flush()
        values = np.memmap(
            self.path,
            dtype=np.float32,
            mode="r+",
            shape=(self.count,),
        )

        requested_indices = []
        positions = []
        for prob in probs:
            pos = prob * (self.count - 1)
            lower = int(np.floor(pos))
            upper = int(np.ceil(pos))
            positions.append((lower, upper, pos - lower))
            requested_indices.extend([lower, upper])

        values.partition(sorted(set(requested_indices)))

        result = []
        for lower, upper, frac in positions:
            lower_value = float(values[lower])
            if upper == lower:
                result.append(lower_value)
            else:
                upper_value = float(values[upper])
                result.append(lower_value + (upper_value - lower_value) * frac)

        del values
        return result

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


def _write_npy_array(folder: str, index: int, array: np.ndarray) -> str:
    path = f"{folder}/{index:06d}.npy"
    np.save(path, np.asarray(array))
    return path


def _write_png_image(folder: str, index: int, image: np.ndarray) -> str:
    path = f"{folder}/{index:06d}.png"
    rgb = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    Image.fromarray(np.rint(rgb * 255.0).astype(np.uint8), mode="RGB").save(
        path,
        format="PNG",
    )
    return path


# ---------------------------------------------------------------------------
# Prefetched sample iteration (worker-parallel I/O)
# ---------------------------------------------------------------------------


def _identity_collate(batch):
    """Pass single-element batches through without tensor collation.

    Samples are heterogeneous dicts of arrays / intrinsics / modality dicts
    and cannot be stacked. The DataLoader is used purely for I/O prefetching,
    so we unwrap the single-element batch back into the original sample dict.
    """
    return batch[0]


def _prefetched_iter(
    dataset,
    num_workers: int,
    skip: int = 0,
) -> Iterator[Tuple[int, dict]]:
    """Yield ``(index, sample)`` pairs with optional worker-based prefetching.

    When ``num_workers > 0`` and the remaining workload outnumbers the
    worker count, a ``torch.utils.data.DataLoader`` drives worker processes
    that call ``dataset[i]`` in parallel, overlapping zip/NFS reads with
    the main process' metric computation. ``num_workers == 0`` (or a
    dataset small enough that worker spawn cost dominates) falls back to
    sequential indexing so tests and tiny datasets avoid the overhead.
    """
    total = len(dataset)
    remaining = total - skip

    if num_workers <= 0 or remaining <= num_workers:
        for i in range(skip, total):
            yield i, dataset[i]
        return

    indices = list(range(skip, total))
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=1,
        num_workers=num_workers,
        prefetch_factor=2,
        collate_fn=_identity_collate,
        persistent_workers=False,
        pin_memory=False,
    )
    try:
        for offset, sample in enumerate(loader):
            yield skip + offset, sample
    finally:
        del loader


# ---------------------------------------------------------------------------
# Batched LPIPS accumulator
# ---------------------------------------------------------------------------


class _LPIPSBatcher:
    """Accumulate (pred, gt) pairs and compute LPIPS in GPU-friendly batches.

    LPIPS is the dominant GPU cost in the per-pair loop. Calling the network
    once per sample leaves the GPU underutilized, while stacking ``batch_size``
    pairs into a single forward pass amortizes launch overhead across more
    work. Each enqueued pair carries a callback that patches the computed
    value into the caller's accumulators (stores + per-file dicts) so the
    calling code stays index-agnostic.

    Memory is bounded: at most ``batch_size`` pairs are buffered; buffers are
    flushed automatically when the batch fills and explicitly via
    :meth:`finalize` at end of loop.
    """

    def __init__(self, metric, batch_size: int = 16):
        self.metric = metric
        self.batch_size = max(1, int(batch_size))
        self._pending: list = []

    def enqueue(self, pred: np.ndarray, gt: np.ndarray, callback) -> None:
        self._pending.append((pred, gt, callback))
        if len(self._pending) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        preds = [p[0] for p in self._pending]
        gts = [p[1] for p in self._pending]
        try:
            values = self.metric.compute_batch(
                preds, gts, batch_size=len(self._pending)
            )
        except Exception:
            values = []
            for pred, gt, _ in self._pending:
                try:
                    values.append(float(self.metric.compute(pred, gt)))
                except Exception:
                    values.append(float("nan"))
        for (_, _, cb), v in zip(self._pending, values):
            cb(v)
        self._pending.clear()

    def finalize(self) -> None:
        self._flush()


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
    is_radial: bool,
    gt_name: str = "GT",
    pred_name: str = "Pred",
    device: str = "cuda",
    batch_size: int = 16,
    num_workers: int = 4,
    verbose: bool = False,
    sanity_checker: Optional[SanityChecker] = None,
    sky_mask_enabled: bool = False,
    alignment_mode: str = "auto_affine",
    benchmark_depth_range: Optional[tuple[float, float]] = None,
) -> dict:
    """Evaluate all depth metrics from a MultiModalDataset.

    The dataset must yield samples with ``"gt"`` and ``"pred"`` keys
    containing depth data (tensors or arrays).

    Args:
        dataset: The MultiModalDataset to iterate.
        is_radial: Whether depth is already radial/euclidean.
        gt_name: GT dataset display name.
        pred_name: Prediction dataset display name.
        device: Computation device.
        batch_size: Batch size for batched metrics.
        num_workers: Data loading workers.
        verbose: Enable verbose output.
        sanity_checker: Optional SanityChecker.
        sky_mask_enabled: If True, use segmentation for sky masking.
        alignment_mode: One of ``none``, ``auto_affine``, ``affine``.
            ``auto_affine`` aligns only if predictions look normalized.
        benchmark_depth_range: Optional ``(min_meters, max_meters)`` tuple.
            When set, also computes depth metrics for pixels within this
            range, subdivided into log-scaled near/mid/far bins.

    Returns:
        Dictionary containing depth aggregate/per-file metrics with:
        ``depth_raw``, ``depth_aligned``, backward-compatible ``depth``,
        and optionally ``depth_benchmark``.
    """
    valid_alignment_modes = {"none", "auto_affine", "affine"}
    if alignment_mode not in valid_alignment_modes:
        raise ValueError(
            f"Invalid alignment_mode '{alignment_mode}'. "
            f"Expected one of {sorted(valid_alignment_modes)}."
        )

    num_samples = len(dataset)
    if num_samples == 0:
        raise ValueError("Dataset has no matched samples")

    print(f"Initializing depth metrics (device: {device})...")
    lpips_metric = LPIPSMetric(device=device)
    fid_kid_metric = FIDKIDMetric(device=device)

    def _init_metric_store(temp_dir: Path, name: str) -> dict:
        return {
            "psnr_values": [],
            "ssim_values": [],
            "lpips_values": [],
            "silog_full_values": [],
            "edge_f1_results": [],
            "pred_depth_paths": [],
            "absrel_store": _StreamingValueStore(str(temp_dir / f"{name}_absrel.bin")),
            "rmse_store": _StreamingValueStore(str(temp_dir / f"{name}_rmse.bin")),
            "silog_store": _StreamingValueStore(str(temp_dir / f"{name}_silog.bin")),
            "normal_store": _StreamingValueStore(str(temp_dir / f"{name}_normal.bin")),
            "normal_below_11_25": 0,
            "normal_below_22_5": 0,
            "normal_below_30": 0,
        }

    def _compute_branch_metrics(
        depth_gt: np.ndarray, depth_pred: np.ndarray, valid_mask: Optional[np.ndarray]
    ) -> dict:
        psnr_val, psnr_meta = compute_psnr(
            depth_pred, depth_gt, valid_mask=valid_mask, return_metadata=True
        )
        ssim_val, ssim_meta = compute_ssim(depth_pred, depth_gt, return_metadata=True)
        absrel_arr, absrel_meta = compute_absrel(
            depth_pred, depth_gt, valid_mask=valid_mask, return_metadata=True
        )
        rmse_arr = compute_rmse_per_pixel(depth_pred, depth_gt, valid_mask=valid_mask)
        silog_arr = compute_silog_per_pixel(depth_pred, depth_gt, valid_mask=valid_mask)
        silog_val = compute_scale_invariant_log_error(
            depth_pred, depth_gt, valid_mask=valid_mask
        )
        normal_angles, normal_meta = compute_normal_angles(
            depth_pred, depth_gt, valid_mask=valid_mask, return_metadata=True
        )
        edge_f1 = compute_depth_edge_f1(depth_pred, depth_gt, valid_mask=valid_mask)

        return {
            "psnr_val": psnr_val,
            "psnr_meta": psnr_meta,
            "ssim_val": ssim_val,
            "ssim_meta": ssim_meta,
            # LPIPS is computed later in batches via _LPIPSBatcher for GPU
            # efficiency; a NaN placeholder keeps _append_metrics and
            # _build_per_file_depth_value index-aligned until the batcher
            # patches the real values.
            "lpips_val": float("nan"),
            "absrel_arr": absrel_arr,
            "absrel_meta": absrel_meta,
            "rmse_arr": rmse_arr,
            "silog_arr": silog_arr,
            "silog_full": silog_val,
            "normal_angles": normal_angles,
            "normal_meta": normal_meta,
            "edge_f1": edge_f1,
        }

    def _append_metrics(store: dict, metrics: dict, pred_depth_path: str) -> None:
        store["psnr_values"].append(metrics["psnr_val"])
        store["ssim_values"].append(metrics["ssim_val"])
        store["lpips_values"].append(metrics["lpips_val"])
        store["silog_full_values"].append(metrics["silog_full"])
        store["edge_f1_results"].append(metrics["edge_f1"])
        store["pred_depth_paths"].append(pred_depth_path)
        store["absrel_store"].append(metrics["absrel_arr"])
        store["rmse_store"].append(np.sqrt(metrics["rmse_arr"]))
        store["silog_store"].append(metrics["silog_arr"])
        store["normal_store"].append(metrics["normal_angles"])
        if len(metrics["normal_angles"]) > 0:
            store["normal_below_11_25"] += int(np.sum(metrics["normal_angles"] < 11.25))
            store["normal_below_22_5"] += int(np.sum(metrics["normal_angles"] < 22.5))
            store["normal_below_30"] += int(np.sum(metrics["normal_angles"] < 30.0))

    def _build_valid_mask(
        depth_gt: np.ndarray, depth_pred: np.ndarray, sky_valid: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        if sky_valid is None:
            return None
        return (
            (depth_gt > 0)
            & (depth_pred > 0)
            & np.isfinite(depth_gt)
            & np.isfinite(depth_pred)
            & sky_valid
        )

    def _safe_mean(values: list[float]) -> Optional[float]:
        valid = [float(v) for v in values if np.isfinite(v)]
        if not valid:
            return None
        return float(np.mean(valid))

    def _build_depth_summary(store: dict, gt_depth_paths: list[str]) -> dict:
        absrel_median, absrel_p90 = store["absrel_store"].quantiles([0.5, 0.9])
        rmse_median, rmse_p90 = store["rmse_store"].quantiles([0.5, 0.9])
        silog_median, silog_p90 = store["silog_store"].quantiles([0.5, 0.9])
        normal_median = store["normal_store"].quantiles([0.5])[0]
        normal_count = store["normal_store"].count
        edge_f1_agg = aggregate_edge_f1(store["edge_f1_results"])

        fid_value = fid_kid_metric.compute_fid(
            gt_depth_paths, store["pred_depth_paths"], batch_size, num_workers
        )
        kid_mean, kid_std = fid_kid_metric.compute_kid(
            gt_depth_paths, store["pred_depth_paths"], batch_size, num_workers
        )

        if normal_count > 0:
            normal_mean = store["normal_store"].mean()
            pct_11_25 = store["normal_below_11_25"] / normal_count * 100.0
            pct_22_5 = store["normal_below_22_5"] / normal_count * 100.0
            pct_30 = store["normal_below_30"] / normal_count * 100.0
        else:
            normal_mean = float("nan")
            pct_11_25 = float("nan")
            pct_22_5 = float("nan")
            pct_30 = float("nan")

        return {
            "image_quality": {
                "psnr": _safe_mean(store["psnr_values"]),
                "ssim": _safe_mean(store["ssim_values"]),
                "lpips": _safe_mean(store["lpips_values"]),
                "fid": fid_value,
                "kid_mean": kid_mean,
                "kid_std": kid_std,
            },
            "depth_metrics": {
                "absrel": {"median": absrel_median, "p90": absrel_p90},
                "rmse": {"median": rmse_median, "p90": rmse_p90},
                "silog": {
                    "mean": _safe_mean(store["silog_full_values"]),
                    "median": silog_median,
                    "p90": silog_p90,
                },
            },
            "geometric_metrics": {
                "normal_consistency": {
                    "mean_angle": normal_mean,
                    "median_angle": normal_median,
                    "percent_below_11_25": pct_11_25,
                    "percent_below_22_5": pct_22_5,
                    "percent_below_30": pct_30,
                },
                "depth_edge_f1": {
                    "precision": edge_f1_agg["precision"],
                    "recall": edge_f1_agg["recall"],
                    "f1": edge_f1_agg["f1"],
                },
            },
        }

    def _build_per_file_depth_value(metrics: dict) -> dict:
        absrel_arr = metrics["absrel_arr"]
        rmse_arr = metrics["rmse_arr"]
        normal_angles = metrics["normal_angles"]
        edge_f1 = metrics["edge_f1"]

        return {
            "image_quality": {
                "psnr": float(metrics["psnr_val"])
                if np.isfinite(metrics["psnr_val"])
                else None,
                "ssim": float(metrics["ssim_val"])
                if np.isfinite(metrics["ssim_val"])
                else None,
                "lpips": float(metrics["lpips_val"])
                if np.isfinite(metrics["lpips_val"])
                else None,
            },
            "depth_metrics": {
                "absrel": float(np.mean(absrel_arr)) if len(absrel_arr) > 0 else None,
                "rmse": float(np.sqrt(np.mean(rmse_arr))) if len(rmse_arr) > 0 else None,
                "silog": float(metrics["silog_full"])
                if np.isfinite(metrics["silog_full"])
                else None,
            },
            "geometric_metrics": {
                "normal_consistency": {
                    "mean_angle": float(np.mean(normal_angles))
                    if len(normal_angles) > 0
                    else None,
                },
                "depth_edge_f1": {
                    "precision": float(edge_f1["precision"]),
                    "recall": float(edge_f1["recall"]),
                    "f1": float(edge_f1["f1"]),
                },
            },
        }

    logged_stats = False
    logged_alignment = False
    normalized_predictions = False
    alignment_applied = False
    gt_native_dims: Optional[tuple[int, int]] = None
    pred_native_dims: Optional[tuple[int, int]] = None
    spatial_method = "none"

    if alignment_mode == "none":
        print("Depth alignment mode: none (raw predictions only)")
    elif alignment_mode == "auto_affine":
        print("Depth alignment mode: auto_affine (normalized-depth detection)")
    else:
        print("Depth alignment mode: affine (always apply scale-and-shift)")

    with tempfile.TemporaryDirectory(prefix="euler_eval_depth_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        gt_depth_dir = temp_dir / "gt_depth"
        raw_pred_dir = temp_dir / "raw_pred"
        aligned_pred_dir = temp_dir / "aligned_pred"
        gt_depth_dir.mkdir()
        raw_pred_dir.mkdir()
        aligned_pred_dir.mkdir()

        stores = {
            "raw": _init_metric_store(temp_dir, "raw"),
            "aligned": _init_metric_store(temp_dir, "aligned"),
        }
        gt_depth_paths: list[str] = []
        per_file_metrics = {}

        lpips_batcher = _LPIPSBatcher(lpips_metric, batch_size=batch_size)

        benchmark_stores = None
        benchmark_boundaries = None
        if benchmark_depth_range is not None:
            bm_min, bm_max = benchmark_depth_range
            print(
                f"Benchmark depth range: [{bm_min}, {bm_max}] meters "
                f"(log-scaled near/mid/far bins)"
            )
            benchmark_stores = {
                bn: _init_benchmark_bin_store(temp_dir, f"bench_{bn}")
                for bn in _BENCHMARK_BIN_NAMES
            }

        try:
            print("Computing per-image depth metrics...")
            depth_iter = _prefetched_iter(dataset, num_workers)
            for i, sample in tqdm(
                depth_iter,
                total=num_samples,
                desc="Processing depth pairs",
            ):
                hierarchy, entry_id = _extract_hierarchy(sample)

                depth_gt = to_numpy_depth(sample["gt"])
                depth_pred = to_numpy_depth(sample["pred"])

                if i == 0:
                    gt_native_dims = depth_gt.shape[:2]
                    pred_native_dims = depth_pred.shape[:2]
                    spatial_method = classify_spatial_alignment(
                        *gt_native_dims, *pred_native_dims
                    )

                if depth_gt.shape[:2] != depth_pred.shape[:2]:
                    if not logged_alignment:
                        print(
                            f"  Aligning GT {depth_gt.shape[:2]} -> "
                            f"pred {depth_pred.shape[:2]}"
                        )
                        logged_alignment = True
                    depth_gt = align_to_prediction(depth_gt, depth_pred)

                intrinsics_K = _get_intrinsics_K(sample)

                sky_valid = None
                if sky_mask_enabled:
                    sky_valid = _get_sky_mask(sample)
                    if (
                        sky_valid is not None
                        and sky_valid.shape[:2] != depth_pred.shape[:2]
                    ):
                        sky_valid = align_to_prediction(sky_valid, depth_pred)

                if alignment_mode == "auto_affine" and i == 0:
                    pred_min = float(np.nanmin(depth_pred))
                    pred_max = float(np.nanmax(depth_pred))
                    if pred_max <= 1.0 + 1e-3 and pred_min >= -1.0 - 1e-3:
                        normalized_predictions = True
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

                depth_gt = process_depth(depth_gt, 1.0, is_radial, intrinsics_K)
                depth_pred_raw = process_depth(depth_pred, 1.0, is_radial, intrinsics_K)

                if alignment_mode == "none":
                    depth_pred_aligned = depth_pred_raw
                else:
                    do_alignment = alignment_mode == "affine" or normalized_predictions
                    if do_alignment:
                        if sky_mask_enabled and i == 0:
                            print(
                                "  Scale-and-shift: fitting on GT depth <= P95 "
                                "when sky masking is enabled"
                            )
                        fit_source = depth_pred if normalized_predictions else depth_pred_raw
                        fit_mask = (
                            (depth_gt > 0)
                            & np.isfinite(depth_gt)
                            & np.isfinite(fit_source)
                        )
                        if sky_valid is not None:
                            fit_mask = fit_mask & sky_valid
                        depth_pred_aligned, s, t = compute_scale_and_shift(
                            fit_source,
                            depth_gt,
                            fit_mask,
                            max_gt_percentile=(
                                SKY_MASK_ALIGNMENT_MAX_GT_PERCENTILE
                                if sky_mask_enabled
                                else None
                            ),
                        )
                        alignment_applied = True
                        if verbose and not logged_stats:
                            print(f"  Fitted scale={s:.4f}, shift={t:.4f}")
                    else:
                        depth_pred_aligned = depth_pred_raw

                if verbose and not logged_stats:
                    _log_sample_stats(depth_gt, depth_pred_aligned, "DEPTH")
                    logged_stats = True

                gt_depth_path = _write_npy_array(str(gt_depth_dir), i, depth_gt)
                raw_pred_path = _write_npy_array(str(raw_pred_dir), i, depth_pred_raw)
                gt_depth_paths.append(gt_depth_path)

                raw_valid_mask = _build_valid_mask(depth_gt, depth_pred_raw, sky_valid)
                raw_metrics = _compute_branch_metrics(depth_gt, depth_pred_raw, raw_valid_mask)
                _append_metrics(stores["raw"], raw_metrics, raw_pred_path)

                if depth_pred_aligned is depth_pred_raw:
                    aligned_metrics = raw_metrics
                else:
                    aligned_valid_mask = _build_valid_mask(
                        depth_gt, depth_pred_aligned, sky_valid
                    )
                    aligned_metrics = _compute_branch_metrics(
                        depth_gt, depth_pred_aligned, aligned_valid_mask
                    )
                    aligned_pred_path = _write_npy_array(
                        str(aligned_pred_dir), i, depth_pred_aligned
                    )
                    _append_metrics(stores["aligned"], aligned_metrics, aligned_pred_path)

                raw_value = _build_per_file_depth_value(raw_metrics)
                aligned_value = (
                    raw_value
                    if aligned_metrics is raw_metrics
                    else _build_per_file_depth_value(aligned_metrics)
                )
                set_value(
                    per_file_metrics,
                    hierarchy,
                    entry_id,
                    {
                        "id": entry_id,
                        "metrics": {
                            "depth": aligned_value,
                            "depth_raw": raw_value,
                            "depth_aligned": aligned_value,
                        },
                    },
                )

                # -- Enqueue batched LPIPS; callbacks patch the placeholders --
                raw_lpips_slot = len(stores["raw"]["lpips_values"]) - 1

                def _raw_lpips_cb(
                    v,
                    _store=stores["raw"],
                    _slot=raw_lpips_slot,
                    _per_file=raw_value,
                ):
                    val = float(v) if np.isfinite(v) else float("nan")
                    _store["lpips_values"][_slot] = val
                    _per_file["image_quality"]["lpips"] = (
                        val if np.isfinite(val) else None
                    )

                lpips_batcher.enqueue(depth_pred_raw, depth_gt, _raw_lpips_cb)

                if aligned_metrics is not raw_metrics:
                    aligned_lpips_slot = len(stores["aligned"]["lpips_values"]) - 1

                    def _aligned_lpips_cb(
                        v,
                        _store=stores["aligned"],
                        _slot=aligned_lpips_slot,
                        _per_file=aligned_value,
                    ):
                        val = float(v) if np.isfinite(v) else float("nan")
                        _store["lpips_values"][_slot] = val
                        _per_file["image_quality"]["lpips"] = (
                            val if np.isfinite(val) else None
                        )

                    lpips_batcher.enqueue(
                        depth_pred_aligned, depth_gt, _aligned_lpips_cb
                    )

                # -- Benchmark depth-range metrics (aligned only) --
                if benchmark_stores is not None:
                    bm_bins = get_benchmark_depth_bins(
                        depth_gt, benchmark_depth_range[0], benchmark_depth_range[1]
                    )
                    if benchmark_boundaries is None:
                        benchmark_boundaries = bm_bins["boundaries"]

                    for bn in _BENCHMARK_BIN_NAMES:
                        bin_mask = bm_bins[bn].copy()
                        bin_mask &= (depth_pred_aligned > 0) & np.isfinite(
                            depth_pred_aligned
                        )
                        if sky_valid is not None:
                            bin_mask &= sky_valid
                        if not bin_mask.any():
                            continue

                        bm_store = benchmark_stores[bn]

                        bm_absrel = compute_absrel(
                            depth_pred_aligned, depth_gt, valid_mask=bin_mask
                        )
                        bm_rmse = compute_rmse_per_pixel(
                            depth_pred_aligned, depth_gt, valid_mask=bin_mask
                        )
                        bm_silog_arr = compute_silog_per_pixel(
                            depth_pred_aligned, depth_gt, valid_mask=bin_mask
                        )
                        bm_silog_val = compute_scale_invariant_log_error(
                            depth_pred_aligned, depth_gt, valid_mask=bin_mask
                        )
                        bm_normals = compute_normal_angles(
                            depth_pred_aligned, depth_gt, valid_mask=bin_mask
                        )

                        bm_store["absrel_store"].append(bm_absrel)
                        bm_store["rmse_store"].append(np.sqrt(bm_rmse))
                        bm_store["silog_store"].append(bm_silog_arr)
                        bm_store["silog_full_values"].append(bm_silog_val)
                        bm_store["normal_store"].append(bm_normals)
                        if len(bm_normals) > 0:
                            bm_store["normal_below_11_25"] += int(
                                np.sum(bm_normals < 11.25)
                            )
                            bm_store["normal_below_22_5"] += int(
                                np.sum(bm_normals < 22.5)
                            )
                            bm_store["normal_below_30"] += int(
                                np.sum(bm_normals < 30.0)
                            )

                if sanity_checker is not None:
                    sanity_checker.validate_depth_input(
                        depth_gt, depth_pred_aligned, entry_id
                    )
                    pm = aligned_metrics["psnr_meta"]
                    if pm["max_val_used"] is not None:
                        sanity_checker.validate_depth_psnr(
                            aligned_metrics["psnr_val"], pm["max_val_used"], entry_id
                        )
                    sm = aligned_metrics["ssim_meta"]
                    if sm["depth_range"] is not None:
                        sanity_checker.validate_depth_ssim(
                            aligned_metrics["ssim_val"], sm["depth_range"], entry_id
                        )
                    am = aligned_metrics["absrel_meta"]
                    if am["median"] is not None:
                        sanity_checker.validate_depth_absrel(
                            am["median"], am["p90"], entry_id
                        )
                    if sm["depth_range"] is not None and len(aligned_metrics["rmse_arr"]) > 0:
                        rmse_val = float(np.sqrt(np.mean(aligned_metrics["rmse_arr"])))
                        sanity_checker.validate_depth_rmse(
                            rmse_val, sm["depth_range"], entry_id
                        )
                    sv = aligned_metrics["silog_full"]
                    if np.isfinite(sv):
                        sanity_checker.validate_depth_silog(sv, entry_id)
                    nm = aligned_metrics["normal_meta"]
                    if nm["mean_angle"] is not None:
                        sanity_checker.validate_normal_consistency(
                            nm["mean_angle"], nm["valid_pixels_after_erosion"], entry_id
                        )
                    ef = aligned_metrics["edge_f1"]
                    sanity_checker.validate_depth_edge_f1(
                        ef["pred_edge_pixels"],
                        ef["gt_edge_pixels"],
                        ef["total_pixels"],
                        ef["f1"],
                        entry_id,
                    )

            print("Computing batched LPIPS (tail flush)...")
            lpips_batcher.finalize()

            print("Computing FID/KID (this may take a while)...")
            print("Aggregating depth results...")

            depth_raw = _build_depth_summary(stores["raw"], gt_depth_paths)
            if alignment_applied:
                depth_aligned = _build_depth_summary(stores["aligned"], gt_depth_paths)
            else:
                depth_aligned = copy.deepcopy(depth_raw)

            # -- Benchmark aggregation --
            depth_benchmark = None
            if benchmark_stores is not None:
                print("Aggregating benchmark depth results...")
                depth_benchmark = {
                    "boundaries": benchmark_boundaries,
                }
                for bn in _BENCHMARK_BIN_NAMES:
                    depth_benchmark[bn] = _build_benchmark_bin_summary(
                        benchmark_stores[bn]
                    )

            result = {
                "depth_raw": depth_raw,
                "depth_aligned": depth_aligned,
                "depth": depth_aligned,
                "depth_benchmark": depth_benchmark,
                "per_file_metrics": per_file_metrics,
                "dataset_info": {
                    "num_pairs": num_samples,
                    "gt_name": gt_name,
                    "pred_name": pred_name,
                },
                "alignment": {
                    "mode": alignment_mode,
                    "applied": alignment_applied,
                },
                "spatial_info": {
                    "gt_dimensions": {"height": gt_native_dims[0], "width": gt_native_dims[1]}
                    if gt_native_dims
                    else None,
                    "pred_dimensions": {"height": pred_native_dims[0], "width": pred_native_dims[1]}
                    if pred_native_dims
                    else None,
                    "method": spatial_method,
                    "evaluated_dimensions": {
                        "height": pred_native_dims[0],
                        "width": pred_native_dims[1],
                    }
                    if pred_native_dims
                    else None,
                },
            }
            return result
        finally:
            for branch_store in stores.values():
                branch_store["absrel_store"].close()
                branch_store["rmse_store"].close()
                branch_store["silog_store"].close()
                branch_store["normal_store"].close()
            if benchmark_stores is not None:
                _close_benchmark_stores(benchmark_stores)


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
    fid_backend: str = "builtin",
    benchmark_depth_range: Optional[tuple[float, float]] = None,
) -> dict:
    """Evaluate all RGB metrics from a MultiModalDataset.

    The dataset must yield samples with ``"gt"`` and ``"pred"`` keys
    containing RGB data. Optionally ``"gt_depth"`` for depth-binned
    metrics, ``"segmentation"`` for sky masking, and ``"calibration"``.

    Args:
        dataset: The MultiModalDataset to iterate.
        depth_meta: Depth metadata dict containing ``radial_depth`` for
                    depth-binned metrics. None to skip.
        gt_name: GT dataset display name.
        pred_name: Prediction dataset display name.
        device: Computation device.
        batch_size: Batch size for batched metrics.
        num_workers: Data loading workers.
        verbose: Enable verbose output.
        sanity_checker: Optional SanityChecker.
        sky_mask_enabled: If True, use segmentation for sky masking.
        fid_backend: RGB FID backend. One of ``"builtin"`` or ``"clean-fid"``.
        benchmark_depth_range: Optional ``(min_meters, max_meters)`` tuple.
            When set and depth is available, computes RGB photometric error
            for pixels within this range, subdivided into log-scaled bins.

    Returns:
        Dictionary containing aggregate and per-file metrics, and
        optionally ``rgb_benchmark``.
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

    valid_fid_backends = {"builtin", "clean-fid"}
    if fid_backend not in valid_fid_backends:
        raise ValueError(
            f"Invalid fid_backend '{fid_backend}'. "
            f"Expected one of {sorted(valid_fid_backends)}."
        )

    has_depth = "gt_depth" in dataset.modality_paths() and depth_meta is not None

    print(f"Initializing RGB metrics (device: {device})...")
    try:
        lpips_metric = RGBLPIPSMetric(device=device)
    except Exception as exc:
        print(f"Warning: Failed to initialize LPIPS metric: {exc}")
        lpips_metric = None
    fid_metric = FIDKIDMetric(device=device) if fid_backend == "builtin" else None

    psnr_values = []
    ssim_values = []
    lpips_values = []
    sce_values = []
    edge_f1_results = []
    high_freq_results = []
    depth_binned_results = []
    benchmark_rgb_results = []
    benchmark_rgb_boundaries = None

    logged_stats = False
    logged_alignment = False
    gt_native_dims: Optional[tuple[int, int]] = None
    pred_native_dims: Optional[tuple[int, int]] = None
    spatial_method = "none"

    with tempfile.TemporaryDirectory(prefix="euler_eval_rgb_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        tail_error_store = _StreamingValueStore(str(temp_dir / "tail_errors.bin"))
        fid_gt_dir = temp_dir / "fid_gt"
        fid_pred_dir = temp_dir / "fid_pred"
        fid_gt_dir.mkdir()
        fid_pred_dir.mkdir()
        fid_gt_inputs: list[str] = []
        fid_pred_inputs: list[str] = []
        per_file_metrics = {}

        rgb_lpips_batcher = (
            _LPIPSBatcher(lpips_metric, batch_size=batch_size)
            if lpips_metric is not None
            else None
        )

        try:
            print("Computing per-image RGB metrics...")
            rgb_iter = _prefetched_iter(dataset, num_workers)
            for i, sample in tqdm(
                rgb_iter,
                total=num_samples,
                desc="Processing RGB pairs",
            ):
                hierarchy, entry_id = _extract_hierarchy(sample)

                img_gt = to_numpy_rgb(sample["gt"])
                img_pred = to_numpy_rgb(sample["pred"])

                if i == 0:
                    gt_native_dims = img_gt.shape[:2]
                    pred_native_dims = img_pred.shape[:2]
                    spatial_method = classify_spatial_alignment(
                        *gt_native_dims, *pred_native_dims
                    )

                if img_gt.shape[:2] != img_pred.shape[:2]:
                    if not logged_alignment:
                        print(
                            f"  Aligning GT {img_gt.shape[:2]} -> pred {img_pred.shape[:2]}"
                        )
                        logged_alignment = True
                    img_gt = align_to_prediction(img_gt, img_pred)

                if not logged_stats:
                    _log_sample_stats(img_gt, img_pred, "RGB")
                    logged_stats = True

                sky_valid = None
                if sky_mask_enabled:
                    sky_valid = _get_sky_mask(sample)
                    if sky_valid is not None and sky_valid.shape[:2] != img_pred.shape[:2]:
                        sky_valid = align_to_prediction(sky_valid, img_pred)

                if sanity_checker is not None:
                    sanity_checker.validate_rgb_input(img_gt, img_pred, entry_id)

                gt_masked = img_gt
                pred_masked = img_pred
                if sky_valid is not None:
                    mask_3c = np.stack([sky_valid] * 3, axis=-1)
                    gt_masked = img_gt * mask_3c
                    pred_masked = img_pred * mask_3c

                if fid_backend == "builtin":
                    fid_gt_inputs.append(_write_npy_array(str(fid_gt_dir), i, gt_masked))
                    fid_pred_inputs.append(
                        _write_npy_array(str(fid_pred_dir), i, pred_masked)
                    )
                else:
                    _write_png_image(str(fid_gt_dir), i, gt_masked)
                    _write_png_image(str(fid_pred_dir), i, pred_masked)

                psnr_val = _safe_compute(
                    "psnr", entry_id, compute_rgb_psnr, pred_masked, gt_masked
                )
                ssim_val = _safe_compute(
                    "ssim", entry_id, compute_rgb_ssim, pred_masked, gt_masked
                )
                sce_val = _safe_compute(
                    "sce", entry_id, compute_sce, pred_masked, gt_masked
                )
                # LPIPS is computed later in batches via _LPIPSBatcher for GPU
                # efficiency. The placeholder is patched in the enqueue callback.
                lpips_val = float("nan") if lpips_metric is not None else None

                edge_f1 = _safe_compute(
                    "edge_f1", entry_id, compute_rgb_edge_f1, pred_masked, gt_masked
                )
                tail_arr = _safe_compute(
                    "tail_errors",
                    entry_id,
                    lambda pred, gt: np.abs(pred - gt).mean(axis=-1),
                    pred_masked,
                    gt_masked,
                )
                if tail_arr is not None:
                    tail_error_store.append(tail_arr)
                high_freq = _safe_compute(
                    "high_frequency",
                    entry_id,
                    compute_high_freq_energy_comparison,
                    pred_masked,
                    gt_masked,
                )

                depth_binned_entry = None
                if has_depth:
                    try:
                        gt_depth_raw = to_numpy_depth(sample["gt_depth"])
                        intrinsics_K = _get_intrinsics_K(sample)
                        gt_depth = process_depth(
                            gt_depth_raw,
                            1.0,
                            depth_meta["radial_depth"],
                            intrinsics_K,
                        )
                        if gt_depth.shape[:2] != img_pred.shape[:2]:
                            gt_depth = align_to_prediction(gt_depth, img_pred)
                        depth_binned_entry = compute_depth_binned_photometric_error(
                            pred_masked, gt_masked, gt_depth
                        )
                        depth_binned_results.append(depth_binned_entry)

                        # Benchmark depth-binned RGB metrics
                        if benchmark_depth_range is not None:
                            bm_bins = get_benchmark_depth_bins(
                                gt_depth,
                                benchmark_depth_range[0],
                                benchmark_depth_range[1],
                            )
                            if benchmark_rgb_boundaries is None:
                                benchmark_rgb_boundaries = bm_bins["boundaries"]
                            abs_error = np.abs(
                                pred_masked.astype(np.float64)
                                - gt_masked.astype(np.float64)
                            ).mean(axis=-1)
                            sq_error = (
                                (
                                    pred_masked.astype(np.float64)
                                    - gt_masked.astype(np.float64)
                                )
                                ** 2
                            ).mean(axis=-1)
                            bm_entry = {"mae": {}, "mse": {}}
                            for bn in _BENCHMARK_BIN_NAMES:
                                bm_mask = bm_bins[bn]
                                if sky_valid is not None:
                                    bm_mask = bm_mask & sky_valid
                                if bm_mask.any():
                                    bm_entry["mae"][bn] = float(
                                        np.mean(abs_error[bm_mask])
                                    )
                                    bm_entry["mse"][bn] = float(
                                        np.mean(sq_error[bm_mask])
                                    )
                                else:
                                    bm_entry["mae"][bn] = 0.0
                                    bm_entry["mse"][bn] = 0.0
                            benchmark_rgb_results.append(bm_entry)
                    except Exception as exc:
                        _warn_metric_failure("depth_binned_photometric", entry_id, exc)

                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)
                sce_values.append(sce_val)
                lpips_values.append(lpips_val)
                edge_f1_results.append(edge_f1)
                high_freq_results.append(high_freq)

                tail_p95 = (
                    float(np.percentile(tail_arr, 95))
                    if tail_arr is not None and len(tail_arr) > 0
                    else None
                )
                tail_p99 = (
                    float(np.percentile(tail_arr, 99))
                    if tail_arr is not None and len(tail_arr) > 0
                    else None
                )

                rgb_metrics = {
                    "image_quality": {
                        "psnr": _none_if_nan(psnr_val),
                        "ssim": _none_if_nan(ssim_val),
                        "sce": _none_if_nan(sce_val),
                        "lpips": _none_if_nan(lpips_val),
                    },
                    "edge_f1": {
                        "precision": _none_if_nan(edge_f1["precision"]) if edge_f1 else None,
                        "recall": _none_if_nan(edge_f1["recall"]) if edge_f1 else None,
                        "f1": _none_if_nan(edge_f1["f1"]) if edge_f1 else None,
                    },
                    "tail_errors": {
                        "p95": tail_p95,
                        "p99": tail_p99,
                    },
                    "high_frequency": {
                        "pred_hf_ratio": _none_if_nan(high_freq["pred_hf_ratio"])
                        if high_freq
                        else None,
                        "gt_hf_ratio": _none_if_nan(high_freq["gt_hf_ratio"])
                        if high_freq
                        else None,
                        "relative_diff": _none_if_nan(high_freq["relative_diff"])
                        if high_freq
                        else None,
                    },
                }
                if depth_binned_entry is not None:
                    rgb_metrics["depth_binned_photometric"] = depth_binned_entry

                set_value(
                    per_file_metrics,
                    hierarchy,
                    entry_id,
                    {"id": entry_id, "metrics": {"rgb": rgb_metrics}},
                )

                # -- Enqueue batched LPIPS; callback patches the placeholder --
                if rgb_lpips_batcher is not None:
                    lpips_slot = len(lpips_values) - 1

                    def _rgb_lpips_cb(
                        v,
                        _slot=lpips_slot,
                        _per_file=rgb_metrics,
                        _entry_id=entry_id,
                    ):
                        val = float(v) if np.isfinite(v) else float("nan")
                        lpips_values[_slot] = val
                        _per_file["image_quality"]["lpips"] = (
                            val if np.isfinite(val) else None
                        )
                        if sanity_checker is not None and np.isfinite(val):
                            sanity_checker.validate_rgb_lpips(val, _entry_id)

                    rgb_lpips_batcher.enqueue(pred_masked, gt_masked, _rgb_lpips_cb)

                if sanity_checker is not None:
                    if psnr_val is not None and np.isfinite(psnr_val):
                        sanity_checker.validate_rgb_psnr(psnr_val, entry_id)
                    if ssim_val is not None and np.isfinite(ssim_val):
                        sanity_checker.validate_rgb_ssim(ssim_val, entry_id)
                    if tail_p99 is not None and np.isfinite(tail_p99):
                        sanity_checker.validate_tail_errors(tail_p99, entry_id)
                    if high_freq is not None and np.isfinite(
                        high_freq.get("relative_diff", float("nan"))
                    ):
                        sanity_checker.validate_high_freq_energy(
                            high_freq["relative_diff"], entry_id
                        )
                    if depth_binned_entry is not None:
                        sanity_checker.validate_depth_binned(depth_binned_entry, entry_id)

            if rgb_lpips_batcher is not None:
                print("Computing batched RGB LPIPS (tail flush)...")
                rgb_lpips_batcher.finalize()

            print(f"Computing RGB FID using backend: {fid_backend}...")
            if fid_backend == "builtin":
                rgb_fid = fid_metric.compute_rgb_fid(
                    fid_gt_inputs, fid_pred_inputs, batch_size, num_workers
                )
            else:
                rgb_fid = compute_clean_fid(
                    str(fid_gt_dir),
                    str(fid_pred_dir),
                    mode="clean",
                    batch_size=batch_size,
                    num_workers=num_workers,
                    device=device,
                    verbose=verbose,
                )

            print("Aggregating RGB results...")

            edge_f1_valid = [r for r in edge_f1_results if r is not None]
            edge_f1_agg = (
                aggregate_rgb_edge_f1(edge_f1_valid)
                if edge_f1_valid
                else {"precision": None, "recall": None, "f1": None}
            )

            tail_p95, tail_p99 = tail_error_store.quantiles([0.95, 0.99])

            high_freq_valid = [r for r in high_freq_results if r is not None]
            high_freq_agg = (
                aggregate_high_freq_metrics(high_freq_valid)
                if high_freq_valid
                else {
                    "pred_hf_ratio_mean": None,
                    "gt_hf_ratio_mean": None,
                    "relative_diff_mean": None,
                }
            )

            rgb_results = {
                "image_quality": {
                    "psnr": _safe_mean(psnr_values, "psnr"),
                    "ssim": _safe_mean(ssim_values, "ssim"),
                    "sce": _safe_mean(sce_values, "sce"),
                    "lpips": _safe_mean(lpips_values, "lpips"),
                    "fid": _none_if_nan(rgb_fid),
                },
                "edge_f1": {
                    "precision": _none_if_nan(edge_f1_agg["precision"]),
                    "recall": _none_if_nan(edge_f1_agg["recall"]),
                    "f1": _none_if_nan(edge_f1_agg["f1"]),
                },
                "tail_errors": {
                    "p95": _none_if_nan(tail_p95),
                    "p99": _none_if_nan(tail_p99),
                },
                "high_frequency": {
                    "pred_hf_ratio": _none_if_nan(high_freq_agg["pred_hf_ratio_mean"]),
                    "gt_hf_ratio": _none_if_nan(high_freq_agg["gt_hf_ratio_mean"]),
                    "relative_diff": _none_if_nan(high_freq_agg["relative_diff_mean"]),
                },
            }

            if depth_binned_results:
                rgb_results["depth_binned_photometric"] = aggregate_depth_binned_errors(
                    depth_binned_results
                )
            elif has_depth:
                print("Warning: No valid depth_binned_photometric results.")

            # -- Benchmark RGB aggregation --
            rgb_benchmark = None
            if benchmark_rgb_results:
                print("Aggregating benchmark RGB results...")
                bm_agg = {"mae": {}, "mse": {}}
                for bn in _BENCHMARK_BIN_NAMES:
                    for metric in ("mae", "mse"):
                        vals = [
                            r[metric][bn]
                            for r in benchmark_rgb_results
                            if bn in r[metric] and np.isfinite(r[metric][bn])
                        ]
                        bm_agg[metric][bn] = (
                            float(np.mean(vals)) if vals else float("nan")
                        )
                rgb_benchmark = {
                    "boundaries": benchmark_rgb_boundaries,
                    **bm_agg,
                }

            return {
                "rgb": rgb_results,
                "rgb_benchmark": rgb_benchmark,
                "per_file_metrics": per_file_metrics,
                "dataset_info": {
                    "num_pairs": num_samples,
                    "gt_name": gt_name,
                    "pred_name": pred_name,
                },
                "spatial_info": {
                    "gt_dimensions": {"height": gt_native_dims[0], "width": gt_native_dims[1]}
                    if gt_native_dims
                    else None,
                    "pred_dimensions": {"height": pred_native_dims[0], "width": pred_native_dims[1]}
                    if pred_native_dims
                    else None,
                    "method": spatial_method,
                    "evaluated_dimensions": {
                        "height": pred_native_dims[0],
                        "width": pred_native_dims[1],
                    }
                    if pred_native_dims
                    else None,
                },
            }
        finally:
            tail_error_store.close()


# ---------------------------------------------------------------------------
# Rays (spherical direction map) evaluation
# ---------------------------------------------------------------------------


def evaluate_rays_samples(
    dataset: MultiModalDataset,
    fov_domain: Optional[str] = None,
    gt_name: str = "GT",
    pred_name: str = "Pred",
    num_workers: int = 4,
    verbose: bool = False,
    sanity_checker: Optional[SanityChecker] = None,
) -> dict:
    """Evaluate spherical direction map (rays) metrics.

    The dataset must yield samples with ``"gt"`` and ``"pred"`` keys
    containing direction map data ``(H, W, 3)`` or ``(3, H, W)``.
    Optionally ``"calibration"`` for automatic FoV domain classification.

    The primary metric is **ρ_A**: the AUC of the angular accuracy curve
    evaluated up to a FoV-dependent threshold (15°/20°/30°).

    Args:
        dataset: The MultiModalDataset to iterate.
        fov_domain: Explicit FoV domain (``"sfov"``, ``"lfov"``, or
            ``"pano"``).  When *None*, the domain is auto-detected from
            calibration intrinsics on the first sample, falling back to
            ``"lfov"`` if no intrinsics are available.
        gt_name: GT dataset display name.
        pred_name: Prediction dataset display name.
        num_workers: Data loading workers for the prefetched per-pair
            iteration. Set to 0 to disable worker-based prefetching.
        verbose: Enable verbose output.
        sanity_checker: Optional SanityChecker.

    Returns:
        Dictionary containing ``rays`` aggregate metrics, ``per_file_metrics``,
        and ``dataset_info``.
    """
    num_samples = len(dataset)
    if num_samples == 0:
        raise ValueError("Dataset has no matched samples")

    rho_a_values: list[float] = []

    # Streaming aggregate stats to avoid storing all per-pixel arrays in
    # memory (which causes OOM on large datasets).
    _agg_total = 0
    _agg_sum = 0.0
    _agg_below_5 = 0
    _agg_below_10 = 0
    _agg_below_15 = 0
    _agg_below_20 = 0
    _agg_below_30 = 0
    # Histogram for approximate median (0.01° bins over [0, 180°])
    _HIST_BINS = 18000
    _HIST_MAX = 180.0
    _agg_hist = np.zeros(_HIST_BINS, dtype=np.int64)

    logged_stats = False
    logged_alignment = False
    gt_native_dims: Optional[tuple[int, int]] = None
    pred_native_dims: Optional[tuple[int, int]] = None
    spatial_method = "none"
    resolved_domain: Optional[str] = fov_domain
    threshold_deg: Optional[float] = None

    if resolved_domain is not None:
        threshold_deg = get_threshold_for_domain(resolved_domain)
        print(f"FoV domain: {resolved_domain} (threshold: {threshold_deg}°)")

    per_file_metrics = {}

    print("Computing per-image rays metrics...")
    rays_iter = _prefetched_iter(dataset, num_workers)
    for i, sample in tqdm(
        rays_iter,
        total=num_samples,
        desc="Processing rays pairs",
    ):
        hierarchy, entry_id = _extract_hierarchy(sample)

        dirs_gt = to_numpy_directions(sample["gt"])
        dirs_pred = to_numpy_directions(sample["pred"])

        # Capture native dimensions from the first sample.
        if i == 0:
            gt_native_dims = dirs_gt.shape[:2]
            pred_native_dims = dirs_pred.shape[:2]
            spatial_method = classify_spatial_alignment(*gt_native_dims, *pred_native_dims)

        # Align GT to prediction dimensions if needed
        if dirs_gt.shape[:2] != dirs_pred.shape[:2]:
            if not logged_alignment:
                print(
                    f"  Aligning GT {dirs_gt.shape[:2]} -> "
                    f"pred {dirs_pred.shape[:2]}"
                )
                logged_alignment = True
            dirs_gt = align_to_prediction(dirs_gt, dirs_pred)

        # Auto-detect FoV domain from intrinsics on first sample
        if resolved_domain is None and i == 0:
            intrinsics_K = _get_intrinsics_K(sample)
            if intrinsics_K is not None:
                h, w = dirs_gt.shape[:2]
                resolved_domain = classify_fov_domain(intrinsics_K, h, w)
                print(
                    f"  FoV domain auto-detected: {resolved_domain} "
                    f"(from intrinsics)"
                )
            else:
                resolved_domain = "lfov"
                print(
                    "  FoV domain: defaulting to lfov "
                    "(no intrinsics available)"
                )
            threshold_deg = get_threshold_for_domain(resolved_domain)
            print(f"  Angular threshold: {threshold_deg}°")

        if verbose and not logged_stats:
            _log_sample_stats(
                np.linalg.norm(dirs_gt, axis=-1),
                np.linalg.norm(dirs_pred, axis=-1),
                "RAYS (norm)",
            )
            logged_stats = True

        # Compute angular errors
        angles, meta = compute_angular_errors(
            dirs_pred, dirs_gt, return_metadata=True
        )

        # Compute per-image metrics immediately (no need to store full array)
        per_image_mean = float(np.mean(angles)) if len(angles) > 0 else None
        per_image_median = (
            float(np.median(angles)) if len(angles) > 0 else None
        )

        # Update streaming aggregate stats
        if len(angles) > 0:
            n = len(angles)
            _agg_total += n
            _agg_sum += float(np.sum(angles))
            _agg_below_5 += int(np.sum(angles < 5.0))
            _agg_below_10 += int(np.sum(angles < 10.0))
            _agg_below_15 += int(np.sum(angles < 15.0))
            _agg_below_20 += int(np.sum(angles < 20.0))
            _agg_below_30 += int(np.sum(angles < 30.0))
            bin_idx = np.clip(
                (angles / _HIST_MAX * _HIST_BINS).astype(np.int64),
                0,
                _HIST_BINS - 1,
            )
            np.add.at(_agg_hist, bin_idx, 1)

        # Compute ρ_A
        rho_a = compute_rho_a(angles, threshold_deg)
        rho_a_values.append(rho_a)

        # Sanity check
        if sanity_checker is not None:
            sanity_checker.validate_rays_input(dirs_gt, dirs_pred, entry_id)
            if meta["mean_angular_error"] is not None:
                sanity_checker.validate_rays_rho_a(
                    rho_a, meta["mean_angular_error"], entry_id
                )

        # Build per-file entry inline
        rays_metrics = {
            "rho_a": float(rho_a) if np.isfinite(rho_a) else None,
            "angular_error": {
                "mean": per_image_mean,
                "median": per_image_median,
            },
        }

        set_value(
            per_file_metrics,
            hierarchy,
            entry_id,
            {"id": entry_id, "metrics": {"rays": rays_metrics}},
        )

    # Aggregate
    print("Aggregating rays results...")
    rho_a_agg = aggregate_rho_a(rho_a_values)

    if _agg_total > 0:
        # Approximate median from histogram
        cumsum = np.cumsum(_agg_hist)
        median_bin = int(np.searchsorted(cumsum, _agg_total / 2.0))
        approx_median = (median_bin + 0.5) * _HIST_MAX / _HIST_BINS

        angular_agg = {
            "mean_angle": _agg_sum / _agg_total,
            "median_angle": approx_median,
            "percent_below_5": _agg_below_5 / _agg_total * 100,
            "percent_below_10": _agg_below_10 / _agg_total * 100,
            "percent_below_15": _agg_below_15 / _agg_total * 100,
            "percent_below_20": _agg_below_20 / _agg_total * 100,
            "percent_below_30": _agg_below_30 / _agg_total * 100,
        }
    else:
        angular_agg = aggregate_angular_errors([])

    rays_results = {
        "rho_a": rho_a_agg,
        "angular_error": angular_agg,
    }

    return {
        "rays": rays_results,
        "per_file_metrics": per_file_metrics,
        "dataset_info": {
            "num_pairs": num_samples,
            "gt_name": gt_name,
            "pred_name": pred_name,
            "fov_domain": resolved_domain,
            "threshold_deg": threshold_deg,
        },
        "spatial_info": {
            "gt_dimensions": {"height": gt_native_dims[0], "width": gt_native_dims[1]}
            if gt_native_dims
            else None,
            "pred_dimensions": {"height": pred_native_dims[0], "width": pred_native_dims[1]}
            if pred_native_dims
            else None,
            "method": spatial_method,
            "evaluated_dimensions": {"height": pred_native_dims[0], "width": pred_native_dims[1]}
            if pred_native_dims
            else None,
        },
    }
