"""Sanity checker for metric validation.

This module provides post-processing validation of metric results against
expected configurations. It collects metadata during metric computation
and validates it after all metrics are computed to avoid GPU performance impact.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class MetricWarning:
    """A single warning about a metric computation."""
    metric_name: str
    warning_type: str
    message: str
    observed_value: Optional[float] = None
    expected_value: Optional[float] = None
    file_id: Optional[str] = None


@dataclass
class MetricMetadata:
    """Metadata collected during metric computation for post-validation."""
    # Depth-specific metadata
    depth_min: Optional[float] = None
    depth_max: Optional[float] = None
    depth_range: Optional[float] = None
    valid_pixel_count: Optional[int] = None
    total_pixel_count: Optional[int] = None

    # PSNR-specific
    psnr_max_val_used: Optional[float] = None

    # Edge-specific
    edge_pixel_count: Optional[int] = None
    gt_edge_pixel_count: Optional[int] = None

    # RGB-specific
    rgb_min: Optional[float] = None
    rgb_max: Optional[float] = None

    # Normal consistency
    valid_pixels_after_erosion: Optional[int] = None


@dataclass
class SanityCheckResult:
    """Aggregated sanity check results for a dataset."""
    warnings: list[MetricWarning] = field(default_factory=list)
    total_samples: int = 0

    def add_warning(self, warning: MetricWarning) -> None:
        """Add a warning to the results."""
        self.warnings.append(warning)

    def get_warning_summary(self) -> dict[str, dict]:
        """Get a summary of warnings grouped by metric and type."""
        summary = {}
        for warning in self.warnings:
            key = f"{warning.metric_name}:{warning.warning_type}"
            if key not in summary:
                summary[key] = {
                    "metric": warning.metric_name,
                    "type": warning.warning_type,
                    "message": warning.message,
                    "count": 0,
                    "sample_values": [],
                }
            summary[key]["count"] += 1
            if warning.observed_value is not None and len(summary[key]["sample_values"]) < 5:
                summary[key]["sample_values"].append(warning.observed_value)
        return summary

    def has_warnings(self) -> bool:
        """Check if any warnings were recorded."""
        return len(self.warnings) > 0


class SanityChecker:
    """Validates metric results against expected configurations.

    This class performs post-processing validation to avoid impacting
    GPU performance during metric computation. Metadata is collected
    during computation and validated afterward.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the sanity checker.

        Args:
            config_path: Path to metrics_config.json. If None, uses default
                        location relative to this file.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "metrics_config.json"

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.depth_results = SanityCheckResult()
        self.rgb_results = SanityCheckResult()

        # Track how many warnings have been reported in intermediate reports
        self._last_reported_depth_warnings = 0
        self._last_reported_rgb_warnings = 0
        self._last_reported_depth_samples = 0
        self._last_reported_rgb_samples = 0

    def _load_config(self) -> dict:
        """Load the metrics configuration file."""
        if not self.config_path.exists():
            print(f"Warning: metrics_config.json not found at {self.config_path}, using defaults")
            return self._get_default_config()

        with open(self.config_path, "r") as f:
            return json.load(f)

    def _get_default_config(self) -> dict:
        """Return default configuration if config file is missing."""
        return {
            "depth": {
                "expected_range": {"min": 0.0, "max": 100.0},
                "psnr": {"expected_max_val": 100.0, "warn_if_max_exceeds": 100.0},
                "ssim": {"min_valid_range": 0.1},
                "absrel": {"warn_if_median_exceeds": 1.0},
                "rmse": {"warn_if_exceeds_fraction_of_range": 0.5},
                "silog": {"warn_if_exceeds": 0.5},
                "normal_consistency": {"warn_if_mean_angle_exceeds": 45.0},
                "depth_edge_f1": {"warn_if_edge_ratio_below": 0.001, "warn_if_edge_ratio_above": 0.5},
            },
            "rgb": {
                "expected_range": {"min": 0.0, "max": 1.0},
                "psnr": {"warn_if_below": 10.0, "warn_if_above": 60.0},
                "ssim": {"warn_if_below": 0.3},
                "lpips": {"warn_if_exceeds": 0.7},
                "tail_errors": {"warn_if_p99_exceeds": 0.5},
                "high_freq_energy": {"warn_if_relative_diff_below": -0.5},
            }
        }

    def reset(self) -> None:
        """Reset all collected results."""
        self.depth_results = SanityCheckResult()
        self.rgb_results = SanityCheckResult()
        self._last_reported_depth_warnings = 0
        self._last_reported_rgb_warnings = 0
        self._last_reported_depth_samples = 0
        self._last_reported_rgb_samples = 0

    # =========================================================================
    # Depth Metric Validation
    # =========================================================================

    def validate_depth_input(
        self,
        depth_gt: np.ndarray,
        depth_pred: np.ndarray,
        file_id: Optional[str] = None,
    ) -> MetricMetadata:
        """Validate depth input data and collect metadata.

        Args:
            depth_gt: Ground truth depth map.
            depth_pred: Predicted depth map.
            file_id: Optional file identifier for warnings.

        Returns:
            MetricMetadata with collected values.
        """
        config = self.config.get("depth", {}).get("expected_range", {})
        expected_max = config.get("max", 100.0)
        expected_min = config.get("min", 0.0)

        # Compute metadata
        valid_mask = (depth_gt > 0) & (depth_pred > 0) & np.isfinite(depth_gt) & np.isfinite(depth_pred)
        valid_count = int(np.sum(valid_mask))
        total_count = depth_gt.size

        metadata = MetricMetadata(
            valid_pixel_count=valid_count,
            total_pixel_count=total_count,
        )

        if valid_count > 0:
            gt_valid = depth_gt[valid_mask]
            pred_valid = depth_pred[valid_mask]

            metadata.depth_min = float(min(gt_valid.min(), pred_valid.min()))
            metadata.depth_max = float(max(gt_valid.max(), pred_valid.max()))
            metadata.depth_range = metadata.depth_max - metadata.depth_min

            # Check for values exceeding expected range
            if metadata.depth_max > expected_max:
                self.depth_results.add_warning(MetricWarning(
                    metric_name="depth_input",
                    warning_type="max_exceeded",
                    message=f"Depth values exceed expected max ({expected_max}m). "
                            f"This may indicate incorrect depth_scale configuration.",
                    observed_value=metadata.depth_max,
                    expected_value=expected_max,
                    file_id=file_id,
                ))

            if metadata.depth_min < expected_min:
                self.depth_results.add_warning(MetricWarning(
                    metric_name="depth_input",
                    warning_type="min_below_expected",
                    message=f"Depth values below expected min ({expected_min}m). "
                            f"Negative depths are invalid.",
                    observed_value=metadata.depth_min,
                    expected_value=expected_min,
                    file_id=file_id,
                ))
        else:
            self.depth_results.add_warning(MetricWarning(
                metric_name="depth_input",
                warning_type="no_valid_pixels",
                message="No valid pixels found in depth maps",
                file_id=file_id,
            ))

        self.depth_results.total_samples += 1
        return metadata

    def validate_depth_psnr(
        self,
        psnr_value: float,
        max_val_used: float,
        file_id: Optional[str] = None,
    ) -> None:
        """Validate PSNR computation for depth maps.

        Args:
            psnr_value: Computed PSNR value.
            max_val_used: The max depth value used in PSNR calculation.
            file_id: Optional file identifier.
        """
        config = self.config.get("depth", {}).get("psnr", {})
        warn_if_exceeds = config.get("warn_if_max_exceeds", 100.0)
        warn_if_below = config.get("warn_if_max_below", 0.1)

        if max_val_used > warn_if_exceeds:
            self.depth_results.add_warning(MetricWarning(
                metric_name="psnr",
                warning_type="max_val_too_high",
                message=f"PSNR max_val ({max_val_used:.2f}m) exceeds expected ({warn_if_exceeds}m). "
                        f"PSNR values may be artificially low. Check depth_scale configuration.",
                observed_value=max_val_used,
                expected_value=warn_if_exceeds,
                file_id=file_id,
            ))

        if max_val_used < warn_if_below:
            self.depth_results.add_warning(MetricWarning(
                metric_name="psnr",
                warning_type="max_val_too_low",
                message=f"PSNR max_val ({max_val_used:.4f}m) below expected ({warn_if_below}m). "
                        f"This may indicate incorrect depth_scale or uniform depth.",
                observed_value=max_val_used,
                expected_value=warn_if_below,
                file_id=file_id,
            ))

    def validate_depth_ssim(
        self,
        ssim_value: float,
        depth_range: float,
        file_id: Optional[str] = None,
    ) -> None:
        """Validate SSIM computation for depth maps.

        Args:
            ssim_value: Computed SSIM value.
            depth_range: Range of valid depth values (max - min).
            file_id: Optional file identifier.
        """
        config = self.config.get("depth", {}).get("ssim", {})
        min_range = config.get("min_valid_range", 0.1)

        if depth_range < min_range:
            self.depth_results.add_warning(MetricWarning(
                metric_name="ssim",
                warning_type="depth_range_too_small",
                message=f"Depth range ({depth_range:.4f}m) below threshold ({min_range}m). "
                        f"SSIM may not be meaningful for near-uniform depth.",
                observed_value=depth_range,
                expected_value=min_range,
                file_id=file_id,
            ))

    def validate_depth_absrel(
        self,
        median_absrel: float,
        p90_absrel: float,
        file_id: Optional[str] = None,
    ) -> None:
        """Validate AbsRel metric results.

        Args:
            median_absrel: Median absolute relative error.
            p90_absrel: 90th percentile absolute relative error.
            file_id: Optional file identifier.
        """
        config = self.config.get("depth", {}).get("absrel", {})
        warn_median = config.get("warn_if_median_exceeds", 1.0)
        warn_p90 = config.get("warn_if_p90_exceeds", 2.0)

        if np.isfinite(median_absrel) and median_absrel > warn_median:
            self.depth_results.add_warning(MetricWarning(
                metric_name="absrel",
                warning_type="median_too_high",
                message=f"Median AbsRel ({median_absrel:.2f}) exceeds {warn_median}. "
                        f"This means typical error exceeds 100% of GT depth - likely scale mismatch.",
                observed_value=median_absrel,
                expected_value=warn_median,
                file_id=file_id,
            ))

        if np.isfinite(p90_absrel) and p90_absrel > warn_p90:
            self.depth_results.add_warning(MetricWarning(
                metric_name="absrel",
                warning_type="p90_too_high",
                message=f"P90 AbsRel ({p90_absrel:.2f}) exceeds {warn_p90}. "
                        f"Significant portion of predictions have >200% relative error.",
                observed_value=p90_absrel,
                expected_value=warn_p90,
                file_id=file_id,
            ))

    def validate_depth_rmse(
        self,
        rmse_value: float,
        depth_range: float,
        file_id: Optional[str] = None,
    ) -> None:
        """Validate RMSE metric results.

        Args:
            rmse_value: Computed RMSE value.
            depth_range: Range of depth values.
            file_id: Optional file identifier.
        """
        config = self.config.get("depth", {}).get("rmse", {})
        fraction = config.get("warn_if_exceeds_fraction_of_range", 0.5)

        if np.isfinite(rmse_value) and np.isfinite(depth_range) and depth_range > 0:
            threshold = fraction * depth_range
            if rmse_value > threshold:
                self.depth_results.add_warning(MetricWarning(
                    metric_name="rmse",
                    warning_type="rmse_too_high",
                    message=f"RMSE ({rmse_value:.2f}m) exceeds {fraction*100:.0f}% of depth range "
                            f"({depth_range:.2f}m). This indicates large prediction errors.",
                    observed_value=rmse_value,
                    expected_value=threshold,
                    file_id=file_id,
                ))

    def validate_depth_silog(
        self,
        silog_value: float,
        file_id: Optional[str] = None,
    ) -> None:
        """Validate SILog metric results.

        Args:
            silog_value: Computed scale-invariant log error.
            file_id: Optional file identifier.
        """
        config = self.config.get("depth", {}).get("silog", {})
        threshold = config.get("warn_if_exceeds", 0.5)

        if np.isfinite(silog_value) and silog_value > threshold:
            self.depth_results.add_warning(MetricWarning(
                metric_name="silog",
                warning_type="silog_too_high",
                message=f"SILog ({silog_value:.3f}) exceeds threshold ({threshold}). "
                        f"This indicates significant scale mismatch between predictions and GT.",
                observed_value=silog_value,
                expected_value=threshold,
                file_id=file_id,
            ))

    def validate_normal_consistency(
        self,
        mean_angle: float,
        valid_pixels_after_erosion: int,
        file_id: Optional[str] = None,
    ) -> None:
        """Validate normal consistency metric results.

        Args:
            mean_angle: Mean angular error in degrees.
            valid_pixels_after_erosion: Number of valid pixels after mask erosion.
            file_id: Optional file identifier.
        """
        config = self.config.get("depth", {}).get("normal_consistency", {})
        warn_angle = config.get("warn_if_mean_angle_exceeds", 45.0)
        warn_pixels = config.get("warn_if_valid_pixels_below", 100)

        if np.isfinite(mean_angle) and mean_angle > warn_angle:
            self.depth_results.add_warning(MetricWarning(
                metric_name="normal_consistency",
                warning_type="mean_angle_too_high",
                message=f"Mean normal angle ({mean_angle:.1f}°) exceeds {warn_angle}°. "
                        f"Surface geometry differs significantly from GT.",
                observed_value=mean_angle,
                expected_value=warn_angle,
                file_id=file_id,
            ))

        if valid_pixels_after_erosion < warn_pixels:
            self.depth_results.add_warning(MetricWarning(
                metric_name="normal_consistency",
                warning_type="insufficient_valid_pixels",
                message=f"Only {valid_pixels_after_erosion} valid pixels after erosion "
                        f"(threshold: {warn_pixels}). Normal consistency may be unreliable.",
                observed_value=float(valid_pixels_after_erosion),
                expected_value=float(warn_pixels),
                file_id=file_id,
            ))

    def validate_depth_edge_f1(
        self,
        pred_edge_count: int,
        gt_edge_count: int,
        total_pixels: int,
        f1_score: float,
        file_id: Optional[str] = None,
    ) -> None:
        """Validate depth edge F1 metric results.

        Args:
            pred_edge_count: Number of predicted edge pixels.
            gt_edge_count: Number of GT edge pixels.
            total_pixels: Total number of pixels.
            f1_score: Computed F1 score.
            file_id: Optional file identifier.
        """
        config = self.config.get("depth", {}).get("depth_edge_f1", {})
        warn_below = config.get("warn_if_edge_ratio_below", 0.001)
        warn_above = config.get("warn_if_edge_ratio_above", 0.5)

        if total_pixels > 0:
            gt_ratio = gt_edge_count / total_pixels
            pred_ratio = pred_edge_count / total_pixels

            if gt_ratio < warn_below:
                self.depth_results.add_warning(MetricWarning(
                    metric_name="depth_edge_f1",
                    warning_type="gt_edges_too_sparse",
                    message=f"GT edge ratio ({gt_ratio:.4f}) below threshold ({warn_below}). "
                            f"Edge threshold may be too high or depth too uniform.",
                    observed_value=gt_ratio,
                    expected_value=warn_below,
                    file_id=file_id,
                ))

            if gt_ratio > warn_above:
                self.depth_results.add_warning(MetricWarning(
                    metric_name="depth_edge_f1",
                    warning_type="gt_edges_too_dense",
                    message=f"GT edge ratio ({gt_ratio:.2f}) exceeds threshold ({warn_above}). "
                            f"Edge threshold may be too low, detecting noise as edges.",
                    observed_value=gt_ratio,
                    expected_value=warn_above,
                    file_id=file_id,
                ))

    # =========================================================================
    # RGB Metric Validation
    # =========================================================================

    def validate_rgb_input(
        self,
        img_gt: np.ndarray,
        img_pred: np.ndarray,
        file_id: Optional[str] = None,
    ) -> MetricMetadata:
        """Validate RGB input data and collect metadata.

        Args:
            img_gt: Ground truth RGB image.
            img_pred: Predicted RGB image.
            file_id: Optional file identifier.

        Returns:
            MetricMetadata with collected values.
        """
        config = self.config.get("rgb", {}).get("expected_range", {})
        expected_max = config.get("max", 1.0)
        expected_min = config.get("min", 0.0)

        metadata = MetricMetadata(
            rgb_min=float(min(img_gt.min(), img_pred.min())),
            rgb_max=float(max(img_gt.max(), img_pred.max())),
        )

        # Check for values outside [0, 1] range
        if metadata.rgb_max > expected_max + 0.01:  # Small tolerance for float precision
            self.rgb_results.add_warning(MetricWarning(
                metric_name="rgb_input",
                warning_type="values_exceed_range",
                message=f"RGB values exceed expected range [0, 1]. Max observed: {metadata.rgb_max:.3f}. "
                        f"Images may not be properly normalized (divide by 255).",
                observed_value=metadata.rgb_max,
                expected_value=expected_max,
                file_id=file_id,
            ))

        if metadata.rgb_min < expected_min - 0.01:
            self.rgb_results.add_warning(MetricWarning(
                metric_name="rgb_input",
                warning_type="values_below_range",
                message=f"RGB values below expected range [0, 1]. Min observed: {metadata.rgb_min:.3f}. "
                        f"Negative values are invalid for RGB images.",
                observed_value=metadata.rgb_min,
                expected_value=expected_min,
                file_id=file_id,
            ))

        self.rgb_results.total_samples += 1
        return metadata

    def validate_rgb_psnr(
        self,
        psnr_value: float,
        file_id: Optional[str] = None,
    ) -> None:
        """Validate RGB PSNR results.

        Args:
            psnr_value: Computed PSNR value.
            file_id: Optional file identifier.
        """
        config = self.config.get("rgb", {}).get("psnr", {})
        warn_below = config.get("warn_if_below", 10.0)
        warn_above = config.get("warn_if_above", 60.0)

        if np.isfinite(psnr_value):
            if psnr_value < warn_below:
                self.rgb_results.add_warning(MetricWarning(
                    metric_name="rgb_psnr",
                    warning_type="psnr_too_low",
                    message=f"RGB PSNR ({psnr_value:.1f} dB) below {warn_below} dB. "
                            f"This indicates very poor reconstruction quality.",
                    observed_value=psnr_value,
                    expected_value=warn_below,
                    file_id=file_id,
                ))

            if psnr_value > warn_above and psnr_value != float("inf"):
                self.rgb_results.add_warning(MetricWarning(
                    metric_name="rgb_psnr",
                    warning_type="psnr_unusually_high",
                    message=f"RGB PSNR ({psnr_value:.1f} dB) above {warn_above} dB. "
                            f"This is unusually high - verify GT and predictions aren't identical or near-identical.",
                    observed_value=psnr_value,
                    expected_value=warn_above,
                    file_id=file_id,
                ))

    def validate_rgb_ssim(
        self,
        ssim_value: float,
        file_id: Optional[str] = None,
    ) -> None:
        """Validate RGB SSIM results.

        Args:
            ssim_value: Computed SSIM value.
            file_id: Optional file identifier.
        """
        config = self.config.get("rgb", {}).get("ssim", {})
        warn_below = config.get("warn_if_below", 0.3)

        if np.isfinite(ssim_value) and ssim_value < warn_below:
            self.rgb_results.add_warning(MetricWarning(
                metric_name="rgb_ssim",
                warning_type="ssim_too_low",
                message=f"RGB SSIM ({ssim_value:.3f}) below {warn_below}. "
                        f"This indicates very poor structural similarity.",
                observed_value=ssim_value,
                expected_value=warn_below,
                file_id=file_id,
            ))

    def validate_rgb_lpips(
        self,
        lpips_value: float,
        file_id: Optional[str] = None,
    ) -> None:
        """Validate RGB LPIPS results.

        Args:
            lpips_value: Computed LPIPS value.
            file_id: Optional file identifier.
        """
        config = self.config.get("rgb", {}).get("lpips", {})
        warn_exceeds = config.get("warn_if_exceeds", 0.7)

        if np.isfinite(lpips_value) and lpips_value > warn_exceeds:
            self.rgb_results.add_warning(MetricWarning(
                metric_name="rgb_lpips",
                warning_type="lpips_too_high",
                message=f"RGB LPIPS ({lpips_value:.3f}) exceeds {warn_exceeds}. "
                        f"This indicates very poor perceptual similarity.",
                observed_value=lpips_value,
                expected_value=warn_exceeds,
                file_id=file_id,
            ))

    def validate_tail_errors(
        self,
        p99_error: float,
        file_id: Optional[str] = None,
    ) -> None:
        """Validate tail error results.

        Args:
            p99_error: 99th percentile error.
            file_id: Optional file identifier.
        """
        config = self.config.get("rgb", {}).get("tail_errors", {})
        warn_exceeds = config.get("warn_if_p99_exceeds", 0.5)

        if np.isfinite(p99_error) and p99_error > warn_exceeds:
            self.rgb_results.add_warning(MetricWarning(
                metric_name="tail_errors",
                warning_type="p99_too_high",
                message=f"Tail error p99 ({p99_error:.3f}) exceeds {warn_exceeds}. "
                        f"Significant outlier errors present in predictions.",
                observed_value=p99_error,
                expected_value=warn_exceeds,
                file_id=file_id,
            ))

    def validate_high_freq_energy(
        self,
        relative_diff: float,
        file_id: Optional[str] = None,
    ) -> None:
        """Validate high-frequency energy comparison results.

        Args:
            relative_diff: Relative difference in HF energy.
            file_id: Optional file identifier.
        """
        config = self.config.get("rgb", {}).get("high_freq_energy", {})
        warn_below = config.get("warn_if_relative_diff_below", -0.5)

        if np.isfinite(relative_diff) and relative_diff < warn_below:
            self.rgb_results.add_warning(MetricWarning(
                metric_name="high_freq_energy",
                warning_type="excessive_smoothing",
                message=f"HF energy relative diff ({relative_diff:.2f}) below {warn_below}. "
                        f"Predictions appear over-smoothed, losing high-frequency detail.",
                observed_value=relative_diff,
                expected_value=warn_below,
                file_id=file_id,
            ))

    def validate_depth_binned(
        self,
        bin_results: dict,
        file_id: Optional[str] = None,
    ) -> None:
        """Validate depth-binned photometric error results.

        Args:
            bin_results: Dictionary with MAE/MSE per bin.
            file_id: Optional file identifier.
        """
        config = self.config.get("rgb", {}).get("depth_binned", {})
        warn_if_empty = config.get("warn_if_bin_empty", True)

        if warn_if_empty and bin_results is not None:
            for metric_type in ["mae", "mse"]:
                if metric_type in bin_results:
                    for bin_name in ["near", "mid", "far"]:
                        if bin_name in bin_results[metric_type]:
                            val = bin_results[metric_type][bin_name]
                            if val is None or (isinstance(val, float) and np.isnan(val)):
                                self.rgb_results.add_warning(MetricWarning(
                                    metric_name="depth_binned",
                                    warning_type=f"{bin_name}_bin_empty",
                                    message=f"Depth bin '{bin_name}' is empty. "
                                            f"Consider adjusting near_threshold/far_threshold in config.",
                                    file_id=file_id,
                                ))

    # =========================================================================
    # Report Generation
    # =========================================================================

    def get_depth_report(self) -> dict:
        """Get the depth metrics sanity check report."""
        summary = self.depth_results.get_warning_summary()
        return {
            "total_samples": self.depth_results.total_samples,
            "total_warnings": len(self.depth_results.warnings),
            "warnings_by_type": summary,
            "has_issues": self.depth_results.has_warnings(),
        }

    def get_rgb_report(self) -> dict:
        """Get the RGB metrics sanity check report."""
        summary = self.rgb_results.get_warning_summary()
        return {
            "total_samples": self.rgb_results.total_samples,
            "total_warnings": len(self.rgb_results.warnings),
            "warnings_by_type": summary,
            "has_issues": self.rgb_results.has_warnings(),
        }

    def print_pair_report(self, pair_name: str, is_depth: bool = True) -> None:
        """Print a sanity check report for just the current dataset pair.

        This prints only the warnings accumulated since the last call to
        print_pair_report, allowing intermediate reports after each evaluation.

        Args:
            pair_name: Name of the dataset pair being evaluated.
            is_depth: True for depth evaluation, False for RGB evaluation.
        """
        if is_depth:
            results = self.depth_results
            last_warnings = self._last_reported_depth_warnings
            last_samples = self._last_reported_depth_samples
            metric_type = "DEPTH"
        else:
            results = self.rgb_results
            last_warnings = self._last_reported_rgb_warnings
            last_samples = self._last_reported_rgb_samples
            metric_type = "RGB"

        # Get warnings added since last report
        new_warnings = results.warnings[last_warnings:]
        new_samples = results.total_samples - last_samples

        if not new_warnings:
            print(f"\n[{metric_type} SANITY CHECK] {pair_name}: All metrics passed (0 warnings)")
            # Update tracking
            if is_depth:
                self._last_reported_depth_warnings = len(results.warnings)
                self._last_reported_depth_samples = results.total_samples
            else:
                self._last_reported_rgb_warnings = len(results.warnings)
                self._last_reported_rgb_samples = results.total_samples
            return

        # Build summary for just the new warnings
        summary = {}
        for warning in new_warnings:
            key = f"{warning.metric_name}:{warning.warning_type}"
            if key not in summary:
                summary[key] = {
                    "metric": warning.metric_name,
                    "type": warning.warning_type,
                    "message": warning.message,
                    "count": 0,
                    "sample_values": [],
                }
            summary[key]["count"] += 1
            if warning.observed_value is not None and len(summary[key]["sample_values"]) < 3:
                summary[key]["sample_values"].append(warning.observed_value)

        # Print the intermediate report
        print(f"\n{'─' * 70}")
        print(f"  [{metric_type} SANITY CHECK] {pair_name}")
        print(f"  {len(new_warnings)} warnings across {new_samples} samples")
        print("─" * 70)

        for key, info in summary.items():
            ratio = info["count"] / new_samples if new_samples > 0 else 0
            print(f"\n  {info['metric']} - {info['type']}:")
            print(f"    Occurrences: {info['count']}/{new_samples} ({ratio*100:.1f}%)")
            print(f"    {info['message']}")
            if info["sample_values"]:
                vals_str = ", ".join(f"{v:.4f}" for v in info["sample_values"])
                print(f"    Sample values: {vals_str}")

        print("─" * 70)

        # Update tracking
        if is_depth:
            self._last_reported_depth_warnings = len(results.warnings)
            self._last_reported_depth_samples = results.total_samples
        else:
            self._last_reported_rgb_warnings = len(results.warnings)
            self._last_reported_rgb_samples = results.total_samples

    def print_report(self, title: str = "SANITY CHECK REPORT") -> None:
        """Print a formatted sanity check report to console.

        Args:
            title: Title for the report.
        """
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print("=" * 70)

        depth_report = self.get_depth_report()
        rgb_report = self.get_rgb_report()

        has_any_issues = depth_report["has_issues"] or rgb_report["has_issues"]

        if not has_any_issues:
            print("\n  All metrics passed sanity checks.")
            print("=" * 70)
            return

        # Print depth warnings
        if depth_report["has_issues"]:
            print(f"\n[DEPTH] {depth_report['total_warnings']} warnings "
                  f"across {depth_report['total_samples']} samples:")
            print("-" * 70)

            for key, info in depth_report["warnings_by_type"].items():
                ratio = info["count"] / depth_report["total_samples"]
                print(f"\n  {info['metric']} - {info['type']}:")
                print(f"    Occurrences: {info['count']}/{depth_report['total_samples']} ({ratio*100:.1f}%)")
                print(f"    {info['message']}")
                if info["sample_values"]:
                    vals_str = ", ".join(f"{v:.4f}" for v in info["sample_values"][:3])
                    print(f"    Sample values: {vals_str}")

        # Print RGB warnings
        if rgb_report["has_issues"]:
            print(f"\n[RGB] {rgb_report['total_warnings']} warnings "
                  f"across {rgb_report['total_samples']} samples:")
            print("-" * 70)

            for key, info in rgb_report["warnings_by_type"].items():
                ratio = info["count"] / rgb_report["total_samples"]
                print(f"\n  {info['metric']} - {info['type']}:")
                print(f"    Occurrences: {info['count']}/{rgb_report['total_samples']} ({ratio*100:.1f}%)")
                print(f"    {info['message']}")
                if info["sample_values"]:
                    vals_str = ", ".join(f"{v:.4f}" for v in info["sample_values"][:3])
                    print(f"    Sample values: {vals_str}")

        print("\n" + "=" * 70)
        print("  Review metrics_config.json to adjust thresholds if needed.")
        print("=" * 70)

    def get_full_report(self) -> dict:
        """Get the complete sanity check report as a dictionary."""
        return {
            "depth": self.get_depth_report(),
            "rgb": self.get_rgb_report(),
        }
