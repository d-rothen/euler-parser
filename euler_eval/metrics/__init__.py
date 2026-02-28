"""Depth and RGB evaluation metrics package.

This package provides various metrics for comparing depth maps and RGB images:

Depth metrics:
- Image quality: PSNR, SSIM, LPIPS, FID, KID
- Depth-specific: AbsRel, RMSE, Scale-Invariant Log Error
- Geometric: Normal Consistency, Depth-Edge F1

RGB metrics:
- Image quality: PSNR, SSIM, LPIPS
- Edge F1
- Tail errors (p95/p99)
- High-frequency energy ratio
- Depth-binned photometric error (MAE/MSE per depth bin)
"""

# Depth metrics
from .psnr import compute_psnr
from .ssim import compute_ssim
from .lpips_metric import compute_lpips, LPIPSMetric
from .fid_kid import compute_fid, compute_kid, FIDKIDMetric
from .absrel import compute_absrel, aggregate_absrel
from .rmse import compute_rmse, compute_rmse_per_pixel, aggregate_rmse
from .scale_invariant_log import (
    compute_scale_invariant_log_error,
    compute_silog_per_pixel,
    aggregate_silog,
)
from .normal_consistency import (
    compute_normal_consistency,
    compute_normal_angles,
    aggregate_normal_consistency,
)
from .depth_edge_f1 import compute_depth_edge_f1, aggregate_edge_f1

# RGB metrics
from .rgb_psnr_ssim import compute_rgb_psnr, compute_rgb_ssim, compute_rgb_psnr_masked
from .rgb_lpips import compute_rgb_lpips, RGBLPIPSMetric
from .daniel_error import compute_sce
from .depth_binned_error import (
    compute_depth_binned_mae,
    compute_depth_binned_mse,
    compute_depth_binned_photometric_error,
    aggregate_depth_binned_errors,
)
from .rgb_edge_f1 import (
    compute_rgb_edge_f1,
    detect_rgb_edges,
    aggregate_rgb_edge_f1,
)
from .tail_errors import (
    compute_tail_errors,
    compute_tail_errors_per_channel,
    get_tail_error_pixels,
    aggregate_tail_errors,
)
from .high_freq_energy import (
    compute_high_freq_energy_ratio,
    compute_high_freq_energy_comparison,
    compute_high_freq_preservation,
    compute_frequency_spectrum_similarity,
    aggregate_high_freq_metrics,
)

# Utilities
from .utils import (
    convert_planar_to_radial,
    normalize_depth_for_visualization,
    get_valid_mask,
    depth_to_3channel,
    get_depth_bins,
)

__all__ = [
    # Depth image quality metrics
    "compute_psnr",
    "compute_ssim",
    "compute_lpips",
    "LPIPSMetric",
    "compute_fid",
    "compute_kid",
    "FIDKIDMetric",
    # Depth-specific metrics
    "compute_absrel",
    "aggregate_absrel",
    "compute_rmse",
    "compute_rmse_per_pixel",
    "aggregate_rmse",
    "compute_scale_invariant_log_error",
    "compute_silog_per_pixel",
    "aggregate_silog",
    # Depth geometric metrics
    "compute_normal_consistency",
    "compute_normal_angles",
    "aggregate_normal_consistency",
    "compute_depth_edge_f1",
    "aggregate_edge_f1",
    # RGB image quality metrics
    "compute_rgb_psnr",
    "compute_rgb_ssim",
    "compute_rgb_psnr_masked",
    "compute_rgb_lpips",
    "RGBLPIPSMetric",
    "compute_sce",
    # RGB depth-binned metrics
    "compute_depth_binned_mae",
    "compute_depth_binned_mse",
    "compute_depth_binned_photometric_error",
    "aggregate_depth_binned_errors",
    # RGB edge metrics
    "compute_rgb_edge_f1",
    "detect_rgb_edges",
    "aggregate_rgb_edge_f1",
    # RGB tail errors
    "compute_tail_errors",
    "compute_tail_errors_per_channel",
    "get_tail_error_pixels",
    "aggregate_tail_errors",
    # RGB high-frequency metrics
    "compute_high_freq_energy_ratio",
    "compute_high_freq_energy_comparison",
    "compute_high_freq_preservation",
    "compute_frequency_spectrum_similarity",
    "aggregate_high_freq_metrics",
    # Utilities
    "convert_planar_to_radial",
    "normalize_depth_for_visualization",
    "get_valid_mask",
    "depth_to_3channel",
    "get_depth_bins",
]
