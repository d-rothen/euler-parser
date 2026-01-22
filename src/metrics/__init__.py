"""Depth evaluation metrics package.

This package provides various metrics for comparing depth maps:
- Image quality metrics: PSNR, SSIM, LPIPS, FID, KID
- Depth-specific metrics: AbsRel, RMSE, Scale-Invariant Log Error
- Geometric metrics: Normal Consistency, Depth-Edge F1
"""

from .psnr import compute_psnr
from .ssim import compute_ssim
from .lpips_metric import compute_lpips, LPIPSMetric
from .fid_kid import compute_fid, compute_kid, FIDKIDMetric
from .absrel import compute_absrel
from .rmse import compute_rmse
from .scale_invariant_log import compute_scale_invariant_log_error
from .normal_consistency import compute_normal_consistency
from .depth_edge_f1 import compute_depth_edge_f1
from .utils import (
    load_depth_file,
    convert_planar_to_radial,
    normalize_depth_for_visualization,
)

__all__ = [
    # Image quality metrics
    "compute_psnr",
    "compute_ssim",
    "compute_lpips",
    "LPIPSMetric",
    "compute_fid",
    "compute_kid",
    "FIDKIDMetric",
    # Depth metrics
    "compute_absrel",
    "compute_rmse",
    "compute_scale_invariant_log_error",
    # Geometric metrics
    "compute_normal_consistency",
    "compute_depth_edge_f1",
    # Utilities
    "load_depth_file",
    "convert_planar_to_radial",
    "normalize_depth_for_visualization",
]
