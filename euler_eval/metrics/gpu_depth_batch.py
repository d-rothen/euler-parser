"""Batched GPU depth metrics (masked PSNR/SSIM + per-pixel AbsRel/RMSE/SILog).

The numpy/scipy per-sample implementations in ``psnr.py``, ``ssim.py``,
``absrel.py``, ``rmse.py``, and ``scale_invariant_log.py`` all consume the
same ``(pred, gt, valid_mask)`` triple and do mostly-elementwise work
plus one Gaussian convolution for SSIM. On a GPU machine we can upload
each triple once, batch across samples that share a spatial shape, and
emit all five metrics in a single pass.

Numerical contract
------------------
The outputs are designed to match the CPU implementations:
- ``compute_psnr`` uses ``max_val = gt_valid.max()`` (valid-only GT).
- ``compute_ssim`` normalizes by ``[min(pred,gt), max(pred,gt)]`` over
  the valid region, zeros out invalid pixels, runs a Gaussian filter
  (``sigma = 11/6``, truncate such that window = 11), and averages the
  SSIM map over ``(gaussian_filter(valid_mask) > 0.5)``.
- ``compute_absrel``: ``|pred-gt| / gt`` over valid pixels, flat array.
- ``compute_rmse_per_pixel``: ``(pred-gt)^2`` over valid pixels.
- ``compute_silog_per_pixel``: ``|log(pred) - log(gt)|`` over valid pixels.
- ``compute_scale_invariant_log_error``:
  ``sqrt(mean(d^2) - mean(d)^2)`` over valid pixels, ``d = log(pred) - log(gt)``.

Edge cases (empty mask, zero range, zero MSE) are preserved.

If torchmetrics-style ops fail or the target device is unusable, the
batcher falls back to the numpy ``compute_*`` functions per sample.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F


_SSIM_WINDOW = 11
_SSIM_SIGMA = _SSIM_WINDOW / 6.0
_SSIM_K1 = 0.01
_SSIM_K2 = 0.03


def _gaussian_kernel_1d(
    window: int, sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Build a normalized 1D Gaussian kernel matching ``scipy.ndimage``.

    ``scipy.ndimage.gaussian_filter`` uses ``truncate = (window-1)/2/sigma``
    so the resulting kernel has ``window`` samples centered at 0. We
    re-derive that kernel here for bit-close parity.
    """
    radius = (window - 1) // 2
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / k.sum()
    return k


def _gaussian_filter_2d(
    x: torch.Tensor, kernel_1d: torch.Tensor
) -> torch.Tensor:
    """Separable Gaussian filter on a (B, H, W) tensor via two conv2d passes."""
    B, H, W = x.shape
    radius = kernel_1d.shape[0] // 2
    k_row = kernel_1d.view(1, 1, 1, -1)
    k_col = kernel_1d.view(1, 1, -1, 1)
    y = x.unsqueeze(1)  # (B, 1, H, W)
    # Horizontal pass: reflect padding matches scipy's default "reflect" mode
    y = F.pad(y, (radius, radius, 0, 0), mode="reflect")
    y = F.conv2d(y, k_row)
    # Vertical pass
    y = F.pad(y, (0, 0, radius, radius), mode="reflect")
    y = F.conv2d(y, k_col)
    return y.squeeze(1)


class GPUDepthMetricsBatcher:
    """Batch PSNR + SSIM + AbsRel + RMSE + SILog on GPU with a shared mask.

    Each enqueued entry becomes one callback invocation carrying a dict:

    .. code-block:: python

        {
            "psnr_val": float,         "psnr_meta": dict,
            "ssim_val": float,         "ssim_meta": dict,
            "absrel_arr": np.ndarray,  "absrel_meta": dict,
            "rmse_arr": np.ndarray,    # per-valid-pixel squared errors
            "silog_arr": np.ndarray,   # per-valid-pixel |log diff|
            "silog_full": float,       # SILog scalar
        }

    The ``*_meta`` dicts mirror the ``return_metadata=True`` contracts of
    the numpy functions so downstream sanity checks keep working.
    """

    def __init__(self, device: str, batch_size: int = 16):
        self.device = torch.device(device)
        self.batch_size = max(1, int(batch_size))
        self._pending: list = []

    @staticmethod
    def is_available(device: str) -> bool:
        dev = torch.device(device)
        if dev.type == "cuda":
            return torch.cuda.is_available()
        if dev.type == "mps":
            return torch.backends.mps.is_available()
        return False

    def enqueue(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        mask: Optional[np.ndarray],
        callback: Callable[[dict], None],
    ) -> None:
        """Enqueue one ``(pred, gt, mask)`` triple for batched GPU computation.

        ``mask`` applies to all five metrics (PSNR, SSIM, AbsRel, RMSE,
        SILog). Passing ``None`` falls back to the default-derived mask
        ``pred>0 & gt>0 & finite`` used by the numpy reference. Note
        that this unifies mask semantics across metrics; in the prior
        CPU-only path ``compute_ssim`` was called without a mask, so
        activating the batcher with a sky-filtered mask now also
        sky-filters SSIM's normalization range — this is more internally
        consistent and matches how every other depth metric in the
        pipeline treats the mask.
        """
        self._pending.append((pred, gt, mask, callback))
        if len(self._pending) >= self.batch_size:
            self._flush()

    def finalize(self) -> None:
        self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return

        shape_groups: dict = {}
        for item in self._pending:
            key = (item[0].shape[0], item[0].shape[1])
            shape_groups.setdefault(key, []).append(item)

        for items in shape_groups.values():
            try:
                self._compute_group(items)
            except Exception:
                # CPU fallback: one sample at a time through the numpy path.
                # Import lazily to avoid circular imports and to respect
                # monkeypatches used in tests.
                from . import (
                    absrel as _absrel_mod,
                    psnr as _psnr_mod,
                    rmse as _rmse_mod,
                    scale_invariant_log as _silog_mod,
                    ssim as _ssim_mod,
                )

                for pred, gt, mask, cb in items:
                    psnr_val, psnr_meta = _psnr_mod.compute_psnr(
                        pred, gt, valid_mask=mask, return_metadata=True
                    )
                    ssim_val, ssim_meta = _ssim_mod.compute_ssim(
                        pred, gt, valid_mask=mask, return_metadata=True
                    )
                    absrel_arr, absrel_meta = _absrel_mod.compute_absrel(
                        pred, gt, valid_mask=mask, return_metadata=True
                    )
                    rmse_arr = _rmse_mod.compute_rmse_per_pixel(
                        pred, gt, valid_mask=mask
                    )
                    silog_arr = _silog_mod.compute_silog_per_pixel(
                        pred, gt, valid_mask=mask
                    )
                    silog_val = _silog_mod.compute_scale_invariant_log_error(
                        pred, gt, valid_mask=mask
                    )
                    cb(
                        {
                            "psnr_val": psnr_val,
                            "psnr_meta": psnr_meta,
                            "ssim_val": ssim_val,
                            "ssim_meta": ssim_meta,
                            "absrel_arr": absrel_arr,
                            "absrel_meta": absrel_meta,
                            "rmse_arr": rmse_arr,
                            "silog_arr": silog_arr,
                            "silog_full": silog_val,
                        }
                    )

        self._pending.clear()

    def _compute_group(self, items: list) -> None:
        B = len(items)
        # Stack on CPU then transfer blocking — MPS non_blocking=True can
        # alias successive host→device copies.
        preds = torch.stack(
            [torch.from_numpy(np.ascontiguousarray(p)).float() for p, _, _, _ in items]
        ).to(self.device)
        gts = torch.stack(
            [torch.from_numpy(np.ascontiguousarray(g)).float() for _, g, _, _ in items]
        ).to(self.device)
        # Mask may be None for a sample (matches numpy semantics: derive
        # default valid_mask internally from pred/gt > 0 and finite).
        masks_np = []
        for pred, gt, mask, _ in items:
            if mask is None:
                m = (
                    (gt > 0)
                    & (pred > 0)
                    & np.isfinite(gt)
                    & np.isfinite(pred)
                )
            else:
                m = mask
            masks_np.append(np.ascontiguousarray(m).astype(bool))
        masks = torch.stack([torch.from_numpy(m) for m in masks_np]).to(self.device)
        mask_f = masks.to(preds.dtype)

        dtype = preds.dtype
        neg_inf = torch.tensor(float("-inf"), device=self.device, dtype=dtype)
        pos_inf = torch.tensor(float("inf"), device=self.device, dtype=dtype)

        # Per-sample valid pixel counts; clamp to 1 for divisions so
        # samples with no valid pixels don't produce NaNs before we
        # overwrite them below.
        valid_count = mask_f.sum(dim=(1, 2))  # (B,)
        any_valid = valid_count > 0
        safe_count = valid_count.clamp(min=1.0)

        # -- PSNR --
        gt_masked_pinf = torch.where(masks, gts, pos_inf)
        gt_masked_ninf = torch.where(masks, gts, neg_inf)
        gt_min = gt_masked_pinf.amin(dim=(1, 2))
        gt_max = gt_masked_ninf.amax(dim=(1, 2))
        diff = preds - gts
        sq = diff * diff
        mse = (sq * mask_f).sum(dim=(1, 2)) / safe_count
        # Match numpy: mse < 1e-10 -> inf; else 10 * log10(max_val^2 / mse)
        zero_mse = mse < 1e-10
        safe_mse = torch.where(zero_mse, torch.ones_like(mse), mse)
        psnr = 10.0 * torch.log10((gt_max * gt_max) / safe_mse)
        psnr = torch.where(
            zero_mse,
            torch.full_like(psnr, float("inf")),
            psnr,
        )
        psnr = torch.where(any_valid, psnr, torch.zeros_like(psnr))

        # -- SSIM --
        pred_masked_pinf = torch.where(masks, preds, pos_inf)
        pred_masked_ninf = torch.where(masks, preds, neg_inf)
        pred_min = pred_masked_pinf.amin(dim=(1, 2))
        pred_max = pred_masked_ninf.amax(dim=(1, 2))
        valid_min = torch.minimum(pred_min, gt_min)
        valid_max = torch.maximum(pred_max, gt_max)
        depth_range = valid_max - valid_min
        # Numpy path returns ssim=1.0 when depth_range < 1e-8 (constant images)
        constant = depth_range < 1e-8
        safe_range = torch.where(
            constant, torch.ones_like(depth_range), depth_range
        )

        norm_pred = (preds - valid_min[:, None, None]) / safe_range[:, None, None]
        norm_gt = (gts - valid_min[:, None, None]) / safe_range[:, None, None]
        norm_pred = norm_pred * mask_f
        norm_gt = norm_gt * mask_f

        kernel = _gaussian_kernel_1d(
            _SSIM_WINDOW, _SSIM_SIGMA, self.device, dtype
        )

        def gf(x: torch.Tensor) -> torch.Tensor:
            return _gaussian_filter_2d(x, kernel)

        mu1 = gf(norm_pred)
        mu2 = gf(norm_gt)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = gf(norm_pred * norm_pred) - mu1_sq
        sigma2_sq = gf(norm_gt * norm_gt) - mu2_sq
        sigma12 = gf(norm_pred * norm_gt) - mu1_mu2

        c1 = _SSIM_K1 * _SSIM_K1  # (k1*L)^2, L = 1
        c2 = _SSIM_K2 * _SSIM_K2
        num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
        den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        ssim_map = num / den

        weight = gf(mask_f) > 0.5
        weight_f = weight.to(dtype)
        weight_sum = weight_f.sum(dim=(1, 2))
        safe_w = weight_sum.clamp(min=1.0)
        ssim_vals = (ssim_map * weight_f).sum(dim=(1, 2)) / safe_w
        ssim_vals = torch.where(
            weight_sum > 0, ssim_vals, torch.zeros_like(ssim_vals)
        )
        # Constant-image fast path and empty-mask sentinel
        ssim_vals = torch.where(
            constant, torch.ones_like(ssim_vals), ssim_vals
        )
        ssim_vals = torch.where(
            any_valid, ssim_vals, torch.zeros_like(ssim_vals)
        )

        # -- AbsRel, RMSE, SILog per-pixel (and SILog scalar) --
        # Safe divisions/logs; invalid pixels are masked out when we extract.
        safe_gt = torch.where(gts > 0, gts, torch.ones_like(gts))
        absrel_map = torch.abs(diff) / safe_gt
        rmse_map = sq  # squared errors
        safe_pred_for_log = torch.where(preds > 0, preds, torch.ones_like(preds))
        safe_gt_for_log = torch.where(gts > 0, gts, torch.ones_like(gts))
        log_diff = torch.log(safe_pred_for_log) - torch.log(safe_gt_for_log)
        silog_pp_map = torch.abs(log_diff)

        # SILog scalar: sqrt(mean(d^2) - mean(d)^2) over valid pixels
        ld_masked = log_diff * mask_f
        ld_sq_masked = log_diff * log_diff * mask_f
        mean_d = ld_masked.sum(dim=(1, 2)) / safe_count
        mean_d_sq = ld_sq_masked.sum(dim=(1, 2)) / safe_count
        var_like = (mean_d_sq - mean_d * mean_d).clamp(min=0.0)
        silog_full = torch.sqrt(var_like)
        silog_full = torch.where(
            any_valid, silog_full, torch.full_like(silog_full, float("nan"))
        )

        # Pull scalars to CPU in one sync; per-sample flat arrays follow below.
        psnr_cpu = psnr.detach().cpu().tolist()
        ssim_cpu = ssim_vals.detach().cpu().tolist()
        silog_full_cpu = silog_full.detach().cpu().tolist()
        gt_max_cpu = gt_max.detach().cpu().tolist()
        gt_min_cpu = gt_min.detach().cpu().tolist()
        valid_min_cpu = valid_min.detach().cpu().tolist()
        valid_max_cpu = valid_max.detach().cpu().tolist()
        depth_range_cpu = depth_range.detach().cpu().tolist()
        valid_count_cpu = valid_count.detach().cpu().tolist()
        any_valid_cpu = any_valid.detach().cpu().tolist()

        # Per-sample flat arrays: gather valid pixels (still on GPU), copy to CPU.
        absrel_flats = [absrel_map[i][masks[i]] for i in range(B)]
        rmse_flats = [rmse_map[i][masks[i]] for i in range(B)]
        silog_flats = [silog_pp_map[i][masks[i]] for i in range(B)]

        for i, (_pred, _gt, _mask, cb) in enumerate(items):
            if not any_valid_cpu[i]:
                psnr_meta = {
                    "max_val_used": None,
                    "valid_pixel_count": 0,
                    "depth_gt_max": None,
                    "depth_gt_min": None,
                }
                ssim_meta = {
                    "valid_pixel_count": 0,
                    "depth_range": None,
                    "depth_min": None,
                    "depth_max": None,
                }
                absrel_meta = {
                    "valid_pixel_count": 0,
                    "median": None,
                    "p90": None,
                }
                cb(
                    {
                        "psnr_val": 0.0,
                        "psnr_meta": psnr_meta,
                        "ssim_val": 0.0,
                        "ssim_meta": ssim_meta,
                        "absrel_arr": np.array([]),
                        "absrel_meta": absrel_meta,
                        "rmse_arr": np.array([]),
                        "silog_arr": np.array([]),
                        "silog_full": float("nan"),
                    }
                )
                continue

            absrel_arr = absrel_flats[i].detach().cpu().numpy()
            rmse_arr = rmse_flats[i].detach().cpu().numpy()
            silog_arr = silog_flats[i].detach().cpu().numpy()

            psnr_meta = {
                "max_val_used": float(gt_max_cpu[i]),
                "valid_pixel_count": int(valid_count_cpu[i]),
                "depth_gt_max": float(gt_max_cpu[i]),
                "depth_gt_min": float(gt_min_cpu[i]),
            }
            ssim_meta = {
                "valid_pixel_count": int(valid_count_cpu[i]),
                "depth_range": float(depth_range_cpu[i]),
                "depth_min": float(valid_min_cpu[i]),
                "depth_max": float(valid_max_cpu[i]),
            }
            absrel_meta = {
                "valid_pixel_count": int(valid_count_cpu[i]),
                "median": float(np.median(absrel_arr)) if absrel_arr.size else None,
                "p90": float(np.percentile(absrel_arr, 90)) if absrel_arr.size else None,
            }

            cb(
                {
                    "psnr_val": float(psnr_cpu[i]),
                    "psnr_meta": psnr_meta,
                    "ssim_val": float(ssim_cpu[i]),
                    "ssim_meta": ssim_meta,
                    "absrel_arr": absrel_arr,
                    "absrel_meta": absrel_meta,
                    "rmse_arr": rmse_arr,
                    "silog_arr": silog_arr,
                    "silog_full": float(silog_full_cpu[i]),
                }
            )
