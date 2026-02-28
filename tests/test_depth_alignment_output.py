"""Tests for depth raw/aligned output structure and alignment behavior."""

import numpy as np

import euler_eval.evaluate as eval_mod


class _DummyDepthDataset:
    def __init__(self, samples):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]


class _DummyLPIPS:
    def __init__(self, device="cpu"):
        self.device = device

    def compute(self, pred, gt):
        return float(np.mean(np.abs(pred - gt)))


class _DummyFIDKID:
    def __init__(self, device="cpu"):
        self.device = device

    def compute_fid(self, all_gt, all_pred, batch_size, num_workers):
        errs = [np.mean(np.abs(g - p)) for g, p in zip(all_gt, all_pred)]
        return float(np.mean(errs))

    def compute_kid(self, all_gt, all_pred, batch_size, num_workers):
        return 0.0, 0.0


def _flatten(values):
    parts = [v.reshape(-1) for v in values if len(v) > 0]
    if not parts:
        return np.array([], dtype=np.float32)
    return np.concatenate(parts)


def _patch_depth_metrics(monkeypatch):
    monkeypatch.setattr(eval_mod, "LPIPSMetric", _DummyLPIPS)
    monkeypatch.setattr(eval_mod, "FIDKIDMetric", _DummyFIDKID)
    monkeypatch.setattr(eval_mod, "tqdm", lambda x, desc=None: x)

    def _psnr(pred, gt, valid_mask=None, return_metadata=False):
        diff = np.abs(pred - gt)
        if valid_mask is not None:
            diff = diff[valid_mask]
        value = float(1.0 / (1.0 + np.mean(diff)))
        meta = {"max_val_used": 1.0}
        return (value, meta) if return_metadata else value

    def _ssim(pred, gt, return_metadata=False):
        diff = float(np.mean(np.abs(pred - gt)))
        value = max(0.0, 1.0 - diff)
        meta = {"depth_range": 1.0}
        return (value, meta) if return_metadata else value

    def _absrel(pred, gt, valid_mask=None, return_metadata=False):
        denom = np.maximum(np.abs(gt), 1e-6)
        rel = np.abs(pred - gt) / denom
        if valid_mask is not None:
            rel = rel[valid_mask]
        meta = {
            "median": float(np.median(rel)) if rel.size else None,
            "p90": float(np.percentile(rel, 90)) if rel.size else None,
        }
        return (
            (rel.astype(np.float32), meta)
            if return_metadata
            else rel.astype(np.float32)
        )

    def _rmse_per_pixel(pred, gt, valid_mask=None):
        sq = (pred - gt) ** 2
        if valid_mask is not None:
            sq = sq[valid_mask]
        return sq.astype(np.float32)

    def _silog_per_pixel(pred, gt, valid_mask=None):
        vals = np.abs(np.log(np.maximum(pred, 1e-6)) - np.log(np.maximum(gt, 1e-6)))
        if valid_mask is not None:
            vals = vals[valid_mask]
        return vals.astype(np.float32)

    def _silog_full(pred, gt, valid_mask=None):
        vals = _silog_per_pixel(pred, gt, valid_mask=valid_mask)
        return float(np.mean(vals)) if vals.size else float("nan")

    def _normal_angles(pred, gt, valid_mask=None, return_metadata=False):
        vals = np.abs(pred - gt).reshape(-1).astype(np.float32)
        if valid_mask is not None:
            vals = np.abs(pred - gt)[valid_mask].reshape(-1).astype(np.float32)
        meta = {
            "mean_angle": float(np.mean(vals)) if vals.size else None,
            "valid_pixels_after_erosion": int(vals.size),
        }
        return (vals, meta) if return_metadata else vals

    def _edge_f1(pred, gt, valid_mask=None):
        err = float(np.mean(np.abs(pred - gt)))
        f1 = max(0.0, 1.0 - err)
        return {
            "precision": f1,
            "recall": f1,
            "f1": f1,
            "pred_edge_pixels": 10,
            "gt_edge_pixels": 10,
            "total_pixels": int(pred.size),
        }

    def _agg(values):
        flat = _flatten(values)
        if flat.size == 0:
            return {"median": None, "p90": None}
        return {
            "median": float(np.median(flat)),
            "p90": float(np.percentile(flat, 90)),
        }

    def _agg_norm(values):
        flat = _flatten(values)
        if flat.size == 0:
            return {
                "mean_angle": None,
                "median_angle": None,
                "percent_below_11_25": None,
                "percent_below_22_5": None,
                "percent_below_30": None,
            }
        return {
            "mean_angle": float(np.mean(flat)),
            "median_angle": float(np.median(flat)),
            "percent_below_11_25": 100.0,
            "percent_below_22_5": 100.0,
            "percent_below_30": 100.0,
        }

    def _agg_edge(values):
        return {
            "precision": float(np.mean([v["precision"] for v in values])),
            "recall": float(np.mean([v["recall"] for v in values])),
            "f1": float(np.mean([v["f1"] for v in values])),
        }

    monkeypatch.setattr(eval_mod, "compute_psnr", _psnr)
    monkeypatch.setattr(eval_mod, "compute_ssim", _ssim)
    monkeypatch.setattr(eval_mod, "compute_absrel", _absrel)
    monkeypatch.setattr(eval_mod, "compute_rmse_per_pixel", _rmse_per_pixel)
    monkeypatch.setattr(eval_mod, "compute_silog_per_pixel", _silog_per_pixel)
    monkeypatch.setattr(eval_mod, "compute_scale_invariant_log_error", _silog_full)
    monkeypatch.setattr(eval_mod, "compute_normal_angles", _normal_angles)
    monkeypatch.setattr(eval_mod, "compute_depth_edge_f1", _edge_f1)
    monkeypatch.setattr(eval_mod, "aggregate_absrel", _agg)
    monkeypatch.setattr(eval_mod, "aggregate_rmse", _agg)
    monkeypatch.setattr(eval_mod, "aggregate_silog", _agg)
    monkeypatch.setattr(eval_mod, "aggregate_normal_consistency", _agg_norm)
    monkeypatch.setattr(eval_mod, "aggregate_edge_f1", _agg_edge)


def _make_dataset():
    gt_a = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    gt_b = np.array([[6.0, 12.0], [18.0, 24.0]], dtype=np.float32)
    pred_a = ((gt_a - 5.0) / 50.0).astype(np.float32)  # normalized
    pred_b = ((gt_b - 3.0) / 30.0).astype(np.float32)  # normalized
    samples = [
        {
            "id": "00001",
            "full_id": "/Scene01/clone/00001",
            "gt": gt_a,
            "pred": pred_a,
        },
        {
            "id": "00002",
            "full_id": "/Scene01/clone/00002",
            "gt": gt_b,
            "pred": pred_b,
        },
    ]
    return _DummyDepthDataset(samples)


def test_depth_output_contains_raw_and_aligned(monkeypatch):
    _patch_depth_metrics(monkeypatch)

    results = eval_mod.evaluate_depth_samples(
        dataset=_make_dataset(),
        is_radial=True,
        device="cpu",
        alignment_mode="auto_affine",
    )

    assert "depth_raw" in results
    assert "depth_aligned" in results
    assert "depth" in results

    raw_absrel = results["depth_raw"]["depth_metrics"]["absrel"]["median"]
    aligned_absrel = results["depth_aligned"]["depth_metrics"]["absrel"]["median"]
    assert aligned_absrel < raw_absrel
    assert results["depth_aligned"]["alignment"]["applied"] is True

    files = results["per_file_metrics"]["children"]["Scene01"]["children"]["clone"][
        "files"
    ]
    per_file = next(item["metrics"] for item in files if item["id"] == "00001")
    assert "depth" in per_file
    assert "depth_raw" in per_file
    assert "depth_aligned" in per_file
    assert (
        per_file["depth_aligned"]["depth_metrics"]["absrel"]
        < per_file["depth_raw"]["depth_metrics"]["absrel"]
    )


def test_depth_alignment_none_keeps_raw_and_aligned_equal(monkeypatch):
    _patch_depth_metrics(monkeypatch)

    results = eval_mod.evaluate_depth_samples(
        dataset=_make_dataset(),
        is_radial=True,
        device="cpu",
        alignment_mode="none",
    )

    raw = results["depth_raw"]["depth_metrics"]["absrel"]["median"]
    aligned = results["depth_aligned"]["depth_metrics"]["absrel"]["median"]
    assert raw == aligned
    assert results["depth_aligned"]["alignment"]["applied"] is False
