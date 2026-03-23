"""Tests for RGB output structure with dataset-level FID."""

import numpy as np
import pytest

import euler_eval.evaluate as eval_mod


class _DummyRGBDataset:
    def __init__(self, samples):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]

    def modality_paths(self):
        return {"gt": "gt", "pred": "pred"}


class _DummyLPIPS:
    def __init__(self, device="cpu"):
        self.device = device

    def compute(self, pred, gt):
        return float(np.mean(np.abs(pred - gt)))


def _make_dataset():
    base = np.linspace(0.0, 1.0, 8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)
    gt_a = base
    pred_a = np.clip(base * 0.9, 0.0, 1.0)
    gt_b = np.flip(base, axis=1).copy()
    pred_b = np.clip(gt_b * 0.8 + 0.05, 0.0, 1.0)
    return _DummyRGBDataset(
        [
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
    )


def test_rgb_output_contains_dataset_fid(monkeypatch):
    calls = []

    class _DummyFIDKID:
        def __init__(self, device="cpu"):
            self.device = device

        def compute_rgb_fid(self, all_gt, all_pred, batch_size, num_workers):
            calls.append((len(all_gt), len(all_pred), batch_size, num_workers))
            return 0.123

    monkeypatch.setattr(eval_mod, "RGBLPIPSMetric", _DummyLPIPS)
    monkeypatch.setattr(eval_mod, "FIDKIDMetric", _DummyFIDKID)
    monkeypatch.setattr(eval_mod, "tqdm", lambda x, desc=None: x)

    results = eval_mod.evaluate_rgb_samples(
        dataset=_make_dataset(),
        device="cpu",
    )

    assert results["rgb"]["image_quality"]["fid"] == pytest.approx(0.123)
    assert "fid" not in results["per_file_metrics"]["children"]["Scene01"]["children"][
        "clone"
    ]["files"][0]["metrics"]["rgb"]["image_quality"]
    assert calls == [(2, 2, 16, 4)]


def test_rgb_clean_fid_backend_bypasses_builtin_metric(monkeypatch):
    calls = []

    class _BuiltinShouldNotBeUsed:
        def __init__(self, device="cpu"):
            raise AssertionError("builtin FID backend should not be initialized")

    def _fake_clean_fid(
        all_gt,
        all_pred,
        *,
        mode,
        batch_size,
        num_workers,
        device,
        verbose,
    ):
        calls.append((len(all_gt), len(all_pred), mode, batch_size, num_workers, device))
        return 0.456

    monkeypatch.setattr(eval_mod, "RGBLPIPSMetric", _DummyLPIPS)
    monkeypatch.setattr(eval_mod, "FIDKIDMetric", _BuiltinShouldNotBeUsed)
    monkeypatch.setattr(eval_mod, "compute_clean_fid", _fake_clean_fid)
    monkeypatch.setattr(eval_mod, "tqdm", lambda x, desc=None: x)

    results = eval_mod.evaluate_rgb_samples(
        dataset=_make_dataset(),
        device="cpu",
        fid_backend="clean-fid",
    )

    assert results["rgb"]["image_quality"]["fid"] == pytest.approx(0.456)
    assert calls == [(2, 2, "clean", 16, 4, "cpu")]
