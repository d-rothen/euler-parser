"""Tests for CLI device resolution and runtime tuning helpers."""

from types import SimpleNamespace

import numpy as np

from euler_eval import cli
from euler_eval.metrics.fid_kid import FIDKIDMetric


class _DummyMatmul:
    allow_tf32 = False


class _DummyCudaBackend:
    matmul = _DummyMatmul()


class _DummyCudnn:
    benchmark = False
    allow_tf32 = False


def test_resolve_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: True)
    assert cli.resolve_device("auto") == "cuda"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
    assert cli.resolve_device("auto") == "cpu"


def test_resolve_device_explicit_cuda_falls_back(monkeypatch, capsys):
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
    assert cli.resolve_device("cuda") == "cpu"
    err = capsys.readouterr().err
    assert "Falling back to CPU" in err


def test_resolve_device_explicit_cpu_passthrough():
    assert cli.resolve_device("cpu") == "cpu"


def test_configure_torch_runtime_sets_cuda_flags(monkeypatch):
    dummy_backends = SimpleNamespace(
        cudnn=_DummyCudnn(),
        cuda=_DummyCudaBackend(),
    )
    monkeypatch.setattr(cli.torch, "backends", dummy_backends, raising=False)

    precision_calls: list[str] = []
    monkeypatch.setattr(
        cli.torch,
        "set_float32_matmul_precision",
        lambda mode: precision_calls.append(mode),
        raising=False,
    )

    cli.configure_torch_runtime("cuda")

    assert dummy_backends.cudnn.benchmark is True
    assert dummy_backends.cudnn.allow_tf32 is True
    assert dummy_backends.cuda.matmul.allow_tf32 is True
    assert precision_calls == ["high"]


def test_configure_torch_runtime_cpu_noop(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        cli.torch,
        "set_float32_matmul_precision",
        lambda mode: calls.append(mode),
        raising=False,
    )

    cli.configure_torch_runtime("cpu")

    assert calls == []


def test_fid_kid_feature_pair_cache_reuses_features(monkeypatch):
    metric = FIDKIDMetric.__new__(FIDKIDMetric)
    metric._cached_pair_key = None
    metric._cached_pair_features = None

    calls: list[tuple[int, int, int]] = []

    def _fake_extract(depths, batch_size, num_workers):
        calls.append((id(depths), batch_size, num_workers))
        value = float(len(calls))
        return np.array([[value]], dtype=np.float32)

    monkeypatch.setattr(metric, "_extract_features", _fake_extract)

    depths_a = [np.zeros((2, 2), dtype=np.float32)]
    depths_b = [np.ones((2, 2), dtype=np.float32)]

    first = metric._get_feature_pair(depths_a, depths_b, batch_size=16, num_workers=4)
    second = metric._get_feature_pair(depths_a, depths_b, batch_size=16, num_workers=4)

    assert len(calls) == 2  # one extraction per side, once
    assert first[0] is second[0]
    assert first[1] is second[1]


def test_resolve_depth_alignment_defaults_to_argument():
    assert cli.resolve_depth_alignment("auto_affine", None) == "auto_affine"
    assert cli.resolve_depth_alignment("affine", None) == "affine"


def test_resolve_depth_alignment_maps_sns_alias(capsys):
    assert cli.resolve_depth_alignment("auto_affine", True) == "auto_affine"
    assert cli.resolve_depth_alignment("auto_affine", False) == "none"
    err = capsys.readouterr().err
    assert "deprecated" in err


def test_resolve_depth_alignment_prefers_explicit_mode_over_sns(capsys):
    resolved = cli.resolve_depth_alignment("affine", False)
    assert resolved == "affine"
    err = capsys.readouterr().err
    assert "ignored" in err
