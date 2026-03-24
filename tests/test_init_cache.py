"""Tests for the init_cache entrypoint helpers."""

import init_cache


def test_configure_torch_home_keeps_existing(monkeypatch):
    monkeypatch.setenv("TORCH_HOME", "/tmp/torch-cache")
    monkeypatch.setenv("HF_HOME", "/tmp/hf-cache")

    path, derived = init_cache._configure_torch_home()

    assert str(path) == "/tmp/torch-cache"
    assert derived is False


def test_configure_torch_home_derives_from_hf_home(monkeypatch):
    monkeypatch.delenv("TORCH_HOME", raising=False)
    monkeypatch.setenv("HF_HOME", "/tmp/hf-cache")

    path, derived = init_cache._configure_torch_home()

    assert str(path) == "/tmp/hf-cache/torch"
    assert derived is True
