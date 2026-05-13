"""Tests for euler-loading modality construction in dataset builders."""

from types import SimpleNamespace

from euler_eval import data


class _CapturedDataset:
    def __init__(self, *, modalities, hierarchical_modalities=None):
        self.modalities = modalities
        self.hierarchical_modalities = hierarchical_modalities or {}


def _install_captured_dataset(monkeypatch):
    monkeypatch.setattr(data, "MultiModalDataset", _CapturedDataset)


def _assert_modality(modality, *, key, split=None, used_as=None, loader=None):
    assert modality.metadata_scope == key
    assert modality.modality_type == key
    assert modality.split == split
    assert modality.used_as == used_as
    if loader is not None:
        assert modality.loader is loader


def test_depth_builder_sets_modality_scopes(monkeypatch):
    _install_captured_dataset(monkeypatch)

    def sky_loader(path, meta=None):
        return None

    monkeypatch.setattr(
        data,
        "_resolve_sky_mask_loader",
        lambda path: sky_loader,
    )

    dataset = data.build_depth_eval_dataset(
        gt_depth_path="/datasets/shared",
        pred_depth_path="/predictions/shared",
        calibration_path="/datasets/calibration",
        segmentation_path="/datasets/segmentation",
        gt_depth_split="test",
        pred_depth_split="val",
        calibration_split="calib",
        segmentation_split="seg",
    )

    _assert_modality(dataset.modalities["gt"], key="depth", split="test")
    _assert_modality(
        dataset.modalities["pred"],
        key="depth",
        split="val",
        used_as="output",
    )
    _assert_modality(
        dataset.hierarchical_modalities["calibration"],
        key="calibration",
        split="calib",
    )
    _assert_modality(
        dataset.hierarchical_modalities["segmentation"],
        key="segmentation",
        split="seg",
        loader=sky_loader,
    )


def test_sparse_depth_builder_sets_projection_modality_scopes(monkeypatch):
    _install_captured_dataset(monkeypatch)

    dataset = data.build_sparse_depth_eval_dataset(
        gt_sparse_depth_path="/datasets/shared",
        pred_depth_path="/predictions/shared",
        intrinsics_path="/datasets/shared",
        camera_extrinsics_path="/datasets/shared",
        gt_sparse_depth_split="test",
        pred_depth_split="val",
        intrinsics_split="cam",
        camera_extrinsics_split="pose",
    )

    _assert_modality(dataset.modalities["gt"], key="sparse_depth", split="test")
    _assert_modality(
        dataset.modalities["pred"],
        key="depth",
        split="val",
        used_as="output",
    )
    _assert_modality(
        dataset.hierarchical_modalities["intrinsics"],
        key="intrinsics",
        split="cam",
    )
    _assert_modality(
        dataset.hierarchical_modalities["camera_extrinsics"],
        key="camera_extrinsics",
        split="pose",
    )


def test_rgb_builder_sets_rgb_and_auxiliary_modality_scopes(monkeypatch):
    _install_captured_dataset(monkeypatch)

    dataset = data.build_rgb_eval_dataset(
        gt_rgb_path="/datasets/shared",
        pred_rgb_path="/predictions/shared",
        gt_depth_path="/datasets/shared",
        gt_rgb_split="test",
        pred_rgb_split="val",
        gt_depth_split="depth",
    )

    _assert_modality(dataset.modalities["gt"], key="rgb", split="test")
    _assert_modality(
        dataset.modalities["pred"],
        key="rgb",
        split="val",
        used_as="output",
    )
    _assert_modality(dataset.modalities["gt_depth"], key="depth", split="depth")


def test_rays_builder_sets_rays_modality_scopes(monkeypatch):
    _install_captured_dataset(monkeypatch)

    dataset = data.build_rays_eval_dataset(
        gt_rays_path="/datasets/shared",
        pred_rays_path="/predictions/shared",
        calibration_path="/datasets/calibration",
        gt_rays_split="test",
        pred_rays_split="val",
        calibration_split="calib",
    )

    _assert_modality(dataset.modalities["gt"], key="rays", split="test")
    _assert_modality(
        dataset.modalities["pred"],
        key="rays",
        split="val",
        used_as="output",
    )
    _assert_modality(
        dataset.hierarchical_modalities["calibration"],
        key="calibration",
        split="calib",
    )


def test_sky_mask_loader_resolution_uses_segmentation_scope(monkeypatch):
    calls = []

    def fake_index_dataset_from_path(path, **kwargs):
        calls.append((path, kwargs))
        return {"euler_loading": {"loader": "vkitti2"}}

    def sky_mask(path, meta=None):
        return None

    monkeypatch.setattr(data, "index_dataset_from_path", fake_index_dataset_from_path)
    monkeypatch.setattr(
        data,
        "resolve_loader_module",
        lambda name: SimpleNamespace(sky_mask=sky_mask),
    )

    assert data._resolve_sky_mask_loader("/datasets/shared") is sky_mask
    assert calls == [
        ("/datasets/shared", {"metadata_scope": "segmentation"}),
    ]


def test_sky_mask_loader_resolution_strips_inline_split(monkeypatch):
    calls = []

    def fake_index_dataset_from_path(path, **kwargs):
        calls.append((path, kwargs))
        return {"euler_loading": {"loader": "vkitti2"}}

    def sky_mask(path, meta=None):
        return None

    monkeypatch.setattr(data, "index_dataset_from_path", fake_index_dataset_from_path)
    monkeypatch.setattr(
        data,
        "resolve_loader_module",
        lambda name: SimpleNamespace(sky_mask=sky_mask),
    )

    assert data._resolve_sky_mask_loader("/datasets/shared.zip:fog_day") is sky_mask
    assert calls == [
        ("/datasets/shared.zip", {"metadata_scope": "segmentation"}),
    ]
