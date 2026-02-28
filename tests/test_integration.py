"""Integration test using euler_loading's MultiModalDataset with stub data files.

Creates a complete on-disk dataset structure with output.json manifests
and tiny stub files, then runs through the full pipeline:
  build dataset → extract metadata → iterate samples → convert → process
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from PIL import Image

from euler_loading import Modality, MultiModalDataset

from euler_eval.data import (
    get_depth_metadata,
    get_rgb_metadata,
    process_depth,
    to_numpy_depth,
    to_numpy_intrinsics,
    to_numpy_mask,
    to_numpy_rgb,
)
from euler_eval.evaluate import _extract_hierarchy, _get_intrinsics_K, _get_sky_mask


# ---------------------------------------------------------------------------
# Constants for the stub data
# ---------------------------------------------------------------------------

H, W = 4, 6
NUM_FILES = 3
SCALE_TO_METERS = 0.01
K_MATRIX = np.array(
    [[525.0, 0.0, 319.5], [0.0, 525.0, 239.5], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Simple test loaders (match the euler_loading loader callable signature)
# ---------------------------------------------------------------------------


def _load_depth(path: str, meta=None) -> np.ndarray:
    return np.load(path).astype(np.float32)


def _load_rgb(path: str, meta=None) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _load_sky_mask(path: str, meta=None) -> np.ndarray:
    return np.load(path).astype(bool)


def _load_intrinsics(path: str, meta=None) -> np.ndarray:
    return np.load(path).astype(np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_output_json(directory: Path, output_dict: dict) -> None:
    """Write output.json at the root of a modality directory."""
    path = directory / "output.json"
    with open(path, "w") as f:
        json.dump(output_dict, f)


def _make_files_list(prefix: str, ext: str, n: int) -> list[dict]:
    return [
        {"path": f"Scene01/clone/{prefix}_{i:05d}{ext}", "id": f"{i:05d}"}
        for i in range(n)
    ]


def _build_output_json(
    name: str, dtype: str, files: list[dict], meta: Optional[dict] = None
) -> dict:
    out: dict = {"name": name, "type": dtype}
    if meta:
        out["meta"] = meta
    out["dataset"] = {
        "children": {"Scene01": {"children": {"clone": {"files": files}}}}
    }
    return out


def _build_hierarchical_output_json(
    name: str, dtype: str, meta: Optional[dict] = None
) -> dict:
    """Calibration lives at the scene level, not file level."""
    out: dict = {"name": name, "type": dtype}
    if meta:
        out["meta"] = meta
    out["dataset"] = {
        "children": {
            "Scene01": {
                "files": [{"path": "Scene01/intrinsics.npy", "id": "intrinsics"}]
            }
        }
    }
    return out


@pytest.fixture
def dataset_root(tmp_path):
    """Create a complete stub dataset tree under tmp_path.

    Layout::

        tmp_path/
          gt_depth/    output.json + Scene01/clone/depth_*.npy
          pred_depth/  output.json + Scene01/clone/depth_*.npy
          gt_rgb/      output.json + Scene01/clone/rgb_*.png
          pred_rgb/    output.json + Scene01/clone/rgb_*.png
          calibration/ output.json + Scene01/intrinsics.npy
          segmentation/output.json + Scene01/clone/seg_*.npy
    """
    np.random.seed(42)

    paths = {}

    # --- GT depth ---
    gt_depth_dir = tmp_path / "gt_depth"
    files_dir = gt_depth_dir / "Scene01" / "clone"
    files_dir.mkdir(parents=True)
    depth_files = _make_files_list("depth", ".npy", NUM_FILES)
    for entry in depth_files:
        arr = np.random.uniform(1000, 5000, size=(H, W)).astype(np.float32)
        np.save(gt_depth_dir / entry["path"], arr)
    _write_output_json(
        gt_depth_dir,
        _build_output_json(
            "test_depth",
            "depth",
            depth_files,
            meta={"scale_to_meters": SCALE_TO_METERS, "radial_depth": False},
        ),
    )
    paths["gt_depth"] = str(gt_depth_dir)

    # --- Pred depth ---
    pred_depth_dir = tmp_path / "pred_depth"
    files_dir = pred_depth_dir / "Scene01" / "clone"
    files_dir.mkdir(parents=True)
    for entry in depth_files:
        arr = np.random.uniform(1000, 5000, size=(H, W)).astype(np.float32)
        np.save(pred_depth_dir / entry["path"], arr)
    _write_output_json(
        pred_depth_dir,
        _build_output_json("test_pred_depth", "depth", depth_files),
    )
    paths["pred_depth"] = str(pred_depth_dir)

    # --- GT RGB ---
    gt_rgb_dir = tmp_path / "gt_rgb"
    files_dir = gt_rgb_dir / "Scene01" / "clone"
    files_dir.mkdir(parents=True)
    rgb_files = _make_files_list("rgb", ".png", NUM_FILES)
    for entry in rgb_files:
        img = Image.fromarray(np.random.randint(0, 255, size=(H, W, 3), dtype=np.uint8))
        img.save(gt_rgb_dir / entry["path"])
    _write_output_json(
        gt_rgb_dir,
        _build_output_json("test_rgb", "rgb", rgb_files, meta={"rgb_range": [0, 1]}),
    )
    paths["gt_rgb"] = str(gt_rgb_dir)

    # --- Pred RGB ---
    pred_rgb_dir = tmp_path / "pred_rgb"
    files_dir = pred_rgb_dir / "Scene01" / "clone"
    files_dir.mkdir(parents=True)
    for entry in rgb_files:
        img = Image.fromarray(np.random.randint(0, 255, size=(H, W, 3), dtype=np.uint8))
        img.save(pred_rgb_dir / entry["path"])
    _write_output_json(
        pred_rgb_dir,
        _build_output_json("test_pred_rgb", "rgb", rgb_files),
    )
    paths["pred_rgb"] = str(pred_rgb_dir)

    # --- Calibration (hierarchical: per-scene, not per-file) ---
    cal_dir = tmp_path / "calibration"
    (cal_dir / "Scene01").mkdir(parents=True)
    np.save(cal_dir / "Scene01" / "intrinsics.npy", K_MATRIX)
    _write_output_json(
        cal_dir,
        _build_hierarchical_output_json("test_calibration", "calibration"),
    )
    paths["calibration"] = str(cal_dir)

    # --- Segmentation ---
    seg_dir = tmp_path / "segmentation"
    files_dir = seg_dir / "Scene01" / "clone"
    files_dir.mkdir(parents=True)
    seg_files = _make_files_list("seg", ".npy", NUM_FILES)
    for entry in seg_files:
        # Top row is sky (True), bottom rows are not
        sky = np.zeros((H, W), dtype=bool)
        sky[0, :] = True
        np.save(seg_dir / entry["path"], sky)
    _write_output_json(
        seg_dir,
        _build_output_json(
            "test_seg", "segmentation", seg_files, meta={"sky_class_id": 29}
        ),
    )
    paths["segmentation"] = str(seg_dir)

    return paths


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestDepthDatasetIntegration:
    """Build a depth MultiModalDataset and verify the full pipeline."""

    def test_dataset_length(self, dataset_root):
        ds = MultiModalDataset(
            modalities={
                "gt": Modality(path=dataset_root["gt_depth"], loader=_load_depth),
                "pred": Modality(path=dataset_root["pred_depth"], loader=_load_depth),
            },
        )
        assert len(ds) == NUM_FILES

    def test_depth_metadata_from_dataset(self, dataset_root):
        ds = MultiModalDataset(
            modalities={
                "gt": Modality(path=dataset_root["gt_depth"], loader=_load_depth),
                "pred": Modality(path=dataset_root["pred_depth"], loader=_load_depth),
            },
        )
        meta = get_depth_metadata(ds)
        assert meta["scale_to_meters"] == 1.0
        assert meta["radial_depth"] is False

    def test_sample_structure(self, dataset_root):
        ds = MultiModalDataset(
            modalities={
                "gt": Modality(path=dataset_root["gt_depth"], loader=_load_depth),
                "pred": Modality(path=dataset_root["pred_depth"], loader=_load_depth),
            },
        )
        sample = ds[0]

        # Must have gt, pred, id, full_id, meta
        assert "gt" in sample
        assert "pred" in sample
        assert "id" in sample
        assert "full_id" in sample
        assert "meta" in sample

        # gt and pred are numpy arrays from loader
        assert isinstance(sample["gt"], np.ndarray)
        assert isinstance(sample["pred"], np.ndarray)

    def test_depth_conversion_pipeline(self, dataset_root):
        ds = MultiModalDataset(
            modalities={
                "gt": Modality(path=dataset_root["gt_depth"], loader=_load_depth),
                "pred": Modality(path=dataset_root["pred_depth"], loader=_load_depth),
            },
        )
        meta = get_depth_metadata(ds)
        sample = ds[0]

        depth_gt = to_numpy_depth(sample["gt"])
        assert depth_gt.shape == (H, W)
        assert depth_gt.dtype == np.float32

        # Raw values are already in meters (1000-5000 in this fixture)
        processed = process_depth(
            depth_gt,
            scale_to_meters=meta["scale_to_meters"],
            is_radial=meta["radial_depth"],
        )
        assert processed.dtype == np.float32
        # Without K, planar→radial is skipped, so values should stay unchanged.
        assert processed.max() <= 5000.0
        assert processed.min() >= 1000.0

    def test_hierarchy_extraction(self, dataset_root):
        ds = MultiModalDataset(
            modalities={
                "gt": Modality(path=dataset_root["gt_depth"], loader=_load_depth),
                "pred": Modality(path=dataset_root["pred_depth"], loader=_load_depth),
            },
        )
        sample = ds[0]
        hierarchy, file_id = _extract_hierarchy(sample)

        # full_id should be like /Scene01/clone/00000
        assert "Scene01" in hierarchy
        assert "clone" in hierarchy
        assert file_id in [f"{i:05d}" for i in range(NUM_FILES)]


class TestDepthWithCalibrationIntegration:
    """Depth dataset with hierarchical calibration modality."""

    def test_intrinsics_extraction(self, dataset_root):
        ds = MultiModalDataset(
            modalities={
                "gt": Modality(path=dataset_root["gt_depth"], loader=_load_depth),
                "pred": Modality(path=dataset_root["pred_depth"], loader=_load_depth),
            },
            hierarchical_modalities={
                "calibration": Modality(
                    path=dataset_root["calibration"], loader=_load_intrinsics
                ),
            },
        )
        sample = ds[0]

        # calibration should be a dict from hierarchical modality
        assert "calibration" in sample
        assert isinstance(sample["calibration"], dict)

        K = _get_intrinsics_K(sample)
        assert K is not None
        assert K.shape == (3, 3)
        assert K.dtype == np.float32
        np.testing.assert_array_equal(K, K_MATRIX)

    def test_process_depth_with_planar_to_radial(self, dataset_root):
        ds = MultiModalDataset(
            modalities={
                "gt": Modality(path=dataset_root["gt_depth"], loader=_load_depth),
                "pred": Modality(path=dataset_root["pred_depth"], loader=_load_depth),
            },
            hierarchical_modalities={
                "calibration": Modality(
                    path=dataset_root["calibration"], loader=_load_intrinsics
                ),
            },
        )
        meta = get_depth_metadata(ds)
        sample = ds[0]

        depth_gt = to_numpy_depth(sample["gt"])
        K = _get_intrinsics_K(sample)

        processed = process_depth(
            depth_gt,
            scale_to_meters=meta["scale_to_meters"],
            is_radial=meta["radial_depth"],  # False → triggers conversion
            intrinsics_K=K,
        )
        # After planar→radial, values should be >= the planar depth.
        scaled = depth_gt
        assert np.all(processed >= scaled - 1e-5)


class TestDepthWithSegmentationIntegration:
    """Depth dataset with segmentation for sky masking."""

    def test_sky_mask_extraction(self, dataset_root):
        ds = MultiModalDataset(
            modalities={
                "gt": Modality(path=dataset_root["gt_depth"], loader=_load_depth),
                "pred": Modality(path=dataset_root["pred_depth"], loader=_load_depth),
            },
            hierarchical_modalities={
                "segmentation": Modality(
                    path=dataset_root["segmentation"], loader=_load_sky_mask
                ),
            },
        )
        sample = ds[0]

        assert "segmentation" in sample
        valid_mask = _get_sky_mask(sample)
        assert valid_mask is not None
        assert valid_mask.shape == (H, W)
        assert valid_mask.dtype == bool

        # Top row was set as sky → inverted to False (invalid)
        assert not valid_mask[0, :].any()
        # Bottom rows are not sky → True (valid)
        assert valid_mask[1:, :].all()


class TestRgbDatasetIntegration:
    """Build an RGB MultiModalDataset and verify the pipeline."""

    def test_dataset_length(self, dataset_root):
        ds = MultiModalDataset(
            modalities={
                "gt": Modality(path=dataset_root["gt_rgb"], loader=_load_rgb),
                "pred": Modality(path=dataset_root["pred_rgb"], loader=_load_rgb),
            },
        )
        assert len(ds) == NUM_FILES

    def test_rgb_metadata_from_dataset(self, dataset_root):
        ds = MultiModalDataset(
            modalities={
                "gt": Modality(path=dataset_root["gt_rgb"], loader=_load_rgb),
                "pred": Modality(path=dataset_root["pred_rgb"], loader=_load_rgb),
            },
        )
        meta = get_rgb_metadata(ds)
        assert meta["rgb_range"] == [0, 1]

    def test_rgb_conversion_pipeline(self, dataset_root):
        ds = MultiModalDataset(
            modalities={
                "gt": Modality(path=dataset_root["gt_rgb"], loader=_load_rgb),
                "pred": Modality(path=dataset_root["pred_rgb"], loader=_load_rgb),
            },
        )
        sample = ds[0]

        img_gt = to_numpy_rgb(sample["gt"])
        assert img_gt.shape == (H, W, 3)
        assert img_gt.dtype == np.float32
        assert img_gt.min() >= 0.0
        assert img_gt.max() <= 1.0

    def test_rgb_with_depth_modality(self, dataset_root):
        """RGB dataset can include gt_depth as third modality."""
        ds = MultiModalDataset(
            modalities={
                "gt": Modality(path=dataset_root["gt_rgb"], loader=_load_rgb),
                "pred": Modality(path=dataset_root["pred_rgb"], loader=_load_rgb),
                "gt_depth": Modality(path=dataset_root["gt_depth"], loader=_load_depth),
            },
        )
        sample = ds[0]

        img_gt = to_numpy_rgb(sample["gt"])
        assert img_gt.shape == (H, W, 3)

        depth_gt = to_numpy_depth(sample["gt_depth"])
        assert depth_gt.shape == (H, W)

        # Both come from the same hierarchy, so IDs match
        assert "gt_depth" in ds.modality_paths()


class TestFullPipelineIntegration:
    """End-to-end: depth + calibration + segmentation all together."""

    def test_all_modalities_combined(self, dataset_root):
        ds = MultiModalDataset(
            modalities={
                "gt": Modality(path=dataset_root["gt_depth"], loader=_load_depth),
                "pred": Modality(path=dataset_root["pred_depth"], loader=_load_depth),
            },
            hierarchical_modalities={
                "calibration": Modality(
                    path=dataset_root["calibration"], loader=_load_intrinsics
                ),
                "segmentation": Modality(
                    path=dataset_root["segmentation"], loader=_load_sky_mask
                ),
            },
        )

        meta = get_depth_metadata(ds)
        assert meta["scale_to_meters"] == 1.0
        assert meta["radial_depth"] is False

        for i in range(len(ds)):
            sample = ds[i]

            # Convert depth
            depth_gt = to_numpy_depth(sample["gt"])
            depth_pred = to_numpy_depth(sample["pred"])
            assert depth_gt.shape == (H, W)
            assert depth_pred.shape == (H, W)

            # Extract intrinsics
            K = _get_intrinsics_K(sample)
            assert K is not None
            np.testing.assert_array_equal(K, K_MATRIX)

            # Process depth (planar→radial)
            processed_gt = process_depth(
                depth_gt, meta["scale_to_meters"], meta["radial_depth"], K
            )
            assert processed_gt.dtype == np.float32
            assert np.all(np.isfinite(processed_gt))

            # Extract sky mask
            valid_mask = _get_sky_mask(sample)
            assert valid_mask is not None
            assert valid_mask.shape == (H, W)
            # Top row is sky → excluded
            assert not valid_mask[0, :].any()
            assert valid_mask[1:, :].all()

            # Build combined mask
            combined = (processed_gt > 0) & np.isfinite(processed_gt) & valid_mask
            # Top row excluded by sky mask
            assert not combined[0, :].any()

            # Hierarchy extraction
            hierarchy, fid = _extract_hierarchy(sample)
            assert len(hierarchy) >= 1
            assert fid in [f"{j:05d}" for j in range(NUM_FILES)]
