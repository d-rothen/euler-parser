"""Tests for evaluate.py helper functions: sky mask, intrinsics, hierarchy."""

import numpy as np
import pytest
import torch

from euler_eval.evaluate import _extract_hierarchy, _get_intrinsics_K, _get_sky_mask


# ---------------------------------------------------------------------------
# _get_sky_mask
# ---------------------------------------------------------------------------


class TestGetSkyMask:
    def test_returns_none_when_no_segmentation(self):
        sample = {"gt": None, "pred": None}
        assert _get_sky_mask(sample) is None

    def test_returns_none_when_segmentation_is_empty_dict(self):
        sample = {"segmentation": {}}
        assert _get_sky_mask(sample) is None

    def test_from_hierarchical_dict(self):
        """Segmentation from hierarchical modality is a dict {file_id: mask}."""
        sky = np.array([[True, False], [False, True]], dtype=bool)
        sample = {"segmentation": {"scene_mask": sky}}
        result = _get_sky_mask(sample)
        assert result is not None
        # Inverted: True where NOT sky
        expected = ~sky
        np.testing.assert_array_equal(result, expected)

    def test_from_direct_array(self):
        """Segmentation as a direct numpy array (non-hierarchical)."""
        sky = np.array([[True, True], [False, False]], dtype=bool)
        sample = {"segmentation": sky}
        result = _get_sky_mask(sample)
        expected = ~sky
        np.testing.assert_array_equal(result, expected)

    def test_from_tensor_1hw(self):
        """Segmentation as a (1, H, W) tensor from GPU loader."""
        sky_np = np.array([[True, False], [True, False]], dtype=bool)
        sky_tensor = torch.from_numpy(sky_np).unsqueeze(0)  # (1, 2, 2)
        sample = {"segmentation": {"mask_id": sky_tensor}}
        result = _get_sky_mask(sample)
        expected = ~sky_np
        np.testing.assert_array_equal(result, expected)

    def test_all_sky(self):
        """All-sky mask should return all-False valid mask."""
        sky = np.ones((3, 3), dtype=bool)
        sample = {"segmentation": sky}
        result = _get_sky_mask(sample)
        assert not result.any()

    def test_no_sky(self):
        """No-sky mask should return all-True valid mask."""
        sky = np.zeros((3, 3), dtype=bool)
        sample = {"segmentation": sky}
        result = _get_sky_mask(sample)
        assert result.all()

    def test_uses_first_dict_entry(self):
        """When multiple entries, picks the first (iter order)."""
        sky_a = np.array([[True, False]], dtype=bool)
        sky_b = np.array([[False, True]], dtype=bool)
        sample = {"segmentation": {"a": sky_a, "b": sky_b}}
        result = _get_sky_mask(sample)
        # Should use sky_a (first in dict)
        np.testing.assert_array_equal(result, ~sky_a)


# ---------------------------------------------------------------------------
# _get_intrinsics_K
# ---------------------------------------------------------------------------


class TestGetIntrinsicsK:
    def test_returns_none_when_no_calibration(self):
        sample = {"gt": None, "pred": None}
        assert _get_intrinsics_K(sample) is None

    def test_returns_none_when_calibration_is_empty_dict(self):
        sample = {"calibration": {}}
        assert _get_intrinsics_K(sample) is None

    def test_from_hierarchical_dict_numpy(self, sample_K):
        sample = {"calibration": {"intrinsics": sample_K}}
        result = _get_intrinsics_K(sample)
        assert result is not None
        assert result.shape == (3, 3)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, sample_K)

    def test_from_hierarchical_dict_tensor(self, sample_K):
        K_tensor = torch.from_numpy(sample_K)
        sample = {"calibration": {"intrinsics": K_tensor}}
        result = _get_intrinsics_K(sample)
        assert result is not None
        assert result.shape == (3, 3)
        np.testing.assert_array_equal(result, sample_K)

    def test_from_direct_array(self, sample_K):
        sample = {"calibration": sample_K}
        result = _get_intrinsics_K(sample)
        assert result is not None
        np.testing.assert_array_equal(result, sample_K)

    def test_extracts_correct_values(self):
        K = np.array([
            [700.0, 0.0, 640.0],
            [0.0, 700.0, 480.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        sample = {"calibration": {"cam0": K}}
        result = _get_intrinsics_K(sample)
        assert result[0, 0] == 700.0  # fx
        assert result[1, 1] == 700.0  # fy
        assert result[0, 2] == 640.0  # cx
        assert result[1, 2] == 480.0  # cy


# ---------------------------------------------------------------------------
# _extract_hierarchy
# ---------------------------------------------------------------------------


class TestExtractHierarchy:
    def test_deep_hierarchy(self):
        sample = {
            "id": "00042",
            "full_id": "/Scene06/clone/Camera_0/00042",
        }
        hierarchy, file_id = _extract_hierarchy(sample)
        assert hierarchy == ["Scene06", "clone", "Camera_0"]
        assert file_id == "00042"

    def test_single_level(self):
        sample = {
            "id": "frame_001",
            "full_id": "/images/frame_001",
        }
        hierarchy, file_id = _extract_hierarchy(sample)
        assert hierarchy == ["images"]
        assert file_id == "frame_001"

    def test_flat_no_hierarchy(self):
        sample = {
            "id": "frame_001",
            "full_id": "/frame_001",
        }
        hierarchy, file_id = _extract_hierarchy(sample)
        assert hierarchy == []
        assert file_id == "frame_001"

    def test_missing_full_id_falls_back(self):
        sample = {"id": "test_file"}
        hierarchy, file_id = _extract_hierarchy(sample)
        assert hierarchy == []
        assert file_id == "test_file"

    def test_two_level_hierarchy(self):
        sample = {
            "id": "00000",
            "full_id": "/Scene01/clone/00000",
        }
        hierarchy, file_id = _extract_hierarchy(sample)
        assert hierarchy == ["Scene01", "clone"]
        assert file_id == "00000"

    def test_trailing_slash_ignored(self):
        sample = {
            "id": "00001",
            "full_id": "/Scene01/clone/00001/",
        }
        hierarchy, file_id = _extract_hierarchy(sample)
        assert hierarchy == ["Scene01", "clone"]
        assert file_id == "00001"
