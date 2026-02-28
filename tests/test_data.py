"""Tests for euler_eval/data.py -- conversions, metadata resolution, process_depth."""

import numpy as np
import pytest
import torch

from euler_eval.data import (
    get_depth_metadata,
    get_rgb_metadata,
    process_depth,
    to_numpy_depth,
    to_numpy_intrinsics,
    to_numpy_mask,
    to_numpy_rgb,
)


# ---------------------------------------------------------------------------
# to_numpy_depth
# ---------------------------------------------------------------------------


class TestToNumpyDepth:
    def test_tensor_1hw(self):
        t = torch.rand(1, 4, 6)
        result = to_numpy_depth(t)
        assert result.shape == (4, 6)
        assert result.dtype == np.float32

    def test_tensor_hw(self):
        t = torch.rand(4, 6)
        result = to_numpy_depth(t)
        assert result.shape == (4, 6)
        assert result.dtype == np.float32

    def test_numpy_passthrough(self):
        arr = np.random.rand(4, 6).astype(np.float64)
        result = to_numpy_depth(arr)
        assert result.shape == (4, 6)
        assert result.dtype == np.float32

    def test_values_preserved(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        t = torch.from_numpy(arr).unsqueeze(0)  # (1, 2, 2)
        result = to_numpy_depth(t)
        np.testing.assert_array_equal(result, arr)

    def test_numpy_hwc_single_channel(self):
        arr = np.random.rand(4, 6, 1).astype(np.float32)
        result = to_numpy_depth(arr)
        assert result.shape == (4, 6)
        np.testing.assert_allclose(result, arr[..., 0], atol=1e-7)


# ---------------------------------------------------------------------------
# to_numpy_rgb
# ---------------------------------------------------------------------------


class TestToNumpyRgb:
    def test_tensor_chw_to_hwc(self):
        t = torch.rand(3, 4, 6)
        result = to_numpy_rgb(t)
        assert result.shape == (4, 6, 3)
        assert result.dtype == np.float32

    def test_tensor_hwc_passthrough(self):
        t = torch.rand(4, 6, 3)
        result = to_numpy_rgb(t)
        assert result.shape == (4, 6, 3)
        assert result.dtype == np.float32

    def test_numpy_hwc_passthrough(self):
        arr = np.random.rand(4, 6, 3).astype(np.float32)
        result = to_numpy_rgb(arr)
        assert result.shape == (4, 6, 3)
        np.testing.assert_array_equal(result, arr)

    def test_chw_values_transposed_correctly(self):
        # Create a CHW tensor with distinct channel values
        t = torch.zeros(3, 2, 2)
        t[0, :, :] = 0.1  # R
        t[1, :, :] = 0.2  # G
        t[2, :, :] = 0.3  # B
        result = to_numpy_rgb(t)
        np.testing.assert_allclose(result[:, :, 0], 0.1, atol=1e-6)
        np.testing.assert_allclose(result[:, :, 1], 0.2, atol=1e-6)
        np.testing.assert_allclose(result[:, :, 2], 0.3, atol=1e-6)

    def test_tensor_chw_with_width_three_still_transposes(self):
        t = torch.rand(3, 4, 3)
        result = to_numpy_rgb(t)
        assert result.shape == (4, 3, 3)
        expected = np.transpose(t.detach().cpu().numpy(), (1, 2, 0))
        np.testing.assert_allclose(result, expected, atol=1e-7)


# ---------------------------------------------------------------------------
# to_numpy_mask
# ---------------------------------------------------------------------------


class TestToNumpyMask:
    def test_tensor_1hw(self):
        t = torch.tensor([[[True, False], [False, True]]])
        result = to_numpy_mask(t)
        assert result.shape == (2, 2)
        assert result.dtype == bool
        assert result[0, 0] is np.True_
        assert result[0, 1] is np.False_

    def test_tensor_hw(self):
        t = torch.tensor([[True, False], [False, True]])
        result = to_numpy_mask(t)
        assert result.shape == (2, 2)
        assert result.dtype == bool

    def test_numpy_passthrough(self):
        arr = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        result = to_numpy_mask(arr)
        assert result.dtype == bool
        assert result[0, 0] is np.True_
        assert result[0, 1] is np.False_

    def test_integer_mask_converted_to_bool(self):
        arr = np.array([[29, 0], [0, 29]], dtype=np.int64)
        result = to_numpy_mask(arr)
        assert result.dtype == bool
        # Any nonzero is True
        assert result[0, 0] is np.True_
        assert result[0, 1] is np.False_

    def test_numpy_hwc_single_channel(self):
        arr = np.array([[[1], [0]], [[0], [1]]], dtype=np.uint8)
        result = to_numpy_mask(arr)
        assert result.shape == (2, 2)
        assert result.dtype == bool
        assert result[0, 0] is np.True_
        assert result[0, 1] is np.False_


# ---------------------------------------------------------------------------
# to_numpy_intrinsics
# ---------------------------------------------------------------------------


class TestToNumpyIntrinsics:
    def test_tensor_3x3(self, sample_K):
        t = torch.from_numpy(sample_K)
        result = to_numpy_intrinsics(t)
        assert result.shape == (3, 3)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, sample_K)

    def test_numpy_3x3(self, sample_K):
        result = to_numpy_intrinsics(sample_K)
        assert result.shape == (3, 3)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# process_depth
# ---------------------------------------------------------------------------


class TestProcessDepth:
    def test_scale_only(self):
        depth = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32)
        result = process_depth(depth, scale_to_meters=0.01, is_radial=True)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np.testing.assert_allclose(result, expected)

    def test_scale_identity(self):
        depth = np.array([[1.0, 2.0]], dtype=np.float32)
        result = process_depth(depth, scale_to_meters=1.0, is_radial=True)
        np.testing.assert_array_equal(result, depth)

    def test_radial_skips_conversion(self, sample_K):
        depth = np.ones((4, 6), dtype=np.float32) * 10.0
        result = process_depth(
            depth, scale_to_meters=1.0, is_radial=True, intrinsics_K=sample_K
        )
        # When radial=True, no conversion should happen
        np.testing.assert_array_equal(result, depth)

    def test_planar_to_radial_conversion(self, sample_K):
        depth = np.ones((4, 6), dtype=np.float32) * 10.0
        result = process_depth(
            depth, scale_to_meters=1.0, is_radial=False, intrinsics_K=sample_K
        )
        # Center pixel (cx=319.5, cy=239.5) is far from (4,6) image, so all
        # pixels have significant correction. Result should be >= input.
        assert result.dtype == np.float32
        assert np.all(result >= depth)
        # For a 4x6 image with cx=319.5, cy=239.5, correction is large
        assert np.all(result > depth)

    def test_planar_without_K_does_nothing(self):
        depth = np.ones((2, 2), dtype=np.float32) * 5.0
        result = process_depth(
            depth, scale_to_meters=1.0, is_radial=False, intrinsics_K=None
        )
        # No K provided, so planar-to-radial skipped even though is_radial=False
        np.testing.assert_array_equal(result, depth)

    def test_scale_and_convert_combined(self, sample_K):
        depth = np.ones((4, 6), dtype=np.float32) * 1000.0
        result = process_depth(
            depth, scale_to_meters=0.001, is_radial=False, intrinsics_K=sample_K
        )
        # After scaling: 1.0 everywhere, then planar→radial should increase
        assert np.all(result >= 1.0)


# ---------------------------------------------------------------------------
# get_depth_metadata / get_rgb_metadata
# ---------------------------------------------------------------------------


class TestGetDepthMetadata:
    def test_reads_from_output_json(self, mock_dataset, depth_index_output):
        ds = mock_dataset({"gt": depth_index_output})
        meta = get_depth_metadata(ds)
        assert meta["scale_to_meters"] == 0.01
        assert meta["radial_depth"] is False

    def test_defaults_when_no_meta(self, mock_dataset):
        ds = mock_dataset({"gt": {"dataset": {}}})
        meta = get_depth_metadata(ds)
        assert meta["scale_to_meters"] == 1.0
        assert meta["radial_depth"] is True

    def test_defaults_when_modality_missing(self, mock_dataset):
        ds = mock_dataset({})
        meta = get_depth_metadata(ds)
        assert meta["scale_to_meters"] == 1.0
        assert meta["radial_depth"] is True

    def test_partial_meta(self, mock_dataset):
        ds = mock_dataset({"gt": {"meta": {"scale_to_meters": 0.5}}})
        meta = get_depth_metadata(ds)
        assert meta["scale_to_meters"] == 0.5
        assert meta["radial_depth"] is True  # default


class TestGetRgbMetadata:
    def test_reads_from_output_json(self, mock_dataset, rgb_index_output):
        ds = mock_dataset({"gt": rgb_index_output})
        meta = get_rgb_metadata(ds)
        assert meta["rgb_range"] == [0, 1]

    def test_defaults_when_no_meta(self, mock_dataset):
        ds = mock_dataset({"gt": {"dataset": {}}})
        meta = get_rgb_metadata(ds)
        assert meta["rgb_range"] == [0.0, 1.0]

    def test_custom_rgb_range(self, mock_dataset):
        ds = mock_dataset({"gt": {"meta": {"rgb_range": [0, 255]}}})
        meta = get_rgb_metadata(ds)
        assert meta["rgb_range"] == [0, 255]
