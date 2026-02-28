"""Tests for dimension alignment and scale-and-shift depth alignment."""

import numpy as np
import pytest

from euler_eval.data import align_to_prediction, compute_scale_and_shift


# ---------------------------------------------------------------------------
# align_to_prediction
# ---------------------------------------------------------------------------


class TestAlignToPredicition:
    """Tests for VAE-crop-aware dimension alignment."""

    def test_same_dims_noop(self):
        gt = np.ones((64, 128), dtype=np.float32)
        pred = np.ones((64, 128), dtype=np.float32)
        result = align_to_prediction(gt, pred)
        assert result is gt  # exact same object, no copy

    def test_vae_crop_depth(self):
        """GT 375x1242 -> pred 368x1240 (multiple-of-8 crop)."""
        gt = np.random.rand(375, 1242).astype(np.float32)
        pred = np.random.rand(368, 1240).astype(np.float32)
        result = align_to_prediction(gt, pred)
        assert result.shape == (368, 1240)
        # Should be a top-left crop, not a resize
        np.testing.assert_array_equal(result, gt[:368, :1240])

    def test_vae_crop_rgb(self):
        """Same crop logic works for (H, W, 3) RGB arrays."""
        gt = np.random.rand(375, 1242, 3).astype(np.float32)
        pred = np.random.rand(368, 1240, 3).astype(np.float32)
        result = align_to_prediction(gt, pred)
        assert result.shape == (368, 1240, 3)
        np.testing.assert_array_equal(result, gt[:368, :1240])

    def test_vae_crop_bool_mask(self):
        """Bool masks are cropped identically."""
        gt = np.ones((375, 1242), dtype=bool)
        gt[370:, :] = False
        pred = np.zeros((368, 1240), dtype=np.float32)
        result = align_to_prediction(gt, pred)
        assert result.shape == (368, 1240)
        assert result.dtype == bool
        np.testing.assert_array_equal(result, gt[:368, :1240])

    def test_vae_crop_only_height(self):
        """Crop needed on height only (width already multiple of 8)."""
        gt = np.random.rand(133, 256).astype(np.float32)
        pred = np.random.rand(128, 256).astype(np.float32)
        result = align_to_prediction(gt, pred)
        assert result.shape == (128, 256)
        np.testing.assert_array_equal(result, gt[:128, :256])

    def test_vae_crop_only_width(self):
        """Crop needed on width only (height already multiple of 8)."""
        gt = np.random.rand(256, 133).astype(np.float32)
        pred = np.random.rand(256, 128).astype(np.float32)
        result = align_to_prediction(gt, pred)
        assert result.shape == (256, 128)
        np.testing.assert_array_equal(result, gt[:256, :128])

    def test_no_crop_when_delta_too_large(self):
        """Delta >= 8 should fall back to resize, not crop."""
        gt = np.random.rand(200, 200).astype(np.float32)
        pred = np.random.rand(184, 184).astype(np.float32)  # delta=16
        result = align_to_prediction(gt, pred)
        assert result.shape == (184, 184)
        # Should NOT be a simple slice (resize changes values)
        assert not np.array_equal(result, gt[:184, :184])

    def test_resize_fallback_preserves_dtype(self):
        """Resize fallback returns float32 for float32 input."""
        gt = np.random.rand(200, 300).astype(np.float32)
        pred = np.random.rand(100, 150).astype(np.float32)
        result = align_to_prediction(gt, pred)
        assert result.shape == (100, 150)
        assert result.dtype == np.float32

    def test_resize_fallback_bool_mask(self):
        """Bool masks use nearest interpolation and come back as bool."""
        gt = np.ones((200, 300), dtype=bool)
        gt[100:, :] = False
        pred = np.zeros((100, 150), dtype=np.float32)
        result = align_to_prediction(gt, pred)
        assert result.shape == (100, 150)
        assert result.dtype == bool

    def test_pred_not_multiple_of_8_uses_resize(self):
        """If pred dims aren't multiples of 8, can't be a VAE crop."""
        gt = np.random.rand(103, 205).astype(np.float32)
        pred = np.random.rand(99, 201).astype(np.float32)  # not mult of 8
        result = align_to_prediction(gt, pred)
        assert result.shape == (99, 201)

    def test_pred_larger_than_gt_uses_resize(self):
        """Edge case: pred larger than GT (negative delta) -> resize."""
        gt = np.random.rand(64, 64).astype(np.float32)
        pred = np.random.rand(128, 128).astype(np.float32)
        result = align_to_prediction(gt, pred)
        assert result.shape == (128, 128)


# ---------------------------------------------------------------------------
# compute_scale_and_shift
# ---------------------------------------------------------------------------


class TestComputeScaleAndShift:
    """Tests for least-squares affine alignment of depth maps."""

    def test_perfect_affine_recovery(self):
        """Recovers exact (s, t) when pred = (gt - t) / s."""
        gt = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        s_true, t_true = 50.0, 5.0
        pred = (gt - t_true) / s_true  # pred in ~[0.1, 0.7]

        aligned, s, t = compute_scale_and_shift(pred, gt)
        np.testing.assert_allclose(s, s_true, rtol=1e-5)
        np.testing.assert_allclose(t, t_true, rtol=1e-5)
        np.testing.assert_allclose(aligned, gt, rtol=1e-5)

    def test_identity_when_already_aligned(self):
        """If pred == gt, fitted params should be s=1, t=0."""
        gt = np.random.rand(16, 16).astype(np.float32) * 100 + 1
        aligned, s, t = compute_scale_and_shift(gt.copy(), gt)
        np.testing.assert_allclose(s, 1.0, atol=1e-4)
        np.testing.assert_allclose(t, 0.0, atol=1e-3)
        np.testing.assert_allclose(aligned, gt, rtol=1e-5)

    def test_normalized_01_to_metric(self):
        """Typical case: pred in [0, 1], GT in metres."""
        rng = np.random.RandomState(42)
        gt = rng.uniform(5.0, 80.0, size=(64, 64)).astype(np.float32)
        # Simulate model normalization
        pred = ((gt - gt.min()) / (gt.max() - gt.min())).astype(np.float32)

        aligned, s, t = compute_scale_and_shift(pred, gt)
        np.testing.assert_allclose(aligned, gt, rtol=1e-4)
        assert s > 0  # positive scale

    def test_normalized_neg1_pos1_to_metric(self):
        """Pred in [-1, 1], GT in metres."""
        rng = np.random.RandomState(42)
        gt = rng.uniform(5.0, 80.0, size=(64, 64)).astype(np.float32)
        pred = (2.0 * (gt - gt.min()) / (gt.max() - gt.min()) - 1.0).astype(
            np.float32
        )

        aligned, s, t = compute_scale_and_shift(pred, gt)
        np.testing.assert_allclose(aligned, gt, rtol=1e-4)

    def test_inverted_depth(self):
        """Model outputs inverted depth (far=1, near=0) -> negative scale."""
        gt = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
        pred = np.array([[1.0, 0.75, 0.5, 0.25]], dtype=np.float32)  # inverted

        aligned, s, t = compute_scale_and_shift(pred, gt)
        assert s < 0  # negative scale for inversion
        np.testing.assert_allclose(aligned, gt, rtol=1e-5)

    def test_valid_mask_excludes_pixels(self):
        """Only masked pixels contribute to the fit."""
        gt = np.array([[10.0, 20.0], [30.0, 999.0]], dtype=np.float32)
        pred = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        mask = np.array([[True, True], [True, False]], dtype=bool)

        aligned, s, t = compute_scale_and_shift(pred, gt, valid_mask=mask)
        # The fit should match the 3 valid pixels, not the outlier
        np.testing.assert_allclose(aligned[0, 0], gt[0, 0], rtol=1e-4)
        np.testing.assert_allclose(aligned[0, 1], gt[0, 1], rtol=1e-4)
        np.testing.assert_allclose(aligned[1, 0], gt[1, 0], rtol=1e-4)

    def test_sky_mask_excluded_from_fit(self):
        """Simulates sky pixels (large GT depth) excluded via mask."""
        rng = np.random.RandomState(123)
        gt = rng.uniform(5.0, 50.0, size=(32, 32)).astype(np.float32)
        # Top rows are "sky" with bogus depth
        gt[:8, :] = 1000.0

        pred = ((gt - 5.0) / 45.0).astype(np.float32)  # normalized
        pred[:8, :] = 0.99  # sky region has arbitrary pred values

        sky_valid = np.ones((32, 32), dtype=bool)
        sky_valid[:8, :] = False  # exclude sky

        fit_mask = (gt > 0) & np.isfinite(gt) & np.isfinite(pred) & sky_valid
        aligned, s, t = compute_scale_and_shift(pred, gt, valid_mask=fit_mask)

        # Non-sky pixels should be well-aligned
        non_sky = gt[8:, :]
        aligned_non_sky = aligned[8:, :]
        np.testing.assert_allclose(aligned_non_sky, non_sky, rtol=1e-3)

    def test_default_mask_excludes_zeros_and_nans(self):
        """When no mask is given, zeros and NaN in GT are auto-excluded."""
        gt = np.array([[0.0, 10.0], [20.0, np.nan]], dtype=np.float32)
        pred = np.array([[0.5, 0.1], [0.2, 0.3]], dtype=np.float32)

        aligned, s, t = compute_scale_and_shift(pred, gt)
        # Should fit on the 2 valid pixels: (0.1, 10) and (0.2, 20)
        np.testing.assert_allclose(s, 100.0, rtol=1e-4)
        np.testing.assert_allclose(t, 0.0, atol=1e-3)

    def test_constant_pred_degeneracy(self):
        """Constant prediction -> aligned output is constant at mean(gt)."""
        gt = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        pred = np.full_like(gt, 0.5)

        aligned, s, t = compute_scale_and_shift(pred, gt)
        # Best fit maps constant pred to mean(gt); aligned should be uniform
        expected = np.mean(gt)
        np.testing.assert_allclose(aligned, expected, atol=1e-4)

    def test_too_few_valid_pixels_returns_unchanged(self):
        """< 2 valid pixels -> returns pred unchanged, s=1, t=0."""
        gt = np.array([[0.0, np.nan], [0.0, 0.0]], dtype=np.float32)
        pred = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)

        aligned, s, t = compute_scale_and_shift(pred, gt)
        np.testing.assert_array_equal(aligned, pred)
        assert s == 1.0
        assert t == 0.0

    def test_output_dtype_float32(self):
        """Aligned output should always be float32."""
        gt = np.random.rand(8, 8).astype(np.float32) * 50 + 1
        pred = np.random.rand(8, 8).astype(np.float32)

        aligned, _, _ = compute_scale_and_shift(pred, gt)
        assert aligned.dtype == np.float32

    def test_large_image_performance(self):
        """Runs on a realistic-size image without error."""
        rng = np.random.RandomState(0)
        gt = rng.uniform(1.0, 300.0, size=(375, 1242)).astype(np.float32)
        pred = ((gt - gt.min()) / (gt.max() - gt.min())).astype(np.float32)

        aligned, s, t = compute_scale_and_shift(pred, gt)
        assert aligned.shape == (375, 1242)
        np.testing.assert_allclose(aligned, gt, rtol=1e-4)
