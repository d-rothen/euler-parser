"""Tests for sparse pointcloud depth projection and evaluation."""

import numpy as np

from euler_eval.data import (
    project_point_cloud_to_depth_map,
    to_numpy_extrinsics,
    to_numpy_point_cloud,
)
from euler_eval.evaluate import evaluate_sparse_depth_samples


def test_to_numpy_point_cloud_accepts_extra_columns():
    cloud = np.array([[1.0, 2.0, 3.0, 0.5, 4.0, 0.0]], dtype=np.float64)

    result = to_numpy_point_cloud(cloud)

    assert result.shape == (1, 6)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[:, :3], cloud[:, :3])


def test_to_numpy_extrinsics_expands_3x4_matrix():
    transform = np.eye(3, 4, dtype=np.float32)

    result = to_numpy_extrinsics(transform)

    assert result.shape == (4, 4)
    np.testing.assert_array_equal(result[3], np.array([0.0, 0.0, 0.0, 1.0]))


def test_project_point_cloud_to_depth_map_uses_nearest_point_per_pixel():
    intrinsics = np.eye(3, dtype=np.float32)
    extrinsics = np.eye(4, dtype=np.float32)
    point_cloud = np.array(
        [
            [4.0, 4.0, 4.0, 0.0],  # Same pixel as next point, farther away.
            [2.0, 2.0, 2.0, 0.0],
            [4.0, 2.0, 2.0, 0.0],
            [1.0, 1.0, -1.0, 0.0],  # Behind camera.
            [20.0, 20.0, 1.0, 0.0],  # Outside image.
        ],
        dtype=np.float32,
    )

    depth, mask, meta = project_point_cloud_to_depth_map(
        point_cloud,
        intrinsics,
        extrinsics,
        image_shape=(3, 3),
    )

    assert int(mask.sum()) == 2
    assert meta["input_points"] == 5
    assert meta["projected_pixels"] == 2
    assert mask[1, 1]
    assert mask[1, 2]
    np.testing.assert_allclose(depth[1, 1], np.sqrt(12.0), rtol=1e-6)
    np.testing.assert_allclose(depth[1, 2], np.sqrt(24.0), rtol=1e-6)


class _OneSampleSparseDataset:
    def __init__(self):
        self.intrinsics = np.eye(3, dtype=np.float32)
        self.extrinsics = np.eye(4, dtype=np.float32)
        self.point_cloud = np.array(
            [
                [2.0, 2.0, 2.0, 0.0],
                [4.0, 2.0, 2.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.pred = np.zeros((3, 3), dtype=np.float32)
        self.pred[1, 1] = np.sqrt(12.0)
        self.pred[1, 2] = np.sqrt(24.0)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return {
            "id": "000000",
            "full_id": "/000000",
            "gt": self.point_cloud,
            "pred": self.pred,
            "intrinsics": {"intrinsics": self.intrinsics},
            "camera_extrinsics": {"lidar2rgb": self.extrinsics},
        }

    def modality_paths(self):
        return {"gt": {"path": "/gt"}, "pred": {"path": "/pred"}}


def test_sparse_depth_eval_reports_only_pointwise_depth_metrics():
    dataset = _OneSampleSparseDataset()

    result = evaluate_sparse_depth_samples(
        dataset,
        pred_is_radial=True,
        num_workers=0,
        alignment_mode="none",
    )

    metrics = result["sparse_depth_metric"]
    assert result["dataset_info"]["projected_pixels"] == 2
    assert result["dataset_info"]["evaluated_pixels"] == 2
    assert "image_quality" not in metrics
    assert "geometric_metrics" not in metrics
    np.testing.assert_allclose(metrics["standard"]["pixel_pool"]["absrel"], 0.0)
    np.testing.assert_allclose(metrics["standard"]["pixel_pool"]["delta1"], 1.0)
    np.testing.assert_allclose(metrics["depth_metrics"]["rmse"]["median"], 0.0)
