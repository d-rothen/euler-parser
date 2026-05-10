"""Tests for shared metric utility helpers."""

import numpy as np

from euler_eval.metrics.utils import get_benchmark_depth_bins


def test_benchmark_depth_bins_for_001_to_80m_range():
    range_min = 0.01
    range_max = 80.0
    sqrt_min = np.sqrt(range_min)
    sqrt_max = np.sqrt(range_max)
    step = (sqrt_max - sqrt_min) / 3.0
    near_max = float((sqrt_min + step) ** 2)
    mid_max = float((sqrt_min + 2 * step) ** 2)

    depth = np.array(
        [
            0.0,
            range_min,
            np.nextafter(near_max, range_min),
            near_max,
            np.nextafter(mid_max, near_max),
            mid_max,
            range_max,
            np.nextafter(range_max, np.inf),
        ],
        dtype=np.float64,
    )

    bins = get_benchmark_depth_bins(depth, range_min, range_max)

    assert bins["boundaries"] == {
        "range": [range_min, range_max],
        "near": [range_min, near_max],
        "mid": [near_max, mid_max],
        "far": [mid_max, range_max],
    }
    assert np.isclose(near_max, 9.290856529333297)
    assert np.isclose(mid_max, 35.95418986266663)

    np.testing.assert_array_equal(
        bins["all"],
        np.array([False, True, True, True, True, True, True, False]),
    )
    np.testing.assert_array_equal(
        bins["near"],
        np.array([False, True, True, False, False, False, False, False]),
    )
    np.testing.assert_array_equal(
        bins["mid"],
        np.array([False, False, False, True, True, False, False, False]),
    )
    np.testing.assert_array_equal(
        bins["far"],
        np.array([False, False, False, False, False, True, True, False]),
    )
