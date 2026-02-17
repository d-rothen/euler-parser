"""Shared fixtures for depth-eval tests."""

import json
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


MOCK_FILES = Path(__file__).parent / "mock_files"


@pytest.fixture
def depth_index_output():
    """Parsed depth output.json with scale_to_meters and radial_depth meta."""
    with open(MOCK_FILES / "depth_output.json") as f:
        return json.load(f)


@pytest.fixture
def rgb_index_output():
    """Parsed RGB output.json with rgb_range meta."""
    with open(MOCK_FILES / "rgb_output.json") as f:
        return json.load(f)


@pytest.fixture
def calibration_index_output():
    """Parsed calibration output.json."""
    with open(MOCK_FILES / "calibration_output.json") as f:
        return json.load(f)


@pytest.fixture
def segmentation_index_output():
    """Parsed segmentation output.json."""
    with open(MOCK_FILES / "segmentation_output.json") as f:
        return json.load(f)


@pytest.fixture
def sample_K():
    """A realistic 3x3 intrinsics matrix."""
    return np.array([
        [525.0, 0.0, 319.5],
        [0.0, 525.0, 239.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


@pytest.fixture
def mock_dataset():
    """Create a mock MultiModalDataset with configurable metadata.

    Returns a factory function that takes index_outputs dict and returns
    a MagicMock with get_modality_metadata and modality_paths wired up.
    """
    def _make(index_outputs: dict[str, dict], modality_names: Optional[list[str]] = None):
        ds = MagicMock()
        ds._index_outputs = index_outputs

        def get_modality_metadata(name):
            return index_outputs.get(name, {}).get("meta", {})

        ds.get_modality_metadata = get_modality_metadata

        names = modality_names or list(index_outputs.keys())
        ds.modality_paths.return_value = {n: f"/fake/{n}" for n in names}

        return ds

    return _make
