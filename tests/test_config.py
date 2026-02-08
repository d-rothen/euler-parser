"""Tests for main.py config validation."""

import json
import pytest

from main import load_config, validate_dataset_entry, validate_gt_config


# ---------------------------------------------------------------------------
# validate_gt_config
# ---------------------------------------------------------------------------


class TestValidateGtConfig:
    def test_valid_minimal(self, tmp_path):
        """Minimal valid GT config with rgb and depth paths."""
        rgb_path = tmp_path / "rgb"
        depth_path = tmp_path / "depth"
        rgb_path.mkdir()
        depth_path.mkdir()

        gt = {
            "rgb": {"path": str(rgb_path)},
            "depth": {"path": str(depth_path)},
        }
        validate_gt_config(gt)  # should not raise

    def test_valid_with_all_modalities(self, tmp_path):
        """Full GT config with all optional modalities."""
        for name in ("rgb", "depth", "segmentation", "calibration"):
            (tmp_path / name).mkdir()

        gt = {
            "rgb": {"path": str(tmp_path / "rgb")},
            "depth": {"path": str(tmp_path / "depth")},
            "segmentation": {"path": str(tmp_path / "segmentation")},
            "calibration": {"path": str(tmp_path / "calibration")},
        }
        validate_gt_config(gt)

    def test_missing_rgb(self, tmp_path):
        depth_path = tmp_path / "depth"
        depth_path.mkdir()
        gt = {"depth": {"path": str(depth_path)}}
        with pytest.raises(ValueError, match="gt.rgb.path"):
            validate_gt_config(gt)

    def test_missing_depth(self, tmp_path):
        rgb_path = tmp_path / "rgb"
        rgb_path.mkdir()
        gt = {"rgb": {"path": str(rgb_path)}}
        with pytest.raises(ValueError, match="gt.depth.path"):
            validate_gt_config(gt)

    def test_nonexistent_path(self, tmp_path):
        rgb_path = tmp_path / "rgb"
        rgb_path.mkdir()
        gt = {
            "rgb": {"path": str(rgb_path)},
            "depth": {"path": str(tmp_path / "nonexistent")},
        }
        with pytest.raises(ValueError, match="does not exist"):
            validate_gt_config(gt)


# ---------------------------------------------------------------------------
# validate_dataset_entry
# ---------------------------------------------------------------------------


class TestValidateDatasetEntry:
    def test_valid_depth_only(self, tmp_path):
        depth_path = tmp_path / "depth"
        depth_path.mkdir()
        entry = {
            "name": "model_a",
            "depth": {"path": str(depth_path)},
        }
        validate_dataset_entry(entry, 0)

    def test_valid_rgb_only(self, tmp_path):
        rgb_path = tmp_path / "rgb"
        rgb_path.mkdir()
        entry = {
            "name": "model_a",
            "rgb": {"path": str(rgb_path)},
        }
        validate_dataset_entry(entry, 0)

    def test_valid_both(self, tmp_path):
        for name in ("rgb", "depth"):
            (tmp_path / name).mkdir()
        entry = {
            "name": "model_a",
            "rgb": {"path": str(tmp_path / "rgb")},
            "depth": {"path": str(tmp_path / "depth")},
        }
        validate_dataset_entry(entry, 0)

    def test_missing_name(self, tmp_path):
        depth_path = tmp_path / "depth"
        depth_path.mkdir()
        entry = {"depth": {"path": str(depth_path)}}
        with pytest.raises(ValueError, match="must have a 'name'"):
            validate_dataset_entry(entry, 0)

    def test_no_modalities(self):
        entry = {"name": "empty"}
        with pytest.raises(ValueError, match="at least"):
            validate_dataset_entry(entry, 0)

    def test_nonexistent_depth_path(self, tmp_path):
        entry = {
            "name": "bad",
            "depth": {"path": str(tmp_path / "nonexistent")},
        }
        with pytest.raises(ValueError, match="does not exist"):
            validate_dataset_entry(entry, 0)

    def test_index_in_error_message(self, tmp_path):
        entry = {"name": "test"}
        with pytest.raises(ValueError, match=r"datasets\[3\]"):
            validate_dataset_entry(entry, 3)


# ---------------------------------------------------------------------------
# load_config  (end-to-end validation)
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def _write_config(self, tmp_path, config_dict):
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f)
        return str(config_path)

    def test_valid_config(self, tmp_path):
        for name in ("gt_rgb", "gt_depth", "pred_depth"):
            (tmp_path / name).mkdir()

        config = {
            "gt": {
                "rgb": {"path": str(tmp_path / "gt_rgb")},
                "depth": {"path": str(tmp_path / "gt_depth")},
            },
            "datasets": [
                {
                    "name": "test_model",
                    "depth": {"path": str(tmp_path / "pred_depth")},
                }
            ],
        }
        path = self._write_config(tmp_path, config)
        result = load_config(path)
        assert result["gt"]["rgb"]["path"] == str(tmp_path / "gt_rgb")
        assert len(result["datasets"]) == 1

    def test_missing_gt_section(self, tmp_path):
        config = {"datasets": [{"name": "x", "depth": {"path": "/fake"}}]}
        path = self._write_config(tmp_path, config)
        with pytest.raises(ValueError, match="'gt' section"):
            load_config(path)

    def test_missing_datasets_section(self, tmp_path):
        for name in ("rgb", "depth"):
            (tmp_path / name).mkdir()
        config = {
            "gt": {
                "rgb": {"path": str(tmp_path / "rgb")},
                "depth": {"path": str(tmp_path / "depth")},
            },
        }
        path = self._write_config(tmp_path, config)
        with pytest.raises(ValueError, match="'datasets'"):
            load_config(path)

    def test_empty_datasets_array(self, tmp_path):
        for name in ("rgb", "depth"):
            (tmp_path / name).mkdir()
        config = {
            "gt": {
                "rgb": {"path": str(tmp_path / "rgb")},
                "depth": {"path": str(tmp_path / "depth")},
            },
            "datasets": [],
        }
        path = self._write_config(tmp_path, config)
        with pytest.raises(ValueError, match="'datasets'"):
            load_config(path)

    def test_invalid_json(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text("not json{{{")
        with pytest.raises((json.JSONDecodeError, ValueError)):
            load_config(str(config_path))

    def test_multiple_datasets(self, tmp_path):
        for name in ("gt_rgb", "gt_depth", "pred_a_rgb", "pred_b_depth"):
            (tmp_path / name).mkdir()

        config = {
            "gt": {
                "rgb": {"path": str(tmp_path / "gt_rgb")},
                "depth": {"path": str(tmp_path / "gt_depth")},
            },
            "datasets": [
                {
                    "name": "model_a",
                    "rgb": {"path": str(tmp_path / "pred_a_rgb")},
                },
                {
                    "name": "model_b",
                    "depth": {"path": str(tmp_path / "pred_b_depth")},
                },
            ],
        }
        path = self._write_config(tmp_path, config)
        result = load_config(path)
        assert len(result["datasets"]) == 2
        assert result["datasets"][0]["name"] == "model_a"
        assert result["datasets"][1]["name"] == "model_b"
