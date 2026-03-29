"""Tests for the structured meta block in eval.json output."""

from euler_eval.cli import _clean_metric_tree


class TestMetaBlockStructure:
    """Verify _clean_metric_tree handles meta blocks correctly."""

    def test_meta_block_preserved(self):
        """Core meta fields survive cleaning."""
        meta = {
            "version": "1.4.1",
            "modality": "depth",
            "device": "cuda",
            "gt": {
                "path": "/data/gt",
                "split": "test",
                "dimensions": {"height": 1024, "width": 2048},
            },
            "pred": {
                "path": "/data/pred",
                "dimensions": {"height": 370, "width": 780},
            },
            "spatial_alignment": {
                "method": "resize",
                "evaluated_dimensions": {"height": 370, "width": 780},
            },
            "modality_params": {
                "radial_depth": True,
                "scale_to_meters": 1.0,
            },
            "eval_params": {
                "sky_masking": False,
                "depth_alignment_mode": "auto_affine",
                "batch_size": 16,
                "num_workers": 4,
            },
        }
        cleaned = _clean_metric_tree(meta)
        assert cleaned["version"] == "1.4.1"
        assert cleaned["gt"]["dimensions"]["height"] == 1024
        assert cleaned["pred"]["dimensions"]["width"] == 780
        assert cleaned["spatial_alignment"]["method"] == "resize"
        assert cleaned["modality_params"]["radial_depth"] is True
        assert cleaned["eval_params"]["sky_masking"] is False

    def test_none_split_pruned(self):
        """None-valued splits are pruned by _clean_metric_tree (expected)."""
        meta = {
            "gt": {"path": "/data/gt", "split": None},
            "pred": {"path": "/data/pred", "split": "test"},
        }
        cleaned = _clean_metric_tree(meta)
        assert "split" not in cleaned["gt"]
        assert cleaned["pred"]["split"] == "test"

    def test_empty_eval_params_pruned(self):
        """Empty eval_params dict is pruned."""
        meta = {
            "version": "1.0.0",
            "eval_params": {},
        }
        cleaned = _clean_metric_tree(meta)
        assert "eval_params" not in cleaned

    def test_meta_in_full_save_dict(self):
        """meta sits alongside metricSet and dataset_info without conflict."""
        save_dict = {
            "metricSet": {
                "metricNamespace": "depth.eval",
                "producerKey": "euler-eval",
            },
            "dataset_info": {"num_pairs": 10},
            "meta": {
                "version": "1.4.1",
                "gt": {"dimensions": {"height": 100, "width": 200}},
                "pred": {"dimensions": {"height": 50, "width": 100}},
                "spatial_alignment": {"method": "resize"},
            },
            "depth": {"eval": {"raw": {}, "aligned": {}}},
        }
        cleaned = _clean_metric_tree(save_dict)
        assert "meta" in cleaned
        assert "metricSet" in cleaned
        assert "dataset_info" in cleaned
        assert cleaned["meta"]["gt"]["dimensions"]["height"] == 100

    def test_rgb_meta_with_rgb_range(self):
        """RGB meta includes rgb_range in modality_params."""
        meta = {
            "version": "1.4.1",
            "modality": "rgb",
            "modality_params": {"rgb_range": [0.0, 1.0]},
        }
        cleaned = _clean_metric_tree(meta)
        assert cleaned["modality_params"]["rgb_range"] == [0.0, 1.0]

    def test_rays_meta_with_fov_domain(self):
        """Rays meta includes fov_domain and threshold_deg."""
        meta = {
            "version": "1.4.1",
            "modality": "rays",
            "modality_params": {
                "fov_domain": "lfov",
                "threshold_deg": 20.0,
            },
        }
        cleaned = _clean_metric_tree(meta)
        assert cleaned["modality_params"]["fov_domain"] == "lfov"
        assert cleaned["modality_params"]["threshold_deg"] == 20.0
