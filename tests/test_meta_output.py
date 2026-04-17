"""Tests for the structured meta block in eval.json output."""

from euler_eval.cli import (
    _clean_metric_tree,
    _depth_eval_axes,
    _DEPTH_EVAL_DESCRIPTIONS,
    _rgb_eval_axes,
    _RGB_EVAL_DESCRIPTIONS,
    _RAYS_EVAL_AXES,
    _RAYS_EVAL_DESCRIPTIONS,
)

# Build base (non-benchmark) axis dicts for testing
_DEPTH_EVAL_AXES = _depth_eval_axes()
_RGB_EVAL_AXES = _rgb_eval_axes()


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
            "depth": {"eval": {"native": {}, "metric": {}}},
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


class TestAxisDeclarations:
    """Verify axis declarations follow the metric-namespacing convention."""

    def test_depth_axes_structure(self):
        """depth.eval declares space, category, and reduction axes."""
        assert "space" in _DEPTH_EVAL_AXES
        assert "category" in _DEPTH_EVAL_AXES
        assert "reduction" in _DEPTH_EVAL_AXES

        space = _DEPTH_EVAL_AXES["space"]
        assert space.position == 0
        assert space.optional is False
        assert "native" in space.values
        assert "metric" in space.values

        category = _DEPTH_EVAL_AXES["category"]
        assert category.position == 1
        assert category.optional is True
        assert set(category.values) == {
            "image_quality",
            "standard",
            "depth_metrics",
            "geometric_metrics",
        }

        reduction = _DEPTH_EVAL_AXES["reduction"]
        assert reduction.position == 2
        assert reduction.optional is True
        assert set(reduction.values) == {
            "image_mean",
            "image_median",
            "pixel_pool",
        }

    def test_rgb_axes_structure(self):
        """rgb.eval declares a single optional category axis."""
        assert "category" in _RGB_EVAL_AXES
        assert len(_RGB_EVAL_AXES) == 1

        category = _RGB_EVAL_AXES["category"]
        assert category.position == 0
        assert category.optional is True
        assert "image_quality" in category.values
        assert "edge_f1" in category.values
        assert "tail_errors" in category.values
        assert "high_frequency" in category.values
        assert "depth_binned_photometric" in category.values

    def test_rays_axes_empty(self):
        """rays.eval has no axes (flat namespace)."""
        assert _RAYS_EVAL_AXES == {}

    def test_axis_positions_are_contiguous(self):
        """Axis positions start at 0 and are contiguous."""
        for axes in (_DEPTH_EVAL_AXES, _RGB_EVAL_AXES):
            if not axes:
                continue
            positions = sorted(a.position for a in axes.values())
            assert positions == list(range(len(positions)))

    def test_axis_values_are_nonempty_strings(self):
        """Every axis has at least one value, all lowercase strings."""
        for axes in (_DEPTH_EVAL_AXES, _RGB_EVAL_AXES, _RAYS_EVAL_AXES):
            for name, decl in axes.items():
                assert len(decl.values) >= 1, f"axis {name} has no values"
                for v in decl.values:
                    assert isinstance(v, str) and v == v.lower(), (
                        f"axis {name} value {v!r} must be lowercase string"
                    )

    def test_depth_benchmark_axes(self):
        """When benchmark=True, depth axes include a bin axis."""
        axes = _depth_eval_axes(benchmark=True)
        assert "bin" in axes
        bin_axis = axes["bin"]
        assert bin_axis.position == 3
        assert bin_axis.optional is True
        assert set(bin_axis.values) == {"all", "near", "mid", "far"}


class TestMetricDescriptions:
    """Verify metric descriptions have valid structure and directions."""

    def _check_descriptions(self, descriptions):
        valid_scales = {"linear", "log", "percentage", "binary"}
        for key, desc in descriptions.items():
            assert isinstance(key, str) and len(key) > 0
            if desc.is_higher_better is not None:
                assert isinstance(desc.is_higher_better, bool)
            if desc.scale is not None:
                assert desc.scale in valid_scales, (
                    f"{key}: invalid scale {desc.scale!r}"
                )
            if desc.min_value is not None:
                assert isinstance(desc.min_value, (int, float))
            if desc.max_value is not None:
                assert isinstance(desc.max_value, (int, float))
            if desc.display_name is not None:
                assert isinstance(desc.display_name, str)
            if desc.unit is not None:
                assert isinstance(desc.unit, str)

    def test_depth_descriptions_valid(self):
        self._check_descriptions(_DEPTH_EVAL_DESCRIPTIONS)

    def test_rgb_descriptions_valid(self):
        self._check_descriptions(_RGB_EVAL_DESCRIPTIONS)

    def test_rays_descriptions_valid(self):
        self._check_descriptions(_RAYS_EVAL_DESCRIPTIONS)

    def test_depth_key_metrics_have_direction(self):
        """Core depth metrics declare is_higher_better."""
        for key in ("psnr", "ssim", "lpips", "absrel", "rmse", "delta1"):
            assert _DEPTH_EVAL_DESCRIPTIONS[key].is_higher_better is not None, (
                f"depth description {key} missing is_higher_better"
            )

    def test_rgb_key_metrics_have_direction(self):
        """Core RGB metrics declare is_higher_better."""
        for key in ("psnr", "ssim", "lpips", "f1"):
            assert _RGB_EVAL_DESCRIPTIONS[key].is_higher_better is not None

    def test_rays_key_metrics_have_direction(self):
        """Core rays metrics declare is_higher_better."""
        for key in ("rho_a.mean", "angular_error.mean_angle"):
            assert _RAYS_EVAL_DESCRIPTIONS[key].is_higher_better is not None

    def test_metricset_envelope_via_namespace(self):
        """MetricNamespace.metric_set_envelope() produces correct structure."""
        from euler_eval.cli import _EvalNamespace, _get_version

        ns = _EvalNamespace(
            producer="euler-eval",
            producer_version="1.7.0",
            modalities=("depth",),
            axes=_depth_eval_axes(),
            descriptions=_DEPTH_EVAL_DESCRIPTIONS,
        )
        envelope = ns.metric_set_envelope("depth", metadata={})
        assert "axes" in envelope
        assert "metricDescriptions" in envelope
        assert envelope["axes"]["space"]["position"] == 0
        assert envelope["axes"]["reduction"]["position"] == 2
        assert "psnr" in envelope["metricDescriptions"]
