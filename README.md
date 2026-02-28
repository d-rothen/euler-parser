# euler-eval

A comprehensive evaluation toolkit for comparing predicted depth maps and RGB images against ground truth, powered by [euler_loading](https://github.com/d-rothen/euler-loading) for flexible dataset loading.

## Features

- **Depth metrics**: PSNR, SSIM, LPIPS, FID, KID, AbsRel, RMSE, Scale-Invariant Log Error, Normal Consistency, Depth Edge F1
- **RGB metrics**: PSNR, SSIM, LPIPS, SCE (Structural Chromatic Error), Edge F1, Tail Errors (p95/p99), High-Frequency Energy Ratio, Depth-Binned Photometric Error
- **Sanity checking**: Automatic validation of metric results against configurable thresholds, with detailed warning reports
- **Sky masking**: Optional exclusion of sky regions from metrics using GT segmentation
- **Flexible dataset loading**: Automatic loader resolution via euler_loading and ds-crawler index metadata
- **Per-file and aggregate results**: Outputs both per-image metrics and dataset-level aggregates to JSON
- **euler_train integration**: Optional experiment logging via [euler_train](https://github.com/d-rothen/euler-train)

## Installation

Requires Python 3.9+.

```bash
uv pip install "euler-eval @ git+https://github.com/d-rothen/euler-parser.git"

# with euler_train logging support
uv pip install "euler-eval[logging] @ git+https://github.com/d-rothen/euler-parser.git"
```

Or install in editable mode:

```bash
pip install -e .
```

### Dependencies

Core:
- numpy, scipy, Pillow
- torch, torchvision
- lpips
- tqdm
- [euler-loading](https://github.com/d-rothen/euler-loading), [ds-crawler](https://github.com/d-rothen/ds-crawler)

Optional:
- [euler-train](https://github.com/d-rothen/euler-train) (install via `[logging]` extra)

## Usage

The package provides a `depth-eval` console script:

```bash
depth-eval <config> [options]
```

Or run directly:

```bash
python main.py <config> [options]
```

### Positional arguments

| Argument | Description |
|---|---|
| `config` | Path to a JSON configuration file (see [Configuration](#configuration)) |

### Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--device` | `{auto,cuda,cpu}` | `auto` | Compute device (`auto` prefers CUDA when available) |
| `--batch-size` | `int` | `16` | Batch size for metrics that support batching |
| `--num-workers` | `int` | `4` | Number of data loading workers |
| `--verbose`, `-v` | flag | off | Enable verbose output |
| `--skip-depth` | flag | off | Skip depth evaluation |
| `--skip-rgb` | flag | off | Skip RGB evaluation |
| `--mask-sky` | flag | off | Mask sky regions from metrics using GT segmentation |
| `--no-sanity-check` | flag | off | Disable sanity checking of metric configurations |
| `--metrics-config` | `str` | auto-detect | Path to `metrics_config.json` for sanity checking |
| `--depth-alignment` | `{none,auto_affine,affine}` | `auto_affine` | Depth alignment mode (`depth` output uses aligned branch) |

### Examples

```bash
# Evaluate with default settings (auto-selects CUDA when available)
depth-eval config.json --batch-size 32

# Evaluate with sky masking enabled (requires gt.segmentation in config)
depth-eval config.json --mask-sky -v

# Skip RGB evaluation, only evaluate depth
depth-eval config.json --skip-rgb

# Disable sanity checking
depth-eval config.json --no-sanity-check

# Disable depth alignment
depth-eval config.json --depth-alignment none

# Force affine scale+shift alignment on all depth predictions
depth-eval config.json --depth-alignment affine
```

## Configuration

### `config.json`

Defines GT modalities, prediction datasets to evaluate, and optional euler_train logging. See [example_config.json](example_config.json).

```json
{
  "euler_train": {
    "dir": "runs/my_project",
    "run_id": null,
    "run_name": null
  },
  "gt": {
    "rgb":          { "path": "/data/gt/rgb" },
    "depth":        { "path": "/data/gt/depth" },
    "segmentation": { "path": "/data/gt/segmentation" },
    "calibration":  { "path": "/data/gt/calibration" }
  },
  "datasets": [
    {
      "name": "model_a",
      "rgb":   { "path": "/data/model_a/rgb" },
      "depth": { "path": "/data/model_a/depth" },
      "output_file": "/path/to/output/model_a_eval.json"
    },
    {
      "name": "model_b_depth_only",
      "depth": { "path": "/data/model_b/depth" }
    },
    {
      "name": "model_c_rgb_only",
      "rgb": { "path": "/data/model_c/rgb" }
    }
  ]
}
```

#### GT section

| Field | Required | Description |
|---|---|---|
| `gt.rgb.path` | yes | Path to GT RGB dataset |
| `gt.depth.path` | yes | Path to GT depth dataset |
| `gt.segmentation.path` | no | Path to GT segmentation (needed for `--mask-sky`) |
| `gt.calibration.path` | no | Path to calibration data (camera intrinsics matrices) |
| `gt.name` | no | Display name for ground truth (default: `"GT"`) |

#### Prediction datasets

Each entry in `datasets` can include `rgb`, `depth`, or both:

| Field | Required | Description |
|---|---|---|
| `name` | yes | Display name for this prediction dataset |
| `rgb.path` | no\* | Path to predicted RGB dataset |
| `depth.path` | no\* | Path to predicted depth dataset |
| `output_file` | no | Custom output path for results JSON (default: `eval.json` inside the first available modality path) |

\* At least one of `rgb.path` or `depth.path` is required.

#### `euler_train` section (optional)

When present, evaluation results are logged to an [euler_train](https://github.com/d-rothen/euler-train) run. Requires the `euler-train` package to be installed (`pip install euler-eval[logging]`).

| Field | Required | Description |
|---|---|---|
| `euler_train.dir` | yes | euler_train project directory |
| `euler_train.run_id` | no | Existing run ID to resume (if `null`, a new run is created) |
| `euler_train.run_name` | no | Human-readable run label |

When `run_id` is provided, the run is detached after evaluation (the run remains active for further use). When `run_id` is `null`, a new run is created and finished upon completion.

### Loader resolution

Loaders are resolved automatically by euler_loading from each dataset directory's ds-crawler index metadata. The index's `euler_loading.loader` and `euler_loading.function` fields determine which loader module and function to use (e.g. `"vkitti2"` maps to `euler_loading.loaders.gpu.vkitti2`).

No manual loader selection is required. Each dataset directory declares its own loader through its ds-crawler configuration.

Dataset metadata (e.g. `radial_depth`, `rgb_range`) is read automatically from the dataset's `output.json` manifest via `get_modality_metadata()`. Depth is assumed to already be in meters.

### Dataset manifest (`output.json`)

Each dataset directory must contain an `output.json` manifest (generated by [ds-crawler](https://github.com/d-rothen/ds-crawler)) describing its hierarchical file structure:

```json
{
  "dataset": {
    "children": {
      "scene_01": {
        "files": [
          { "id": "frame_0001", "path": "scene_01/frame_0001.png" },
          { "id": "frame_0002", "path": "scene_01/frame_0002.png" }
        ]
      }
    }
  }
}
```

GT and prediction datasets are matched by hierarchy path and file ID through `MultiModalDataset`.

### `metrics_config.json`

Controls sanity check thresholds. See [metrics_config.json](metrics_config.json) for all available options. When `--metrics-config` is not specified, the tool auto-detects `metrics_config.json` at the project root. If not found, built-in defaults are used.

## Metrics

### Depth metrics

| Metric | Key | Description |
|---|---|---|
| PSNR | `depth.image_quality.psnr` | Peak Signal-to-Noise Ratio (dB), using max depth as dynamic range |
| SSIM | `depth.image_quality.ssim` | Structural Similarity Index |
| LPIPS | `depth.image_quality.lpips` | Learned Perceptual Image Patch Similarity |
| FID | `depth.image_quality.fid` | Fréchet Inception Distance (dataset-level distribution metric) |
| KID | `depth.image_quality.kid_mean`, `kid_std` | Kernel Inception Distance (mean and std) |
| AbsRel | `depth.depth_metrics.absrel` | Absolute Relative Error (\|pred-gt\|/gt), reported as median and p90 |
| RMSE | `depth.depth_metrics.rmse` | Root Mean Square Error, reported as median and p90 |
| SILog | `depth.depth_metrics.silog` | Scale-Invariant Log Error, reported as mean, median, and p90 |
| Normal Consistency | `depth.geometric_metrics.normal_consistency` | Surface normal angular error (degrees) via finite differences; includes mean, median, and percent below 11.25°/22.5°/30° |
| Depth Edge F1 | `depth.geometric_metrics.depth_edge_f1` | Edge detection precision/recall/F1 for depth discontinuities |

### RGB metrics

| Metric | Key | Description |
|---|---|---|
| PSNR | `rgb.image_quality.psnr` | Peak Signal-to-Noise Ratio (dB) |
| SSIM | `rgb.image_quality.ssim` | Structural Similarity Index |
| SCE | `rgb.image_quality.sce` | Structural Chromatic Error |
| LPIPS | `rgb.image_quality.lpips` | Learned Perceptual Image Patch Similarity |
| Edge F1 | `rgb.edge_f1` | Edge preservation precision/recall/F1 |
| Tail Errors | `rgb.tail_errors` | 95th and 99th percentile per-pixel errors |
| High-Frequency Energy | `rgb.high_frequency` | HF energy preservation ratio (pred vs GT) and relative difference |
| Depth-Binned Photometric Error | `rgb.depth_binned_photometric` | MAE/MSE in near/mid/far depth bins (requires GT depth) |

## Output

Results are saved as JSON per prediction dataset. Default path: `eval.json` inside the first available modality path of the dataset, unless overridden by `output_file` in the config.

### Output structure

```json
{
  "depth_raw": { "...": "metrics without alignment" },
  "depth_aligned": { "...": "metrics with selected alignment mode" },
  "depth": {
    "...": "backward-compatible alias of depth_aligned"
  },
  "rgb": {
    "...": "..."
  },
  "per_file_metrics": {
    "children": {
      "scene_01": {
        "children": {
          "camera_0": {
            "files": [
              {
                "id": "frame_0001",
                "metrics": {
                  "depth": { "...": "aligned (alias)" },
                  "depth_raw": { "...": "raw" },
                  "depth_aligned": { "...": "aligned" },
                  "rgb": { "...": "..." }
                }
              }
            ]
          }
        }
      }
    }
  }
}
```

For depth outputs:
- `depth_raw`: metric-space depth without any post-hoc alignment.
- `depth_aligned`: metric-space depth after configured alignment mode.
- `depth`: backward-compatible alias of `depth_aligned`.

Previous single-depth structure (kept under `depth`) is:

```json
{
  "depth": {
    "image_quality": {
      "psnr": 28.5,
      "ssim": 0.92,
      "lpips": 0.08,
      "fid": 12.3,
      "kid_mean": 0.005,
      "kid_std": 0.002
    },
    "depth_metrics": {
      "absrel": { "median": 0.05, "p90": 0.12 },
      "rmse":   { "median": 1.2, "p90": 3.1 },
      "silog":  { "mean": 0.08, "median": 0.06, "p90": 0.15 }
    },
    "geometric_metrics": {
      "normal_consistency": {
        "mean_angle": 12.3,
        "median_angle": 9.8,
        "percent_below_11_25": 55.2,
        "percent_below_22_5": 82.1,
        "percent_below_30": 91.5
      },
      "depth_edge_f1": {
        "precision": 0.72,
        "recall": 0.68,
        "f1": 0.70
      }
    },
    "dataset_info": {
      "num_pairs": 500,
      "gt_name": "GT",
      "pred_name": "model_a"
    }
  },
  "rgb": { "...": "unchanged" }
}
```

### Sanity check report

When sanity checking is enabled (the default), a `sanity_check_report.json` is saved to the current working directory containing warnings grouped by metric type.

## License

MIT
