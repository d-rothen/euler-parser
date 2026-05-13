# euler-eval

A comprehensive evaluation toolkit for comparing predicted depth maps, RGB images, and camera ray direction maps against ground truth, powered by [euler_loading](https://github.com/d-rothen/euler-loading) for flexible dataset loading.

## Features

- **Depth metrics**: PSNR, SSIM, LPIPS, FID, KID, AbsRel, RMSE, Scale-Invariant Log Error, Normal Consistency, Depth Edge F1
- **RGB metrics**: PSNR, SSIM, LPIPS, FID, SCE (Structural Chromatic Error), Edge F1, Tail Errors (p95/p99), High-Frequency Energy Ratio, Depth-Binned Photometric Error
- **Rays metrics**: ρ_A (AUC of angular accuracy curve), Angular Error statistics and threshold percentages
- **Benchmark binning**: Optional depth-range benchmark that subdivides metrics into square-root-scaled near/mid/far bins
- **Sanity checking**: Automatic validation of metric results against configurable thresholds, with detailed warning reports
- **Sky masking**: Optional exclusion of sky regions from metrics using GT segmentation
- **Flexible dataset loading**: Automatic loader resolution via euler_loading and ds-crawler index metadata
- **Per-file and aggregate results**: Outputs both per-image metrics and dataset-level aggregates to JSON, saved per-modality
- **euler_train integration**: Optional experiment logging via [euler_train](https://github.com/d-rothen/euler-train)

## Installation

Requires Python 3.9+.

```bash
uv pip install "euler-eval @ git+https://github.com/d-rothen/euler-parser.git"

# with euler_train logging support
uv pip install "euler-eval[logging] @ git+https://github.com/d-rothen/euler-parser.git"

# with clean-fid RGB FID backend support
uv pip install "euler-eval[fid] @ git+https://github.com/d-rothen/euler-parser.git"
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

It also provides a cache warmup helper for offline environments:

```bash
euler-eval.init
```

Or run directly:

```bash
python main.py <config> [options]
```

Before running on offline compute nodes, you can warm caches on a machine with network access:

```bash
HF_HOME=/shared/cache/hf \
TORCH_HOME=/shared/cache/hf/torch \
CLEANFID_CACHE_DIR=/shared/cache/clean-fid \
euler-eval.init
```

This pre-downloads:
- torchvision AlexNet weights
- torchvision Inception v3 weights
- LPIPS AlexNet weights
- the clean-fid inception checkpoint, if `clean-fid` is installed

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
| `--skip-rays` | flag | off | Skip rays (spherical direction map) evaluation |
| `--mask-sky` | flag | off | Mask sky regions from metrics using GT segmentation |
| `--no-sanity-check` | flag | off | Disable sanity checking of metric configurations |
| `--metrics-config` | `str` | auto-detect | Path to `metrics_config.json` for sanity checking |
| `--depth-alignment` | `{none,auto_affine,affine}` | `auto_affine` | Depth calibration mode; outputs are emitted in semantic `native`/`metric` spaces and `depth` aliases the canonical branch |
| `--rgb-fid-backend` | `{builtin,clean-fid}` | `builtin` | RGB FID backend; `clean-fid` requires optional dependency |
| `--benchmark-depth-range` | `float float` | none | Depth range `[MIN, MAX]` in meters for benchmark evaluation; computes depth and RGB metrics for pixels within this range, subdivided into square-root-scaled near/mid/far bins (additive to regular metrics) |

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

# Use clean-fid for RGB FID computation
depth-eval config.json --rgb-fid-backend clean-fid

# Benchmark depth and RGB metrics within a depth range (near/mid/far bins)
depth-eval config.json --benchmark-depth-range 0.01 80.0

# Evaluate dense depth predictions against sparse pointcloud GT
depth-eval example_sparse_depth_config.json --skip-rgb --skip-rays

# Skip rays evaluation
depth-eval config.json --skip-rays
```

### Benchmark Depth Bins

`--benchmark-depth-range MIN MAX` first filters valid GT depth pixels to
`MIN <= depth <= MAX`. The `near`, `mid`, and `far` bins are then computed as
three equal-width intervals in square-root depth space:

```text
sqrt_min = sqrt(MIN)
sqrt_max = sqrt(MAX)
step = (sqrt_max - sqrt_min) / 3
near_max = (sqrt_min + step)^2
mid_max = (sqrt_min + 2 * step)^2
```

The interval bounds are `near=[MIN, near_max)`, `mid=[near_max, mid_max)`, and
`far=[mid_max, MAX]`. For `--benchmark-depth-range 0.01 80.0`, this gives:

| Bin | Depth interval |
|---|---|
| `all` | `[0.01, 80.0]` |
| `near` | `[0.01, 9.290856529)` |
| `mid` | `[9.290856529, 35.954189863)` |
| `far` | `[35.954189863, 80.0]` |

## Configuration

### `config.json`

Defines GT modalities, prediction datasets to evaluate, and optional euler_train logging. See [example_config.json](example_config.json). For sparse pointcloud depth GT, see [example_sparse_depth_config.json](example_sparse_depth_config.json).

```json
{
  "euler_train": {
    "dir": "runs/my_project"
  },
  "gt": {
    "rgb":          { "path": "/data/gt/rgb" },
    "depth":        { "path": "/data/gt/depth" },
    "rays":         { "path": "/data/gt/rays" },
    "segmentation": { "path": "/data/gt/segmentation" },
    "calibration":  { "path": "/data/gt/calibration" }
  },
  "datasets": [
    {
      "name": "model_a",
      "rgb":   { "path": "/data/model_a/rgb" },
      "depth": { "path": "/data/model_a/depth" },
      "rays":  { "path": "/data/model_a/rays" },
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

Each modality entry can optionally include a `split` field to select a specific split from the dataset (e.g. `{ "path": "/data/gt/depth", "split": "test" }`). euler-loading inline selectors are also accepted in `path`, such as `/data/muses.zip:test` or `/data/muses.zip:test#scope=rgb`.

For sparse pointcloud depth evaluation, use `gt.sparse_depth` instead of `gt.depth`. The prediction uses a dense depth-like map under `datasets[].depth`, `datasets[].relative_depth`, or `datasets[].affine_depth`. The evaluator projects the sparse GT point cloud into the prediction image plane using `gt.intrinsics` and `gt.camera_extrinsics`, then computes pointwise depth metrics only at projected valid pixels.

#### GT section

| Field | Required | Description |
|---|---|---|
| `gt.rgb.path` | no\* | Path to GT RGB dataset |
| `gt.depth.path` | no\* | Path to GT depth dataset |
| `gt.sparse_depth.path` | no\* | Path to sparse pointcloud GT dataset, e.g. `sparse_depth` with `(N,C)` points whose first columns are `x,y,z` in meters |
| `gt.rays.path` | no\* | Path to GT ray direction map dataset (for rays evaluation) |
| `gt.segmentation.path` | no | Path to GT segmentation (needed for `--mask-sky`) |
| `gt.calibration.path` | no | Path to calibration data (camera intrinsics matrices) |
| `gt.intrinsics.path` | required with `gt.sparse_depth` | Path to camera intrinsics matrices for pointcloud projection |
| `gt.camera_extrinsics.path` | required with `gt.sparse_depth` | Path to source-to-camera extrinsics, e.g. MUSES `lidar2rgb`, for pointcloud projection |
| `gt.name` | no | Display name for ground truth (default: `"GT"`) |

\* At least one of `gt.rgb.path`, `gt.depth.path`, `gt.sparse_depth.path`, or `gt.rays.path` is required.

#### Prediction datasets

Each entry in `datasets` can include `rgb`, one dense depth-like prediction (`depth`, `relative_depth`, or `affine_depth`), `rays`, or any combination:

| Field | Required | Description |
|---|---|---|
| `name` | yes | Display name for this prediction dataset |
| `rgb.path` | no\* | Path to predicted RGB dataset |
| `depth.path` | no\* | Path to predicted dense metric depth dataset; also used when evaluating against sparse pointcloud GT |
| `relative_depth.path` | no\* | Path to predicted dense relative depth dataset; evaluated through the same depth pipeline with scale/shift alignment support |
| `affine_depth.path` | no\* | Path to predicted dense affine-depth dataset; evaluated through the same depth pipeline with scale/shift alignment support |
| `rays.path` | no\* | Path to predicted ray direction map dataset |
| `output_file` | no | Custom output path for results JSON (default: `eval.json` inside the first available modality path) |

\* At least one of `rgb.path`, `depth.path`, `relative_depth.path`, `affine_depth.path`, or `rays.path` is required. Use only one dense depth-like entry per prediction dataset.

When `relative_depth` or `affine_depth` is used, `--depth-alignment auto_affine` treats the entry as declared non-metric depth and runs scale/shift alignment even if the raw value range is not normalized. `--depth-alignment none` still disables calibration.

#### `euler_train` section (optional)

When present, evaluation results are logged to an [euler_train](https://github.com/d-rothen/euler-train) run. Requires the `euler-train` package to be installed (`pip install euler-eval[logging]`).

| Field | Required | Description |
|---|---|---|
| `euler_train.dir` | yes | Project directory (creates a new run) **or** full path to an existing run directory (resumes it) |

euler_train auto-detects whether the path is a run directory by checking for `meta.json`. When resuming an existing run, the run is detached after evaluation (the run remains active for further use). When a new run is created, it is finished upon completion.

### Loader resolution

Loaders are resolved automatically by euler_loading from each dataset directory's ds-crawler index metadata. The index's `euler_loading.loader` and `euler_loading.function` fields determine which loader module and function to use (e.g. `"vkitti2"` maps to `euler_loading.loaders.gpu.vkitti2`).

No manual loader selection is required. Each dataset directory declares its own loader through its ds-crawler configuration.

Dataset metadata (e.g. `radial_depth`, `rgb_range`, sparse point columns, and coordinate units) is read automatically from each dataset's ds-crawler metadata via `get_modality_metadata()`. Depth and point-cloud coordinates are assumed to already be in meters.

### Dataset Metadata

Each dataset directory or archive must contain ds-crawler metadata artifacts, typically generated under `.ds_crawler/`:

```text
dataset-root/
  .ds_crawler/
    dataset-head.json
    ds-crawler.json
    index.json
```

When one physical root or archive contains several logical modalities, the artifacts may instead be scoped under `.ds_crawler/<modality>/` with a `.ds_crawler/scopes.json` manifest. `euler-eval` passes modality scopes to euler-loading so paths such as `/data/muses.zip:test` can load `.ds_crawler/rgb/index.json`, `.ds_crawler/depth/index.json`, and related scoped artifacts from the same archive.

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
| Standard depth metrics | `depth.standard.{image_mean,image_median,pixel_pool}.*` | Monocular-depth metrics with explicit reducers: `absrel`, `sqrel`, `mae`, `rmse`, `rmse_log`, `log10`, `silog`, `delta1`, `delta2`, `delta3` |
| AbsRel | `depth.depth_metrics.absrel` | Absolute Relative Error (\|pred-gt\|/gt), reported as median and p90 |
| RMSE | `depth.depth_metrics.rmse` | Root Mean Square Error, reported as median and p90 |
| SILog | `depth.depth_metrics.silog` | Scale-Invariant Log Error, reported as mean, median, and p90 |
| Normal Consistency | `depth.geometric_metrics.normal_consistency` | Surface normal angular error (degrees) via finite differences; includes mean, median, and percent below 11.25°/22.5°/30° |
| Depth Edge F1 | `depth.geometric_metrics.depth_edge_f1` | Edge detection precision/recall/F1 for depth discontinuities |

### Sparse Depth Metrics

Sparse pointcloud GT evaluation reports only metrics that remain meaningful at isolated projected points. Dense image-quality and geometric metrics such as SSIM, LPIPS, FID, normals, and edge F1 are intentionally skipped.
Serialized `eval.json` sparse-depth metrics use the namespace root `sparsedepth` (without an underscore), so flattened metric names match `metricSet.metricNamespace = "sparsedepth.eval"`.

| Metric | Key | Description |
|---|---|---|
| Standard depth metrics | `sparsedepth.eval.{native,metric}.standard.{image_mean,image_median,pixel_pool}.*` | Same monocular-depth reducers as dense depth, computed only at projected sparse GT pixels |
| AbsRel | `sparsedepth.eval.{native,metric}.depth_metrics.absrel` | Absolute Relative Error (\|pred-gt\|/gt), reported as median and p90 over projected sparse pixels |
| RMSE | `sparsedepth.eval.{native,metric}.depth_metrics.rmse` | Root Mean Square Error, reported as median and p90 over projected sparse pixels |
| SILog | `sparsedepth.eval.{native,metric}.depth_metrics.silog` | Scale-Invariant Log Error, reported as mean, median, and p90 over projected sparse pixels |

### RGB metrics

| Metric | Key | Description |
|---|---|---|
| PSNR | `rgb.image_quality.psnr` | Peak Signal-to-Noise Ratio (dB) |
| SSIM | `rgb.image_quality.ssim` | Structural Similarity Index |
| SCE | `rgb.image_quality.sce` | Structural Chromatic Error |
| LPIPS | `rgb.image_quality.lpips` | Learned Perceptual Image Patch Similarity |
| FID | `rgb.image_quality.fid` | Fréchet Inception Distance (dataset-level distribution metric) |
| Edge F1 | `rgb.edge_f1` | Edge preservation precision/recall/F1 |
| Tail Errors | `rgb.tail_errors` | 95th and 99th percentile per-pixel errors |
| High-Frequency Energy | `rgb.high_frequency` | HF energy preservation ratio (pred vs GT) and relative difference |
| Depth-Binned Photometric Error | `rgb.depth_binned_photometric` | MAE/MSE in near/mid/far depth bins (requires GT depth) |

### Rays metrics

| Metric | Key | Description |
|---|---|---|
| ρ_A | `rays.rho_a.mean`, `rho_a.median` | Area Under the angular accuracy Curve — fraction of pixels with angular error ≤ threshold, integrated from 0 to a FoV-dependent threshold (S.FoV: 15°, L.FoV: 20°, Pano: 30°) |
| Angular Error | `rays.angular_error.mean_angle`, `median_angle` | Per-pixel angular error between predicted and GT camera ray directions (degrees) |
| Angular Error Thresholds | `rays.angular_error.percent_below_*` | Percentage of pixels with angular error below 5°, 10°, 15°, 20°, 30° |

## Output

Results are saved as JSON per modality per prediction dataset (one file for depth or sparse depth, one for RGB, one for rays). Default path: `eval.json` inside each modality's dataset path, unless overridden by `output_file` in the config.

For RGB FID, two backends are available:
- `builtin`: in-process Inception-based implementation in this repository.
- `clean-fid`: delegates folder-vs-folder FID computation to [clean-fid](https://github.com/GaParmar/clean-fid). This backend requires installing the optional `fid` extra and is recommended when you need scores closer to standard published FID numbers.

When `--rgb-fid-backend clean-fid` is used, `euler-eval` will honor `CLEANFID_CACHE_DIR` if set:
- If `CLEANFID_CACHE_DIR/inception-2015-12-05.pt` exists, it is staged into the location `clean-fid` expects before evaluation.
- If it does not exist and the machine is online, `euler-eval` asks `clean-fid` to download it into `CLEANFID_CACHE_DIR`.
- Without `CLEANFID_CACHE_DIR`, `clean-fid` falls back to its own default local path handling.

### Output structure

```json
{
  "depth_native": { "...": "native model depth space, if diagnostically meaningful" },
  "depth_metric": { "...": "metric depth space, if available" },
  "depth": {
    "...": "canonical alias of depth_metric when present, else depth_native"
  },
  "sparsedepth": {
    "eval": { "...": "serialized sparse-depth metrics when gt.sparse_depth is used" }
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
                  "depth": { "...": "canonical alias" },
                  "depth_native": { "...": "native, when emitted" },
                  "depth_metric": { "...": "metric, when emitted" },
                  "sparsedepth": { "eval": { "...": "serialized sparse-depth metrics" } },
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
- `depth_native`: the model's native depth space after spatial/radial preprocessing, emitted only when it is diagnostically distinct.
- `depth_metric`: the comparable metric-depth branch. This is either the native prediction itself or the calibrated scale-shift result.
- `depth`: canonical alias of `depth_metric` when available, otherwise `depth_native`.
- `standard`: explicit monocular-depth metrics with three reducers:
  `image_mean`, `image_median`, and `pixel_pool`.

For sparse depth outputs, the internal Python result dict still uses `sparse_depth_native`, `sparse_depth_metric`, and `sparse_depth`, but serialized `eval.json` metric paths are rooted at `sparsedepth.eval` to satisfy external namespace validation. Only `standard` and `depth_metrics` categories are present.

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
    "standard": {
      "image_mean": {
        "absrel": 0.08,
        "sqrel": 0.04,
        "mae": 0.62,
        "rmse": 1.20,
        "rmse_log": 0.11,
        "log10": 0.04,
        "silog": 0.08,
        "delta1": 0.91,
        "delta2": 0.97,
        "delta3": 0.99
      },
      "image_median": {
        "absrel": 0.07,
        "sqrel": 0.03,
        "mae": 0.58,
        "rmse": 1.10,
        "rmse_log": 0.10,
        "log10": 0.04,
        "silog": 0.07,
        "delta1": 0.92,
        "delta2": 0.98,
        "delta3": 0.99
      },
      "pixel_pool": {
        "absrel": 0.08,
        "sqrel": 0.04,
        "mae": 0.61,
        "rmse": 1.18,
        "rmse_log": 0.11,
        "log10": 0.04,
        "silog": 0.08,
        "delta1": 0.91,
        "delta2": 0.97,
        "delta3": 0.99
      }
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
