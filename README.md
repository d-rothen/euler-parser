# depth-eval

A comprehensive evaluation toolkit for comparing predicted depth maps and RGB images against ground truth, with built-in sanity checking.

## Features

- **Depth metrics**: PSNR, SSIM, LPIPS, FID, KID, AbsRel, RMSE, Scale-Invariant Log Error, Normal Consistency, Depth Edge F1
- **RGB metrics**: PSNR, SSIM, LPIPS, SCE (Structural Chromatic Error), Edge F1, Tail Errors (p95/p99), High-Frequency Energy Ratio, Depth-Binned Photometric Error
- **Sanity checking**: Automatic validation of metric results against configurable thresholds, with detailed warning reports
- **Hierarchical dataset matching**: Datasets are matched by ID through `output.json` manifests supporting nested hierarchy structures
- **Per-file and aggregate results**: Outputs both per-image metrics and dataset-level aggregates to JSON

## Installation

Requires Python 3.10+.

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

### Dependencies

- numpy, scipy, Pillow
- torch, torchvision
- lpips
- tqdm

## Usage

```bash
python main.py <config.json> [options]
```

### Options

| Flag | Description | Default |
|---|---|---|
| `--device` | `cuda` or `cpu` | `cuda` |
| `--batch-size` | Batch size for batched metrics | `16` |
| `--num-workers` | Data loading workers | `4` |
| `-v, --verbose` | Verbose output | off |
| `--skip-depth` | Skip depth evaluation | off |
| `--skip-rgb` | Skip RGB evaluation | off |
| `--no-sanity-check` | Disable sanity checking | off |
| `--metrics-config` | Path to `metrics_config.json` | auto-detect |

### Example

```bash
python main.py example_config.json --device cuda --batch-size 32 -v
```

## Configuration

### `config.json`

Defines which datasets to evaluate. See [example_config.json](example_config.json) for a full example.

```json
{
  "depth": {
    "gt_dataset": {
      "name": "depth_ground_truth",
      "path": "/path/to/depth_gt",
      "depth_scale": 0.001,
      "intrinsics": { "fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5 }
    },
    "datasets": [
      {
        "name": "model_a",
        "path": "/path/to/model_a_depth",
        "depth_scale": 1.0,
        "output_file": "/path/to/output.json"
      }
    ]
  },
  "rgb": {
    "gt_dataset": {
      "name": "rgb_ground_truth",
      "path": "/path/to/rgb_gt",
      "pixel_value_max": 255
    },
    "datasets": [
      {
        "name": "model_a",
        "path": "/path/to/model_a_rgb",
        "dim": [368, 1240],
        "pixel_value_max": 255
      }
    ]
  }
}
```

#### Dataset fields

| Field | Required | Description |
|---|---|---|
| `name` | yes | Display name |
| `path` | yes | Path to dataset root (must contain `output.json`) |
| `depth_scale` | no | Multiplier to convert raw depth values to meters (default: `1.0`) |
| `pixel_value_max` | no | Max pixel value for RGB normalization (default: `255`) |
| `intrinsics` | no | Camera intrinsics (`fx`, `fy`, `cx`, `cy`) for planar-to-radial conversion |
| `dim` | no | Target `[height, width]` for RGB GT resizing (RGB datasets only) |
| `output_file` | no | Custom output path for results (prediction datasets only) |

### Dataset manifest (`output.json`)

Each dataset directory must contain an `output.json` manifest describing its hierarchical file structure:

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

Ground truth and prediction datasets are matched by hierarchy path and file ID.

### `metrics_config.json`

Controls sanity check thresholds. See [metrics_config.json](metrics_config.json) for all available options. Example thresholds:

- Depth AbsRel median > 1.0 triggers a warning (error exceeds 100% of GT)
- RGB PSNR outside 10-60 dB is flagged as unusual
- High-frequency energy relative diff below -0.5 warns of over-smoothing

## Output

Results are saved as JSON (to the dataset path or a custom `output_file`) containing:

- **Aggregate metrics** under `depth` / `rgb` keys
- **Per-file metrics** under `per_file_metrics` with the same hierarchy structure as the input manifest
- **Sanity check report** saved to `sanity_check_report.json`

## License

MIT
