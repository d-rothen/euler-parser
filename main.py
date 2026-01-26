#!/usr/bin/env python3
"""Main entry point for depth and RGB evaluation.

Parses config.json and runs evaluation between datasets.
"""

import argparse
import json
import sys
from pathlib import Path

from src.evaluate import evaluate_depth_datasets, evaluate_rgb_datasets
from src.sanity_checker import SanityChecker


def validate_dataset_config(
    dataset: dict,
    name: str,
    allow_output: bool = True,
    allow_dim: bool = False,
) -> None:
    """Validate a single dataset configuration.

    Args:
        dataset: Dataset configuration dictionary.
        name: Name for error messages.
        allow_output: Whether output_file is allowed.

    Raises:
        ValueError: If configuration is invalid.
    """
    if "name" not in dataset:
        raise ValueError(f"{name} must have a 'name' field")
    if "path" not in dataset:
        raise ValueError(f"{name} must have a 'path' field")

    path = Path(dataset["path"])
    if not path.exists():
        raise ValueError(f"Dataset path does not exist: {path}")

    if not allow_output and "output_file" in dataset:
        raise ValueError(f"{name}: output_file is only allowed on non-GT datasets")

    if "dim" in dataset:
        if not allow_dim:
            raise ValueError(f"{name}: dim is only allowed on rgb.datasets entries")
        dim = dataset["dim"]
        if not isinstance(dim, list) or len(dim) != 2:
            raise ValueError(f"{name}: dim must be a list like [height, width]")
        for value in dim:
            if not isinstance(value, (int, float)) or value <= 0 or int(value) != value:
                raise ValueError(f"{name}: dim values must be positive integers")

    if "intrinsics" in dataset:
        intrinsics = dataset["intrinsics"]
        required_keys = ["fx", "fy", "cx", "cy"]
        for key in required_keys:
            if key not in intrinsics:
                raise ValueError(f"{name} intrinsics missing '{key}'")

    if "pixel_value_max" in dataset:
        pixel_value_max = dataset["pixel_value_max"]
        if not isinstance(pixel_value_max, (int, float)) or pixel_value_max <= 0:
            raise ValueError(f"{name}: pixel_value_max must be a positive number")


def load_config(config_path: str) -> dict:
    """Load and validate configuration from JSON file.

    Args:
        config_path: Path to config.json file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        ValueError: If configuration is invalid.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    # Validate depth section if present
    if "depth" in config:
        depth_config = config["depth"]
        if "gt_dataset" not in depth_config:
            raise ValueError("depth section must have 'gt_dataset'")
        if "datasets" not in depth_config:
            raise ValueError("depth section must have 'datasets' array")

        validate_dataset_config(
            depth_config["gt_dataset"], "depth.gt_dataset", allow_output=False
        )
        for i, dataset in enumerate(depth_config["datasets"]):
            validate_dataset_config(dataset, f"depth.datasets[{i}]", allow_output=True)

    # Validate RGB section if present
    if "rgb" in config:
        rgb_config = config["rgb"]
        if "gt_dataset" not in rgb_config:
            raise ValueError("rgb section must have 'gt_dataset'")
        if "datasets" not in rgb_config:
            raise ValueError("rgb section must have 'datasets' array")

        validate_dataset_config(
            rgb_config["gt_dataset"], "rgb.gt_dataset", allow_output=False
        )
        for i, dataset in enumerate(rgb_config["datasets"]):
            validate_dataset_config(
                dataset, f"rgb.datasets[{i}]", allow_output=True, allow_dim=True
            )

    if "depth" not in config and "rgb" not in config:
        raise ValueError("Config must contain at least 'depth' or 'rgb' section")

    return config


def save_results(results: dict, dataset_config: dict, default_path: Path) -> Path:
    """Save results to output file.

    Args:
        results: Results dictionary.
        dataset_config: Dataset configuration.
        default_path: Default path if output_file not specified.

    Returns:
        Path where results were saved.
    """
    output_file = dataset_config.get("output_file")
    if output_file is None:
        output_file = default_path / "eval.json"
    else:
        output_file = Path(output_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    return output_file


def print_results(results: dict, title: str) -> None:
    """Print results summary."""
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)

    def print_dict(d: dict, indent: int = 0) -> None:
        prefix = "  " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                print_dict(value, indent + 1)
            elif isinstance(value, float):
                print(f"{prefix}{key}: {value:.6f}")
            else:
                print(f"{prefix}{key}: {value}")

    print_dict(results)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate depth and RGB datasets"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to config.json file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for computation (default: cuda)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for metrics that support batching (default: 16)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--skip-depth",
        action="store_true",
        help="Skip depth evaluation",
    )
    parser.add_argument(
        "--skip-rgb",
        action="store_true",
        help="Skip RGB evaluation",
    )
    parser.add_argument(
        "--no-sanity-check",
        action="store_true",
        help="Disable sanity checking of metric configurations",
    )
    parser.add_argument(
        "--metrics-config",
        type=str,
        default=None,
        help="Path to metrics_config.json for sanity checking (default: auto-detect)",
    )

    args = parser.parse_args()

    # Load and validate config
    try:
        config = load_config(args.config)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Device: {args.device}")
    print("-" * 60)

    # Initialize sanity checker if not disabled
    sanity_checker = None
    if not args.no_sanity_check:
        config_path = Path(args.metrics_config) if args.metrics_config else None
        sanity_checker = SanityChecker(config_path)
        print("Sanity checking enabled")
    else:
        print("Sanity checking disabled")
    print("-" * 60)

    # Get depth GT config for RGB depth-binned metrics
    depth_gt_config = None
    if "depth" in config:
        depth_gt_config = config["depth"]["gt_dataset"]

    # Evaluate depth datasets
    if "depth" in config and not args.skip_depth:
        depth_config = config["depth"]
        gt_config = depth_config["gt_dataset"]

        print(f"\n[DEPTH] Ground Truth: '{gt_config['name']}' ({gt_config['path']})")

        for dataset_config in depth_config["datasets"]:
            print(f"\n[DEPTH] Evaluating: '{dataset_config['name']}'")
            print(f"  Path: {dataset_config['path']}")

            results = evaluate_depth_datasets(
                gt_config=gt_config,
                pred_config=dataset_config,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                verbose=args.verbose,
                sanity_checker=sanity_checker,
            )

            output_path = save_results(
                results, dataset_config, Path(dataset_config["path"])
            )
            print(f"  Results saved to: {output_path}")
            print_results(results, f"DEPTH: {dataset_config['name']}")

    # Evaluate RGB datasets
    if "rgb" in config and not args.skip_rgb:
        rgb_config = config["rgb"]
        gt_config = rgb_config["gt_dataset"]

        print(f"\n[RGB] Ground Truth: '{gt_config['name']}' ({gt_config['path']})")

        for dataset_config in rgb_config["datasets"]:
            print(f"\n[RGB] Evaluating: '{dataset_config['name']}'")
            print(f"  Path: {dataset_config['path']}")

            results = evaluate_rgb_datasets(
                gt_config=gt_config,
                pred_config=dataset_config,
                depth_gt_config=depth_gt_config,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                verbose=args.verbose,
                sanity_checker=sanity_checker,
            )

            output_path = save_results(
                results, dataset_config, Path(dataset_config["path"])
            )
            print(f"  Results saved to: {output_path}")
            print_results(results, f"RGB: {dataset_config['name']}")

    # Print sanity check report at the end
    if sanity_checker is not None:
        sanity_checker.print_report()

        # Save sanity check report to file
        report = sanity_checker.get_full_report()
        report_path = Path("sanity_check_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSanity check report saved to: {report_path}")


if __name__ == "__main__":
    main()
