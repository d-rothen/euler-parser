#!/usr/bin/env python3
"""Main entry point for depth evaluation.

Parses config.json and runs evaluation between two datasets.
"""

import argparse
import json
import sys
from pathlib import Path

from src.evaluate import evaluate_datasets


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

    if "datasets" not in config:
        raise ValueError("Config must contain 'datasets' key")

    datasets = config["datasets"]
    if len(datasets) != 2:
        raise ValueError("Config must contain exactly 2 datasets for comparison")

    # Validate each dataset config
    for i, dataset in enumerate(datasets):
        if "name" not in dataset:
            raise ValueError(f"Dataset {i} must have a 'name' field")
        if "path" not in dataset:
            raise ValueError(f"Dataset {i} must have a 'path' field")

        # Validate path exists
        path = Path(dataset["path"])
        if not path.exists():
            raise ValueError(f"Dataset path does not exist: {path}")

        # Validate intrinsics if provided
        if "intrinsics" in dataset:
            intrinsics = dataset["intrinsics"]
            required_keys = ["fx", "fy", "cx", "cy"]
            for key in required_keys:
                if key not in intrinsics:
                    raise ValueError(
                        f"Dataset '{dataset['name']}' intrinsics missing '{key}'"
                    )

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate depth estimation between two datasets"
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

    args = parser.parse_args()

    # Load and validate config
    try:
        config = load_config(args.config)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract dataset configurations
    dataset1_config = config["datasets"][0]
    dataset2_config = config["datasets"][1]

    print(f"Evaluating: '{dataset1_config['name']}' vs '{dataset2_config['name']}'")
    print(f"Dataset 1 path: {dataset1_config['path']}")
    print(f"Dataset 2 path: {dataset2_config['path']}")
    print(f"Device: {args.device}")
    print("-" * 60)

    # Run evaluation
    results = evaluate_datasets(
        dataset1_config=dataset1_config,
        dataset2_config=dataset2_config,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        verbose=args.verbose,
    )

    # Save results
    for dataset_config, result in zip(config["datasets"], [results, results]):
        output_file = dataset_config.get("output_file")
        if output_file is None:
            output_file = Path(dataset_config["path"]) / "metrics.json"
        else:
            output_file = Path(output_file)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for category, metrics in results.items():
        print(f"\n{category}:")
        if isinstance(metrics, dict):
            for name, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {name}: {value:.6f}")
                else:
                    print(f"  {name}: {value}")
        else:
            print(f"  {metrics}")


if __name__ == "__main__":
    main()
