#!/usr/bin/env python3
"""Main entry point for depth and RGB evaluation.

Parses config.json and runs evaluation using euler_loading datasets.
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Optional

import torch

from .data import (
    build_depth_eval_dataset,
    build_rgb_eval_dataset,
    get_depth_metadata,
    get_rgb_metadata,
)
from .evaluate import evaluate_depth_samples, evaluate_rgb_samples
from .sanity_checker import SanityChecker

try:
    import euler_train as _euler_train
except ImportError:
    _euler_train = None


def resolve_device(device: str) -> str:
    """Resolve runtime device with graceful CUDA fallback."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print(
            "Warning: --device cuda requested but CUDA is unavailable. "
            "Falling back to CPU.",
            file=sys.stderr,
        )
        return "cpu"
    return device


def configure_torch_runtime(device: str) -> None:
    """Enable CUDA performance knobs when running on GPU."""
    if device != "cuda":
        return

    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def print_device_info(requested_device: str, resolved_device: str) -> None:
    """Print effective runtime device info."""
    if requested_device == resolved_device:
        print(f"Device: {resolved_device}")
    else:
        print(f"Device: {requested_device} -> {resolved_device}")

    if resolved_device == "cuda":
        try:
            idx = torch.cuda.current_device()
            print(f"GPU: {torch.cuda.get_device_name(idx)}")
        except Exception:
            pass


def resolve_depth_alignment(depth_alignment: str, sns: Optional[bool]) -> str:
    """Resolve depth alignment mode with deprecated --sns compatibility."""
    if sns is None:
        return depth_alignment

    mapped = "auto_affine" if sns else "none"
    if depth_alignment != "auto_affine":
        print(
            "Warning: --sns/--no-sns is ignored because --depth-alignment is set.",
            file=sys.stderr,
        )
        return depth_alignment

    print(
        "Warning: --sns/--no-sns is deprecated. Use --depth-alignment instead.",
        file=sys.stderr,
    )
    return mapped


def validate_gt_config(gt: dict) -> None:
    """Validate the ``gt`` section of the configuration.

    Raises:
        ValueError: If required fields are missing or paths do not exist.
    """
    if "rgb" not in gt or "path" not in gt["rgb"]:
        raise ValueError("gt.rgb.path is required")
    if "depth" not in gt or "path" not in gt["depth"]:
        raise ValueError("gt.depth.path is required")

    for modality in ("rgb", "depth", "segmentation", "calibration"):
        if modality in gt:
            p = Path(gt[modality]["path"])
            if not p.exists():
                raise ValueError(f"gt.{modality}.path does not exist: {p}")


def validate_dataset_entry(entry: dict, index: int) -> None:
    """Validate a single prediction dataset entry.

    Raises:
        ValueError: If the entry is malformed.
    """
    label = f"datasets[{index}]"
    if "name" not in entry:
        raise ValueError(f"{label} must have a 'name' field")

    has_rgb = "rgb" in entry and "path" in entry.get("rgb", {})
    has_depth = "depth" in entry and "path" in entry.get("depth", {})
    if not has_rgb and not has_depth:
        raise ValueError(f"{label} must have at least 'rgb.path' or 'depth.path'")

    for modality in ("rgb", "depth"):
        if modality in entry and "path" in entry[modality]:
            p = Path(entry[modality]["path"])
            if not p.exists():
                raise ValueError(f"{label}.{modality}.path does not exist: {p}")


def validate_euler_train_config(et_config: dict) -> None:
    """Validate the optional ``euler_train`` config section.

    Raises:
        ValueError: If required fields are missing or euler_train is not installed.
    """
    if "dir" not in et_config:
        raise ValueError(
            "euler_train.dir is required when euler_train logging is enabled"
        )
    if _euler_train is None:
        raise ValueError(
            "euler_train logging is configured but the 'euler-train' package is not "
            "installed. Install it with: pip install euler-train"
        )


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

    if "gt" not in config:
        raise ValueError("Config must contain a 'gt' section")
    if "datasets" not in config or not config["datasets"]:
        raise ValueError("Config must contain a non-empty 'datasets' array")

    validate_gt_config(config["gt"])
    for i, entry in enumerate(config["datasets"]):
        validate_dataset_entry(entry, i)

    if "euler_train" in config:
        validate_euler_train_config(config["euler_train"])

    return config


def _find_zip_ancestor(path: Path):
    """Find a ``.zip`` file among the ancestors of *path*.

    Returns:
        ``(zip_path, internal_name)`` when a zip archive is detected in the
        path hierarchy, or ``(None, None)`` for plain filesystem paths.
    """
    parts = path.parts
    for i in range(1, len(parts) + 1):
        candidate = Path(*parts[:i])
        if candidate.suffix.lower() == ".zip" and candidate.is_file():
            remainder = "/".join(parts[i:])
            return candidate, remainder
    return None, None


def _save_json_to_zip(zip_path: Path, internal_name: str, data: dict) -> None:
    """Write JSON *data* into an existing zip archive at *internal_name*.

    If an entry with the same name already exists it is replaced.
    """
    json_bytes = json.dumps(data, indent=2).encode("utf-8")

    with zipfile.ZipFile(zip_path, "r") as zf:
        needs_replace = internal_name in zf.namelist()

    if needs_replace:
        tmp = zip_path.with_suffix(".zip.tmp")
        with zipfile.ZipFile(zip_path, "r") as zf_in:
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf_out:
                for item in zf_in.infolist():
                    if item.filename != internal_name:
                        zf_out.writestr(item, zf_in.read(item.filename))
                zf_out.writestr(internal_name, json_bytes)
        tmp.replace(zip_path)
    else:
        with zipfile.ZipFile(zip_path, "a") as zf:
            zf.writestr(internal_name, json_bytes)


def save_results(results: dict, dataset_config: dict) -> Path:
    """Save results to output file.

    Handles both plain directories and zip archives.  When the resolved
    output path passes through a ``.zip`` file the results are written
    into the archive instead of onto the filesystem.

    Args:
        results: Results dictionary.
        dataset_config: Dataset configuration entry.

    Returns:
        Path where results were saved.
    """
    output_file = dataset_config.get("output_file")
    if output_file is None:
        # Default: save alongside first available modality path
        for modality in ("depth", "rgb"):
            if modality in dataset_config and "path" in dataset_config[modality]:
                output_file = Path(dataset_config[modality]["path"]) / "eval.json"
                break
        if output_file is None:
            output_file = Path("eval.json")
    else:
        output_file = Path(output_file)

    zip_path, internal_name = _find_zip_ancestor(output_file)
    if zip_path is not None and internal_name:
        _save_json_to_zip(zip_path, internal_name, results)
    else:
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
        description="Evaluate depth and RGB datasets using euler_loading"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to config.json file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for computation (default: auto)",
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
        "--mask-sky",
        action="store_true",
        help="Mask sky regions from metrics using GT segmentation",
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
    parser.add_argument(
        "--depth-alignment",
        type=str,
        default="auto_affine",
        choices=["none", "auto_affine", "affine"],
        help="Depth alignment mode: none, auto_affine (default), or affine",
    )
    parser.add_argument(
        "--sns",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Deprecated alias for --depth-alignment auto_affine/none",
    )

    args = parser.parse_args()
    args.depth_alignment = resolve_depth_alignment(args.depth_alignment, args.sns)

    # Load and validate config
    try:
        config = load_config(args.config)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    requested_device = args.device
    args.device = resolve_device(requested_device)
    configure_torch_runtime(args.device)
    print_device_info(requested_device, args.device)
    print(f"Depth alignment: {args.depth_alignment}")
    print("-" * 60)

    # Check sky masking prerequisites
    if args.mask_sky:
        if "segmentation" not in config["gt"]:
            print(
                "Warning: --mask-sky requires gt.segmentation in config. "
                "Sky masking disabled.",
                file=sys.stderr,
            )
            args.mask_sky = False
        else:
            print("Sky masking enabled")
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

    # Initialize euler_train logging if configured
    et_config = config.get("euler_train")
    et_run = None
    if et_config is not None:
        et_run = _euler_train.init(
            dir=et_config["dir"],
            run_id=et_config.get("run_id"),
            run_name=et_config.get("run_name"),
        )
        print(f"euler_train logging enabled -> {et_run.dir}")
    print("-" * 60)

    gt = config["gt"]
    gt_depth_path = gt["depth"]["path"]
    gt_rgb_path = gt["rgb"]["path"]
    calibration_path = gt.get("calibration", {}).get("path")
    segmentation_path = (
        gt.get("segmentation", {}).get("path") if args.mask_sky else None
    )

    # Evaluate each prediction dataset
    for dataset_config in config["datasets"]:
        ds_name = dataset_config["name"]
        has_depth = "depth" in dataset_config and "path" in dataset_config["depth"]
        has_rgb = "rgb" in dataset_config and "path" in dataset_config["rgb"]

        all_results = {}
        et_eval_datasets = {}

        # Register evaluation as running before work begins
        if et_run is not None:
            et_run.add_evaluation(ds_name, name=ds_name, status="running")

        # -- Depth evaluation --
        depth_dataset = None
        if has_depth and not args.skip_depth:
            pred_depth_path = dataset_config["depth"]["path"]
            print(f"\n[DEPTH] Evaluating: '{ds_name}'")
            print(f"  GT:   {gt_depth_path}")
            print(f"  Pred: {pred_depth_path}")

            depth_dataset = build_depth_eval_dataset(
                gt_depth_path=gt_depth_path,
                pred_depth_path=pred_depth_path,
                calibration_path=calibration_path,
                segmentation_path=segmentation_path,
            )
            et_eval_datasets["depth"] = depth_dataset

            depth_meta = get_depth_metadata(depth_dataset)
            print(f"  scale_to_meters: {depth_meta['scale_to_meters']}")
            print(f"  radial_depth: {depth_meta['radial_depth']}")
            print(f"  Matched pairs: {len(depth_dataset)}")

            depth_results = evaluate_depth_samples(
                dataset=depth_dataset,
                scale_to_meters=depth_meta["scale_to_meters"],
                is_radial=depth_meta["radial_depth"],
                gt_name=gt.get("name", "GT"),
                pred_name=ds_name,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                verbose=args.verbose,
                sanity_checker=sanity_checker,
                sky_mask_enabled=args.mask_sky,
                alignment_mode=args.depth_alignment,
            )

            if sanity_checker is not None:
                sanity_checker.print_pair_report(ds_name, is_depth=True)

            for depth_key in ("depth", "depth_raw", "depth_aligned"):
                if depth_key in depth_results:
                    all_results[depth_key] = depth_results[depth_key]
            all_results.setdefault("per_file_metrics", {}).update(
                depth_results.get("per_file_metrics", {})
            )
            print_results(
                {k: v for k, v in depth_results.items() if k != "per_file_metrics"},
                f"DEPTH: {ds_name}",
            )

        # -- RGB evaluation --
        rgb_dataset = None
        if has_rgb and not args.skip_rgb:
            pred_rgb_path = dataset_config["rgb"]["path"]
            print(f"\n[RGB] Evaluating: '{ds_name}'")
            print(f"  GT:   {gt_rgb_path}")
            print(f"  Pred: {pred_rgb_path}")

            rgb_dataset = build_rgb_eval_dataset(
                gt_rgb_path=gt_rgb_path,
                pred_rgb_path=pred_rgb_path,
                gt_depth_path=gt_depth_path,
                calibration_path=calibration_path,
                segmentation_path=segmentation_path,
            )
            et_eval_datasets["rgb"] = rgb_dataset

            rgb_meta = get_rgb_metadata(rgb_dataset)
            print(f"  rgb_range: {rgb_meta['rgb_range']}")
            print(f"  Matched pairs: {len(rgb_dataset)}")

            depth_meta = (
                get_depth_metadata(rgb_dataset)
                if "gt_depth" in rgb_dataset.modality_paths()
                else None
            )

            rgb_results = evaluate_rgb_samples(
                dataset=rgb_dataset,
                depth_meta=depth_meta,
                gt_name=gt.get("name", "GT"),
                pred_name=ds_name,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                verbose=args.verbose,
                sanity_checker=sanity_checker,
                sky_mask_enabled=args.mask_sky,
            )

            if sanity_checker is not None:
                sanity_checker.print_pair_report(ds_name, is_depth=False)

            all_results["rgb"] = rgb_results.get("rgb", {})
            all_results.setdefault("per_file_metrics", {}).update(
                rgb_results.get("per_file_metrics", {})
            )
            print_results(
                {k: v for k, v in rgb_results.items() if k != "per_file_metrics"},
                f"RGB: {ds_name}",
            )

        # Save combined results
        if all_results:
            output_path = save_results(all_results, dataset_config)
            print(f"\n  Results saved to: {output_path}")

        # Log to euler_train
        if et_run is not None and et_eval_datasets:
            et_run.add_evaluation(
                ds_name,
                datasets=et_eval_datasets,
                # name=ds_name,
                # status="completed",
                metadata={
                    "results": {
                        k: v for k, v in all_results.items() if k != "per_file_metrics"
                    }
                },
            )
            et_run.finish_evaluation(ds_name)

    # Print sanity check report at the end
    if sanity_checker is not None:
        sanity_checker.print_report()

        report = sanity_checker.get_full_report()
        report_path = Path("sanity_check_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSanity check report saved to: {report_path}")

    if et_run is not None:
        if et_config.get("run_id") is not None:
            et_run.detach()
            print(f"\neuler_train run detached (run still active): {et_run.run_id}")
        else:
            et_run.finish()
            print(f"\neuler_train run finished: {et_run.run_id}")


if __name__ == "__main__":
    main()
