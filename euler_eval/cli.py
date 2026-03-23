#!/usr/bin/env python3
"""Main entry point for depth and RGB evaluation.

Parses config.json and runs evaluation using euler_loading datasets.
"""

import argparse
import importlib.metadata
import json
import math
import sys
import zipfile
from pathlib import Path

import torch

from .data import (
    build_depth_eval_dataset,
    build_rays_eval_dataset,
    build_rgb_eval_dataset,
    get_depth_metadata,
    get_rays_metadata,
    get_rgb_metadata,
)
from .evaluate import evaluate_depth_samples, evaluate_rays_samples, evaluate_rgb_samples
from .sanity_checker import SanityChecker

try:
    import euler_train as _euler_train
except ImportError:
    _euler_train = None


def _get_version() -> str:
    """Return the installed euler-eval version, falling back to ``"0.0.0"``."""
    try:
        return importlib.metadata.version("euler-eval")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def _clean_metric_tree(tree: dict) -> dict:
    """Recursively sanitize a metric dict for JSON schema compliance.

    - Removes entries where the value is ``None``
    - Removes entries where the value is a non-finite float (NaN, Inf)
    - Recursively processes nested dicts and list items
    - Prunes empty dicts after cleaning
    """
    cleaned = {}
    for key, value in tree.items():
        if value is None:
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if isinstance(value, dict):
            sub = _clean_metric_tree(value)
            if sub:
                cleaned[key] = sub
        elif isinstance(value, list):
            cleaned[key] = [
                _clean_metric_tree(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def _wrap_pfm_metrics(pfm: dict, wrapper_fn) -> dict:
    """Walk a ``perFileHierarchyNode`` tree, applying *wrapper_fn* to each
    file entry's ``metrics`` dict.  This restructures per-file metric keys
    to match the declared namespace path without changing ``evaluate.py``.
    """
    result = {}
    if "children" in pfm:
        result["children"] = {
            k: _wrap_pfm_metrics(v, wrapper_fn)
            for k, v in pfm["children"].items()
        }
    if "files" in pfm:
        result["files"] = [
            {"id": f["id"], "metrics": wrapper_fn(f["metrics"])}
            for f in pfm["files"]
        ]
    return result


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


def validate_gt_config(gt: dict) -> None:
    """Validate the ``gt`` section of the configuration.

    Raises:
        ValueError: If required fields are missing or paths do not exist.
    """
    has_rgb = "rgb" in gt and "path" in gt.get("rgb", {})
    has_depth = "depth" in gt and "path" in gt.get("depth", {})
    has_rays = "rays" in gt and "path" in gt.get("rays", {})

    if not has_rgb and not has_depth and not has_rays:
        raise ValueError(
            "gt must have at least one of 'rgb.path', 'depth.path', or 'rays.path'"
        )

    for modality in ("rgb", "depth", "rays", "segmentation", "calibration"):
        if modality in gt and "path" in gt[modality]:
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
    has_rays = "rays" in entry and "path" in entry.get("rays", {})
    if not has_rgb and not has_depth and not has_rays:
        raise ValueError(
            f"{label} must have at least 'rgb.path', 'depth.path', or 'rays.path'"
        )

    for modality in ("rgb", "depth", "rays"):
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
    unknown = set(et_config) - {"dir"}
    if unknown:
        raise ValueError(
            f"Unknown euler_train config keys: {unknown}. "
            f"Only 'dir' is supported — pass a project directory for a new run "
            f"or a run directory (containing meta.json) to resume."
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


def save_results(
    results: dict, dataset_config: dict, modality: str | None = None
) -> Path:
    """Save results to output file.

    Handles both plain directories and zip archives.  When the resolved
    output path passes through a ``.zip`` file the results are written
    into the archive instead of onto the filesystem.

    Args:
        results: Results dictionary.
        dataset_config: Dataset configuration entry.
        modality: When set, save to this specific modality's path
            (e.g. ``"depth"`` or ``"rgb"``).  Falls back to the first
            available modality path when *None*.

    Returns:
        Path where results were saved.
    """
    output_file = dataset_config.get("output_file")
    if output_file is None:
        if (
            modality is not None
            and modality in dataset_config
            and "path" in dataset_config[modality]
        ):
            output_file = Path(dataset_config[modality]["path"]) / "eval.json"
        else:
            # Default: save alongside first available modality path
            for mod in ("depth", "rgb", "rays"):
                if mod in dataset_config and "path" in dataset_config[mod]:
                    output_file = Path(dataset_config[mod]["path"]) / "eval.json"
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
        "--skip-rays",
        action="store_true",
        help="Skip rays (spherical direction map) evaluation",
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
        "--rgb-fid-backend",
        type=str,
        default="builtin",
        choices=["builtin", "clean-fid"],
        help="Backend for RGB FID computation: builtin (default) or clean-fid",
    )

    args = parser.parse_args()

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
    print(f"RGB FID backend: {args.rgb_fid_backend}")
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
    et_resuming = False
    if et_config is not None:
        et_dir = Path(et_config["dir"])
        et_resuming = (et_dir / "meta.json").is_file()
        et_run = _euler_train.init(
            dir=et_dir,
            mode="eval",
        )
        print(f"euler_train logging enabled -> {et_run.dir}")
    print("-" * 60)

    gt = config["gt"]
    gt_depth_path = gt.get("depth", {}).get("path")
    gt_rgb_path = gt.get("rgb", {}).get("path")
    gt_rays_path = gt.get("rays", {}).get("path")
    calibration_path = gt.get("calibration", {}).get("path")
    segmentation_path = (
        gt.get("segmentation", {}).get("path") if args.mask_sky else None
    )

    # Evaluate each prediction dataset
    for dataset_config in config["datasets"]:
        ds_name = dataset_config["name"]
        has_depth = "depth" in dataset_config and "path" in dataset_config["depth"]
        has_rgb = "rgb" in dataset_config and "path" in dataset_config["rgb"]
        has_rays = "rays" in dataset_config and "path" in dataset_config["rays"]

        all_results = {}
        depth_save = {}
        rgb_save = {}
        rays_save = {}
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
            print(f"  radial_depth: {depth_meta['radial_depth']}")
            print(f"  Matched pairs: {len(depth_dataset)}")

            depth_results = evaluate_depth_samples(
                dataset=depth_dataset,
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

            # Build per-modality results for saving.
            # All metric names must be fully-qualified under the declared
            # metricNamespace.  We nest raw/aligned under depth → eval so
            # every flattened path starts with "depth.eval.".
            alignment_info = depth_results.get("alignment", {})
            depth_dataset_info = depth_results.get("dataset_info", {})

            depth_save = {
                "metricSet": {
                    "metricNamespace": "depth.eval",
                    "producerKey": "euler-eval",
                    "producerVersion": _get_version(),
                    "sourceKind": "computed",
                    "metadata": {
                        "alignment_mode": alignment_info.get("mode", "unknown"),
                        "alignment_applied": alignment_info.get("applied", False),
                    },
                },
                "dataset_info": depth_dataset_info,
                "depth": {
                    "eval": {
                        "raw": _clean_metric_tree(depth_results["depth_raw"]),
                        "aligned": _clean_metric_tree(
                            depth_results["depth_aligned"]
                        ),
                    },
                },
            }
            for depth_key in ("depth", "depth_raw", "depth_aligned"):
                if depth_key in depth_results:
                    all_results[depth_key] = depth_results[depth_key]
            depth_pfm = depth_results.get("per_file_metrics", {})
            if depth_pfm:
                depth_save["per_file_metrics"] = _clean_metric_tree(
                    _wrap_pfm_metrics(
                        depth_pfm,
                        lambda m: {
                            "depth": {
                                "eval": {
                                    "raw": m.get("depth_raw", {}),
                                    "aligned": m.get("depth_aligned", {}),
                                },
                            },
                        },
                    )
                )
            all_results.setdefault("per_file_metrics", {}).update(depth_pfm)

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
                fid_backend=args.rgb_fid_backend,
            )

            if sanity_checker is not None:
                sanity_checker.print_pair_report(ds_name, is_depth=False)

            # Build per-modality results for saving.
            # All metric names must be fully-qualified under the declared
            # metricNamespace.  We nest metrics under rgb → eval so every
            # flattened path starts with "rgb.eval.".
            rgb_dataset_info = rgb_results.get("dataset_info", {})

            rgb_save = {
                "metricSet": {
                    "metricNamespace": "rgb.eval",
                    "producerKey": "euler-eval",
                    "producerVersion": _get_version(),
                    "sourceKind": "computed",
                },
                "dataset_info": rgb_dataset_info,
            }
            rgb_metrics = rgb_results.get("rgb", {})
            if rgb_metrics:
                rgb_save["rgb"] = {"eval": _clean_metric_tree(rgb_metrics)}
                all_results["rgb"] = rgb_metrics
            rgb_pfm = rgb_results.get("per_file_metrics", {})
            if rgb_pfm:
                rgb_save["per_file_metrics"] = _clean_metric_tree(
                    _wrap_pfm_metrics(
                        rgb_pfm,
                        lambda m: {"rgb": {"eval": m.get("rgb", {})}},
                    )
                )
            all_results.setdefault("per_file_metrics", {}).update(rgb_pfm)

            print_results(
                {k: v for k, v in rgb_results.items() if k != "per_file_metrics"},
                f"RGB: {ds_name}",
            )

        # -- Rays (spherical direction map) evaluation --
        rays_dataset = None
        if has_rays and gt_rays_path and not args.skip_rays:
            pred_rays_path = dataset_config["rays"]["path"]
            print(f"\n[RAYS] Evaluating: '{ds_name}'")
            print(f"  GT:   {gt_rays_path}")
            print(f"  Pred: {pred_rays_path}")

            rays_dataset = build_rays_eval_dataset(
                gt_rays_path=gt_rays_path,
                pred_rays_path=pred_rays_path,
                calibration_path=calibration_path,
            )
            et_eval_datasets["rays"] = rays_dataset

            rays_meta = get_rays_metadata(rays_dataset)
            print(f"  fov_domain: {rays_meta['fov_domain'] or 'auto-detect'}")
            print(f"  Matched pairs: {len(rays_dataset)}")

            rays_results = evaluate_rays_samples(
                dataset=rays_dataset,
                fov_domain=rays_meta["fov_domain"],
                gt_name=gt.get("name", "GT"),
                pred_name=ds_name,
                verbose=args.verbose,
                sanity_checker=sanity_checker,
            )

            if sanity_checker is not None:
                sanity_checker.print_pair_report(ds_name, modality="rays")

            rays_dataset_info = rays_results.get("dataset_info", {})

            rays_save = {
                "metricSet": {
                    "metricNamespace": "rays.eval",
                    "producerKey": "euler-eval",
                    "producerVersion": _get_version(),
                    "sourceKind": "computed",
                    "metadata": {
                        "fov_domain": rays_dataset_info.get("fov_domain"),
                        "threshold_deg": rays_dataset_info.get("threshold_deg"),
                    },
                },
                "dataset_info": rays_dataset_info,
            }
            rays_metrics = rays_results.get("rays", {})
            if rays_metrics:
                rays_save["rays"] = {"eval": _clean_metric_tree(rays_metrics)}
                all_results["rays"] = rays_metrics
            rays_pfm = rays_results.get("per_file_metrics", {})
            if rays_pfm:
                rays_save["per_file_metrics"] = _clean_metric_tree(
                    _wrap_pfm_metrics(
                        rays_pfm,
                        lambda m: {"rays": {"eval": m.get("rays", {})}},
                    )
                )
            all_results.setdefault("per_file_metrics", {}).update(rays_pfm)

            print_results(
                {k: v for k, v in rays_results.items() if k != "per_file_metrics"},
                f"RAYS: {ds_name}",
            )

        # Save per-modality results to respective dataset paths
        if depth_save:
            depth_out = save_results(depth_save, dataset_config, modality="depth")
            print(f"\n  Depth results saved to: {depth_out}")
        if rgb_save:
            rgb_out = save_results(rgb_save, dataset_config, modality="rgb")
            print(f"\n  RGB results saved to: {rgb_out}")
        if rays_save:
            rays_out = save_results(rays_save, dataset_config, modality="rays")
            print(f"\n  Rays results saved to: {rays_out}")

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
        if et_resuming:
            et_run.detach()
            print(f"\neuler_train run detached (run still active): {et_run.run_id}")
        else:
            et_run.finish()
            print(f"\neuler_train run finished: {et_run.run_id}")


if __name__ == "__main__":
    main()
