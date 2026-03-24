"""Cache warmup entrypoint for offline cluster jobs.

This script pre-downloads the model artifacts used by euler-eval:
- torchvision AlexNet weights (used by LPIPS)
- torchvision Inception v3 weights (used by builtin FID/KID)
- LPIPS AlexNet weights
- clean-fid inception checkpoint when clean-fid is installed
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _configure_torch_home() -> tuple[Path | None, bool]:
    """Derive TORCH_HOME from HF_HOME when TORCH_HOME is unset."""
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        return Path(torch_home), False

    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        return None, False

    derived = Path(hf_home) / "torch"
    os.environ["TORCH_HOME"] = str(derived)
    return derived, True


def _warm_torchvision_models() -> list[str]:
    """Download torchvision weights used by the evaluator."""
    from torchvision.models import (
        AlexNet_Weights,
        Inception_V3_Weights,
        alexnet,
        inception_v3,
    )

    alexnet(weights=AlexNet_Weights.DEFAULT)
    inception_v3(weights=Inception_V3_Weights.DEFAULT)
    return [
        "torchvision AlexNet weights",
        "torchvision Inception v3 weights",
    ]


def _warm_lpips() -> str:
    """Download or initialize LPIPS AlexNet weights."""
    import lpips

    model = lpips.LPIPS(net="alex")
    model.eval()
    return "LPIPS AlexNet weights"


def _warm_clean_fid() -> str | None:
    """Download the clean-fid inception checkpoint if clean-fid is installed."""
    try:
        from cleanfid import downloads_helper as clean_downloads
    except ImportError:
        return None

    cache_dir_value = os.environ.get("CLEANFID_CACHE_DIR")
    if cache_dir_value:
        cache_dir = Path(cache_dir_value)
    else:
        from euler_eval.metrics.fid_kid import _get_clean_fid_model_target_dir

        cache_dir = _get_clean_fid_model_target_dir()

    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = clean_downloads.check_download_inception(str(cache_dir))
    return f"clean-fid inception checkpoint: {checkpoint}"


def main() -> None:
    """Warm the caches needed for offline evaluation."""
    torch_home, derived_torch_home = _configure_torch_home()

    if derived_torch_home:
        print(f"TORCH_HOME not set; deriving it from HF_HOME -> {torch_home}")
    elif torch_home is not None:
        print(f"Using TORCH_HOME={torch_home}")
    else:
        print("Using torch default cache location (TORCH_HOME and HF_HOME unset)")

    cleanfid_cache = os.environ.get("CLEANFID_CACHE_DIR")
    if cleanfid_cache:
        print(f"Using CLEANFID_CACHE_DIR={cleanfid_cache}")

    warmed: list[str] = []
    failures: list[str] = []

    try:
        warmed.extend(_warm_torchvision_models())
    except Exception as exc:  # pragma: no cover - environment dependent
        failures.append(f"torchvision weights: {exc}")

    try:
        warmed.append(_warm_lpips())
    except Exception as exc:  # pragma: no cover - environment dependent
        failures.append(f"LPIPS weights: {exc}")

    try:
        clean_fid_result = _warm_clean_fid()
    except Exception as exc:  # pragma: no cover - environment dependent
        failures.append(f"clean-fid checkpoint: {exc}")
    else:
        if clean_fid_result is None:
            print("clean-fid not installed; skipping optional clean-fid checkpoint")
        else:
            warmed.append(clean_fid_result)

    if warmed:
        print("Warmup complete:")
        for item in warmed:
            print(f"  - {item}")

    if failures:
        print("Warmup failed for:", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
