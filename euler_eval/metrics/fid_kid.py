"""Fréchet Inception Distance (FID) and Kernel Inception Distance (KID) metrics."""

import tempfile
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Inception_V3_Weights, inception_v3
from tqdm import tqdm


def _resolve_device(device: Union[str, torch.device]) -> torch.device:
    """Resolve a metric runtime device with CUDA fallback."""
    if isinstance(device, torch.device):
        requested = device
    else:
        name = str(device)
        if name == "auto":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        requested = torch.device(name)

    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def _normalize_depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    """Convert a depth map into a normalized 3-channel image."""
    valid_mask = (depth > 0) & np.isfinite(depth)
    if valid_mask.any():
        min_val = depth[valid_mask].min()
        max_val = depth[valid_mask].max()
        if max_val - min_val > 1e-8:
            normalized = (depth - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(depth)
    else:
        normalized = np.zeros_like(depth)

    normalized = np.clip(normalized, 0, 1).astype(np.float32)
    return np.stack([normalized, normalized, normalized], axis=-1)


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Convert an RGB float image in [0, 1] to uint8 for file-based FID tools."""
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(
            f"Unsupported RGB image shape {arr.shape}. Expected (H, W, 3)."
        )
    arr = np.clip(arr, 0.0, 1.0)
    return np.rint(arr * 255.0).astype(np.uint8)


def _write_rgb_image_folder(images: list[np.ndarray], folder: str) -> None:
    """Write a list of RGB images to a folder as lossless PNG files."""
    for index, image in enumerate(images):
        path = f"{folder}/{index:06d}.png"
        Image.fromarray(_to_uint8_rgb(image), mode="RGB").save(path, format="PNG")


def compute_clean_fid(
    images1: list[np.ndarray],
    images2: list[np.ndarray],
    *,
    mode: str = "clean",
    batch_size: int = 32,
    num_workers: int = 12,
    device: Union[str, torch.device] = "cuda",
    verbose: bool = False,
) -> float:
    """Compute RGB FID via the official clean-fid implementation.

    Images are written to temporary PNG folders so clean-fid can apply its
    own quantization and resizing pipeline.
    """
    try:
        from cleanfid import fid as clean_fid
    except ImportError as exc:
        raise ImportError(
            "clean-fid backend requested, but the 'clean-fid' package is not "
            "installed. Install it with `pip install clean-fid`."
        ) from exc

    runtime_device = _resolve_device(device)

    with tempfile.TemporaryDirectory(prefix="euler_eval_cleanfid_gt_") as gt_dir:
        with tempfile.TemporaryDirectory(prefix="euler_eval_cleanfid_pred_") as pred_dir:
            _write_rgb_image_folder(images1, gt_dir)
            _write_rgb_image_folder(images2, pred_dir)
            return float(
                clean_fid.compute_fid(
                    gt_dir,
                    pred_dir,
                    mode=mode,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    device=runtime_device,
                    verbose=verbose,
                )
            )


class DepthDataset(Dataset):
    """Dataset wrapper for depth maps."""

    def __init__(
        self,
        depth_maps: list[np.ndarray],
        transform: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
    ):
        self.depth_maps = depth_maps
        self.transform = transform

    def __len__(self) -> int:
        return len(self.depth_maps)

    def __getitem__(self, idx: int) -> torch.Tensor:
        depth = self.depth_maps[idx]
        if self.transform:
            return self.transform(depth)

        img = _normalize_depth_to_rgb(depth)
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()


class RGBDataset(Dataset):
    """Dataset wrapper for RGB images."""

    def __init__(
        self,
        images: list[np.ndarray],
        transform: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
    ):
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = np.asarray(self.images[idx], dtype=np.float32)
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(
                f"Unsupported RGB image shape {img.shape}. Expected (H, W, 3)."
            )
        if self.transform:
            return self.transform(img)

        img = np.clip(img, 0, 1).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()


class FIDKIDMetric:
    """FID and KID metric calculator using Inception v3 features."""

    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
    ):
        """Initialize FID/KID metric.

        Args:
            device: Device to run computation on.
        """
        self.device = _resolve_device(device)
        self._non_blocking = self.device.type == "cuda"

        # Load Inception v3 and modify for feature extraction
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.model.fc = nn.Identity()  # Remove final classifier
        self.model = self.model.to(self.device)
        self.model.eval()

        self.inception_input_size = 299
        self._imagenet_mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32
        ).view(1, 3, 1, 1)
        self._imagenet_std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32
        ).view(1, 3, 1, 1)
        self._cached_pair_key: tuple[int, int, int, int, int, int, str] | None = None
        self._cached_pair_features: tuple[np.ndarray, np.ndarray] | None = None

    def _build_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        collate_fn: Optional[Callable[[list[torch.Tensor]], torch.Tensor]] = None,
    ) -> DataLoader:
        """Create a DataLoader tuned for the active device."""
        kwargs: dict[str, object] = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": self.device.type == "cuda",
        }
        if collate_fn is not None:
            kwargs["collate_fn"] = collate_fn

        if num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 4

        if self.device.type == "cuda":
            kwargs["pin_memory_device"] = "cuda"

        try:
            return DataLoader(dataset, **kwargs)
        except TypeError:
            kwargs.pop("pin_memory_device", None)
            return DataLoader(dataset, **kwargs)

    def _normalize_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization to a batched tensor."""
        mean = self._imagenet_mean.to(device=batch.device, dtype=batch.dtype)
        std = self._imagenet_std.to(device=batch.device, dtype=batch.dtype)
        return (batch - mean) / std

    def _resize_tensor(
        self,
        tensor: torch.Tensor,
        preserve_aspect: bool,
    ) -> torch.Tensor:
        """Resize a CHW tensor for Inception feature extraction."""
        if preserve_aspect:
            height, width = tensor.shape[-2:]
            short_side = min(height, width)
            if short_side <= 0:
                raise ValueError(
                    f"Invalid input shape {(height, width)} for FID preprocessing."
                )
            scale = self.inception_input_size / float(short_side)
            target_h = max(1, int(round(height * scale)))
            target_w = max(1, int(round(width * scale)))
        else:
            target_h = self.inception_input_size
            target_w = self.inception_input_size

        batch = tensor.unsqueeze(0)
        try:
            resized = F.interpolate(
                batch,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        except TypeError:
            resized = F.interpolate(
                batch,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
        return resized.squeeze(0)

    def _prepare_depth_input(self, depth: np.ndarray) -> torch.Tensor:
        """Convert a depth map into a resized tensor for Inception."""
        img = _normalize_depth_to_rgb(depth)
        tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
        return self._resize_tensor(tensor, preserve_aspect=False)

    def _prepare_rgb_input(self, img: np.ndarray) -> torch.Tensor:
        """Convert an RGB image into an aspect-preserving tensor for Inception."""
        img = np.asarray(img, dtype=np.float32)
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(
                f"Unsupported RGB image shape {img.shape}. Expected (H, W, 3)."
            )
        img = np.clip(img, 0, 1).astype(np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
        return self._resize_tensor(tensor, preserve_aspect=True)

    def _pad_collate(self, batch: list[torch.Tensor]) -> torch.Tensor:
        """Pad variable-sized CHW tensors to the batch maximum size and stack."""
        if not batch:
            raise ValueError("Cannot collate an empty batch.")

        channels = batch[0].shape[0]
        max_h = max(item.shape[-2] for item in batch)
        max_w = max(item.shape[-1] for item in batch)
        padded = torch.zeros(
            (len(batch), channels, max_h, max_w), dtype=batch[0].dtype
        )

        for index, item in enumerate(batch):
            _, height, width = item.shape
            offset_y = (max_h - height) // 2
            offset_x = (max_w - width) // 2
            padded[index, :, offset_y : offset_y + height, offset_x : offset_x + width] = item

        return padded

    def _extract_features(
        self,
        depth_maps: list[np.ndarray],
        batch_size: int = 16,
        num_workers: int = 4,
    ) -> np.ndarray:
        """Extract Inception features from depth maps.

        Args:
            depth_maps: List of depth maps.
            batch_size: Batch size for processing.
            num_workers: Number of data loading workers.

        Returns:
            Feature array of shape (N, 2048).
        """
        dataset = DepthDataset(depth_maps, transform=self._prepare_depth_input)
        dataloader = self._build_dataloader(
            dataset, batch_size, num_workers, collate_fn=self._pad_collate
        )

        features = []

        with torch.inference_mode():
            for batch in tqdm(dataloader, desc="Extracting features", leave=False):
                batch = batch.to(self.device, non_blocking=self._non_blocking)
                batch = self._normalize_batch(batch)
                feat = self.model(batch)
                features.append(feat.cpu().numpy())

        return np.concatenate(features, axis=0)

    def _extract_rgb_features(
        self,
        images: list[np.ndarray],
        batch_size: int = 16,
        num_workers: int = 4,
    ) -> np.ndarray:
        """Extract Inception features from RGB images."""
        dataset = RGBDataset(images, transform=self._prepare_rgb_input)
        dataloader = self._build_dataloader(
            dataset, batch_size, num_workers, collate_fn=self._pad_collate
        )

        features = []

        with torch.inference_mode():
            for batch in tqdm(dataloader, desc="Extracting features", leave=False):
                batch = batch.to(self.device, non_blocking=self._non_blocking)
                batch = self._normalize_batch(batch)
                feat = self.model(batch)
                features.append(feat.cpu().numpy())

        return np.concatenate(features, axis=0)

    def _get_feature_pair(
        self,
        depths1: list[np.ndarray],
        depths2: list[np.ndarray],
        batch_size: int,
        num_workers: int,
        input_kind: str = "depth",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get feature matrices for a pair of datasets with simple memoization."""
        key = (
            id(depths1),
            len(depths1),
            id(depths2),
            len(depths2),
            batch_size,
            num_workers,
            input_kind,
        )
        if self._cached_pair_key == key and self._cached_pair_features is not None:
            return self._cached_pair_features

        if input_kind == "depth":
            extractor = self._extract_features
        elif input_kind == "rgb":
            extractor = self._extract_rgb_features
        else:
            raise ValueError(
                f"Unknown input_kind '{input_kind}'. Expected 'depth' or 'rgb'."
            )

        features1 = extractor(depths1, batch_size, num_workers)
        features2 = extractor(depths2, batch_size, num_workers)
        self._cached_pair_key = key
        self._cached_pair_features = (features1, features2)
        return features1, features2

    def _compute_fid_from_features(
        self,
        features1: np.ndarray,
        features2: np.ndarray,
    ) -> float:
        """Compute FID from pre-extracted feature matrices."""
        # Compute statistics
        mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
        mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)

        # Compute FID
        diff = mu1 - mu2

        # Compute sqrt of product of covariances
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

        # Handle numerical issues
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

        # Handle complex numbers from numerical errors
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)

        return float(fid)

    def _compute_kid_from_features(
        self,
        features1: np.ndarray,
        features2: np.ndarray,
        num_subsets: int = 100,
        subset_size: int = 1000,
    ) -> tuple[float, float]:
        """Compute KID from pre-extracted feature matrices."""
        # Adjust subset size if needed
        n1, n2 = len(features1), len(features2)
        subset_size = min(subset_size, n1, n2)

        kid_values = []

        for _ in range(num_subsets):
            # Random subsets
            idx1 = np.random.choice(n1, subset_size, replace=False)
            idx2 = np.random.choice(n2, subset_size, replace=False)

            f1 = features1[idx1]
            f2 = features2[idx2]

            # Polynomial kernel: (x^T y / d + 1)^3
            d = features1.shape[1]

            k11 = (f1 @ f1.T / d + 1) ** 3
            k22 = (f2 @ f2.T / d + 1) ** 3
            k12 = (f1 @ f2.T / d + 1) ** 3

            # MMD^2 estimator
            m = subset_size
            mmd2 = (
                (k11.sum() - np.diag(k11).sum()) / (m * (m - 1))
                + (k22.sum() - np.diag(k22).sum()) / (m * (m - 1))
                - 2 * k12.mean()
            )

            kid_values.append(mmd2)

        return float(np.mean(kid_values)), float(np.std(kid_values))

    def compute_fid(
        self,
        depths1: list[np.ndarray],
        depths2: list[np.ndarray],
        batch_size: int = 16,
        num_workers: int = 4,
    ) -> float:
        """Compute FID between two sets of depth maps."""
        features1, features2 = self._get_feature_pair(
            depths1, depths2, batch_size, num_workers
        )
        return self._compute_fid_from_features(features1, features2)

    def compute_rgb_fid(
        self,
        images1: list[np.ndarray],
        images2: list[np.ndarray],
        batch_size: int = 16,
        num_workers: int = 4,
    ) -> float:
        """Compute FID between two sets of RGB images."""
        features1, features2 = self._get_feature_pair(
            images1,
            images2,
            batch_size,
            num_workers,
            input_kind="rgb",
        )
        return self._compute_fid_from_features(features1, features2)

    def compute_kid(
        self,
        depths1: list[np.ndarray],
        depths2: list[np.ndarray],
        batch_size: int = 16,
        num_workers: int = 4,
        num_subsets: int = 100,
        subset_size: int = 1000,
    ) -> tuple[float, float]:
        """Compute KID between two sets of depth maps."""
        features1, features2 = self._get_feature_pair(
            depths1, depths2, batch_size, num_workers
        )
        return self._compute_kid_from_features(
            features1,
            features2,
            num_subsets=num_subsets,
            subset_size=subset_size,
        )

    def compute_fid_kid(
        self,
        depths1: list[np.ndarray],
        depths2: list[np.ndarray],
        batch_size: int = 16,
        num_workers: int = 4,
        num_subsets: int = 100,
        subset_size: int = 1000,
    ) -> tuple[float, float, float]:
        """Compute FID and KID in one pass of feature extraction."""
        features1, features2 = self._get_feature_pair(
            depths1, depths2, batch_size, num_workers
        )
        fid = self._compute_fid_from_features(features1, features2)
        kid_mean, kid_std = self._compute_kid_from_features(
            features1,
            features2,
            num_subsets=num_subsets,
            subset_size=subset_size,
        )
        return fid, kid_mean, kid_std


def compute_fid(
    depths1: list[np.ndarray],
    depths2: list[np.ndarray],
    device: str = "cuda",
    batch_size: int = 16,
) -> float:
    """Compute FID between two sets of depth maps.

    Convenience function that creates a one-time FID/KID metric.

    Args:
        depths1: First set of depth maps.
        depths2: Second set of depth maps.
        device: Device to run computation on.
        batch_size: Batch size for feature extraction.

    Returns:
        FID score. Lower is better.
    """
    metric = FIDKIDMetric(device=device)
    return metric.compute_fid(depths1, depths2, batch_size)


def compute_kid(
    depths1: list[np.ndarray],
    depths2: list[np.ndarray],
    device: str = "cuda",
    batch_size: int = 16,
) -> tuple[float, float]:
    """Compute KID between two sets of depth maps.

    Convenience function that creates a one-time FID/KID metric.

    Args:
        depths1: First set of depth maps.
        depths2: Second set of depth maps.
        device: Device to run computation on.
        batch_size: Batch size for feature extraction.

    Returns:
        Tuple of (KID mean, KID std). Lower is better.
    """
    metric = FIDKIDMetric(device=device)
    return metric.compute_kid(depths1, depths2, batch_size)
