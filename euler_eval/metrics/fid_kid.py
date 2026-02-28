"""Fréchet Inception Distance (FID) and Kernel Inception Distance (KID) metrics."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from typing import Optional, Union
from scipy import linalg
from tqdm import tqdm


class DepthDataset(Dataset):
    """Dataset wrapper for depth maps."""

    def __init__(
        self,
        depth_maps: list[np.ndarray],
        transform: Optional[transforms.Compose] = None,
    ):
        self.depth_maps = depth_maps
        self.transform = transform

    def __len__(self) -> int:
        return len(self.depth_maps)

    def __getitem__(self, idx: int) -> torch.Tensor:
        depth = self.depth_maps[idx]

        # Normalize depth to [0, 1]
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

        # Convert to 3-channel image
        img = np.stack([normalized, normalized, normalized], axis=-1)

        if self.transform:
            img = self.transform(img)

        return img


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
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load Inception v3 and modify for feature extraction
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.model.fc = nn.Identity()  # Remove final classifier
        self.model = self.model.to(self.device)
        self.model.eval()

        # Transform for Inception v3
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((299, 299), antialias=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

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
        dataset = DepthDataset(depth_maps, transform=self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        features = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features", leave=False):
                batch = batch.to(self.device)
                feat = self.model(batch)
                features.append(feat.cpu().numpy())

        return np.concatenate(features, axis=0)

    def compute_fid(
        self,
        depths1: list[np.ndarray],
        depths2: list[np.ndarray],
        batch_size: int = 16,
        num_workers: int = 4,
    ) -> float:
        """Compute FID between two sets of depth maps.

        Args:
            depths1: First set of depth maps.
            depths2: Second set of depth maps.
            batch_size: Batch size for feature extraction.
            num_workers: Number of data loading workers.

        Returns:
            FID score. Lower is better.
        """
        # Extract features
        features1 = self._extract_features(depths1, batch_size, num_workers)
        features2 = self._extract_features(depths2, batch_size, num_workers)

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

    def compute_kid(
        self,
        depths1: list[np.ndarray],
        depths2: list[np.ndarray],
        batch_size: int = 16,
        num_workers: int = 4,
        num_subsets: int = 100,
        subset_size: int = 1000,
    ) -> tuple[float, float]:
        """Compute KID between two sets of depth maps.

        Args:
            depths1: First set of depth maps.
            depths2: Second set of depth maps.
            batch_size: Batch size for feature extraction.
            num_workers: Number of data loading workers.
            num_subsets: Number of subsets for KID computation.
            subset_size: Size of each subset.

        Returns:
            Tuple of (KID mean, KID std). Lower is better.
        """
        # Extract features
        features1 = self._extract_features(depths1, batch_size, num_workers)
        features2 = self._extract_features(depths2, batch_size, num_workers)

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
