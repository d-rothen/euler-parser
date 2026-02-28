"""Learned Perceptual Image Patch Similarity (LPIPS) metric for depth maps."""

from typing import Union

import lpips
import numpy as np
import torch


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


class LPIPSMetric:
    """LPIPS metric calculator with GPU support.

    Uses a pre-trained network (default: AlexNet) to compute perceptual similarity.
    """

    def __init__(
        self,
        net: str = "alex",
        device: Union[str, torch.device] = "cuda",
    ):
        """Initialize LPIPS metric.

        Args:
            net: Network to use ('alex', 'vgg', 'squeeze').
            device: Device to run computation on.
        """
        self.device = _resolve_device(device)
        self._non_blocking = self.device.type == "cuda"
        self.model = lpips.LPIPS(net=net).to(self.device)
        self.model.eval()

    def _prepare_input(self, depth: np.ndarray) -> torch.Tensor:
        """Prepare depth map for LPIPS computation.

        Args:
            depth: Depth map in meters (H, W).

        Returns:
            Normalized tensor in range [-1, 1] with shape (1, 3, H, W).
        """
        # Normalize to [0, 1]
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

        normalized = np.clip(normalized, 0, 1)

        # Convert to [-1, 1] range (LPIPS expects this)
        normalized = normalized * 2 - 1

        # Create 3-channel tensor
        tensor = torch.from_numpy(normalized).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        tensor = tensor.expand(-1, 3, -1, -1)  # (1, 3, H, W)

        return tensor.to(self.device, non_blocking=self._non_blocking)

    def compute(
        self,
        depth_pred: np.ndarray,
        depth_gt: np.ndarray,
    ) -> float:
        """Compute LPIPS between predicted and ground truth depth maps.

        Args:
            depth_pred: Predicted depth map in meters.
            depth_gt: Ground truth depth map in meters.

        Returns:
            LPIPS value. Lower is better (more similar).
        """
        with torch.inference_mode():
            pred_tensor = self._prepare_input(depth_pred)
            gt_tensor = self._prepare_input(depth_gt)

            lpips_value = self.model(pred_tensor, gt_tensor)

        return float(lpips_value.item())

    def compute_batch(
        self,
        depths_pred: list[np.ndarray],
        depths_gt: list[np.ndarray],
        batch_size: int = 16,
    ) -> list[float]:
        """Compute LPIPS for a batch of depth map pairs.

        Args:
            depths_pred: List of predicted depth maps.
            depths_gt: List of ground truth depth maps.
            batch_size: Batch size for processing.

        Returns:
            List of LPIPS values.
        """
        results = []

        for i in range(0, len(depths_pred), batch_size):
            batch_pred = depths_pred[i : i + batch_size]
            batch_gt = depths_gt[i : i + batch_size]

            with torch.inference_mode():
                pred_tensors = torch.cat(
                    [self._prepare_input(d) for d in batch_pred], dim=0
                )
                gt_tensors = torch.cat(
                    [self._prepare_input(d) for d in batch_gt], dim=0
                )

                lpips_values = self.model(pred_tensors, gt_tensors)
                values = lpips_values.reshape(-1).detach().float().cpu().numpy()
                results.extend(values.tolist())

        return results


def compute_lpips(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    net: str = "alex",
    device: str = "cuda",
) -> float:
    """Compute LPIPS between predicted and ground truth depth maps.

    Convenience function that creates a one-time LPIPS metric.
    For multiple computations, use LPIPSMetric class directly.

    Args:
        depth_pred: Predicted depth map in meters.
        depth_gt: Ground truth depth map in meters.
        net: Network to use ('alex', 'vgg', 'squeeze').
        device: Device to run computation on.

    Returns:
        LPIPS value. Lower is better.
    """
    metric = LPIPSMetric(net=net, device=device)
    return metric.compute(depth_pred, depth_gt)
