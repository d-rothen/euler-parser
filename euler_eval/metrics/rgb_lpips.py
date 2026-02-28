"""LPIPS metric for RGB images."""

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


class RGBLPIPSMetric:
    """LPIPS metric calculator for RGB images with GPU support."""

    def __init__(
        self,
        net: str = "alex",
        device: Union[str, torch.device] = "cuda",
    ):
        """Initialize RGB LPIPS metric.

        Args:
            net: Network to use ('alex', 'vgg', 'squeeze').
            device: Device to run computation on.
        """
        self.device = _resolve_device(device)
        self._non_blocking = self.device.type == "cuda"
        self.model = lpips.LPIPS(net=net).to(self.device)
        self.model.eval()

    def _prepare_input(self, img: np.ndarray) -> torch.Tensor:
        """Prepare RGB image for LPIPS computation.

        Args:
            img: RGB image in [0, 1] range with shape (H, W, 3).

        Returns:
            Tensor in range [-1, 1] with shape (1, 3, H, W).
        """
        # Convert to [-1, 1] range (LPIPS expects this)
        normalized = img * 2 - 1

        # Convert to tensor: (H, W, 3) -> (1, 3, H, W)
        tensor = torch.from_numpy(normalized).float()
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)

        return tensor.to(self.device, non_blocking=self._non_blocking)

    def compute(
        self,
        img_pred: np.ndarray,
        img_gt: np.ndarray,
    ) -> float:
        """Compute LPIPS between predicted and ground truth RGB images.

        Args:
            img_pred: Predicted RGB image in [0, 1] range, shape (H, W, 3).
            img_gt: Ground truth RGB image in [0, 1] range, shape (H, W, 3).

        Returns:
            LPIPS value. Lower is better (more similar).
        """
        with torch.inference_mode():
            pred_tensor = self._prepare_input(img_pred)
            gt_tensor = self._prepare_input(img_gt)

            lpips_value = self.model(pred_tensor, gt_tensor)

        return float(lpips_value.item())

    def compute_batch(
        self,
        imgs_pred: list[np.ndarray],
        imgs_gt: list[np.ndarray],
        batch_size: int = 16,
    ) -> list[float]:
        """Compute LPIPS for a batch of RGB image pairs.

        Args:
            imgs_pred: List of predicted RGB images.
            imgs_gt: List of ground truth RGB images.
            batch_size: Batch size for processing.

        Returns:
            List of LPIPS values.
        """
        results = []

        for i in range(0, len(imgs_pred), batch_size):
            batch_pred = imgs_pred[i : i + batch_size]
            batch_gt = imgs_gt[i : i + batch_size]

            with torch.inference_mode():
                pred_tensors = torch.cat(
                    [self._prepare_input(img) for img in batch_pred], dim=0
                )
                gt_tensors = torch.cat(
                    [self._prepare_input(img) for img in batch_gt], dim=0
                )

                lpips_values = self.model(pred_tensors, gt_tensors)
                values = lpips_values.reshape(-1).detach().float().cpu().numpy()
                results.extend(values.tolist())

        return results


def compute_rgb_lpips(
    img_pred: np.ndarray,
    img_gt: np.ndarray,
    net: str = "alex",
    device: str = "cuda",
) -> float:
    """Compute LPIPS between predicted and ground truth RGB images.

    Convenience function that creates a one-time LPIPS metric.
    For multiple computations, use RGBLPIPSMetric class directly.

    Args:
        img_pred: Predicted RGB image in [0, 1] range, shape (H, W, 3).
        img_gt: Ground truth RGB image in [0, 1] range, shape (H, W, 3).
        net: Network to use ('alex', 'vgg', 'squeeze').
        device: Device to run computation on.

    Returns:
        LPIPS value. Lower is better.
    """
    metric = RGBLPIPSMetric(net=net, device=device)
    return metric.compute(img_pred, img_gt)
