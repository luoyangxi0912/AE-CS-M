from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .module import require_torch


@dataclass
class FlattenedWindowBatch:
    values: torch.Tensor
    mask_observed: torch.Tensor
    mask_missing: torch.Tensor
    original_shape: tuple[int, int, int]


def flatten_windows(values: np.ndarray, mask_observed: np.ndarray, device: torch.device) -> FlattenedWindowBatch:
    torch, _ = require_torch()
    values = np.asarray(values, dtype=np.float32)
    mask_observed = np.asarray(mask_observed, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected window values with shape [batch, time, features], got {values.shape}.")
    if mask_observed.shape != values.shape:
        raise ValueError(f"mask_observed shape {mask_observed.shape} does not match values shape {values.shape}.")
    flat_values = values.reshape(values.shape[0], -1)
    flat_observed = mask_observed.reshape(mask_observed.shape[0], -1)
    value_tensor = torch.from_numpy(flat_values).to(device=device, dtype=torch.float32)
    observed_tensor = torch.from_numpy(flat_observed).to(device=device, dtype=torch.float32)
    missing_tensor = 1.0 - observed_tensor
    return FlattenedWindowBatch(
        values=value_tensor,
        mask_observed=observed_tensor,
        mask_missing=missing_tensor,
        original_shape=values.shape,
    )


def reshape_flat_windows(flat_values: torch.Tensor, original_shape: tuple[int, int, int]) -> np.ndarray:
    return flat_values.detach().cpu().numpy().reshape(original_shape).astype(np.float32)


def apply_training_corruption(mask_observed: torch.Tensor, corruption_rate: float) -> tuple[torch.Tensor, torch.Tensor]:
    torch, _ = require_torch()
    if corruption_rate <= 0:
        corrupted_observed = mask_observed
    else:
        keep = (torch.rand_like(mask_observed) > corruption_rate).to(mask_observed.dtype)
        corrupted_observed = mask_observed * keep
    mask_missing = 1.0 - corrupted_observed
    return corrupted_observed, mask_missing
