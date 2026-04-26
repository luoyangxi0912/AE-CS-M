from __future__ import annotations

from typing import Tuple

import numpy as np

from experiments.battery_pack_wltp.dataset import STRIDE, WINDOW_SIZE


def create_windows(
    x: np.ndarray,
    mask_observed: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    mask_observed = np.asarray(mask_observed, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"create_windows() expects x with shape [time, features], got {x.shape}.")
    if mask_observed.shape != x.shape:
        raise ValueError(
            "create_windows() expects mask_observed to have the same shape as x, "
            f"got x={x.shape}, mask_observed={mask_observed.shape}."
        )
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive.")
    if x.shape[0] < window_size:
        raise ValueError(
            "create_windows() generated no windows because the sequence is shorter than window_size. "
            f"sequence_length={x.shape[0]}, window_size={window_size}."
        )
    starts = np.arange(0, x.shape[0] - window_size + 1, stride, dtype=np.int32)
    if starts.size == 0:
        raise ValueError("create_windows() generated no windows. Check window_size and stride.")
    x_windows = np.stack([x[s : s + window_size] for s in starts]).astype(np.float32)
    mask_windows = np.stack([mask_observed[s : s + window_size] for s in starts]).astype(np.float32)
    return x_windows, mask_windows, starts


def reconstruct_from_windows(
    windows: np.ndarray,
    starts: np.ndarray,
    total_length: int,
    n_features: int,
    window_size: int = WINDOW_SIZE,
) -> np.ndarray:
    windows = np.asarray(windows, dtype=np.float32)
    starts = np.asarray(starts, dtype=np.int32)
    if windows.ndim != 3:
        raise ValueError(f"reconstruct_from_windows() expects windows with shape [n, time, features], got {windows.shape}.")
    if windows.shape[0] != starts.shape[0]:
        raise ValueError(
            "reconstruct_from_windows() expects one start per window, "
            f"got windows={windows.shape[0]}, starts={starts.shape[0]}."
        )
    accum = np.zeros((total_length, n_features), dtype=np.float64)
    count = np.zeros((total_length, n_features), dtype=np.float64)
    for idx, start in enumerate(starts):
        end = int(start) + window_size
        accum[int(start) : end] += windows[idx]
        count[int(start) : end] += 1.0
    count = np.maximum(count, 1.0)
    return (accum / count).astype(np.float32)
