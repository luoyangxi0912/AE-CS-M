from __future__ import annotations

from typing import Tuple

import numpy as np

from experiments.battery_pack_wltp.dataset import STRIDE, WINDOW_SIZE


def window_starts(total_length: int, window_size: int = WINDOW_SIZE, stride: int = STRIDE) -> np.ndarray:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive.")
    if total_length < window_size:
        raise ValueError(
            "window_starts() generated no windows because the sequence is shorter than window_size. "
            f"sequence_length={total_length}, window_size={window_size}."
        )
    starts = list(range(0, total_length - window_size + 1, stride))
    final_start = total_length - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return np.asarray(starts, dtype=np.int32)


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
    starts = window_starts(x.shape[0], window_size, stride)
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
    if windows.shape[1] != window_size:
        raise ValueError(
            "reconstruct_from_windows() window_size does not match windows shape, "
            f"got windows.shape[1]={windows.shape[1]}, window_size={window_size}."
        )
    if windows.shape[2] != n_features:
        raise ValueError(
            "reconstruct_from_windows() n_features does not match windows shape, "
            f"got windows.shape[2]={windows.shape[2]}, n_features={n_features}."
        )
    accum = np.zeros((total_length, n_features), dtype=np.float64)
    count = np.zeros((total_length, n_features), dtype=np.float64)
    for idx, start in enumerate(starts):
        end = int(start) + window_size
        if int(start) < 0 or end > total_length:
            raise ValueError(
                "reconstruct_from_windows() received an out-of-range window start, "
                f"start={int(start)}, end={end}, total_length={total_length}."
            )
        accum[int(start) : end] += windows[idx]
        count[int(start) : end] += 1.0
    if np.any(count == 0):
        uncovered_rows = np.where(count[:, 0] == 0)[0]
        raise ValueError(
            "reconstruct_from_windows() cannot reconstruct uncovered rows. "
            f"First uncovered rows: {uncovered_rows[:10].tolist()}."
        )
    return (accum / count).astype(np.float32)
