from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data.loaders.dynamic_profiles_loader import SequenceRecord, iter_feature_frames, slice_record, split_sequence_files


SMOKE_MAX_ROWS = 4096
WINDOW_SIZE = 128
STRIDE = 32


@dataclass
class SequenceBundle:
    file_id: str
    split: str
    protocol: str
    metadata: pd.DataFrame
    feature_names: List[str]
    x_raw: np.ndarray
    x_norm: np.ndarray
    natural_mask_observed: np.ndarray


class StreamingStandardScaler:
    def __init__(self):
        self.count = 0
        self.sum = None
        self.sum_sq = None

    def update(self, frame: pd.DataFrame) -> None:
        arr = frame.to_numpy(dtype=np.float64)
        if self.sum is None:
            self.sum = arr.sum(axis=0)
            self.sum_sq = np.square(arr).sum(axis=0)
        else:
            self.sum += arr.sum(axis=0)
            self.sum_sq += np.square(arr).sum(axis=0)
        self.count += arr.shape[0]

    def finalize(self) -> StandardScaler:
        if self.sum is None or self.sum_sq is None:
            raise ValueError("Cannot finalize scaler from empty training frames.")
        scaler = StandardScaler()
        means = self.sum / max(self.count, 1)
        var = np.maximum(self.sum_sq / max(self.count, 1) - np.square(means), 1e-12)
        scale = np.sqrt(var)
        scaler.mean_ = means.astype(np.float64)
        scaler.scale_ = scale.astype(np.float64)
        scaler.var_ = var.astype(np.float64)
        scaler.n_features_in_ = len(means)
        scaler.n_samples_seen_ = self.count
        return scaler


def build_scaler(smoke: bool = False) -> StandardScaler:
    split_map = split_sequence_files()
    acc = StreamingStandardScaler()
    for _, frame in iter_feature_frames(split_map["train"], smoke=smoke, smoke_max_rows=SMOKE_MAX_ROWS):
        acc.update(frame)
    return acc.finalize()


def load_or_build_scaler(smoke: bool = False) -> StandardScaler:
    return build_scaler(smoke=smoke)


def normalize_frame(frame: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(frame.to_numpy(dtype=np.float32)).astype(np.float32)


def inverse_normalize(x_norm: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.inverse_transform(x_norm).astype(np.float32)


def load_sequence_bundle(record: SequenceRecord, scaler: StandardScaler, smoke: bool = False) -> SequenceBundle:
    metadata, features = slice_record(record, smoke=smoke, smoke_max_rows=SMOKE_MAX_ROWS)
    x_raw = features.to_numpy(dtype=np.float32)
    x_norm = normalize_frame(features, scaler)
    natural_mask_observed = np.ones_like(x_norm, dtype=np.int8)
    return SequenceBundle(
        file_id=record.file_id,
        split=record.split,
        protocol=record.protocol,
        metadata=metadata,
        feature_names=list(features.columns),
        x_raw=x_raw,
        x_norm=x_norm,
        natural_mask_observed=natural_mask_observed,
    )


def load_dataset_bundles(smoke: bool = False) -> Dict[str, object]:
    split_map = split_sequence_files()
    scaler = load_or_build_scaler(smoke=smoke)
    return {
        "train_records": split_map["train"],
        "val_records": split_map["val"],
        "test_records": split_map["test"],
        "supplementary_records": split_map["supplementary"],
        "scaler": scaler,
    }


def _buffered_batch_iterator(
    sample_iter: Iterable[tuple[np.ndarray, np.ndarray]],
    batch_size: int,
    shuffle_buffer: int = 0,
    seed: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed) if shuffle_buffer > 0 else None
    sample_buffer: list[tuple[np.ndarray, np.ndarray]] = []
    x_batch: list[np.ndarray] = []
    mask_observed_batch: list[np.ndarray] = []

    def flush_one():
        if rng is None:
            x_item, mask_observed_item = sample_buffer.pop(0)
        else:
            idx = int(rng.integers(0, len(sample_buffer)))
            x_item, mask_observed_item = sample_buffer.pop(idx)
        x_batch.append(x_item)
        mask_observed_batch.append(mask_observed_item)
        if len(x_batch) == batch_size:
            out = (
                np.stack(x_batch).astype(np.float32),
                np.stack(mask_observed_batch).astype(np.float32),
            )
            x_batch.clear()
            mask_observed_batch.clear()
            return out
        return None

    for sample in sample_iter:
        sample_buffer.append(sample)
        if shuffle_buffer > 0:
            if len(sample_buffer) >= shuffle_buffer:
                out = flush_one()
                if out is not None:
                    yield out
        else:
            out = flush_one()
            if out is not None:
                yield out

    while sample_buffer:
        out = flush_one()
        if out is not None:
            yield out

    if x_batch:
        yield np.stack(x_batch).astype(np.float32), np.stack(mask_observed_batch).astype(np.float32)


def iter_row_batches(
    records: Iterable[SequenceRecord],
    scaler: StandardScaler,
    batch_size: int,
    smoke: bool = False,
    shuffle_buffer: int = 0,
    seed: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    def sample_iter():
        for record in records:
            _, features = slice_record(record, smoke=smoke, smoke_max_rows=SMOKE_MAX_ROWS)
            x_norm = normalize_frame(features, scaler)
            mask_observed = np.ones_like(x_norm, dtype=np.float32)
            for row_x, row_mask_observed in zip(x_norm, mask_observed):
                yield row_x, row_mask_observed

    yield from _buffered_batch_iterator(sample_iter(), batch_size, shuffle_buffer=shuffle_buffer, seed=seed)


def iter_window_batches(
    records: Iterable[SequenceRecord],
    scaler: StandardScaler,
    batch_size: int,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    smoke: bool = False,
    shuffle_buffer: int = 0,
    seed: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    def sample_iter():
        for record in records:
            _, features = slice_record(record, smoke=smoke, smoke_max_rows=SMOKE_MAX_ROWS)
            x_norm = normalize_frame(features, scaler)
            mask_observed = np.ones_like(x_norm, dtype=np.float32)
            starts = range(0, x_norm.shape[0] - window_size + 1, stride)
            for start in starts:
                end = start + window_size
                yield x_norm[start:end], mask_observed[start:end]

    yield from _buffered_batch_iterator(sample_iter(), batch_size, shuffle_buffer=shuffle_buffer, seed=seed)
