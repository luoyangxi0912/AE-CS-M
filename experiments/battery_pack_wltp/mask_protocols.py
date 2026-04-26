from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from data.loaders.dynamic_profiles_loader import CELL_GROUPS, FEATURE_COLUMNS, FEATURE_GROUPS, GROUP_ORDER
from experiments.battery_pack_wltp.configs import MASKS_DIR


def mask_path(file_id: str, config_name: str, seed: int, run_tag: str | None = None) -> Path:
    base_dir = MASKS_DIR / run_tag if run_tag else MASKS_DIR
    return base_dir / f"{file_id}__{config_name}__seed{seed}.npz"


def mask_observed_to_missing(mask_observed: np.ndarray) -> np.ndarray:
    return (1 - mask_observed.astype(np.int8)).astype(np.int8)


def mask_missing_to_observed(mask_missing: np.ndarray) -> np.ndarray:
    return (1 - mask_missing.astype(np.int8)).astype(np.int8)


def generate_mcar_mask_observed(natural_mask_observed: np.ndarray, ratio: float, rng: np.random.RandomState) -> np.ndarray:
    artificial_mask_observed = np.ones_like(natural_mask_observed, dtype=np.int8)
    positions = np.argwhere(natural_mask_observed == 1)
    n_drop = int(len(positions) * ratio)
    if n_drop == 0:
        return artificial_mask_observed
    chosen = rng.choice(len(positions), n_drop, replace=False)
    artificial_mask_observed[positions[chosen, 0], positions[chosen, 1]] = 0
    return artificial_mask_observed


def generate_block_mask_observed(natural_mask_observed: np.ndarray, ratio: float, rng: np.random.RandomState) -> np.ndarray:
    t, n = natural_mask_observed.shape
    target = int(natural_mask_observed.sum() * ratio)
    artificial_mask_observed = np.ones_like(natural_mask_observed, dtype=np.int8)
    current = 0
    while current < target:
        feature = rng.randint(0, n)
        block_len = rng.randint(max(4, t // 100), max(8, t // 20))
        start = rng.randint(0, max(1, t - block_len))
        end = min(t, start + block_len)
        valid = (natural_mask_observed[start:end, feature] == 1) & (artificial_mask_observed[start:end, feature] == 1)
        artificial_mask_observed[start:end, feature][valid] = 0
        current = int(((natural_mask_observed == 1) & (artificial_mask_observed == 0)).sum())
    return artificial_mask_observed


def _column_index(feature_names: Sequence[str] | None = None) -> dict[str, int]:
    names = list(FEATURE_COLUMNS if feature_names is None else feature_names)
    return {name: idx for idx, name in enumerate(names)}


def _group_indices(group_names: Sequence[str], feature_names: Sequence[str] | None = None) -> np.ndarray:
    column_index = _column_index(feature_names)
    indices: list[int] = []
    for name in group_names:
        indices.extend(column_index[col] for col in FEATURE_GROUPS[name] if col in column_index)
    if not indices:
        raise ValueError(f"No feature columns found for groups: {group_names}")
    return np.array(sorted(set(indices)), dtype=np.int32)


def _cell_triplet_indices(feature_names: Sequence[str] | None = None) -> list[np.ndarray]:
    column_index = _column_index(feature_names)
    triplets: list[np.ndarray] = []
    for cell in CELL_GROUPS:
        cols = [cell["voltage"], cell["temp_top"], cell["temp_bottom"]]
        idx = [column_index[col] for col in cols if col in column_index]
        if idx:
            triplets.append(np.array(idx, dtype=np.int32))
    if not triplets:
        raise ValueError("No cell triplet columns found for sensor-drop masks.")
    return triplets


def generate_sensordrop_mask_observed(
    natural_mask_observed: np.ndarray,
    ratio: float,
    rng: np.random.RandomState,
    feature_names: Sequence[str] | None = None,
) -> np.ndarray:
    artificial_mask_observed = np.ones_like(natural_mask_observed, dtype=np.int8)
    triplets = _cell_triplet_indices(feature_names)
    n_groups = max(1, int(len(triplets) * ratio))
    chosen_groups = rng.choice(len(triplets), n_groups, replace=False)
    for group_idx in chosen_groups:
        artificial_mask_observed[:, triplets[int(group_idx)]] = 0
    return artificial_mask_observed


def generate_sensordrop_burst_mask_observed(
    natural_mask_observed: np.ndarray,
    ratio: float,
    rng: np.random.RandomState,
    feature_names: Sequence[str] | None = None,
) -> np.ndarray:
    t, _ = natural_mask_observed.shape
    artificial_mask_observed = np.ones_like(natural_mask_observed, dtype=np.int8)
    target_hidden = max(1, int(natural_mask_observed.sum() * ratio))
    triplets = _cell_triplet_indices(feature_names)
    min_burst = max(6, t // 200)
    max_burst = max(min_burst + 1, t // 20)
    max_burst = min(max_burst, max(min_burst + 1, t // 6))
    desired_rows = max(1, int(0.4 * t))
    min_groups_for_local_burst = int(np.ceil(target_hidden / max(1, 3 * desired_rows)))
    base_groups = int(np.ceil(ratio * len(triplets) * 3.0))
    n_groups = max(1, min_groups_for_local_burst, base_groups)
    n_groups = min(n_groups, max(1, int(0.75 * len(triplets))))
    chosen_groups = rng.choice(len(triplets), size=n_groups, replace=False)

    def apply_one_burst(group_idx: int) -> None:
        burst_len = int(rng.randint(min_burst, max_burst + 1))
        start = int(rng.randint(0, max(1, t - burst_len + 1)))
        end = min(t, start + burst_len)
        for col in triplets[group_idx]:
            valid = (natural_mask_observed[start:end, col] == 1) & (artificial_mask_observed[start:end, col] == 1)
            artificial_mask_observed[start:end, col][valid] = 0

    for group_idx in chosen_groups:
        for _ in range(int(rng.randint(2, 6))):
            apply_one_burst(int(group_idx))

    hidden_now = int(((natural_mask_observed == 1) & (artificial_mask_observed == 0)).sum())
    max_extra = max(32, target_hidden // 16)
    extra = 0
    while hidden_now < target_hidden and extra < max_extra:
        extra += 1
        apply_one_burst(int(rng.choice(chosen_groups)))
        hidden_now = int(((natural_mask_observed == 1) & (artificial_mask_observed == 0)).sum())
    return artificial_mask_observed


def generate_fixed_interval_mask_observed(
    natural_mask_observed: np.ndarray,
    ratio: float,
    rng: np.random.RandomState,
    feature_names: Sequence[str] | None = None,
) -> np.ndarray:
    del ratio, rng
    artificial_mask_observed = np.ones_like(natural_mask_observed, dtype=np.int8)
    interval_map = {
        "cell_voltage": 2,
        "cell_temp_top": 3,
        "cell_temp_bottom": 3,
        "current": 1,
        "voltage_summary": 2,
        "state_env": 4,
    }
    for group_name in GROUP_ORDER:
        indices = _group_indices([group_name], feature_names)
        keep_rows = np.zeros(natural_mask_observed.shape[0], dtype=np.int8)
        keep_rows[:: interval_map[group_name]] = 1
        artificial_mask_observed[:, indices] = artificial_mask_observed[:, indices] * keep_rows[:, None]
    return artificial_mask_observed.astype(np.int8)


def generate_async_interval_jitter_mask_observed(
    natural_mask_observed: np.ndarray,
    ratio: float,
    rng: np.random.RandomState,
    feature_names: Sequence[str] | None = None,
) -> np.ndarray:
    del ratio
    t, _ = natural_mask_observed.shape
    artificial_mask_observed = np.ones_like(natural_mask_observed, dtype=np.int8)
    interval_map = {
        "cell_voltage": 2,
        "cell_temp_top": 3,
        "cell_temp_bottom": 3,
        "current": 1,
        "voltage_summary": 2,
        "state_env": 4,
    }
    jitter_prob = 0.12
    packet_drop_prob = 0.03
    for group_name in GROUP_ORDER:
        interval = int(interval_map[group_name])
        indices = _group_indices([group_name], feature_names)
        if interval <= 1:
            keep_rows = np.ones(t, dtype=np.int8)
            keep_rows[rng.rand(t) < packet_drop_prob] = 0
        else:
            phase = int(rng.randint(0, interval))
            base_positions = np.arange(phase, t, interval, dtype=np.int32)
            shifted = []
            for pos in base_positions:
                shift = int(rng.choice([-1, 1])) if rng.rand() < jitter_prob else 0
                shifted.append(int(np.clip(pos + shift, 0, t - 1)))
            keep_positions = np.unique(np.array(shifted, dtype=np.int32)) if shifted else np.array([], dtype=np.int32)
            keep_rows = np.zeros(t, dtype=np.int8)
            keep_rows[keep_positions] = 1
            if keep_positions.size > 0:
                keep_rows[keep_positions[rng.rand(keep_positions.size) < packet_drop_prob]] = 0
            if keep_rows.sum() == 0:
                keep_rows[phase % t] = 1
        artificial_mask_observed[:, indices] = artificial_mask_observed[:, indices] * keep_rows[:, None]
    return artificial_mask_observed.astype(np.int8)


def split_val_test_missing(
    natural_mask_observed: np.ndarray,
    artificial_mask_observed: np.ndarray,
    rng: np.random.RandomState,
    val_ratio: float = 1.0 / 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden = np.argwhere((natural_mask_observed == 1) & (artificial_mask_observed == 0))
    perm = rng.permutation(len(hidden))
    n_val = int(len(hidden) * val_ratio)
    val_idx = perm[:n_val]
    test_idx = perm[n_val:]
    val_mask_missing = np.zeros_like(natural_mask_observed, dtype=np.int8)
    test_mask_missing = np.zeros_like(natural_mask_observed, dtype=np.int8)
    if len(hidden) > 0:
        val_mask_missing[hidden[val_idx, 0], hidden[val_idx, 1]] = 1
        test_mask_missing[hidden[test_idx, 0], hidden[test_idx, 1]] = 1
    train_mask_observed = ((natural_mask_observed == 1) & (artificial_mask_observed == 1)).astype(np.int8)
    return train_mask_observed, val_mask_missing, test_mask_missing


def parse_config(config_name: str) -> tuple[str, float]:
    kind, ratio = config_name.rsplit("_", 1)
    return kind, float(ratio)


def generate_mask(
    config_name: str,
    natural_mask_observed: np.ndarray,
    seed: int,
    feature_names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    kind, ratio = parse_config(config_name)
    rng = np.random.RandomState(seed)
    if kind == "mcar":
        artificial_mask_observed = generate_mcar_mask_observed(natural_mask_observed, ratio, rng)
    elif kind == "block":
        artificial_mask_observed = generate_block_mask_observed(natural_mask_observed, ratio, rng)
    elif kind == "sensordrop":
        artificial_mask_observed = generate_sensordrop_mask_observed(natural_mask_observed, ratio, rng, feature_names)
    elif kind == "sensordrop_burst":
        artificial_mask_observed = generate_sensordrop_burst_mask_observed(natural_mask_observed, ratio, rng, feature_names)
    elif kind == "fixed_interval":
        artificial_mask_observed = generate_fixed_interval_mask_observed(natural_mask_observed, ratio, rng, feature_names)
    elif kind == "async_interval_jitter":
        artificial_mask_observed = generate_async_interval_jitter_mask_observed(natural_mask_observed, ratio, rng, feature_names)
    else:
        raise ValueError(f"Unknown mask config: {config_name}")
    train_mask_observed, val_mask_missing, test_mask_missing = split_val_test_missing(
        natural_mask_observed,
        artificial_mask_observed,
        rng,
    )
    return {
        "natural_mask_observed": natural_mask_observed.astype(np.int8),
        "artificial_mask_observed": artificial_mask_observed.astype(np.int8),
        "train_mask_observed": train_mask_observed.astype(np.int8),
        "val_mask_missing": val_mask_missing.astype(np.int8),
        "test_mask_missing": test_mask_missing.astype(np.int8),
        "config_name": np.array(config_name),
        "seed": np.array(seed),
    }


def save_mask(file_id: str, config_name: str, seed: int, mask_dict: dict, run_tag: str | None = None) -> Path:
    path = mask_path(file_id, config_name, seed, run_tag=run_tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **mask_dict)
    return path


def load_mask(file_id: str, config_name: str, seed: int, run_tag: str | None = None) -> dict[str, np.ndarray]:
    path = mask_path(file_id, config_name, seed, run_tag=run_tag)
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}
