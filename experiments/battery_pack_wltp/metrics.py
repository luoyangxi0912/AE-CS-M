from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.battery_pack_wltp.configs import RAW_RESULTS_DIR, SUMMARY_RESULTS_DIR


def compute_metrics(gt: np.ndarray, pred: np.ndarray, eval_mask_missing: np.ndarray) -> dict[str, float | int]:
    pos = eval_mask_missing.astype(bool).flatten()
    y_true = gt.flatten()[pos].astype(np.float64)
    y_pred = pred.flatten()[pos].astype(np.float64)
    n_eval = int(pos.sum())
    if n_eval == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n_eval": 0}
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "n_eval": n_eval}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def raw_result_dir(method: str, run_tag: str | None = None) -> Path:
    base_dir = RAW_RESULTS_DIR / run_tag if run_tag else RAW_RESULTS_DIR
    return base_dir / method


def summary_dir(run_tag: str | None = None) -> Path:
    return SUMMARY_RESULTS_DIR / run_tag if run_tag else SUMMARY_RESULTS_DIR


def save_raw_result(
    method: str,
    file_id: str,
    config_name: str,
    seed: int,
    metrics_dict: dict,
    extra: dict | None = None,
    run_tag: str | None = None,
) -> Path:
    out_dir = raw_result_dir(method, run_tag=run_tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "file_id": file_id,
        "config_name": config_name,
        "seed": int(seed),
        **metrics_dict,
        "timestamp": datetime.now().isoformat(),
    }
    if extra:
        payload.update(extra)
    path = out_dir / f"{file_id}__{config_name}__seed{seed}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)
    return path


def load_all_raw_results(run_tag: str | None = None) -> pd.DataFrame:
    base_dir = RAW_RESULTS_DIR / run_tag if run_tag else RAW_RESULTS_DIR
    rows = []
    for path in base_dir.rglob("*.json"):
        with open(path, encoding="utf-8") as f:
            rows.append(json.load(f))
    return pd.DataFrame(rows)


def generate_summary_tables(run_tag: str | None = None) -> dict[str, Path] | None:
    df = load_all_raw_results(run_tag=run_tag)
    if df.empty:
        return None
    out_dir = summary_dir(run_tag=run_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_seed = df.sort_values(["file_id", "config_name", "method", "seed"])
    by_seed_path = out_dir / "by_seed.csv"
    by_seed.to_csv(by_seed_path, index=False)

    grouped = df.groupby(["method", "config_name"]).agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        n_runs=("seed", "count"),
    ).reset_index()
    config_summary_path = out_dir / "config_summary.csv"
    grouped.to_csv(config_summary_path, index=False)

    rows = []
    for method in sorted(df["method"].unique()):
        row = {"method": method}
        for config_name in sorted(df["config_name"].unique()):
            sub = df[(df["method"] == method) & (df["config_name"] == config_name)]
            if sub.empty:
                continue
            row[f"{config_name}_mae"] = float(sub["mae"].mean())
            row[f"{config_name}_rmse"] = float(sub["rmse"].mean())
            row[f"{config_name}_r2"] = float(sub["r2"].mean())
        rows.append(row)
    main_table_path = out_dir / "main_table.csv"
    pd.DataFrame(rows).to_csv(main_table_path, index=False)

    return {
        "by_seed": by_seed_path,
        "config_summary": config_summary_path,
        "main_table": main_table_path,
    }
