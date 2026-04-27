from __future__ import annotations

import argparse
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.battery_pack_wltp import dataset as dataset_module
from experiments.battery_pack_wltp.mask_protocols import generate_mask, save_mask
from experiments.battery_pack_wltp.metrics import compute_metrics, generate_summary_tables, save_raw_result
from experiments.battery_pack_wltp.registry import get_method_spec


DEFAULT_CONFIG_NAME = "mcar_0.1"
DEFAULT_SEED = 0

SMOKE_METHOD_CONFIGS: dict[str, dict[str, Any]] = {
    "aecs": {
        "window_size": 8,
        "stride": 8,
        "batch_size": 2,
        "epochs": 1,
        "hidden_units": 8,
        "latent_dim": 4,
        "dropout_rate": 0.0,
        "shuffle_buffer": 8,
        "p_drop": 0.2,
        "p_consist": 0.1,
    },
    "deep_ae": {
        "window_size": 8,
        "stride": 8,
        "batch_size": 2,
        "epochs": 1,
        "hidden_dim": 8,
        "latent_dim": 4,
        "dropout": 0.0,
        "shuffle_buffer": 8,
        "corruption_rate": 0.1,
        "device": "cpu",
    },
    "sm_dae": {
        "window_size": 8,
        "stride": 8,
        "batch_size": 2,
        "epochs": 1,
        "hidden_dim": 8,
        "latent_dim": 4,
        "dropout": 0.0,
        "shuffle_buffer": 8,
        "corruption_rate": 0.1,
        "device": "cpu",
    },
    "sdai": {
        "window_size": 8,
        "stride": 8,
        "batch_size": 2,
        "epochs": 1,
        "hidden_dim": 8,
        "latent_dim": 4,
        "dropout": 0.0,
        "shuffle_buffer": 8,
        "corruption_rate": 0.1,
        "device": "cpu",
    },
    "trdae": {
        "window_size": 8,
        "stride": 8,
        "batch_size": 2,
        "epochs": 1,
        "hidden_dim": 8,
        "latent_dim": 4,
        "dropout": 0.0,
        "shuffle_buffer": 8,
        "corruption_rate": 0.1,
        "device": "cpu",
        "trdae_exact_max_dim": 2048,
    },
    "gain": {
        "window_size": 8,
        "stride": 8,
        "batch_size": 2,
        "epochs": 1,
        "hidden_dim": 8,
        "dropout": 0.0,
        "shuffle_buffer": 8,
        "corruption_rate": 0.1,
        "alpha": 10.0,
        "hint_drop_rate": 0.2,
        "learning_rate": 1e-4,
        "device": "cpu",
        "d_steps": 1,
        "g_steps": 1,
    },
}


@dataclass(frozen=True)
class ExperimentResult:
    method: str
    file_id: str
    config_name: str
    seed: int
    raw_result_path: Path
    mask_path: Path
    metrics: dict[str, float | int]


def instantiate_imputer(method: str, config: dict[str, Any] | None = None):
    spec = get_method_spec(method)
    if not spec.enabled:
        raise ValueError(f"Method {method!r} is not enabled in the current phase.")
    if not spec.adapter_module or not spec.adapter_class:
        raise ValueError(f"Method {method!r} does not have an adapter registered.")
    module = importlib.import_module(spec.adapter_module)
    cls = getattr(module, spec.adapter_class)
    return cls(config=config)


def method_config(method: str, smoke: bool, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if smoke:
        cfg.update(SMOKE_METHOD_CONFIGS.get(method, {}))
    if overrides:
        cfg.update(overrides)
    return cfg


def _load_json_object(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}.")
    return data


def _select_records(records, max_records: int | None):
    records = list(records)
    if max_records is None or max_records <= 0:
        return records
    return records[:max_records]


def run_single_method(
    method: str,
    config_name: str = DEFAULT_CONFIG_NAME,
    seed: int = DEFAULT_SEED,
    smoke: bool = False,
    run_tag: str | None = None,
    max_test_records: int | None = None,
    imputer_config: dict[str, Any] | None = None,
    smoke_max_rows: int | None = 64,
) -> list[ExperimentResult]:
    np.random.seed(seed)
    if smoke and smoke_max_rows is not None and smoke_max_rows > 0:
        dataset_module.SMOKE_MAX_ROWS = int(smoke_max_rows)
    bundles = dataset_module.load_dataset_bundles(smoke=smoke)
    scaler = bundles["scaler"]
    train_records = list(bundles["train_records"])
    test_records = _select_records(bundles["test_records"], max_test_records)
    if not train_records:
        raise ValueError("No training records found.")
    if not test_records:
        raise ValueError("No test records selected.")

    imputer = instantiate_imputer(method, method_config(method, smoke=smoke, overrides=imputer_config))
    started = time.time()
    imputer.fit(train_records, scaler=scaler, smoke=smoke, metadata={"seed": seed, "config_name": config_name})
    fit_seconds = time.time() - started

    results: list[ExperimentResult] = []
    for record in test_records:
        seq = dataset_module.load_sequence_bundle(record, scaler, smoke=smoke)
        mask_dict = generate_mask(config_name, seq.natural_mask_observed, seed, feature_names=seq.feature_names)
        current_mask_path = save_mask(seq.file_id, config_name, seed, mask_dict, run_tag=run_tag)

        artificial_mask_observed = mask_dict["artificial_mask_observed"].astype(np.float32)
        test_mask_missing = mask_dict["test_mask_missing"].astype(np.int8)
        x_input = np.where(artificial_mask_observed == 1.0, seq.x_norm, 0.0).astype(np.float32)

        infer_started = time.time()
        pred = imputer.impute(x_input, artificial_mask_observed, metadata={"seed": seed, "config_name": config_name})
        infer_seconds = time.time() - infer_started
        metrics_dict = compute_metrics(seq.x_norm, pred, test_mask_missing)
        raw_path = save_raw_result(
            method,
            seq.file_id,
            config_name,
            seed,
            metrics_dict,
            extra={
                "split": record.split,
                "protocol": record.protocol,
                "smoke": smoke,
                "run_tag": run_tag,
                "fit_seconds": fit_seconds,
                "infer_seconds": infer_seconds,
                "sequence_shape": list(seq.x_norm.shape),
                "mask_path": str(current_mask_path),
                "metric_space": "normalized",
            },
            run_tag=run_tag,
        )
        results.append(
            ExperimentResult(
                method=method,
                file_id=seq.file_id,
                config_name=config_name,
                seed=seed,
                raw_result_path=raw_path,
                mask_path=current_mask_path,
                metrics=metrics_dict,
            )
        )

    generate_summary_tables(run_tag=run_tag)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AE-CS-M battery_pack_wltp method-level experiment.")
    parser.add_argument("--method", required=True, help="Method name registered for the current phase.")
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME, help="Mask config name, e.g. mcar_0.1.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke", action="store_true", help="Use smoke-size data and method config.")
    parser.add_argument("--run-tag", default=None, help="Optional result namespace under masks/results.")
    parser.add_argument("--max-test-records", type=int, default=1, help="Limit test records. Use <=0 for all.")
    parser.add_argument("--smoke-max-rows", type=int, default=64, help="Rows per sequence when --smoke is active.")
    parser.add_argument("--imputer-config-json", default=None, help="Optional JSON object with adapter config overrides.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    overrides = _load_json_object(args.imputer_config_json)
    results = run_single_method(
        method=args.method,
        config_name=args.config_name,
        seed=args.seed,
        smoke=args.smoke,
        run_tag=args.run_tag,
        max_test_records=args.max_test_records,
        imputer_config=overrides,
        smoke_max_rows=args.smoke_max_rows,
    )
    for result in results:
        print(
            json.dumps(
                {
                    "method": result.method,
                    "file_id": result.file_id,
                    "config_name": result.config_name,
                    "seed": result.seed,
                    "metrics": result.metrics,
                    "raw_result_path": str(result.raw_result_path),
                    "mask_path": str(result.mask_path),
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
