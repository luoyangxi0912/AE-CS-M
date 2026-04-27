# Phase 2A Execution Plan

Status: planned, not yet executed

Project root: `D:\AE-CS-M`

## 1. Goal

Phase 2A is the first non-smoke verification stage after Phase 1 freeze.

It is intentionally narrow:

- no new methods
- no new datasets
- no `GAN`
- no full benchmark
- one non-smoke config only

The purpose is to verify that selected methods still behave correctly once smoke-only row limits are removed.

## 2. Frozen Phase 2A Scope

Methods:

- `aecs`
- `deep_ae`
- `trdae`
- `gain`

Config:

- `mcar_0.1`

Seeds:

- `7`
- `17`
- `27`

Dataset:

- `battery_pack_wltp_dataset`

## 3. Execution Rules

Phase 2A is non-smoke:

- do not pass `--smoke`
- do not use `smoke_max_rows`

Phase 2A is still intentionally small:

- keep `max_test_records = 1`
- keep a dedicated run tag namespace
- use fixed method-specific adapter configs

Recommended run tag:

- `phase2a_non_smoke_mcar_0_1`

## 4. Fixed Method Configs

### `aecs`

Use:

```json
{
  "epochs": 3,
  "batch_size": 8,
  "window_size": 128,
  "stride": 32,
  "hidden_units": 128,
  "latent_dim": 32,
  "dropout_rate": 0.0,
  "shuffle_buffer": 256,
  "p_drop": 0.2,
  "p_consist": 0.1,
  "learning_rate": 0.001,
  "lambda_consist": 0.05,
  "lambda_space": 0.05,
  "lambda_time": 0.05
}
```

`dropout_rate = 0.0` is the migrated `aecs.dropout0` setting from the old AE-CS project.
It disables neural-network layer dropout only. The AE-CS denoising masks remain enabled through `p_drop = 0.2` and `p_consist = 0.1`.

### `deep_ae`

Use:

```json
{
  "epochs": 3,
  "batch_size": 8,
  "window_size": 32,
  "stride": 16,
  "hidden_dim": 128,
  "latent_dim": 64,
  "dropout": 0.0,
  "shuffle_buffer": 128,
  "corruption_rate": 0.2,
  "learning_rate": 0.0001,
  "device": "cpu"
}
```

### `trdae`

Use:

```json
{
  "epochs": 2,
  "batch_size": 4,
  "window_size": 8,
  "stride": 4,
  "hidden_dim": 64,
  "latent_dim": 32,
  "dropout": 0.0,
  "shuffle_buffer": 64,
  "corruption_rate": 0.2,
  "learning_rate": 0.0001,
  "device": "cpu",
  "trdae_exact_max_dim": 2048
}
```

The smaller `window_size` for `trdae` is deliberate. It keeps flattened input dimensionality inside the current exact leave-one-variable-out limit.

### `gain`

Use:

```json
{
  "epochs": 3,
  "batch_size": 8,
  "window_size": 32,
  "stride": 16,
  "hidden_dim": 128,
  "dropout": 0.0,
  "shuffle_buffer": 128,
  "corruption_rate": 0.2,
  "alpha": 100.0,
  "hint_drop_rate": 0.2,
  "learning_rate": 0.0001,
  "device": "cpu",
  "d_steps": 1,
  "g_steps": 1
}
```

## 5. Success Standard

Phase 2A is successful if:

- all four methods finish for all three seeds
- all runs write raw results under a dedicated Phase 2A run tag
- Phase 2A summary tables are generated
- no run fails due to window coverage, adapter protocol mismatch, or environment switching
- `n_eval > 0` for every run
- metrics are finite

## 6. What Can Be Examined

After Phase 2A completes, it is valid to examine:

- per-seed metric spread
- non-smoke runtime cost
- whether `GAIN` remains numerically stable outside smoke
- whether `TRDAE` remains computationally acceptable
- whether `AECS` still completes cleanly on full-length sequences

## 7. What Must Not Be Used As Main Results

Phase 2A outputs must not be used as a final paper main table because:

- only one config is covered
- only one dataset is covered
- only four methods are included
- only three seeds are used
- no broader benchmark grid has been run

## 8. Execution Entry

The fixed execution entry for Phase 2A is:

```powershell
cd D:\AE-CS-M
.\scripts\run_phase2a.ps1
```
