# Phase 1C GAIN Completion Record

Status: GAIN smoke-level runner reproduction completed

Date: 2026-04-26

Project root: `D:\AE-CS-M`

## Scope

Phase 1C adds only:

- `gain`
- `baselines/mentor_gain_family/`
- `experiments/battery_pack_wltp/imputers/gain_imputer.py`
- `scripts/smoke_phase1c_gain.ps1`

Phase 1C does not add:

- `gan`
- `mentor_gan_family`
- multi-dataset support
- full benchmark sweeps
- paper-scale GAIN tuning

## Protocol Boundary

External interface remains:

```python
fit(train_source, scaler=None, smoke=False, metadata=None)
impute(X, mask_observed, metadata=None)
```

External mask semantics remain:

- `mask_observed = 1` means observed
- `mask_observed = 0` means missing

GAIN internals may use mentor-style missing logic, but this is contained inside:

- `baselines/mentor_gain_family/gain.py`
- `experiments/battery_pack_wltp/imputers/gain_imputer.py`

The experiment layer must not require users to provide mentor `mask_missing = 1` semantics.

## Frozen Smoke Configuration

The Phase 1C smoke run uses:

- `method`: `gain`
- `config_name`: `mcar_0.1`
- `seed`: `7`
- `smoke`: enabled
- `smoke_max_rows`: `64`
- `max_test_records`: `1`
- `run_tag`: `smoke_phase1c_gain`

Run:

```powershell
cd D:\AE-CS-M
.\scripts\smoke_phase1c_gain.ps1
```

To run the full Phase 1 smoke set including Phase 1B methods and Phase 1C `gain`:

```powershell
cd D:\AE-CS-M
.\scripts\smoke_phase1_all.ps1
```

## Environment

GAIN uses the same PyTorch environment recorded for Phase 1B mentor baselines:

- Python: `C:\Users\Henry\AppData\Local\Programs\Python\Python311\python.exe`
- extra package path: `D:\Python_Packages\PyTorch`
- observed PyTorch: `2.8.0+cu126`
- CUDA available: `True`

The old raw data path:

`D:\AE-CS\Lithium-ion battery pack cycling dataset with CC-CV charging and WLTPconstant discharge profiles`

is a known environment coupling for read-only raw data fallback only. It is not an artifact directory.

## Last Known Smoke Result

```csv
method,config_name,mae_mean,mae_std,rmse_mean,rmse_std,r2_mean,r2_std,n_runs
gain,mcar_0.1,1582.931821673404,,9551.55490099349,,-0.012755486315121134,,1
```

This is a smoke artifact. It proves the GAIN adapter, runner registration, mask generation, metrics, raw result writing, and summary generation are connected. It is not a model-quality claim.

## Output Locations

All GAIN smoke artifacts must stay under `D:\AE-CS-M`:

- raw results: `D:\AE-CS-M\experiments\battery_pack_wltp\results\raw\smoke_phase1c_gain\gain`
- summary tables: `D:\AE-CS-M\experiments\battery_pack_wltp\results\summary\smoke_phase1c_gain`
- masks: `D:\AE-CS-M\experiments\battery_pack_wltp\masks\smoke_phase1c_gain`

No Phase 1C artifact may be written to `D:\AE-CS`.

## Acceptance Standard

Phase 1C GAIN is considered complete when:

- `gain` imports successfully.
- `gain` is registered as Phase 1C and enabled.
- `GAINImputer` can be constructed in the PyTorch environment.
- `GAINImputer.fit(...)` and `GAINImputer.impute(...)` pass method-level smoke.
- `scripts/smoke_phase1c_gain.ps1` completes without error.
- `config_summary.csv` exists for `smoke_phase1c_gain`.
- no `GAN` code path is added or executed.

## Known Risks

- The GAIN implementation is a Phase 1 minimum viable migration, not a tuned reproduction of the original mentor training setup.
- The original mentor GAIN updated generator/discriminator inside `forward`; AE-CS-M separates model losses from optimizer steps to fit the project adapter protocol.
- Smoke metrics are unstable and should not be used for performance comparison.
- If future full runs require stronger fidelity to the mentor implementation, compare the generator/discriminator loss flow against `D:\马里兰\torch_fuzz\fuzz\model\variant\gain.py` before tuning hyperparameters.
