# Phase 1B Completion Record

Status: frozen for smoke-level runner reproduction

Freeze date: 2026-04-26

Project root: `D:\AE-CS-M`

## Scope

Phase 1B covers only the following methods:

- `aecs`
- `deep_ae`
- `sm_dae`
- `sdai`
- `trdae`

Phase 1B does not include:

- `gain`
- `gan`
- multi-dataset expansion
- full benchmark sweeps
- paper-scale ablation experiments
- checkpoint training policy

## Frozen Components

The following components are considered the Phase 1B smoke baseline:

- AE-CS model body: `models/aecs/ae_cs.py`
- AE-CS losses: `models/aecs/losses.py`
- AE-CS adapter: `experiments/battery_pack_wltp/imputers/aecs_imputer.py`
- battery_pack_wltp dataset protocol: `experiments/battery_pack_wltp/dataset.py`
- windowing protocol: `experiments/battery_pack_wltp/windowing.py`
- mask protocol: `experiments/battery_pack_wltp/mask_protocols.py`
- metrics and result writing: `experiments/battery_pack_wltp/metrics.py`
- mentor AE-family baseline code: `baselines/mentor_ae_family/`
- mentor AE-family adapter: `experiments/battery_pack_wltp/imputers/mentor_ae_family_imputer.py`
- method registry: `experiments/battery_pack_wltp/registry.py`
- runner entrypoint: `experiments/battery_pack_wltp/run_experiment.py`
- reproducibility smoke script: `scripts/smoke_phase1b_runner.ps1`

## Runner Smoke Configuration

The frozen smoke run uses:

- `config_name`: `mcar_0.1`
- `seed`: `7`
- `smoke`: enabled
- `smoke_max_rows`: `64`
- `max_test_records`: `1`
- `run_tag`: `smoke_phase1b_runner`

The smoke run is intentionally small. It proves that the method-level adapters, dataset protocol, mask generation, metrics, raw result writing, and summary generation are connected. It is not a model-quality result.

## Validation Commands

Run the frozen smoke script from PowerShell:

```powershell
cd D:\AE-CS-M
.\scripts\smoke_phase1b_runner.ps1
```

If the raw dataset is not available in the default location, pass it explicitly:

```powershell
.\scripts\smoke_phase1b_runner.ps1 -DataRoot "D:\path\to\battery_pack_wltp_dataset"
```

The script runs:

- `aecs` with the TensorFlow Python environment
- `deep_ae`, `sm_dae`, `sdai`, `trdae` with the PyTorch Python environment

## Last Known Smoke Results

Last known successful smoke run used the read-only raw dataset at:

`D:\AE-CS\Lithium-ion battery pack cycling dataset with CC-CV charging and WLTPconstant discharge profiles`

This old-project path is a known environment coupling for raw data discovery only. It is allowed as a read-only fallback on the current machine, but it is not part of the Phase 1B artifact protocol and must not be used for results, masks, summaries, or checkpoints.

Summary output:

```csv
method,config_name,mae_mean,mae_std,rmse_mean,rmse_std,r2_mean,r2_std,n_runs
aecs,mcar_0.1,911.0029405392326,,9534.509260219005,,-0.009143998423392308,,1
deep_ae,mcar_0.1,1642.2574002645774,,9631.608486990011,,-0.029802859508987423,,1
sdai,mcar_0.1,1090.048887926611,,9518.373391499752,,-0.005731208967037604,,1
sm_dae,mcar_0.1,1362.8883379774202,,9552.307540256252,,-0.012915097934756092,,1
trdae,mcar_0.1,1000.9019793949344,,9530.282997384777,,-0.008249571153382762,,1
```

These values are smoke artifacts. They should be used to confirm the pipeline shape, not to compare method performance. Small numeric drift is acceptable across reruns unless the pass/fail checks or output contract changes.

## Output Locations

All Phase 1B smoke artifacts must stay under `D:\AE-CS-M`:

- raw results: `D:\AE-CS-M\experiments\battery_pack_wltp\results\raw\smoke_phase1b_runner`
- summary tables: `D:\AE-CS-M\experiments\battery_pack_wltp\results\summary\smoke_phase1b_runner`
- masks: `D:\AE-CS-M\experiments\battery_pack_wltp\masks\smoke_phase1b_runner`

No Phase 1B experiment artifact may be written to `D:\AE-CS`.

## Freeze Rules

After this freeze point:

- Do not modify Phase 1B model or adapter behavior while only trying to reproduce the smoke run.
- If a behavior change is required, document it first and use a new `run_tag`.
- Do not enable `gain` inside Phase 1B.
- Do not add `gan` to Phase 1B.
- Do not make the runner depend on old `D:\AE-CS` experiment artifact paths.
- Treat the old `D:\AE-CS` path only as a possible read-only raw data location.
- Treat `scripts/smoke_phase1b_runner.ps1` as the frozen smoke command for Phase 1B.

## Acceptance Standard

Phase 1B is considered reproducible when:

- `scripts/smoke_phase1b_runner.ps1` completes without error.
- raw result JSON files exist for `aecs`, `deep_ae`, `sm_dae`, `sdai`, and `trdae`.
- `config_summary.csv` exists under the Phase 1B smoke summary directory.
- generated paths are all under `D:\AE-CS-M`.
- no `gain` or `gan` code path is executed.
