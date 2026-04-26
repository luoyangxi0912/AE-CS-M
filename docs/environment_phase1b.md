# Phase 1B Environment Notes

This document records the environment needed to reproduce the frozen Phase 1B smoke run.

Project root:

`D:\AE-CS-M`

## Raw Data Root

The raw dataset is read-only input. It is not an experiment artifact directory.

The runner resolves the raw dataset in this order:

1. `-DataRoot` passed to `scripts\smoke_phase1b_runner.ps1`
2. environment variable `AECSM_BATTERY_PACK_WLTP_DATA_ROOT`
3. `D:\AE-CS-M\battery_pack_wltp_dataset`
4. `D:\AE-CS\Lithium-ion battery pack cycling dataset with CC-CV charging and WLTPconstant discharge profiles`

The fourth option is allowed only as a read-only raw data location. It must not be used for results, masks, summaries, or checkpoints.

Known environment coupling:

- `D:\AE-CS\Lithium-ion battery pack cycling dataset with CC-CV charging and WLTPconstant discharge profiles` is currently retained only as a read-only raw data fallback for this machine.
- This fallback is not part of the AE-CS-M artifact protocol.
- Formal experiment artifacts must still be written only under `D:\AE-CS-M`.

Set the data root explicitly when needed:

```powershell
$env:AECSM_BATTERY_PACK_WLTP_DATA_ROOT = "D:\path\to\battery_pack_wltp_dataset"
```

## TensorFlow Environment

Used for:

- `aecs`

Executable:

`C:\Python310\python.exe`

Observed package:

- TensorFlow `2.10.0`

Check command:

```powershell
& "C:\Python310\python.exe" -c "import sys; import tensorflow as tf; print(sys.executable); print(tf.__version__)"
```

Important constraint:

- This environment does not provide the PyTorch setup used by mentor AE-family baselines.

## PyTorch Environment

Used for:

- `deep_ae`
- `sm_dae`
- `sdai`
- `trdae`

Executable:

`C:\Users\Henry\AppData\Local\Programs\Python\Python311\python.exe`

Extra package path:

`D:\Python_Packages\PyTorch`

Observed package:

- PyTorch `2.8.0+cu126`
- CUDA available: `True`

Check command:

```powershell
$env:PYTHONPATH = "D:\Python_Packages\PyTorch;D:\AE-CS-M"
& "C:\Users\Henry\AppData\Local\Programs\Python\Python311\python.exe" -c "import sys; import torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
```

Important constraint:

- This Python environment does not provide TensorFlow.
- Do not import `AECSImputer` eagerly when running mentor baselines.
- `experiments\battery_pack_wltp\imputers\__init__.py` uses lazy exports to prevent TensorFlow dependency leakage into the PyTorch baseline path.

## Why Two Python Environments Are Used

No currently confirmed Python executable has both:

- TensorFlow needed by `AECSImputer`
- PyTorch needed by mentor AE-family baselines

The frozen Phase 1B smoke therefore runs one TensorFlow process for `aecs` and separate PyTorch processes for mentor AE-family methods.

This is acceptable for Phase 1B because the public experiment artifacts are still written through the same project runner into the same `D:\AE-CS-M` result namespace.

## Frozen Smoke Script

Run:

```powershell
cd D:\AE-CS-M
.\scripts\smoke_phase1b_runner.ps1
```

Equivalent explicit form:

```powershell
.\scripts\smoke_phase1b_runner.ps1 `
  -DataRoot "D:\AE-CS\Lithium-ion battery pack cycling dataset with CC-CV charging and WLTPconstant discharge profiles" `
  -TfPython "C:\Python310\python.exe" `
  -TorchPython "C:\Users\Henry\AppData\Local\Programs\Python\Python311\python.exe" `
  -TorchPackages "D:\Python_Packages\PyTorch" `
  -RunTag "smoke_phase1b_runner" `
  -ConfigName "mcar_0.1" `
  -Seed 7 `
  -SmokeMaxRows 64 `
  -MaxTestRecords 1
```

## Expected Artifacts

Expected raw result directories:

- `experiments\battery_pack_wltp\results\raw\smoke_phase1b_runner\aecs`
- `experiments\battery_pack_wltp\results\raw\smoke_phase1b_runner\deep_ae`
- `experiments\battery_pack_wltp\results\raw\smoke_phase1b_runner\sm_dae`
- `experiments\battery_pack_wltp\results\raw\smoke_phase1b_runner\sdai`
- `experiments\battery_pack_wltp\results\raw\smoke_phase1b_runner\trdae`

Expected summary files:

- `experiments\battery_pack_wltp\results\summary\smoke_phase1b_runner\by_seed.csv`
- `experiments\battery_pack_wltp\results\summary\smoke_phase1b_runner\config_summary.csv`
- `experiments\battery_pack_wltp\results\summary\smoke_phase1b_runner\main_table.csv`

Expected mask file:

- `experiments\battery_pack_wltp\masks\smoke_phase1b_runner\Qtzl_Cycle_010_WLTP_partial_data__mcar_0.1__seed7.npz`

## Guardrails

- Do not enable `gain` in Phase 1B.
- Do not add `gan` to Phase 1B.
- Do not write results to `D:\AE-CS`.
- Do not use old `experiments.dynamic_profiles` imports.
- Do not treat smoke metrics as final performance.
- If environment paths change, update this document before changing runner or adapter code.
