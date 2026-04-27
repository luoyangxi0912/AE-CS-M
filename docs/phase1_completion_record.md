# Phase 1 Completion Record

Status: complete and frozen

Freeze date: 2026-04-26

Project root: `D:\AE-CS-M`

## 1. Scope

Phase 1 includes only the following public methods:

- `aecs`
- `deep_ae`
- `sm_dae`
- `sdai`
- `trdae`
- `gain`

Phase 1 does not include:

- `gan`
- multiple datasets
- multiple experiment shells
- full benchmark sweeps
- paper-scale ablations

## 2. Public Protocol

Unified external interface:

```python
fit(train_source, scaler=None, smoke=False, metadata=None)
impute(X, mask_observed, metadata=None)
```

Unified public mask semantics:

- `mask_observed = 1` means observed
- `mask_observed = 0` means missing

Mentor-style missing-indicator semantics are allowed only inside baseline adapters and baseline internal modules.

## 3. Completed Sub-Phases

### Phase 1A

Completed:

- documentation contract
- directory contract
- project skeleton
- import-level smoke

### Phase 1B

Completed:

- AE-CS body migration
- Deep_AE / SM_DAE / SDAi / TRDAE migration
- AE-family adapter
- dataset, mask, metrics, runner minimal loop
- Phase 1B smoke runner

### Phase 1C

Completed:

- GAIN migration
- GAIN adapter
- registry `gain` support
- GAIN smoke
- unified Phase 1 smoke entry

## 4. Fixed Functional Issue Before Freeze

Before final freeze, `windowing.py` was corrected so that:

- sliding-window generation always includes the trailing window start at `total_length - window_size`
- reconstruction no longer leaves uncovered tail rows as implicit zero predictions
- reconstruction raises an explicit error if uncovered rows remain

This fix was followed by rerunning the unified Phase 1 smoke pipeline.

## 5. Frozen Smoke Commands

Phase 1B smoke:

```powershell
cd D:\AE-CS-M
.\scripts\smoke_phase1b_runner.ps1
```

Phase 1C GAIN smoke:

```powershell
cd D:\AE-CS-M
.\scripts\smoke_phase1c_gain.ps1
```

Unified Phase 1 smoke:

```powershell
cd D:\AE-CS-M
.\scripts\smoke_phase1_all.ps1
```

## 6. Artifact Contract

All Phase 1 experiment artifacts must stay under `D:\AE-CS-M`:

- `experiments\battery_pack_wltp\results\raw\`
- `experiments\battery_pack_wltp\results\summary\`
- `experiments\battery_pack_wltp\masks\`
- `experiments\battery_pack_wltp\checkpoints\`

No Phase 1 artifact may be written to `D:\AE-CS`.

## 7. Known Environment Coupling

The old path:

`D:\AE-CS\Lithium-ion battery pack cycling dataset with CC-CV charging and WLTPconstant discharge profiles`

may still be used as a read-only raw data fallback on the current machine.

This is an environment coupling only.
It is not part of the AE-CS-M artifact protocol.

## 8. Acceptance Result

Phase 1 is considered complete because:

- all six public methods are registered and enabled as intended
- the trailing-window reconstruction bug is fixed
- unified Phase 1 smoke has been rerun after the fix
- smoke-level raw results, masks, and summaries are generated under `AE-CS-M`
- no `GAN` path was introduced
- no multi-dataset or full benchmark expansion was introduced

## 9. Freeze Rule

After this file is accepted:

- do not add new methods into Phase 1
- do not add `GAN` into Phase 1
- do not reinterpret smoke metrics as final benchmark claims
- any post-freeze behavior change should move to Phase 2 or later
