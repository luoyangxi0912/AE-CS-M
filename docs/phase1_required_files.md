# AE-CS-M Phase 1 Required Files

## 1. Purpose

This document records the required file set for Phase 1 and explicitly separates:

- `Phase 1A`: documentation, directory contract, project skeleton
- `Phase 1B`: `AE-CS + Deep_AE / SM_DAE / SDAi / TRDAE`
- `Phase 1C`: `GAIN`

This file is planning and control documentation only.
It does not authorize early implementation of `Phase 1B` or `Phase 1C`.

## 2. Phase 1A Files

### Already required in Phase 1A

| File Path | Type | Status | Purpose |
|---|---|---|---|
| `D:\AE-CS-M\CLAUDE.md` | document | required | top-level project contract |
| `D:\AE-CS-M\docs\phase1_mvp_plan.md` | document | required | phase plan and scope |
| `D:\AE-CS-M\docs\protocol_contract.md` | document | required | public protocol contract |
| `D:\AE-CS-M\docs\phase1_required_files.md` | document | required | file inventory and phase ownership |
| `D:\AE-CS-M\data\__init__.py` | package placeholder | required | data package marker |
| `D:\AE-CS-M\data\loaders\__init__.py` | package placeholder | required | loader package marker |
| `D:\AE-CS-M\models\__init__.py` | package placeholder | required | models package marker |
| `D:\AE-CS-M\models\aecs\__init__.py` | package placeholder | required | AE-CS package marker only |
| `D:\AE-CS-M\baselines\__init__.py` | package placeholder | required | baselines package marker |
| `D:\AE-CS-M\baselines\mentor_ae_family\__init__.py` | package placeholder | required | AE-family placeholder only |
| `D:\AE-CS-M\experiments\__init__.py` | package placeholder | required | experiments package marker |
| `D:\AE-CS-M\experiments\battery_pack_wltp\__init__.py` | package placeholder | required | experiment-shell marker |
| `D:\AE-CS-M\experiments\battery_pack_wltp\imputers\__init__.py` | package placeholder | required | adapter package marker |
| `D:\AE-CS-M\experiments\battery_pack_wltp\configs.py` | skeleton | required | path constants and smoke constants |
| `D:\AE-CS-M\experiments\battery_pack_wltp\registry.py` | skeleton | required | method placeholder registry |
| `D:\AE-CS-M\experiments\battery_pack_wltp\imputers\base.py` | skeleton | required | unified external adapter interface |
| `D:\AE-CS-M\scripts\smoke_phase1a.ps1` | skeleton script | required | directory and import validation only |

### Explicitly forbidden in Phase 1A

These files must not be implemented in `Phase 1A`:

- `models/aecs/ae_cs.py`
- `models/aecs/losses.py`
- `baselines/mentor_ae_family/module.py`
- `baselines/mentor_ae_family/impu_module.py`
- `baselines/mentor_ae_family/imputation_dataset.py`
- `baselines/mentor_ae_family/dae.py`
- `baselines/mentor_ae_family/trdae.py`
- `baselines/mentor_gain_family/*`
- `experiments/battery_pack_wltp/dataset.py`
- `experiments/battery_pack_wltp/mask_protocols.py`
- `experiments/battery_pack_wltp/metrics.py`
- `experiments/battery_pack_wltp/run_experiment.py`
- `experiments/battery_pack_wltp/imputers/aecs_imputer.py`
- `experiments/battery_pack_wltp/imputers/mentor_ae_family_imputer.py`
- `experiments/battery_pack_wltp/imputers/gain_imputer.py`

## 3. Phase 1B Files

### Must be implemented in Phase 1B

| File Path | Source | Purpose |
|---|---|---|
| `D:\AE-CS-M\data\loaders\dynamic_profiles_loader.py` | old AE-CS | WLTP data loading |
| `D:\AE-CS-M\models\aecs\ae_cs.py` | old AE-CS | AE-CS method body |
| `D:\AE-CS-M\models\aecs\losses.py` | old AE-CS | AE-CS losses |
| `D:\AE-CS-M\baselines\mentor_ae_family\module.py` | mentor project | AE-family base module |
| `D:\AE-CS-M\baselines\mentor_ae_family\impu_module.py` | mentor project | imputation support base |
| `D:\AE-CS-M\baselines\mentor_ae_family\imputation_dataset.py` | mentor project | AE-family internal data layer |
| `D:\AE-CS-M\baselines\mentor_ae_family\dae.py` | mentor project | Deep_AE / SM_DAE / SDAi |
| `D:\AE-CS-M\baselines\mentor_ae_family\trdae.py` | mentor project | TRDAE |
| `D:\AE-CS-M\experiments\battery_pack_wltp\dataset.py` | old AE-CS | experiment dataset protocol |
| `D:\AE-CS-M\experiments\battery_pack_wltp\mask_protocols.py` | old AE-CS | experiment mask protocol |
| `D:\AE-CS-M\experiments\battery_pack_wltp\metrics.py` | old AE-CS | metrics and summaries |
| `D:\AE-CS-M\experiments\battery_pack_wltp\run_experiment.py` | old AE-CS | unified runner |
| `D:\AE-CS-M\experiments\battery_pack_wltp\imputers\aecs_imputer.py` | old AE-CS | AE-CS adapter |
| `D:\AE-CS-M\experiments\battery_pack_wltp\imputers\mentor_ae_family_imputer.py` | new write | AE-family adapter |
| `D:\AE-CS-M\scripts\smoke_phase1b.ps1` | new write | Phase 1B smoke |

### Phase 1B exit criteria

`Phase 1B` is not complete until:

- `aecs` import succeeds
- `deep_ae / sm_dae / sdai / trdae` registration works
- single dataset, single config, single seed smoke works
- raw results and summary files are written only under `D:\AE-CS-M`

## 4. Phase 1C Files

### Must be implemented in Phase 1C

| File Path | Source | Purpose |
|---|---|---|
| `D:\AE-CS-M\baselines\mentor_gain_family\__init__.py` | new write | GAIN family package marker |
| `D:\AE-CS-M\baselines\mentor_gain_family\gain.py` | mentor project | GAIN implementation |
| `D:\AE-CS-M\experiments\battery_pack_wltp\imputers\gain_imputer.py` | new write | GAIN adapter |
| `D:\AE-CS-M\scripts\smoke_phase1c_gain.ps1` | new write | GAIN smoke |

### Phase 1C entry rule

`Phase 1C` may begin only if:

- `Phase 1A` is complete
- `Phase 1B` is stable
- public protocol remains unchanged

### Phase 1C stop rule

Pause `Phase 1C` immediately if:

- `gain` requires experiment-layer users to adopt mentor missing-indicator public semantics
- `gain` cannot fit into the unified external adapter interface
- `gain` forces a rewrite of already-stable `Phase 1B` public protocol

## 5. Directory Ownership By Phase

### Created in Phase 1A

- `docs/`
- `data/loaders/`
- `models/aecs/`
- `baselines/mentor_ae_family/`
- `experiments/battery_pack_wltp/`
- `experiments/battery_pack_wltp/imputers/`
- `experiments/battery_pack_wltp/results/raw/`
- `experiments/battery_pack_wltp/results/summary/`
- `experiments/battery_pack_wltp/masks/`
- `experiments/battery_pack_wltp/checkpoints/`
- `scripts/`

### Deferred to Phase 1C

- `baselines/mentor_gain_family/`

## 6. Current Phase 1A Closure Checklist

- `CLAUDE.md` exists and matches v2 contract
- `docs/phase1_mvp_plan.md` exists
- `docs/protocol_contract.md` exists
- `docs/phase1_required_files.md` exists
- `experiments/battery_pack_wltp/configs.py` imports successfully
- `experiments/battery_pack_wltp/registry.py` imports successfully
- `experiments/battery_pack_wltp/imputers/base.py` imports successfully
- all Phase 1A path constants point to `D:\AE-CS-M`
- no artifact path points to `D:\AE-CS`
- `mentor_gain_family/` has not been created yet
