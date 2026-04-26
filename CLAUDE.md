# AE-CS-M Project Contract v2

## 1. Project Positioning

`AE-CS-M` is a new and fully independent imputation experiment project.

Phase 1 remains a minimum viable engineering stage, but it is split into three sub-phases:

- `Phase 1A`: documentation, directory contract, project skeleton
- `Phase 1B`: `AE-CS + Deep_AE / SM_DAE / SDAi / TRDAE`
- `Phase 1C`: `GAIN`

Phase 1 scope rules:

- the external protocol is AE-CS-first
- baselines may keep native internal input and mask semantics inside adapters
- all methods must expose the same external interface
- `GAIN` is included in Phase 1, but only as the high-risk final sub-phase `Phase 1C`
- `GAN` is not allowed into Phase 1

## 2. Relationship To `D:\AE-CS`

- `D:\AE-CS` is read-only reference
- it provides:
  - AE-CS method reference
  - old experiment protocol reference
  - `battery_pack_wltp_dataset` loading reference
- `AE-CS-M` must not share any experiment artifact directories with `D:\AE-CS`
- forbidden shared artifact categories:
  - `results`
  - `masks`
  - `checkpoints`
  - `cache`

Code may be migrated or rewritten from the old project, but runtime artifacts must be isolated.

## 3. Relationship To Mentor Project

The mentor project provides:

- baseline families
- old imputation-shell ideas
- missing-value write-back training references
- old experiment organization references

The mentor project is not the public protocol authority of `AE-CS-M`.

`AE-CS-M` does not directly inherit the following mentor defaults:

- `mask = 1` means missing
- flatten/vector as the global public input protocol
- write-back during training as the default AE-CS training definition

Mentor baselines may preserve native internals inside adapters, but they must not leak old public semantics into the experiment layer.

## 4. Root Whitelist

Only the following first-level entries are allowed at repository root:

- `CLAUDE.md`
- `docs/`
- `data/`
- `models/`
- `baselines/`
- `experiments/`
- `scripts/`
- read-only raw external data directory or its read-only mapping

If a new first-level directory is needed, update this document first.

## 5. First-Level Responsibilities

- `docs/`: documentation only
- `data/`: data loading and reproducible data-layer helpers only
- `models/`: AE-CS method body and direct dependencies only
- `baselines/`: baseline families and compatibility code only
- `experiments/`: experiment protocol, adapters, runner, metrics, results
- `scripts/`: executable scripts only

## 6. Hard Boundary Between `models/` And `baselines/`

- `models/` only contains AE-CS method body
- mentor baselines and compatibility code are forbidden under `models/`
- mentor baselines must live under `baselines/`
- experiment layer must call baselines through unified adapters under `experiments/.../imputers/`

Phase 1 baseline family locations:

- `baselines/mentor_ae_family/`
- `baselines/mentor_gain_family/`

`baselines/mentor_gain_family/` is reserved for `Phase 1C`

## 7. File Naming Rules

- directory names: lowercase English with underscores
- Python files: lowercase English with underscores
- docs files: lowercase English with underscores
- adapter files: `<method>_imputer.py`
- registry file: `registry.py`
- runner file: `run_experiment.py`
- dataset protocol file: `dataset.py`
- mask protocol file: `mask_protocols.py`
- metrics file: `metrics.py`

## 8. Baseline Family Naming Rules

Family layout uses "family directory + stable public method names".

Phase 1 public method names:

- `aecs`
- `deep_ae`
- `sm_dae`
- `sdai`
- `trdae`
- `gain`

Family rules:

- `Deep_AE / SM_DAE / SDAi` belong to one AE-family implementation
- `TRDAE` is exposed as an independent public method
- `GAIN` belongs to `mentor_gain_family`
- family internals may share low-level implementation
- public method names must not use mentor `model_id`

Phase rules:

- `aecs / deep_ae / sm_dae / sdai / trdae` belong to `Phase 1B`
- `gain` belongs to `Phase 1C`
- `gan` is excluded from Phase 1

## 9. Unified External Interface

All methods must expose:

- `fit(train_source, scaler=None, smoke=False, metadata=None)`
- `impute(X, mask_observed, metadata=None)`

Unified mask contract:

- `mask_observed = 1` means observed
- `mask_observed = 0` means missing
- if a baseline internally needs missing-indicator:
  - `mask_missing = 1 - mask_observed`

Modules are forbidden from guessing mask semantics by ambiguous naming.

## 10. Mask Naming Rules

Allowed public names:

- `mask_observed`
- `mask_missing`

Allowed derived names:

- `train_mask_observed`
- `eval_mask_missing`
- `test_mask_missing`

Forbidden long-lived ambiguous names:

- `mask`
- `nan`

except for tiny local scope with explicit comment.

## 11. AE-CS-P vs AE-CS-M

- `AE-CS-P`: old project protocol and artifact system in `D:\AE-CS`
- `AE-CS-M`: current independent project and its experiment shell

They must not mix:

- results directories
- mask directories
- checkpoint directories

All docs, logs and tables must clearly distinguish them.

## 12. Result Artifact Rules

Phase 1 experiment artifacts may only be written to:

- `experiments/battery_pack_wltp/results/raw/`
- `experiments/battery_pack_wltp/results/summary/`
- `experiments/battery_pack_wltp/masks/`
- `experiments/battery_pack_wltp/checkpoints/`

Naming rules:

- raw result: `<file_id>__<config_name>__seed<seed>.json`
- mask: `<file_id>__<config_name>__seed<seed>.npz`
- summary: `by_seed.csv`, `config_summary.csv`, `main_table.csv`

Forbidden write-back targets:

- old project directories
- raw data directory
- root-level scattered artifact files

## 13. Smoke / Full Run Rules

- `smoke` is part of the protocol, not an ad hoc tiny run
- `smoke` must use isolated artifact space
- `full run` artifacts must never be overwritten by `smoke`
- Phase 1 may begin with single-config single-seed smoke

Phase-aware smoke requirements:

- `Phase 1A`: skeleton smoke
- `Phase 1B`: AE-family smoke
- `Phase 1C`: GAIN smoke

`gain` smoke is only allowed after `Phase 1A` and `Phase 1B` are stable.

## 14. Raw Data Read-Only Rule

`battery_pack_wltp_dataset` is read-only input.

Never write back:

- mask
- checkpoint
- json
- csv
- npz
- logs
- cache

All derived artifacts must stay inside `AE-CS-M`.

## 15. Development Rules

- update docs before code
- lock protocol before implementation
- complete the smallest stable loop before expansion
- if directory structure, naming rules or artifact rules change, update this file first
- no baseline migration may rewrite AE-CS method definition
- do not start `Phase 1C` before `Phase 1A` and `Phase 1B` are stable
- do not admit `GAN` into Phase 1
