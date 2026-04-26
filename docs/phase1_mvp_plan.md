# AE-CS-M Phase 1 MVP Plan v2

## 1. Goal

Phase 1 remains a minimum viable engineering stage.

It must:

- keep AE-CS as the external protocol center
- allow baselines to preserve native internals inside adapters
- keep experiment artifacts fully isolated from `D:\AE-CS`
- include `GAIN`, but only as the final high-risk sub-phase
- exclude `GAN`

## 2. Phase 1 Sub-Phases

### Phase 1A

- documentation revision
- directory contract
- project skeleton
- initial smoke scaffolding

### Phase 1B

- AE-CS body migration
- Deep_AE / SM_DAE / SDAi / TRDAE migration
- AE-family adapter
- runner, registry, dataset, mask, metrics minimal loop

### Phase 1C

- GAIN migration
- GAIN adapter
- registry `gain` support
- GAIN-specific smoke

Entry rule:

- `Phase 1C` may begin only after `Phase 1A` and `Phase 1B` are stable

## 3. Revised Phase 1 Minimal Directory Tree

```text
D:\AE-CS-M
├─ CLAUDE.md
├─ docs/
│  ├─ phase1_mvp_plan.md
│  └─ protocol_contract.md
├─ data/
│  └─ loaders/
├─ models/
│  └─ aecs/
├─ baselines/
│  ├─ mentor_ae_family/
│  └─ mentor_gain_family/
├─ experiments/
│  └─ battery_pack_wltp/
│     ├─ imputers/
│     ├─ results/
│     │  ├─ raw/
│     │  └─ summary/
│     ├─ masks/
│     └─ checkpoints/
└─ scripts/
```

Responsibilities:

- `docs/`: phase documents and protocol docs
- `data/loaders/`: WLTP data loading only
- `models/aecs/`: AE-CS method body only
- `baselines/mentor_ae_family/`: mentor AE family implementation and compatibility code
- `baselines/mentor_gain_family/`: mentor GAIN implementation and compatibility code
- `experiments/battery_pack_wltp/`: sole Phase 1 experiment shell
- `experiments/battery_pack_wltp/imputers/`: unified external adapters
- `results/raw/`: per-run raw results
- `results/summary/`: summaries
- `masks/`: AE-CS-M-owned mask artifacts
- `checkpoints/`: AE-CS-M-owned checkpoints
- `scripts/`: smoke and helper scripts

Build timing:

- create in `Phase 1A`:
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
- create in `Phase 1C`:
  - `baselines/mentor_gain_family/`

## 4. Phase 1 Required Capabilities

Phase 1A must prove:

- docs and directory contract are fixed
- skeleton smoke path exists

Phase 1B must prove:

- AE-CS import succeeds
- Deep_AE family import succeeds
- TRDAE import succeeds
- runner can register methods
- single dataset, single config, single seed can output raw result
- summary tables can be generated

Phase 1C must prove:

- GAIN import succeeds
- `gain` can be registered
- `gain` can pass its own smoke without leaking mentor public semantics

## 5. Revised Phase 1 Baseline Scope

Included:

- `aecs`
- `deep_ae`
- `sm_dae`
- `sdai`
- `trdae`
- `gain`

Excluded:

- `gan`
- `igani`
- `am_dae`
- old-project `vae / brits / am_dae`

Priority rule:

- `gain` is included in Phase 1
- `gain` must be implemented last
- `gain` must not run in parallel with `Phase 1B` core migration

## 6. Revised Phase 1 Not-Doing List

- no multiple datasets
- no multiple experiment shells
- no top-level `training/`
- no top-level `evaluation/`
- no `GAN`
- no AE-CS training-enhancement studies
- no 5-seed full benchmark
- no paper extension ablations

## 7. Construction Rules

- revise docs before code
- unify mask semantics before connecting baselines
- finish the smallest stable loop before expanding
- baselines must not live under `models/`
- AE-CS-M must not share artifact directories with `D:\AE-CS`
- `GAIN` must wait until `Phase 1A` and `Phase 1B` are stable
- if `GAIN` starts exposing mentor public semantics, stop `Phase 1C`

## 8. Revised Phase 1 Completion Standard

Phase 1 is complete only if all of the following are true:

- `Phase 1A` is complete and directory/protocol docs are frozen
- `Phase 1B` is stable:
  - `aecs / deep_ae / sm_dae / sdai / trdae` register successfully
  - at least one end-to-end smoke loop succeeds
  - `results/raw` and `results/summary` are independently written under `AE-CS-M`
- `Phase 1C` is stable:
  - `gain` registers successfully
  - `gain` still obeys unified external protocol
  - `gain` passes its dedicated smoke
- `GAN` is still excluded from Phase 1
