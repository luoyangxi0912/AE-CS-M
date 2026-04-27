# Phase 2A MCAR Result Note

Status: recorded

Date: 2026-04-27

Project root: `D:\AE-CS-M`

## Scope

This note records the current Phase 2A non-smoke MCAR results after migrating AE-CS to the `aecs.dropout0` mainline.

Included methods:

- `aecs`
- `deep_ae`
- `trdae`
- `gain`

Included configs:

- `mcar_0.1`
- `mcar_0.3`

Seeds:

- `7`
- `17`
- `27`

Dataset:

- `battery_pack_wltp_dataset`

Test selection:

- single test record
- `max_test_records = 1`

Metric space:

- normalized

## AE-CS Mainline

The current AE-CS-M mainline is `aecs.dropout0`.

This means:

- `dropout_rate = 0.0`
- neural-network hidden-layer dropout is disabled
- AE-CS denoising corruption is still enabled through `p_drop = 0.2`
- AE-CS consistency-mask augmentation is still enabled through `p_consist = 0.1`

This does not remove missing-value training. It only removes extra stochastic dropout inside the neural network layers.

## Result Summary

### `mcar_0.1`

Run tag:

- `phase2a_non_smoke_mcar_0_1_dropout0`

| method | MAE mean | MAE std | RMSE mean | RMSE std | R2 mean | R2 std | n_runs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `aecs` | 0.335657 | 0.069047 | 0.489386 | 0.092393 | 0.850346 | 0.057937 | 3 |
| `deep_ae` | 0.457671 | 0.015403 | 0.708818 | 0.039303 | 0.692800 | 0.034812 | 3 |
| `gain` | 0.511088 | 0.099004 | 0.803690 | 0.186406 | 0.592045 | 0.184603 | 3 |
| `trdae` | 0.813742 | 0.106611 | 1.278782 | 0.181343 | -0.010677 | 0.272592 | 3 |

### `mcar_0.3`

Run tag:

- `phase2a_non_smoke_mcar_0_3_dropout0`

| method | MAE mean | MAE std | RMSE mean | RMSE std | R2 mean | R2 std | n_runs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `aecs` | 0.294902 | 0.016023 | 0.434961 | 0.022443 | 0.884450 | 0.012234 | 3 |
| `gain` | 0.400758 | 0.040588 | 0.620858 | 0.074300 | 0.762820 | 0.055708 | 3 |
| `deep_ae` | 0.435705 | 0.047878 | 0.640323 | 0.054432 | 0.748854 | 0.043436 | 3 |
| `trdae` | 0.730649 | 0.106313 | 1.152636 | 0.217363 | 0.171069 | 0.302141 | 3 |

## Current Reading

`aecs.dropout0` is the strongest method in both MCAR settings in Phase 2A.

For `mcar_0.1`, `aecs.dropout0` has the best MAE, RMSE, and R2.

For `mcar_0.3`, `aecs.dropout0` also has the best MAE, RMSE, and R2.

`gain` becomes the second-best method on `mcar_0.3`, ahead of `deep_ae`, but remains more seed-sensitive than `aecs`.

`trdae` remains weak under the current migrated adapter and Phase 2A configuration. This should not yet be interpreted as a final statement about the original TRDAE method, because the current adapter still differs from the full mentor training semantics.

## Usage Boundary

These results are valid for Phase 2A inspection and method-direction decisions.

They should not yet be used as a final paper main table because:

- only MCAR configs are covered
- only one dataset is covered
- only one test record is covered
- only four methods are included
- only three seeds are used
- broader missing-pattern protocols have not been run

## Artifact Locations

Summaries:

- `experiments\battery_pack_wltp\results\summary\phase2a_non_smoke_mcar_0_1_dropout0`
- `experiments\battery_pack_wltp\results\summary\phase2a_non_smoke_mcar_0_3_dropout0`

Raw results:

- `experiments\battery_pack_wltp\results\raw\phase2a_non_smoke_mcar_0_1_dropout0`
- `experiments\battery_pack_wltp\results\raw\phase2a_non_smoke_mcar_0_3_dropout0`

Masks:

- `experiments\battery_pack_wltp\masks\phase2a_non_smoke_mcar_0_1_dropout0`
- `experiments\battery_pack_wltp\masks\phase2a_non_smoke_mcar_0_3_dropout0`

Experiment artifacts are intentionally ignored by Git and must be regenerated when needed.
