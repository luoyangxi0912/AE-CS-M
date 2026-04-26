from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "docs"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
BASELINES_DIR = PROJECT_ROOT / "baselines"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "battery_pack_wltp"

RESULTS_DIR = EXPERIMENT_DIR / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
SUMMARY_RESULTS_DIR = RESULTS_DIR / "summary"
MASKS_DIR = EXPERIMENT_DIR / "masks"
CHECKPOINTS_DIR = EXPERIMENT_DIR / "checkpoints"

RAW_DATASET_NAME = "battery_pack_wltp_dataset"

PHASE_1A_METHOD_PLACEHOLDERS = (
    "aecs",
    "deep_ae",
    "sm_dae",
    "sdai",
    "trdae",
    "gain",
)

PHASE_1A_SMOKE = {
    "phase": "1A",
    "split": "skeleton",
    "config_name": "skeleton_smoke",
    "seed": 0,
    "check_training": False,
}

