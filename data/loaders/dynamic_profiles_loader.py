from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import pandas as pd

from experiments.battery_pack_wltp.configs import PROJECT_ROOT, RAW_DATASET_NAME


META_COLUMNS = [
    "Timestamp",
    "Cycle",
    "Semicycle",
]

CELL_VOLTAGE_COLUMNS = [
    f"Voltage_Cell_P{pack}S{cell} [V]"
    for pack in range(1, 4)
    for cell in range(1, 13)
]

CELL_TEMP_TOP_COLUMNS = [
    f"Temperature_Cell_Top_P{pack}S{cell} [degC]"
    for pack in range(1, 4)
    for cell in range(1, 13)
]

CELL_TEMP_BOTTOM_COLUMNS = [
    f"Temperature_Cell_Bottom_P{pack}S{cell} [degC]"
    for pack in range(1, 4)
    for cell in range(1, 13)
]

CURRENT_COLUMNS = [
    "Current_Actual_Battery [A]",
    "Current_Actual_P1 [A]",
    "Current_Actual_P2 [A]",
    "Current_Actual_P3 [A]",
    "Current_Max_Charge [A]",
    "Current_Max_Discharge [A]",
]

VOLTAGE_SUMMARY_COLUMNS = [
    "Voltage_Actual_Battery [V]",
    "Voltage_Actual_Battery_IT5101 [V]",
    "Voltage_Actual_P1 [V]",
    "Voltage_Actual_P2 [V]",
    "Voltage_Actual_P3 [V]",
    "Voltage_Avg_Cell [V]",
    "Voltage_Max_Battery [V]",
    "Voltage_Max_Cell [V]",
    "Voltage_Min_Battery [V]",
    "Voltage_Min_Cell [V]",
]

STATE_ENV_COLUMNS = [
    "SoC_Actual_Battery [percent]",
    "Resistance_Actual_Battery_IT5101 [Ohm]",
    "Temperature_IN_Chamber [degC]",
    "Temperature_OUT_Chamber [degC]",
    "Humidity_IN_Chamber [RH_percent]",
]

FEATURE_GROUPS = {
    "cell_voltage": CELL_VOLTAGE_COLUMNS,
    "cell_temp_top": CELL_TEMP_TOP_COLUMNS,
    "cell_temp_bottom": CELL_TEMP_BOTTOM_COLUMNS,
    "current": CURRENT_COLUMNS,
    "voltage_summary": VOLTAGE_SUMMARY_COLUMNS,
    "state_env": STATE_ENV_COLUMNS,
}

FEATURE_COLUMNS = (
    CELL_VOLTAGE_COLUMNS
    + CELL_TEMP_TOP_COLUMNS
    + CELL_TEMP_BOTTOM_COLUMNS
    + CURRENT_COLUMNS
    + VOLTAGE_SUMMARY_COLUMNS
    + STATE_ENV_COLUMNS
)

GROUP_ORDER = [
    "cell_voltage",
    "cell_temp_top",
    "cell_temp_bottom",
    "current",
    "voltage_summary",
    "state_env",
]

CELL_GROUPS = [
    {
        "cell_id": f"P{pack}S{cell}",
        "voltage": f"Voltage_Cell_P{pack}S{cell} [V]",
        "temp_top": f"Temperature_Cell_Top_P{pack}S{cell} [degC]",
        "temp_bottom": f"Temperature_Cell_Bottom_P{pack}S{cell} [degC]",
    }
    for pack in range(1, 4)
    for cell in range(1, 13)
]


@dataclass(frozen=True)
class SequenceRecord:
    file_id: str
    split: str
    protocol: str
    path: Path


def dataset_root() -> Path:
    env_root = os.environ.get("AECSM_BATTERY_PACK_WLTP_DATA_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return PROJECT_ROOT / RAW_DATASET_NAME


def list_sequence_files(root: Path | None = None) -> List[Path]:
    root = dataset_root() if root is None else Path(root)
    return sorted(root.glob("*.parquet"))


def split_sequence_files(root: Path | None = None) -> Dict[str, List[SequenceRecord]]:
    files = list_sequence_files(root)
    wltp = [p for p in files if "WLTP" in p.name]
    capacity = [p for p in files if "Capacity_check" in p.name]
    if len(wltp) < 8:
        raise FileNotFoundError(
            "Expected at least 8 WLTP parquet files under the read-only dataset root. "
            f"Resolved root: {dataset_root() if root is None else Path(root)}"
        )
    return {
        "train": [SequenceRecord(p.stem, "train", "wltp_main", p) for p in wltp[:6]],
        "val": [SequenceRecord(wltp[6].stem, "val", "wltp_main", wltp[6])],
        "test": [SequenceRecord(wltp[7].stem, "test", "wltp_main", wltp[7])],
        "supplementary": [SequenceRecord(p.stem, "supplementary", "capacity_check", p) for p in capacity],
    }


def read_sequence_file(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def infer_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in META_COLUMNS]


def available_feature_columns(df: pd.DataFrame, preferred: Sequence[str] = FEATURE_COLUMNS) -> List[str]:
    columns = [c for c in preferred if c in df.columns]
    if columns:
        return columns
    return infer_feature_columns(df)


def slice_record(
    record: SequenceRecord,
    smoke: bool = False,
    smoke_max_rows: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = read_sequence_file(record.path)
    if smoke and smoke_max_rows is not None:
        df = df.iloc[:smoke_max_rows].reset_index(drop=True)
    missing_meta = [c for c in META_COLUMNS if c not in df.columns]
    if missing_meta:
        raise ValueError(f"Missing metadata columns in {record.path.name}: {missing_meta}")
    feature_columns = available_feature_columns(df)
    if not feature_columns:
        raise ValueError(f"No feature columns found in {record.path.name}.")
    return df[META_COLUMNS].copy(), df[feature_columns].copy()


def iter_feature_frames(
    records: Iterable[SequenceRecord],
    smoke: bool = False,
    smoke_max_rows: int | None = None,
) -> Iterator[tuple[SequenceRecord, pd.DataFrame]]:
    for record in records:
        _, features = slice_record(record, smoke=smoke, smoke_max_rows=smoke_max_rows)
        yield record, features
