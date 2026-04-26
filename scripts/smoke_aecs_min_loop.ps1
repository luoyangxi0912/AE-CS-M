$ErrorActionPreference = "Stop"

$projectRoot = "D:\AE-CS-M"
$defaultDataRoot = Join-Path $projectRoot "battery_pack_wltp_dataset"

if (-not $env:AECSM_BATTERY_PACK_WLTP_DATA_ROOT) {
    if (Test-Path -LiteralPath $defaultDataRoot -PathType Container) {
        $env:AECSM_BATTERY_PACK_WLTP_DATA_ROOT = $defaultDataRoot
    }
    else {
        throw "Set AECSM_BATTERY_PACK_WLTP_DATA_ROOT to the read-only battery_pack_wltp_dataset directory."
    }
}

Push-Location $projectRoot
try {
@'
from experiments.battery_pack_wltp.dataset import load_dataset_bundles, load_sequence_bundle
from experiments.battery_pack_wltp.imputers.aecs_imputer import AECSImputer
from experiments.battery_pack_wltp.windowing import create_windows, reconstruct_from_windows

smoke = True
seed = 0

dataset = load_dataset_bundles(smoke=smoke)
train_records = dataset["train_records"][:1]
test_record = dataset["test_records"][0]
scaler = dataset["scaler"]

bundle = load_sequence_bundle(test_record, scaler, smoke=smoke)

config = {
    "epochs": 1,
    "batch_size": 2,
    "window_size": 128,
    "stride": 64,
    "shuffle_buffer": 0,
    "hidden_units": 16,
    "latent_dim": 8,
}

imputer = AECSImputer(config=config)
windows_x, windows_m, starts = create_windows(bundle.x_norm, bundle.natural_mask_observed, config["window_size"], config["stride"])
reconstructed = reconstruct_from_windows(windows_x, starts, bundle.x_norm.shape[0], bundle.x_norm.shape[1], config["window_size"])

imputer.fit(train_records, scaler=scaler, smoke=smoke, metadata={"seed": seed})
filled = imputer.impute(bundle.x_norm, bundle.natural_mask_observed, metadata={"seed": seed})

assert windows_x.ndim == 3
assert windows_m.shape == windows_x.shape
assert reconstructed.shape == bundle.x_norm.shape
assert filled.shape == bundle.x_norm.shape

print("AE-CS minimal loop passed.")
print(f"train_records={len(train_records)}")
print(f"file_id={bundle.file_id}")
print(f"sequence_shape={bundle.x_norm.shape}")
print(f"window_shape={windows_x.shape}")
print(f"filled_shape={filled.shape}")
'@ | python -
if ($LASTEXITCODE -ne 0) {
    throw "AE-CS minimal loop failed with Python exit code $LASTEXITCODE."
}
}
finally {
    Pop-Location
}
