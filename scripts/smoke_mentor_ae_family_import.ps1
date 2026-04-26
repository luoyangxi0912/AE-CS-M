$ErrorActionPreference = "Stop"

$projectRoot = "D:\AE-CS-M"

Push-Location $projectRoot
try {
@'
from baselines.mentor_ae_family import Deep_AE, TRDAE, build_deep_ae
from experiments.battery_pack_wltp.imputers.mentor_ae_family_imputer import (
    DeepAEImputer,
    SMDAEImputer,
    SDAIImputer,
    TRDAEImputer,
)
from experiments.battery_pack_wltp.registry import get_method_spec

instances = [
    DeepAEImputer(config={"device": "cpu", "window_size": 4, "stride": 2, "batch_size": 2}),
    SMDAEImputer(config={"device": "cpu", "window_size": 4, "stride": 2, "batch_size": 2}),
    SDAIImputer(config={"device": "cpu", "window_size": 4, "stride": 2, "batch_size": 2}),
    TRDAEImputer(config={"device": "cpu", "window_size": 4, "stride": 2, "batch_size": 2}),
]

assert [item.name for item in instances] == ["deep_ae", "sm_dae", "sdai", "trdae"]
assert get_method_spec("deep_ae").enabled is True
assert get_method_spec("sm_dae").enabled is True
assert get_method_spec("sdai").enabled is True
assert get_method_spec("trdae").enabled is True
assert get_method_spec("gain").enabled is False

print("Mentor AE-family import smoke passed.")
print([item.name for item in instances])
'@ | python -
if ($LASTEXITCODE -ne 0) {
    throw "Mentor AE-family import smoke failed with Python exit code $LASTEXITCODE."
}
}
finally {
    Pop-Location
}
