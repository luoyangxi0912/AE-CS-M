$ErrorActionPreference = "Stop"

$projectRoot = "D:\AE-CS-M"

Write-Host "[Phase1A] Checking required directories..."
$requiredDirs = @(
    "$projectRoot\docs",
    "$projectRoot\data",
    "$projectRoot\data\loaders",
    "$projectRoot\models",
    "$projectRoot\models\aecs",
    "$projectRoot\baselines",
    "$projectRoot\baselines\mentor_ae_family",
    "$projectRoot\experiments",
    "$projectRoot\experiments\battery_pack_wltp",
    "$projectRoot\experiments\battery_pack_wltp\imputers",
    "$projectRoot\experiments\battery_pack_wltp\results\raw",
    "$projectRoot\experiments\battery_pack_wltp\results\summary",
    "$projectRoot\experiments\battery_pack_wltp\masks",
    "$projectRoot\experiments\battery_pack_wltp\checkpoints",
    "$projectRoot\scripts"
)

foreach ($dir in $requiredDirs) {
    if (-not (Test-Path -LiteralPath $dir -PathType Container)) {
        throw "Missing required directory: $dir"
    }
}

Write-Host "[Phase1A] Checking required files..."
$requiredFiles = @(
    "$projectRoot\CLAUDE.md",
    "$projectRoot\docs\phase1_mvp_plan.md",
    "$projectRoot\docs\protocol_contract.md",
    "$projectRoot\experiments\battery_pack_wltp\configs.py",
    "$projectRoot\experiments\battery_pack_wltp\registry.py",
    "$projectRoot\experiments\battery_pack_wltp\imputers\base.py"
)

foreach ($file in $requiredFiles) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing required file: $file"
    }
}

Write-Host "[Phase1A] Import checks..."
Push-Location $projectRoot
try {
@'
from experiments.battery_pack_wltp import configs, registry
from experiments.battery_pack_wltp.imputers.base import BaseImputer

assert str(configs.PROJECT_ROOT) == r"D:\AE-CS-M"
assert str(configs.EXPERIMENT_DIR).startswith(r"D:\AE-CS-M")
assert str(configs.RESULTS_DIR).startswith(r"D:\AE-CS-M")
assert str(configs.MASKS_DIR).startswith(r"D:\AE-CS-M")
assert str(configs.CHECKPOINTS_DIR).startswith(r"D:\AE-CS-M")
assert str(configs.RESULTS_DIR).startswith(r"D:\AE-CS-M")
assert str(configs.MASKS_DIR).startswith(r"D:\AE-CS-M")
assert str(configs.CHECKPOINTS_DIR).startswith(r"D:\AE-CS-M")

names = registry.list_method_names()
assert names == ("aecs", "deep_ae", "sm_dae", "sdai", "trdae", "gain")
assert BaseImputer.name == "base"

print("Phase 1A smoke passed.")
'@ | python -
}
finally {
    Pop-Location
}
