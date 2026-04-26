param(
    [string]$ProjectRoot = "D:\AE-CS-M",
    [string]$DataRoot = $env:AECSM_BATTERY_PACK_WLTP_DATA_ROOT,
    [string]$RunTag = "smoke_phase1c_gain",
    [string]$ConfigName = "mcar_0.1",
    [int]$Seed = 7,
    [int]$SmokeMaxRows = 64,
    [int]$MaxTestRecords = 1,
    [string]$TorchPython = "C:\Users\Henry\AppData\Local\Programs\Python\Python311\python.exe",
    [string]$TorchPackages = "D:\Python_Packages\PyTorch"
)

$ErrorActionPreference = "Stop"

function Resolve-DataRoot {
    param([string]$RequestedRoot, [string]$ProjectRoot)

    if ($RequestedRoot -and (Test-Path -LiteralPath $RequestedRoot -PathType Container)) {
        return (Resolve-Path -LiteralPath $RequestedRoot).Path
    }

    $localRoot = Join-Path $ProjectRoot "battery_pack_wltp_dataset"
    if (Test-Path -LiteralPath $localRoot -PathType Container) {
        return (Resolve-Path -LiteralPath $localRoot).Path
    }

    $readonlyRoot = "D:\AE-CS\Lithium-ion battery pack cycling dataset with CC-CV charging and WLTPconstant discharge profiles"
    if (Test-Path -LiteralPath $readonlyRoot -PathType Container) {
        return (Resolve-Path -LiteralPath $readonlyRoot).Path
    }

    throw "Set AECSM_BATTERY_PACK_WLTP_DATA_ROOT or pass -DataRoot to the read-only battery_pack_wltp_dataset directory."
}

function Invoke-Checked {
    param(
        [string]$Label,
        [string]$Exe,
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "[Phase1C] $Label"
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw "Missing project root: $ProjectRoot"
}
if (-not (Test-Path -LiteralPath $TorchPython -PathType Leaf)) {
    throw "Missing Torch Python executable: $TorchPython"
}
if (-not (Test-Path -LiteralPath $TorchPackages -PathType Container)) {
    throw "Missing Torch package directory: $TorchPackages"
}

$resolvedDataRoot = Resolve-DataRoot -RequestedRoot $DataRoot -ProjectRoot $ProjectRoot
$env:AECSM_BATTERY_PACK_WLTP_DATA_ROOT = $resolvedDataRoot

Write-Host "[Phase1C] ProjectRoot=$ProjectRoot"
Write-Host "[Phase1C] DataRoot=$resolvedDataRoot"
Write-Host "[Phase1C] RunTag=$RunTag"
Write-Host "[Phase1C] ConfigName=$ConfigName Seed=$Seed SmokeMaxRows=$SmokeMaxRows MaxTestRecords=$MaxTestRecords"

Push-Location $ProjectRoot
try {
    $oldPythonPath = $env:PYTHONPATH
    try {
        if ($oldPythonPath) {
            $env:PYTHONPATH = "$TorchPackages;$ProjectRoot;$oldPythonPath"
        }
        else {
            $env:PYTHONPATH = "$TorchPackages;$ProjectRoot"
        }

        Invoke-Checked -Label "PyTorch environment check" -Exe $TorchPython -Arguments @(
            "-c", "import torch; print('torch=' + torch.__version__); print('cuda_available=' + str(torch.cuda.is_available()))"
        )

        Invoke-Checked -Label "GAIN import and registry check" -Exe $TorchPython -Arguments @(
            "-c", "from experiments.battery_pack_wltp.registry import get_method_spec; from experiments.battery_pack_wltp.imputers.gain_imputer import GAINImputer; spec=get_method_spec('gain'); assert spec.enabled and spec.adapter_class == 'GAINImputer'; imp=GAINImputer({'device':'cpu'}); assert imp.name == 'gain'; print('gain registry/import passed')"
        )

        Invoke-Checked -Label "runner smoke: gain" -Exe $TorchPython -Arguments @(
            "-m", "experiments.battery_pack_wltp.run_experiment",
            "--method", "gain",
            "--config-name", $ConfigName,
            "--seed", "$Seed",
            "--smoke",
            "--smoke-max-rows", "$SmokeMaxRows",
            "--run-tag", $RunTag,
            "--max-test-records", "$MaxTestRecords"
        )
    }
    finally {
        $env:PYTHONPATH = $oldPythonPath
    }

    $summaryDir = Join-Path $ProjectRoot "experiments\battery_pack_wltp\results\summary\$RunTag"
    $summaryFile = Join-Path $summaryDir "config_summary.csv"
    if (-not (Test-Path -LiteralPath $summaryFile -PathType Leaf)) {
        throw "Missing summary file after GAIN smoke run: $summaryFile"
    }

    Write-Host ""
    Write-Host "[Phase1C] Summary:"
    Get-Content -LiteralPath $summaryFile
    Write-Host ""
    Write-Host "[Phase1C] GAIN smoke passed."
}
finally {
    Pop-Location
}
