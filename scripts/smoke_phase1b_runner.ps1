param(
    [string]$ProjectRoot = "D:\AE-CS-M",
    [string]$DataRoot = $env:AECSM_BATTERY_PACK_WLTP_DATA_ROOT,
    [string]$RunTag = "smoke_phase1b_runner",
    [string]$ConfigName = "mcar_0.1",
    [int]$Seed = 7,
    [int]$SmokeMaxRows = 64,
    [int]$MaxTestRecords = 1,
    [string]$TfPython = "C:\Python310\python.exe",
    [string]$TorchPython = "C:\Users\Henry\AppData\Local\Programs\Python\Python311\python.exe",
    [string]$TorchPackages = "D:\Python_Packages\PyTorch",
    [switch]$SkipAecs,
    [switch]$SkipMentorAEFamily
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
    Write-Host "[Phase1B] $Label"
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw "Missing project root: $ProjectRoot"
}

$resolvedDataRoot = Resolve-DataRoot -RequestedRoot $DataRoot -ProjectRoot $ProjectRoot
$env:AECSM_BATTERY_PACK_WLTP_DATA_ROOT = $resolvedDataRoot

Write-Host "[Phase1B] ProjectRoot=$ProjectRoot"
Write-Host "[Phase1B] DataRoot=$resolvedDataRoot"
Write-Host "[Phase1B] RunTag=$RunTag"
Write-Host "[Phase1B] ConfigName=$ConfigName Seed=$Seed SmokeMaxRows=$SmokeMaxRows MaxTestRecords=$MaxTestRecords"

Push-Location $ProjectRoot
try {
    $commonArgs = @(
        "-m", "experiments.battery_pack_wltp.run_experiment",
        "--config-name", $ConfigName,
        "--seed", "$Seed",
        "--smoke",
        "--smoke-max-rows", "$SmokeMaxRows",
        "--run-tag", $RunTag,
        "--max-test-records", "$MaxTestRecords"
    )

    if (-not $SkipAecs) {
        Invoke-Checked -Label "TensorFlow environment check" -Exe $TfPython -Arguments @(
            "-c", "import tensorflow as tf; print('tensorflow=' + tf.__version__)"
        )
        Invoke-Checked -Label "runner smoke: aecs" -Exe $TfPython -Arguments (@("-m", "experiments.battery_pack_wltp.run_experiment", "--method", "aecs") + $commonArgs[2..($commonArgs.Length - 1)])
    }

    if (-not $SkipMentorAEFamily) {
        if (-not (Test-Path -LiteralPath $TorchPython -PathType Leaf)) {
            throw "Missing Torch Python executable: $TorchPython"
        }
        if (-not (Test-Path -LiteralPath $TorchPackages -PathType Container)) {
            throw "Missing Torch package directory: $TorchPackages"
        }

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

            foreach ($method in @("deep_ae", "sm_dae", "sdai", "trdae")) {
                Invoke-Checked -Label "runner smoke: $method" -Exe $TorchPython -Arguments (@("-m", "experiments.battery_pack_wltp.run_experiment", "--method", $method) + $commonArgs[2..($commonArgs.Length - 1)])
            }
        }
        finally {
            $env:PYTHONPATH = $oldPythonPath
        }
    }

    $summaryDir = Join-Path $ProjectRoot "experiments\battery_pack_wltp\results\summary\$RunTag"
    $summaryFile = Join-Path $summaryDir "config_summary.csv"
    if (-not (Test-Path -LiteralPath $summaryFile -PathType Leaf)) {
        throw "Missing summary file after smoke run: $summaryFile"
    }

    Write-Host ""
    Write-Host "[Phase1B] Summary:"
    Get-Content -LiteralPath $summaryFile
    Write-Host ""
    Write-Host "[Phase1B] Runner smoke passed."
}
finally {
    Pop-Location
}
