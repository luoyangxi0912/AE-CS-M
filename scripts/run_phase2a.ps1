param(
    [string]$ProjectRoot = "D:\AE-CS-M",
    [string]$DataRoot = $env:AECSM_BATTERY_PACK_WLTP_DATA_ROOT,
    [string]$ConfigName = "mcar_0.1",
    [string]$RunTag = "phase2a_non_smoke_mcar_0_1",
    [string]$TfPython = "C:\Python310\python.exe",
    [string]$TorchPython = "C:\Users\Henry\AppData\Local\Programs\Python\Python311\python.exe",
    [string]$TorchPackages = "D:\Python_Packages\PyTorch",
    [int]$MaxTestRecords = 1
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

function New-TempConfigFile {
    param(
        [string]$ProjectRoot,
        [string]$Method,
        [string]$Json
    )

    $tmpDir = Join-Path $ProjectRoot ".codex_tmp\phase2a"
    New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
    $path = Join-Path $tmpDir "$Method.json"
    Set-Content -LiteralPath $path -Value $Json -Encoding UTF8
    return $path
}

function Invoke-Checked {
    param(
        [string]$Label,
        [string]$Exe,
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "[Phase2A] $Label"
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
$originalPythonPath = $env:PYTHONPATH

$seeds = @(7, 17, 27)

$aecsConfig = @'
{
  "epochs": 3,
  "batch_size": 8,
  "window_size": 128,
  "stride": 32,
  "hidden_units": 128,
  "latent_dim": 32,
  "dropout_rate": 0.0,
  "shuffle_buffer": 256,
  "p_drop": 0.2,
  "p_consist": 0.1,
  "learning_rate": 0.001,
  "lambda_consist": 0.05,
  "lambda_space": 0.05,
  "lambda_time": 0.05
}
'@

$deepAeConfig = @'
{
  "epochs": 3,
  "batch_size": 8,
  "window_size": 32,
  "stride": 16,
  "hidden_dim": 128,
  "latent_dim": 64,
  "dropout": 0.0,
  "shuffle_buffer": 128,
  "corruption_rate": 0.2,
  "learning_rate": 0.0001,
  "device": "cpu"
}
'@

$trdaeConfig = @'
{
  "epochs": 2,
  "batch_size": 4,
  "window_size": 8,
  "stride": 4,
  "hidden_dim": 64,
  "latent_dim": 32,
  "dropout": 0.0,
  "shuffle_buffer": 64,
  "corruption_rate": 0.2,
  "learning_rate": 0.0001,
  "device": "cpu",
  "trdae_exact_max_dim": 2048
}
'@

$gainConfig = @'
{
  "epochs": 3,
  "batch_size": 8,
  "window_size": 32,
  "stride": 16,
  "hidden_dim": 128,
  "dropout": 0.0,
  "shuffle_buffer": 128,
  "corruption_rate": 0.2,
  "alpha": 100.0,
  "hint_drop_rate": 0.2,
  "learning_rate": 0.0001,
  "device": "cpu",
  "d_steps": 1,
  "g_steps": 1
}
'@

$aecsConfigPath = New-TempConfigFile -ProjectRoot $ProjectRoot -Method "aecs" -Json $aecsConfig
$deepAeConfigPath = New-TempConfigFile -ProjectRoot $ProjectRoot -Method "deep_ae" -Json $deepAeConfig
$trdaeConfigPath = New-TempConfigFile -ProjectRoot $ProjectRoot -Method "trdae" -Json $trdaeConfig
$gainConfigPath = New-TempConfigFile -ProjectRoot $ProjectRoot -Method "gain" -Json $gainConfig

Write-Host "[Phase2A] ProjectRoot=$ProjectRoot"
Write-Host "[Phase2A] DataRoot=$resolvedDataRoot"
Write-Host "[Phase2A] RunTag=$RunTag"
Write-Host "[Phase2A] ConfigName=$ConfigName"
Write-Host "[Phase2A] Seeds=$($seeds -join ', ')"
Write-Host "[Phase2A] MaxTestRecords=$MaxTestRecords"

Push-Location $ProjectRoot
try {
    $env:PYTHONPATH = $ProjectRoot

    Invoke-Checked -Label "TensorFlow environment check" -Exe $TfPython -Arguments @(
        "-c", "import tensorflow as tf; print('tensorflow=' + tf.__version__)"
    )

    foreach ($seed in $seeds) {
        Invoke-Checked -Label "run aecs seed=$seed" -Exe $TfPython -Arguments @(
            "-m", "experiments.battery_pack_wltp.run_experiment",
            "--method", "aecs",
            "--config-name", $ConfigName,
            "--seed", "$seed",
            "--run-tag", $RunTag,
            "--max-test-records", "$MaxTestRecords",
            "--imputer-config-json", $aecsConfigPath
        )
    }

    try {
        if ($originalPythonPath) {
            $env:PYTHONPATH = "$TorchPackages;$ProjectRoot;$originalPythonPath"
        }
        else {
            $env:PYTHONPATH = "$TorchPackages;$ProjectRoot"
        }

        Invoke-Checked -Label "PyTorch environment check" -Exe $TorchPython -Arguments @(
            "-c", "import torch; print('torch=' + torch.__version__); print('cuda_available=' + str(torch.cuda.is_available()))"
        )

        foreach ($seed in $seeds) {
            Invoke-Checked -Label "run deep_ae seed=$seed" -Exe $TorchPython -Arguments @(
                "-m", "experiments.battery_pack_wltp.run_experiment",
                "--method", "deep_ae",
                "--config-name", $ConfigName,
                "--seed", "$seed",
                "--run-tag", $RunTag,
                "--max-test-records", "$MaxTestRecords",
                "--imputer-config-json", $deepAeConfigPath
            )

            Invoke-Checked -Label "run trdae seed=$seed" -Exe $TorchPython -Arguments @(
                "-m", "experiments.battery_pack_wltp.run_experiment",
                "--method", "trdae",
                "--config-name", $ConfigName,
                "--seed", "$seed",
                "--run-tag", $RunTag,
                "--max-test-records", "$MaxTestRecords",
                "--imputer-config-json", $trdaeConfigPath
            )

            Invoke-Checked -Label "run gain seed=$seed" -Exe $TorchPython -Arguments @(
                "-m", "experiments.battery_pack_wltp.run_experiment",
                "--method", "gain",
                "--config-name", $ConfigName,
                "--seed", "$seed",
                "--run-tag", $RunTag,
                "--max-test-records", "$MaxTestRecords",
                "--imputer-config-json", $gainConfigPath
            )
        }
    }
    finally {
        $env:PYTHONPATH = $originalPythonPath
    }

    $summaryDir = Join-Path $ProjectRoot "experiments\battery_pack_wltp\results\summary\$RunTag"
    $summaryFile = Join-Path $summaryDir "config_summary.csv"
    if (-not (Test-Path -LiteralPath $summaryFile -PathType Leaf)) {
        throw "Missing Phase 2A summary file: $summaryFile"
    }

    Write-Host ""
    Write-Host "[Phase2A] Summary:"
    Get-Content -LiteralPath $summaryFile
    Write-Host ""
    Write-Host "[Phase2A] Non-smoke single-config run completed."
}
finally {
    $env:PYTHONPATH = $originalPythonPath
    Pop-Location
}
