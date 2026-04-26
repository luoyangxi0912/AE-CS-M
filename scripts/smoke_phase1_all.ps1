param(
    [string]$ProjectRoot = "D:\AE-CS-M",
    [string]$DataRoot = $env:AECSM_BATTERY_PACK_WLTP_DATA_ROOT,
    [string]$ConfigName = "mcar_0.1",
    [int]$Seed = 7,
    [int]$SmokeMaxRows = 64,
    [int]$MaxTestRecords = 1,
    [string]$TfPython = "C:\Python310\python.exe",
    [string]$TorchPython = "C:\Users\Henry\AppData\Local\Programs\Python\Python311\python.exe",
    [string]$TorchPackages = "D:\Python_Packages\PyTorch",
    [string]$Phase1BRunTag = "smoke_phase1b_runner",
    [string]$Phase1CGainRunTag = "smoke_phase1c_gain"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw "Missing project root: $ProjectRoot"
}

$phase1bScript = Join-Path $ProjectRoot "scripts\smoke_phase1b_runner.ps1"
$phase1cGainScript = Join-Path $ProjectRoot "scripts\smoke_phase1c_gain.ps1"

if (-not (Test-Path -LiteralPath $phase1bScript -PathType Leaf)) {
    throw "Missing Phase 1B smoke script: $phase1bScript"
}
if (-not (Test-Path -LiteralPath $phase1cGainScript -PathType Leaf)) {
    throw "Missing Phase 1C GAIN smoke script: $phase1cGainScript"
}

Write-Host "[Phase1] ProjectRoot=$ProjectRoot"
Write-Host "[Phase1] ConfigName=$ConfigName Seed=$Seed SmokeMaxRows=$SmokeMaxRows MaxTestRecords=$MaxTestRecords"
Write-Host "[Phase1] Phase1BRunTag=$Phase1BRunTag"
Write-Host "[Phase1] Phase1CGainRunTag=$Phase1CGainRunTag"

Write-Host ""
Write-Host "[Phase1] Phase 1B methods: aecs/deep_ae/sm_dae/sdai/trdae"
if ($DataRoot) {
    & $phase1bScript -ProjectRoot $ProjectRoot -DataRoot $DataRoot -RunTag $Phase1BRunTag -ConfigName $ConfigName -Seed $Seed -SmokeMaxRows $SmokeMaxRows -MaxTestRecords $MaxTestRecords -TfPython $TfPython -TorchPython $TorchPython -TorchPackages $TorchPackages
}
else {
    & $phase1bScript -ProjectRoot $ProjectRoot -RunTag $Phase1BRunTag -ConfigName $ConfigName -Seed $Seed -SmokeMaxRows $SmokeMaxRows -MaxTestRecords $MaxTestRecords -TfPython $TfPython -TorchPython $TorchPython -TorchPackages $TorchPackages
}
if ($LASTEXITCODE -ne 0) {
    throw "Phase 1B smoke failed with exit code $LASTEXITCODE."
}

Write-Host ""
Write-Host "[Phase1] Phase 1C method: gain"
if ($DataRoot) {
    & $phase1cGainScript -ProjectRoot $ProjectRoot -DataRoot $DataRoot -RunTag $Phase1CGainRunTag -ConfigName $ConfigName -Seed $Seed -SmokeMaxRows $SmokeMaxRows -MaxTestRecords $MaxTestRecords -TorchPython $TorchPython -TorchPackages $TorchPackages
}
else {
    & $phase1cGainScript -ProjectRoot $ProjectRoot -RunTag $Phase1CGainRunTag -ConfigName $ConfigName -Seed $Seed -SmokeMaxRows $SmokeMaxRows -MaxTestRecords $MaxTestRecords -TorchPython $TorchPython -TorchPackages $TorchPackages
}
if ($LASTEXITCODE -ne 0) {
    throw "Phase 1C GAIN smoke failed with exit code $LASTEXITCODE."
}

Write-Host ""
Write-Host "[Phase1] All Phase 1 smoke checks passed."
