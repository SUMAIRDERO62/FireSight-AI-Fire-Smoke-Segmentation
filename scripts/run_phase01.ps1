$ErrorActionPreference = "Stop"

$ProjectRoot = "E:\BEST\FireSight AI"
Set-Location $ProjectRoot

Write-Host "`n[FireSight AI] Phase 01 dataset audit" -ForegroundColor Cyan

$RequiredFolders = @(
    "data\raw\D-Fire",
    "data\raw\DataCluster",
    "data\inspection\phase01",
    "scripts",
    "docs\datasets"
)

foreach ($Folder in $RequiredFolders) {
    if (-not (Test-Path $Folder)) {
        New-Item -ItemType Directory -Path $Folder -Force | Out-Null
        Write-Host "Created: $Folder" -ForegroundColor Yellow
    } else {
        Write-Host "Found:   $Folder" -ForegroundColor Green
    }
}

if (-not (Test-Path "scripts\inspect_datasets.py")) {
    throw "Missing scripts\inspect_datasets.py. Extract the Phase 01 kit into the project root first."
}

Write-Host "`nRunning read-only dataset audit..." -ForegroundColor Cyan
python "scripts\inspect_datasets.py"

if ($LASTEXITCODE -ne 0) {
    throw "Dataset audit failed. Read the Python error above."
}

Write-Host "`nAudit completed. Reports:" -ForegroundColor Green
Get-ChildItem "data\inspection\phase01" -File | Select-Object Name, Length, LastWriteTime

Write-Host "`nNext: upload phase01_dataset_audit.txt and phase01_dataset_audit.json for review." -ForegroundColor Cyan
