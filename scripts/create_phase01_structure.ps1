$ErrorActionPreference = "Stop"
$ProjectRoot = "E:\BEST\FireSight AI"
Set-Location $ProjectRoot

# Preserve the existing project architecture; create only missing Phase 01 folders.
$Folders = @(
    "data\raw\D-Fire",
    "data\raw\DataCluster",
    "data\inspection\phase01",
    "data\processed",
    "data\annotations",
    "scripts",
    "docs\datasets",
    "reports\datasets"
)

foreach ($Folder in $Folders) {
    New-Item -ItemType Directory -Path $Folder -Force | Out-Null
}

Write-Host "`nPhase 01 folders are ready:" -ForegroundColor Green
$Folders | ForEach-Object { Write-Host " - $_" }

Write-Host "`nDo not move your existing raw datasets. Place/extract them under data\raw\D-Fire and data\raw\DataCluster if they are not already there." -ForegroundColor Yellow
