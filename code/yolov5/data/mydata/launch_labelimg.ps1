# Launch LabelImg for mydata (Phase C)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Resolve-Path (Join-Path $Here "..\..\..\..")
$VenvLabelImg = Join-Path $Repo ".venv\Scripts\labelImg.exe"
$Images = Join-Path $Here "images"
$Classes = Join-Path $Here "predefined_classes.txt"
$Xml = Join-Path $Here "xml"

if (-not (Test-Path $VenvLabelImg)) {
    Write-Host "labelImg not found. Run from repo root:"
    Write-Host '  uv pip install labelImg "PyQt5==5.15.11" "PyQt5-Qt5==5.15.2" lxml'
    exit 1
}
if (-not (Test-Path $Images)) { New-Item -ItemType Directory -Path $Images | Out-Null }
if (-not (Test-Path $Xml)) { New-Item -ItemType Directory -Path $Xml | Out-Null }

Write-Host "Open Dir : $Images"
Write-Host "Save Dir : $Xml"
Write-Host "  (in LabelImg: Change Save Dir -> xml folder)"
Write-Host "Format   : PascalVOC"
Write-Host "Classes  : person / car"
Write-Host "Hotkeys  : W draw | Ctrl+S save | D next | A prev"
& $VenvLabelImg $Images $Classes
