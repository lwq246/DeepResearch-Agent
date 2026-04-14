param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "[1/4] Starting Qdrant via Docker Compose..." -ForegroundColor Cyan
docker compose up -d qdrant

$venvPath = Join-Path $root "backend\.venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$createdVenv = $false

if (-not (Test-Path $venvPython)) {
    Write-Host "[2/4] Creating backend virtual environment..." -ForegroundColor Cyan
    py -m venv $venvPath
    $createdVenv = $true
}

if (-not (Test-Path $venvPython)) {
    throw "Could not find Python at $venvPython"
}

if ($createdVenv -or -not $SkipInstall) {
    Write-Host "[3/4] Ensuring backend dependencies are installed..." -ForegroundColor Cyan
    & $venvPython -m pip install -r (Join-Path $root "backend\requirements.txt")
}

if (-not (Test-Path (Join-Path $root "frontend\node_modules"))) {
    Write-Host "[3b/4] Installing frontend dependencies..." -ForegroundColor Cyan
    Set-Location (Join-Path $root "frontend")
    npm install
    Set-Location $root
}

$backendCommand = "Set-Location '$root\backend'; & '$venvPython' -m uvicorn main:app --reload --port 8000"
$frontendCommand = "Set-Location '$root\frontend'; npm run dev"

Write-Host "[4/4] Launching backend and frontend terminals..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCommand | Out-Null
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCommand | Out-Null

Write-Host "Done." -ForegroundColor Green
Write-Host "Backend: http://localhost:8000" -ForegroundColor Green
Write-Host "Frontend: http://localhost:3000" -ForegroundColor Green
Write-Host "Debug API: POST http://localhost:8000/chat/debug" -ForegroundColor Green
