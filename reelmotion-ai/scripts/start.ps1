# =============================================================================
# ReelMotion AI — Start Script (Windows PowerShell)
# =============================================================================
# Starts both the backend server and frontend dev server.
# Usage: .\scripts\start.ps1
# =============================================================================

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ReelMotion AI — Starting Application  " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# --- Check prerequisites ---
Write-Host "[1/5] Checking prerequisites..." -ForegroundColor Yellow

# Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Python not found. Install Python 3.11+ from python.org" -ForegroundColor Red
    exit 1
}

# Node
try {
    $nodeVersion = node --version 2>&1
    Write-Host "  Node.js: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Node.js not found. Install from nodejs.org" -ForegroundColor Red
    exit 1
}

# Virtual environment
$VenvPath = Join-Path $ProjectRoot ".venv"
if (-not (Test-Path (Join-Path $VenvPath "Scripts\python.exe"))) {
    Write-Host "  Virtual environment not found. Creating..." -ForegroundColor Yellow
    python -m venv $VenvPath
    & "$VenvPath\Scripts\pip.exe" install -e "$ProjectRoot[dev]" 2>&1 | Out-Null
    Write-Host "  Virtual environment created and dependencies installed." -ForegroundColor Green
} else {
    Write-Host "  Virtual environment: OK" -ForegroundColor Green
}

# Frontend node_modules
$NodeModules = Join-Path $ProjectRoot "frontend\node_modules"
if (-not (Test-Path $NodeModules)) {
    Write-Host "  Frontend dependencies not installed. Running npm install..." -ForegroundColor Yellow
    Push-Location (Join-Path $ProjectRoot "frontend")
    npm install 2>&1 | Out-Null
    Pop-Location
    Write-Host "  Frontend dependencies installed." -ForegroundColor Green
} else {
    Write-Host "  Frontend dependencies: OK" -ForegroundColor Green
}

# --- Create data directories ---
Write-Host ""
Write-Host "[2/5] Creating data directories..." -ForegroundColor Yellow
$dataDirs = @("models", "characters", "recordings", "exports", "logs")
foreach ($dir in $dataDirs) {
    $dirPath = Join-Path $ProjectRoot $dir
    if (-not (Test-Path $dirPath)) {
        New-Item -ItemType Directory -Force -Path $dirPath | Out-Null
    }
}
Write-Host "  Data directories: OK" -ForegroundColor Green

# --- Copy .env if not exists ---
Write-Host ""
Write-Host "[3/5] Checking configuration..." -ForegroundColor Yellow
$envFile = Join-Path $ProjectRoot ".env"
$envExample = Join-Path $ProjectRoot ".env.example"
if (-not (Test-Path $envFile) -and (Test-Path $envExample)) {
    Copy-Item $envExample $envFile
    Write-Host "  Created .env from .env.example" -ForegroundColor Green
} else {
    Write-Host "  Configuration: OK" -ForegroundColor Green
}

# --- Start backend ---
Write-Host ""
Write-Host "[4/5] Starting backend server..." -ForegroundColor Yellow
$backendDir = Join-Path $ProjectRoot "backend"
$backendJob = Start-Job -ScriptBlock {
    param($venv, $backendDir, $projectRoot)
    Set-Location $projectRoot
    $env:PYTHONPATH = $projectRoot
    & "$venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload 2>&1
} -ArgumentList $VenvPath, $backendDir, $ProjectRoot
Write-Host "  Backend starting on http://127.0.0.1:8000" -ForegroundColor Green

# --- Start frontend ---
Write-Host ""
Write-Host "[5/5] Starting frontend dev server..." -ForegroundColor Yellow
$frontendDir = Join-Path $ProjectRoot "frontend"
$frontendJob = Start-Job -ScriptBlock {
    param($frontendDir)
    Set-Location $frontendDir
    npm run dev 2>&1
} -ArgumentList $frontendDir
Write-Host "  Frontend starting on http://localhost:5173" -ForegroundColor Green

# --- Summary ---
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ReelMotion AI is starting!            " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Backend:  http://127.0.0.1:8000" -ForegroundColor White
Write-Host "  Frontend: http://localhost:5173" -ForegroundColor White
Write-Host "  API docs: http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "  Press Ctrl+C to stop both servers." -ForegroundColor DarkGray
Write-Host ""

# Wait for either job to finish or user interrupt
try {
    while ($true) {
        # Check backend job
        if ($backendJob.State -eq "Failed") {
            Write-Host "Backend server failed!" -ForegroundColor Red
            Receive-Job $backendJob
            break
        }
        # Check frontend job
        if ($frontendJob.State -eq "Failed") {
            Write-Host "Frontend server failed!" -ForegroundColor Red
            Receive-Job $frontendJob
            break
        }
        Start-Sleep -Seconds 2
    }
} finally {
    Write-Host ""
    Write-Host "Stopping servers..." -ForegroundColor Yellow
    Stop-Job $backendJob -ErrorAction SilentlyContinue
    Stop-Job $frontendJob -ErrorAction SilentlyContinue
    Remove-Job $backendJob -ErrorAction SilentlyContinue
    Remove-Job $frontendJob -ErrorAction SilentlyContinue
    Write-Host "Servers stopped." -ForegroundColor Green
}

