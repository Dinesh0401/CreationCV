# =============================================================================
# ReelMotion AI — Test Script (Windows PowerShell)
# =============================================================================
# Runs all automated tests and reports results.
# Usage: .\scripts\test.ps1
# =============================================================================

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ReelMotion AI — Test Suite            " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check virtual environment
if (-not (Test-Path $VenvPython)) {
    Write-Host "ERROR: Virtual environment not found at .venv\" -ForegroundColor Red
    Write-Host "Run: python -m venv .venv && .venv\Scripts\pip.exe install -e '.[dev]'" -ForegroundColor Yellow
    exit 1
}

# Set PYTHONPATH
$env:PYTHONPATH = $ProjectRoot

# Run backend tests
Write-Host "[Backend Tests]" -ForegroundColor Yellow
Write-Host "Running pytest..." -ForegroundColor DarkGray
Write-Host ""

& $VenvPython -m pytest tests/backend/ -v --tb=short 2>&1 | ForEach-Object { Write-Host $_ }
$backendResult = $LASTEXITCODE

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan

if ($backendResult -eq 0) {
    Write-Host "  ALL TESTS PASSED ✓                    " -ForegroundColor Green
} else {
    Write-Host "  SOME TESTS FAILED ✗                   " -ForegroundColor Red
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

exit $backendResult

