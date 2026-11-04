# ReelMotion AI - Test Script
$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Write-Host ""
Write-Host "========================================"
Write-Host "  ReelMotion AI - Test Suite            "
Write-Host "========================================"
Write-Host ""

$PythonExe = "python"
if (Test-Path $VenvPython) {
    $PythonExe = $VenvPython
}

$env:PYTHONPATH = $ProjectRoot

Write-Host "[Backend Tests] Running pytest..."
if ($PythonExe -eq "python") {
    python -m pytest tests/backend/ -v --tb=short
} else {
    & $PythonExe -m pytest tests/backend/ -v --tb=short
}
$backendResult = $LASTEXITCODE

Write-Host ""
Write-Host "========================================"
if ($backendResult -eq 0) {
    Write-Host "  ALL TESTS PASSED"
} else {
    Write-Host "  SOME TESTS FAILED"
}
Write-Host "========================================"
Write-Host ""

exit $backendResult
