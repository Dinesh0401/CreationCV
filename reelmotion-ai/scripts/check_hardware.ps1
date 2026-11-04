# =============================================================================
# ReelMotion AI — Hardware Check Script
# =============================================================================
# Detects and reports all hardware relevant to ReelMotion AI.
# Usage: powershell -ExecutionPolicy Bypass -File .\scripts\check_hardware.ps1
# =============================================================================

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ReelMotion AI — Hardware Report       " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# --- GPU via WMI ---
Write-Host "[GPU (WMI)]" -ForegroundColor Yellow
try {
    $gpuList = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "NVIDIA|AMD|Intel" }
    if ($gpuList) {
        foreach ($g in $gpuList) {
            Write-Host "  Name:        $($g.Name)" -ForegroundColor White
            $vramGB = [math]::Round($g.AdapterRAM / 1GB, 2)
            Write-Host "  VRAM (WMI):  $vramGB GB" -ForegroundColor White
            Write-Host "  Driver:      $($g.DriverVersion)" -ForegroundColor White
        }
    } else {
        Write-Host "  No dedicated GPU detected via WMI" -ForegroundColor Red
    }
} catch {
    Write-Host "  GPU detection failed: $_" -ForegroundColor Red
}

# --- nvidia-smi ---
Write-Host ""
Write-Host "[NVIDIA SMI]" -ForegroundColor Yellow
try {
    $smiOut = & nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version --format=csv,noheader,nounits 2>&1
    if ($LASTEXITCODE -eq 0 -and $smiOut) {
        $parts = $smiOut -split ","
        if ($parts.Count -ge 5) {
            Write-Host "  Name:        $($parts[0].Trim())" -ForegroundColor White
            Write-Host "  VRAM Total:  $($parts[1].Trim()) MB" -ForegroundColor White
            Write-Host "  VRAM Used:   $($parts[2].Trim()) MB" -ForegroundColor White
            Write-Host "  VRAM Free:   $($parts[3].Trim()) MB" -ForegroundColor White
            Write-Host "  Driver:      $($parts[4].Trim())" -ForegroundColor White
        }
    } else {
        Write-Host "  nvidia-smi not on PATH or not responding" -ForegroundColor DarkGray
    }
} catch {
    Write-Host "  nvidia-smi check skipped" -ForegroundColor DarkGray
}

# --- CPU ---
Write-Host ""
Write-Host "[CPU]" -ForegroundColor Yellow
try {
    $cpu = Get-CimInstance Win32_Processor
    Write-Host "  Name:        $($cpu.Name.Trim())" -ForegroundColor White
    Write-Host "  Cores:       $($cpu.NumberOfCores)" -ForegroundColor White
    Write-Host "  Threads:     $($cpu.NumberOfLogicalProcessors)" -ForegroundColor White
} catch {
    Write-Host "  CPU detection failed: $_" -ForegroundColor Red
}

# --- RAM ---
Write-Host ""
Write-Host "[RAM]" -ForegroundColor Yellow
try {
    $os = Get-CimInstance Win32_OperatingSystem
    $totalGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
    $freeGB = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
    $usedGB = [math]::Round($totalGB - $freeGB, 2)
    Write-Host "  Total:       $totalGB GB" -ForegroundColor White
    Write-Host "  Used:        $usedGB GB" -ForegroundColor White
    Write-Host "  Available:   $freeGB GB" -ForegroundColor White
} catch {
    Write-Host "  RAM detection failed: $_" -ForegroundColor Red
}

# --- OS ---
Write-Host ""
Write-Host "[OS]" -ForegroundColor Yellow
try {
    $osInfo = Get-CimInstance Win32_OperatingSystem
    Write-Host "  Caption:     $($osInfo.Caption)" -ForegroundColor White
    Write-Host "  Version:     $($osInfo.Version)" -ForegroundColor White
} catch {
    Write-Host "  OS detection failed" -ForegroundColor Red
}

# --- Python ---
Write-Host ""
Write-Host "[Python]" -ForegroundColor Yellow
try {
    $pyVer = python --version 2>&1
    Write-Host "  Version:     $pyVer" -ForegroundColor White
} catch {
    Write-Host "  Python not found" -ForegroundColor Red
}

# --- PyTorch & CUDA via Python ---
Write-Host ""
Write-Host "[PyTorch / CUDA]" -ForegroundColor Yellow
$pyScript = @'
try:
    import torch
    print("  PyTorch:     " + torch.__version__)
    cuda_ok = torch.cuda.is_available()
    print("  CUDA:        " + ("Available" if cuda_ok else "Not Available (CPU Build)"))
    if cuda_ok:
        print("  CUDA Vers:   " + str(torch.version.cuda))
        print("  Device:      " + torch.cuda.get_device_name(0))
except ImportError:
    print("  PyTorch:     Not installed")
except Exception as e:
    print("  PyTorch Err: " + str(e))
'@
python -c $pyScript

# --- Node.js & npm ---
Write-Host ""
Write-Host "[Node.js]" -ForegroundColor Yellow
try {
    $nodeVer = node --version 2>&1
    $npmVer = npm --version 2>&1
    Write-Host "  Node.js:     $nodeVer" -ForegroundColor White
    Write-Host "  npm:         $npmVer" -ForegroundColor White
} catch {
    Write-Host "  Node.js:     NOT INSTALLED" -ForegroundColor Red
}

# --- FFmpeg ---
Write-Host ""
Write-Host "[FFmpeg]" -ForegroundColor Yellow
try {
    $ffVer = ffmpeg -version 2>&1 | Select-Object -First 1
    if ($LASTEXITCODE -eq 0 -and $ffVer) {
        Write-Host "  $ffVer" -ForegroundColor White
    } else {
        Write-Host "  FFmpeg:      NOT INSTALLED (Needed for Phase 7 Video Export)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  FFmpeg:      NOT INSTALLED (Needed for Phase 7 Video Export)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Hardware Check Complete               " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
