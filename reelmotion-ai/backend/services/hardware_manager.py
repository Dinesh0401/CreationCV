"""Hardware detection and monitoring for ReelMotion AI.

Detects GPU (NVIDIA via GPUtil/nvidia-smi/WMI), CPU, RAM, CUDA, FFmpeg,
and Python version — all via real system introspection, never hardcoded.
"""

from __future__ import annotations

import hashlib
import logging
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class HardwareTier(str, Enum):
    """Hardware capability tier based on available VRAM."""

    CPU_ONLY = "cpu_only"
    LOW_VRAM = "low_vram"      # 4-6 GB
    MID_VRAM = "mid_vram"      # 8-12 GB
    HIGH_VRAM = "high_vram"    # 16+ GB


class GPUInfo(BaseModel):
    """GPU detection result."""

    detected: bool = False
    name: str = "No GPU detected"
    vram_total_mb: int = 0
    vram_used_mb: int = 0
    vram_free_mb: int = 0
    driver_version: str = ""
    detection_method: str = "none"


class CPUInfo(BaseModel):
    """CPU detection result."""

    name: str = "Unknown"
    physical_cores: int = 0
    logical_cores: int = 0
    architecture: str = ""
    max_frequency_mhz: float = 0.0


class RAMInfo(BaseModel):
    """RAM detection result."""

    total_mb: int = 0
    used_mb: int = 0
    available_mb: int = 0
    percent_used: float = 0.0


class SoftwareInfo(BaseModel):
    """Software availability detection."""

    python_version: str = ""
    cuda_available: bool = False
    cuda_version: str = ""
    torch_installed: bool = False
    torch_version: str = ""
    ffmpeg_available: bool = False
    ffmpeg_version: str = ""


class HardwareProfile(BaseModel):
    """Complete hardware profile for the system."""

    gpu: GPUInfo = Field(default_factory=GPUInfo)
    cpu: CPUInfo = Field(default_factory=CPUInfo)
    ram: RAMInfo = Field(default_factory=RAMInfo)
    software: SoftwareInfo = Field(default_factory=SoftwareInfo)
    tier: HardwareTier = HardwareTier.CPU_ONLY
    detected_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    platform: str = Field(default_factory=lambda: platform.system())


def _detect_gpu_gputil() -> GPUInfo | None:
    """Try GPU detection via GPUtil library."""
    try:
        import GPUtil  # type: ignore[import-untyped]

        gpus = GPUtil.getGPUs()
        if not gpus:
            return None
        gpu = gpus[0]  # Primary GPU
        return GPUInfo(
            detected=True,
            name=gpu.name,
            vram_total_mb=int(gpu.memoryTotal),
            vram_used_mb=int(gpu.memoryUsed),
            vram_free_mb=int(gpu.memoryFree),
            driver_version=str(gpu.driver) if hasattr(gpu, "driver") else "",
            detection_method="GPUtil",
        )
    except ImportError:
        logger.debug("GPUtil not installed, trying fallback")
        return None
    except Exception as exc:
        logger.debug("GPUtil detection failed: %s", exc)
        return None


def _detect_gpu_nvidia_smi() -> GPUInfo | None:
    """Try GPU detection via nvidia-smi subprocess."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        if result.returncode != 0:
            return None

        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            return None

        return GPUInfo(
            detected=True,
            name=parts[0],
            vram_total_mb=int(float(parts[1])),
            vram_used_mb=int(float(parts[2])),
            vram_free_mb=int(float(parts[3])),
            driver_version=parts[4] if len(parts) > 4 else "",
            detection_method="nvidia-smi",
        )
    except FileNotFoundError:
        logger.debug("nvidia-smi not found on PATH")
        return None
    except Exception as exc:
        logger.debug("nvidia-smi detection failed: %s", exc)
        return None


def _detect_gpu_wmi() -> GPUInfo | None:
    """Try GPU detection via WMI (Windows only) using PowerShell."""
    if sys.platform != "win32":
        return None
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "Get-CimInstance Win32_VideoController | "
                    "Select-Object -First 1 Name, AdapterRAM, DriverVersion | "
                    "ConvertTo-Json"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode != 0:
            return None

        import json

        data = json.loads(result.stdout.strip())
        # Handle list or single object
        if isinstance(data, list):
            data = data[0] if data else {}

        name = data.get("Name", "Unknown GPU")
        adapter_ram = data.get("AdapterRAM", 0)
        driver_version = data.get("DriverVersion", "")

        # AdapterRAM is in bytes; convert to MB
        # WMI caps at 4GB for AdapterRAM on some drivers (uint32 limit)
        vram_total_mb = int(adapter_ram) // (1024 * 1024) if adapter_ram else 0

        # WMI doesn't provide used/free — only total
        return GPUInfo(
            detected=True if vram_total_mb > 0 else False,
            name=name,
            vram_total_mb=vram_total_mb,
            vram_used_mb=0,
            vram_free_mb=vram_total_mb,
            driver_version=str(driver_version),
            detection_method="WMI",
        )
    except Exception as exc:
        logger.debug("WMI GPU detection failed: %s", exc)
        return None


def detect_gpu() -> GPUInfo:
    """Detect GPU using multiple fallback strategies.

    Order: GPUtil → nvidia-smi → WMI → No GPU
    """
    for detector in (_detect_gpu_gputil, _detect_gpu_nvidia_smi, _detect_gpu_wmi):
        result = detector()
        if result is not None and result.detected:
            logger.info("GPU detected via %s: %s (%d MB VRAM)",
                        result.detection_method, result.name, result.vram_total_mb)
            return result

    logger.info("No GPU detected by any method")
    return GPUInfo()


def detect_cpu() -> CPUInfo:
    """Detect CPU information via psutil and platform."""
    info = CPUInfo(
        architecture=platform.machine(),
    )

    # CPU name — platform.processor() is often empty on Windows
    processor = platform.processor()
    if not processor or processor == "":
        # Try to get a better name on Windows
        if sys.platform == "win32":
            try:
                result = subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-Command",
                        "(Get-CimInstance Win32_Processor).Name",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                if result.returncode == 0 and result.stdout.strip():
                    processor = result.stdout.strip()
            except Exception:
                pass
    info.name = processor or platform.processor() or "Unknown CPU"

    try:
        import psutil

        info.physical_cores = psutil.cpu_count(logical=False) or 0
        info.logical_cores = psutil.cpu_count(logical=True) or 0
        freq = psutil.cpu_freq()
        if freq:
            info.max_frequency_mhz = freq.max or freq.current or 0.0
    except ImportError:
        logger.warning("psutil not installed — CPU core count unavailable")
        # Fallback to os.cpu_count()
        import os
        info.logical_cores = os.cpu_count() or 0
    except Exception as exc:
        logger.warning("CPU detection error: %s", exc)

    return info


def detect_ram() -> RAMInfo:
    """Detect RAM information via psutil."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return RAMInfo(
            total_mb=int(vm.total / (1024 * 1024)),
            used_mb=int(vm.used / (1024 * 1024)),
            available_mb=int(vm.available / (1024 * 1024)),
            percent_used=vm.percent,
        )
    except ImportError:
        logger.warning("psutil not installed — RAM detection unavailable")
        return RAMInfo()
    except Exception as exc:
        logger.warning("RAM detection error: %s", exc)
        return RAMInfo()


def detect_software() -> SoftwareInfo:
    """Detect software availability (Python, CUDA, torch, FFmpeg)."""
    info = SoftwareInfo(
        python_version=platform.python_version(),
    )

    # CUDA / PyTorch
    try:
        import torch

        info.torch_installed = True
        info.torch_version = torch.__version__
        info.cuda_available = torch.cuda.is_available()
        if info.cuda_available:
            info.cuda_version = torch.version.cuda or ""
    except ImportError:
        logger.debug("PyTorch not installed — CUDA detection skipped")
    except Exception as exc:
        logger.debug("PyTorch/CUDA detection error: %s", exc)

    # FFmpeg
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        if result.returncode == 0:
            info.ffmpeg_available = True
            # Extract version from first line, e.g. "ffmpeg version 6.1.1 ..."
            first_line = result.stdout.split("\n")[0]
            match = re.search(r"version\s+([\S]+)", first_line)
            info.ffmpeg_version = match.group(1) if match else first_line.strip()
    except FileNotFoundError:
        logger.debug("FFmpeg not found on PATH")
    except Exception as exc:
        logger.debug("FFmpeg detection error: %s", exc)

    return info


def determine_tier(gpu: GPUInfo, force_cpu: bool = False) -> HardwareTier:
    """Determine hardware tier based on GPU VRAM.

    Args:
        gpu: Detected GPU information.
        force_cpu: If True, always return CPU_ONLY.

    Returns:
        Hardware tier classification.
    """
    if force_cpu or not gpu.detected:
        return HardwareTier.CPU_ONLY

    vram = gpu.vram_total_mb
    if vram >= 16384:  # 16 GB+
        return HardwareTier.HIGH_VRAM
    elif vram >= 8192:  # 8-16 GB
        return HardwareTier.MID_VRAM
    elif vram >= 4096:  # 4-8 GB
        return HardwareTier.LOW_VRAM
    else:
        return HardwareTier.CPU_ONLY


class HardwareManager:
    """Manages hardware detection and periodic refresh.

    Provides a cached hardware profile that can be refreshed
    on demand or periodically (for the monitoring WebSocket).
    """

    def __init__(self, force_cpu: bool = False) -> None:
        self._force_cpu = force_cpu
        self._profile: HardwareProfile | None = None

    def detect(self) -> HardwareProfile:
        """Run full hardware detection and cache the result."""
        gpu = detect_gpu()
        cpu = detect_cpu()
        ram = detect_ram()
        software = detect_software()
        tier = determine_tier(gpu, self._force_cpu)

        self._profile = HardwareProfile(
            gpu=gpu,
            cpu=cpu,
            ram=ram,
            software=software,
            tier=tier,
        )

        logger.info(
            "Hardware profile: tier=%s, GPU=%s (%d MB), CPU=%s (%d cores), RAM=%d MB",
            tier.value,
            gpu.name,
            gpu.vram_total_mb,
            cpu.name,
            cpu.logical_cores,
            ram.total_mb,
        )
        return self._profile

    def refresh(self) -> HardwareProfile:
        """Re-detect hardware (for periodic monitoring)."""
        return self.detect()

    @property
    def profile(self) -> HardwareProfile:
        """Get the current hardware profile, detecting if not yet done."""
        if self._profile is None:
            self.detect()
        assert self._profile is not None
        return self._profile

    def get_live_stats(self) -> dict[str, Any]:
        """Get lightweight live stats for WebSocket telemetry.

        Only refreshes RAM and GPU memory (fast), not full detection.
        """
        stats: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Quick RAM refresh
        try:
            import psutil

            vm = psutil.virtual_memory()
            stats["ram"] = {
                "used_mb": int(vm.used / (1024 * 1024)),
                "available_mb": int(vm.available / (1024 * 1024)),
                "percent": vm.percent,
            }
            stats["cpu_percent"] = psutil.cpu_percent(interval=0)
        except ImportError:
            stats["ram"] = {"error": "psutil not installed"}
            stats["cpu_percent"] = 0.0
        except Exception as exc:
            stats["ram"] = {"error": str(exc)}
            stats["cpu_percent"] = 0.0

        # Quick GPU refresh — only via nvidia-smi for speed
        gpu_info = _detect_gpu_nvidia_smi() or _detect_gpu_gputil()
        if gpu_info and gpu_info.detected:
            stats["gpu"] = {
                "vram_used_mb": gpu_info.vram_used_mb,
                "vram_free_mb": gpu_info.vram_free_mb,
                "vram_total_mb": gpu_info.vram_total_mb,
            }
        else:
            stats["gpu"] = {"detected": False}

        return stats
