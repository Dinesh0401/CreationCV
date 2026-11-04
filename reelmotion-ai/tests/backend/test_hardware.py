"""Tests for hardware detection service and endpoints."""

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from backend.services.hardware_manager import (
    HardwareManager,
    HardwareProfile,
    HardwareTier,
    GPUInfo,
)


def test_hardware_manager_detection():
    """Test that real hardware detection returns valid metrics."""
    mgr = HardwareManager()
    profile = mgr.detect()

    assert isinstance(profile, HardwareProfile)
    # CPU must report at least 1 core
    assert profile.cpu.logical_cores > 0
    # RAM must report positive memory
    assert profile.ram.total_mb > 0
    # Tier must be a valid HardwareTier
    assert isinstance(profile.tier, HardwareTier)
    # Platform must be detected
    assert len(profile.platform) > 0


def test_hardware_endpoints(client: TestClient):
    """Test /api/hardware and /api/hardware/gpu endpoints."""
    # Full hardware profile
    res = client.get("/api/hardware")
    assert res.status_code == 200
    data = res.json()
    assert "gpu" in data
    assert "cpu" in data
    assert "ram" in data
    assert "tier" in data

    # GPU-only endpoint
    gpu_res = client.get("/api/hardware/gpu")
    assert gpu_res.status_code == 200
    gpu_data = gpu_res.json()
    assert "gpu" in gpu_data
    assert "detected" in gpu_data["gpu"]


def test_hardware_compatibility_endpoint(client: TestClient):
    """Test /api/hardware/compatibility endpoint."""
    res = client.get("/api/hardware/compatibility")
    assert res.status_code == 200
    data = res.json()
    assert "tier" in data
    assert "compatible_models" in data
    assert isinstance(data["compatible_models"], list)


def test_hardware_failure_fallback_no_gpu():
    """Failure test: When no GPU is detected, tier should fall back to CPU_ONLY gracefully."""
    with patch(
        "backend.services.hardware_manager._detect_gpu_nvidia_smi", return_value=None
    ), patch(
        "backend.services.hardware_manager._detect_gpu_gputil", return_value=None
    ), patch(
        "backend.services.hardware_manager._detect_gpu_wmi", return_value=None
    ):
        mgr = HardwareManager()
        profile = mgr.detect()

        assert profile.gpu.detected is False
        assert profile.gpu.name == "No GPU detected"
        assert profile.tier == HardwareTier.CPU_ONLY
