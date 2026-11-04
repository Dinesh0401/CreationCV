"""Health and hardware API endpoints.

Provides server health checks, hardware profile, and compatibility info.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from backend import __version__
from backend.services.hardware_manager import GPUInfo, HardwareProfile, HardwareManager

router = APIRouter(prefix="/api", tags=["health"])

# Startup timestamp set in main.py lifespan
_start_time: float = 0.0


def set_start_time(t: float) -> None:
    """Set the server start time (called during startup)."""
    global _start_time
    _start_time = t


class HealthResponse(BaseModel):
    """Server health response."""

    status: str
    version: str
    uptime_seconds: float
    debug: bool


class HardwareGPUResponse(BaseModel):
    """GPU-specific response."""

    gpu: GPUInfo


class CompatibilityResponse(BaseModel):
    """Hardware compatibility response."""

    tier: str
    compatible_models: list[dict[str, Any]]
    preset: dict[str, Any] | None


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Server health check — status, uptime, version."""
    settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        version=__version__,
        uptime_seconds=round(time.time() - _start_time, 2),
        debug=settings.server.debug,
    )


@router.get("/hardware", response_model=HardwareProfile)
async def get_hardware(request: Request) -> HardwareProfile:
    """Full hardware profile from real detection."""
    hw_manager: HardwareManager = request.app.state.hardware_manager
    return hw_manager.profile


@router.get("/hardware/gpu", response_model=HardwareGPUResponse)
async def get_gpu_info(request: Request) -> HardwareGPUResponse:
    """GPU-specific hardware information."""
    hw_manager: HardwareManager = request.app.state.hardware_manager
    return HardwareGPUResponse(gpu=hw_manager.profile.gpu)


@router.get("/hardware/compatibility", response_model=CompatibilityResponse)
async def get_compatibility(request: Request) -> CompatibilityResponse:
    """Check model compatibility with detected hardware."""
    hw_manager: HardwareManager = request.app.state.hardware_manager
    model_manager = request.app.state.model_manager

    profile = hw_manager.profile
    compat_list = model_manager.get_compatible_models(profile)
    preset = model_manager.get_preset(profile.tier)

    return CompatibilityResponse(
        tier=profile.tier.value,
        compatible_models=[c.model_dump() for c in compat_list],
        preset=preset.model_dump() if preset else None,
    )
