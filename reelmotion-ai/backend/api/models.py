"""Model management API endpoints.

Provides listing, details, compatibility, and status for AI models
declared in the model manifest.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.services.hardware_manager import HardwareManager
from backend.services.model_manager import (
    ModelCompatibility,
    ModelInfo,
    ModelManager,
    ModelStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])


class ModelStatusResponse(BaseModel):
    """Model status response."""

    model_id: str
    status: str
    on_disk: bool


class ModelListResponse(BaseModel):
    """Model list response."""

    models: list[ModelInfo]
    total: int


class CompatibleModelsResponse(BaseModel):
    """Compatible models response."""

    tier: str
    models: list[ModelCompatibility]


@router.get("", response_model=ModelListResponse)
async def list_models(request: Request) -> ModelListResponse:
    """List all models from the manifest."""
    model_manager: ModelManager = request.app.state.model_manager
    models = model_manager.list_models()
    return ModelListResponse(models=models, total=len(models))


@router.get("/compatible", response_model=CompatibleModelsResponse)
async def get_compatible_models(request: Request) -> CompatibleModelsResponse:
    """Get models compatible with current hardware."""
    hw_manager: HardwareManager = request.app.state.hardware_manager
    model_manager: ModelManager = request.app.state.model_manager

    profile = hw_manager.profile
    compat = model_manager.get_compatible_models(profile)

    return CompatibleModelsResponse(
        tier=profile.tier.value,
        models=compat,
    )


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(
    model_id: str,
    request: Request,
) -> ModelInfo:
    """Get model details by ID."""
    model_manager: ModelManager = request.app.state.model_manager
    model = model_manager.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return model


@router.get("/{model_id}/status", response_model=ModelStatusResponse)
async def get_model_status(
    model_id: str,
    request: Request,
) -> ModelStatusResponse:
    """Check the on-disk status of a model (downloaded, corrupted, etc.)."""
    model_manager: ModelManager = request.app.state.model_manager
    model = model_manager.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    status = model_manager.check_model_status(model_id)
    return ModelStatusResponse(
        model_id=model_id,
        status=status.value,
        on_disk=status == ModelStatus.READY,
    )
