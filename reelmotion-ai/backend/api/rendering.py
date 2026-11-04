"""Rendering API endpoints — Phase 1 stubs.

Provides render job queue management. Actual rendering logic
will be implemented in Phase 5+.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.services.job_manager import Job, JobManager, JobState, JobType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/render", tags=["rendering"])


class RenderStartRequest(BaseModel):
    """Request to start a render job."""

    character_id: str
    recording_id: str | None = None
    output_width: int = 1080
    output_height: int = 1920
    output_fps: int = 30
    output_format: str = "mp4"


class RenderStartResponse(BaseModel):
    """Response for starting a render."""

    job_id: str
    state: str
    message: str


class RenderQueueResponse(BaseModel):
    """Render queue listing."""

    jobs: list[Job]
    stats: dict[str, int]


@router.get("/queue", response_model=RenderQueueResponse)
async def get_render_queue(request: Request) -> RenderQueueResponse:
    """List all render jobs in the queue."""
    job_manager: JobManager = request.app.state.job_manager
    jobs = job_manager.list_jobs(job_type=JobType.RENDER)
    stats = job_manager.get_queue_stats()
    return RenderQueueResponse(jobs=jobs, stats=stats)


@router.post("/start", response_model=RenderStartResponse, status_code=202)
async def start_render(
    body: RenderStartRequest,
    request: Request,
) -> RenderStartResponse:
    """Start a render job (stub — actual rendering in Phase 5+).

    Creates a queued job. The render pipeline will pick it up when implemented.
    """
    job_manager: JobManager = request.app.state.job_manager

    job = job_manager.create_job(
        job_type=JobType.RENDER,
        metadata={
            "character_id": body.character_id,
            "recording_id": body.recording_id,
            "output_width": body.output_width,
            "output_height": body.output_height,
            "output_fps": body.output_fps,
            "output_format": body.output_format,
        },
    )

    logger.info(
        "Render job queued: %s (character=%s)", job.id, body.character_id
    )
    return RenderStartResponse(
        job_id=job.id,
        state=job.state.value,
        message="Render job queued. Rendering pipeline not yet implemented (Phase 5+).",
    )


@router.get("/{job_id}/status", response_model=Job)
async def get_render_status(
    job_id: str,
    request: Request,
) -> Job:
    """Get render job status."""
    job_manager: JobManager = request.app.state.job_manager
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Render job not found")
    return job


@router.delete("/{job_id}", status_code=200)
async def cancel_render(
    job_id: str,
    request: Request,
) -> dict[str, str]:
    """Cancel a render job."""
    job_manager: JobManager = request.app.state.job_manager
    job = job_manager.cancel_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Render job not found or cannot be cancelled",
        )
    return {"message": f"Render job {job_id} cancelled"}
