"""Rendering API endpoints for ReelMotion AI.

Provides full render job lifecycle management:
- Start render pipeline (POST /api/render/start)
- Queue inspection (GET /api/render/queue)
- Real-time progress and state querying (GET /api/render/{job_id}/status)
- Render cancellation (DELETE /api/render/{job_id})
- Exported video download (GET /api/render/{job_id}/download)
- Completed exports listing (GET /api/render/exports)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.services.job_manager import Job, JobManager, JobState, JobType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/render", tags=["rendering"])


class RenderStartRequest(BaseModel):
    """Request to start a render job."""

    character_id: str
    recording_id: Optional[str] = None
    output_width: int = 1080
    output_height: int = 1920
    output_fps: int = 30
    output_format: str = "mp4"
    aspect_ratio_mode: str = "fit_blur_pad"


class RenderStartResponse(BaseModel):
    """Response returned upon successfully queueing a render job."""

    job_id: str
    state: str
    message: str


class RenderQueueResponse(BaseModel):
    """Render queue listing with summary metrics."""

    jobs: list[Job]
    stats: dict[str, int]


class ExportItem(BaseModel):
    """Metadata for an exported video file."""

    filename: str
    file_size_mb: float
    created_at: str
    download_url: str


@router.get("/queue", response_model=RenderQueueResponse)
async def get_render_queue(request: Request) -> RenderQueueResponse:
    """List all render jobs currently in the queue with statistics."""
    job_manager: JobManager = request.app.state.job_manager
    jobs = job_manager.list_jobs(job_type=JobType.RENDER)
    stats = job_manager.get_queue_stats()
    return RenderQueueResponse(jobs=jobs, stats=stats)


@router.post("/start", response_model=RenderStartResponse, status_code=202)
async def start_render(
    body: RenderStartRequest,
    request: Request,
) -> RenderStartResponse:
    """Start an end-to-end character replacement and video rendering job.

    Queues the job and schedules asynchronous execution in the background.
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
            "aspect_ratio_mode": body.aspect_ratio_mode,
        },
    )

    # Schedule background execution
    job_manager.enqueue_render_job(job.id)

    logger.info("Render job queued and launched: %s (character=%s)", job.id, body.character_id)
    return RenderStartResponse(
        job_id=job.id,
        state=job.state.value,
        message="Render job accepted and processing started.",
    )


@router.get("/exports", response_model=list[ExportItem])
async def list_exports(request: Request) -> list[ExportItem]:
    """List all exported MP4 videos from the exports directory."""
    settings = request.app.state.settings
    exports_dir: Path = settings.resolved_paths().get("exports_dir")

    if not exports_dir or not exports_dir.exists():
        return []

    items: list[ExportItem] = []
    for f in sorted(exports_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = f.stat()
        items.append(
            ExportItem(
                filename=f.name,
                file_size_mb=round(stat.st_size / (1024 * 1024), 2),
                created_at=str(stat.st_mtime),
                download_url=f"/api/render/download/file/{f.name}",
            )
        )
    return items


@router.get("/{job_id}/status", response_model=Job)
async def get_render_status(
    job_id: str,
    request: Request,
) -> Job:
    """Get the current status, progress percentage, and metadata for a render job."""
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
    """Cancel a running or queued render job."""
    job_manager: JobManager = request.app.state.job_manager
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Render job not found")

    cancelled_job = job_manager.cancel_job(job_id)
    if cancelled_job is None:
        raise HTTPException(
            status_code=400,
            detail=f"Render job cannot be cancelled — already in terminal state '{job.state.value}'",
        )
    return {"message": f"Render job {job_id} cancelled successfully"}


@router.get("/{job_id}/download")
async def download_render_result(
    job_id: str,
    request: Request,
) -> FileResponse:
    """Download the completed 1080x1920 MP4 video for a completed render job."""
    job_manager: JobManager = request.app.state.job_manager
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Render job not found")

    if job.state == JobState.RUNNING or job.state == JobState.QUEUED:
        raise HTTPException(status_code=400, detail="Render job is still in progress")

    if job.state == JobState.FAILED:
        raise HTTPException(status_code=400, detail=f"Render job failed: {job.error}")

    if job.state == JobState.CANCELLED:
        raise HTTPException(status_code=400, detail="Render job was cancelled")

    export_path_str = job.metadata.get("export_path")
    if not export_path_str:
        raise HTTPException(status_code=404, detail="Export video path not recorded in job")

    export_path = Path(export_path_str)
    if not export_path.exists():
        raise HTTPException(status_code=404, detail=f"Exported video file not found at {export_path}")

    return FileResponse(
        path=str(export_path),
        media_type="video/mp4",
        filename=export_path.name,
    )


@router.get("/download/file/{filename}")
async def download_file_by_name(
    filename: str,
    request: Request,
) -> FileResponse:
    """Directly download an exported video by its filename."""
    settings = request.app.state.settings
    exports_dir: Path = settings.resolved_paths().get("exports_dir")

    file_path = exports_dir / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Exported file {filename} not found")

    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=filename,
    )
