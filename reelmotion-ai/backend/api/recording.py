"""Recording management API endpoints — Phase 1 stubs.

Recording creation is browser-side (Phase 2). These endpoints
provide listing, retrieval, and deletion of stored recordings.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/recordings", tags=["recordings"])


class RecordingInfo(BaseModel):
    """Recording metadata."""

    id: str
    name: str
    duration_seconds: float = 0.0
    resolution: list[int] = [1280, 720]
    fps: int = 30
    file_size_mb: float = 0.0
    created_at: str = ""
    format: str = "mp4"


def _get_recordings_dir(request: Request) -> Path:
    """Get the resolved recordings directory."""
    settings = request.app.state.settings
    paths = settings.resolved_paths()
    return paths["recordings_dir"]


def _load_recording_meta(rec_dir: Path) -> RecordingInfo | None:
    """Load recording metadata from a recording directory."""
    meta_path = rec_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return RecordingInfo(**data)
    except Exception as exc:
        logger.warning("Failed to load recording metadata from %s: %s", meta_path, exc)
        return None


@router.get("", response_model=list[RecordingInfo])
async def list_recordings(request: Request) -> list[RecordingInfo]:
    """List all recordings from the recordings directory."""
    recs_dir = _get_recordings_dir(request)
    if not recs_dir.exists():
        return []

    recordings: list[RecordingInfo] = []
    for entry in sorted(recs_dir.iterdir()):
        if entry.is_dir():
            info = _load_recording_meta(entry)
            if info:
                recordings.append(info)
    return recordings


@router.get("/{recording_id}", response_model=RecordingInfo)
async def get_recording(
    recording_id: str,
    request: Request,
) -> RecordingInfo:
    """Get recording info by ID."""
    recs_dir = _get_recordings_dir(request)
    rec_dir = recs_dir / recording_id

    if not rec_dir.exists():
        raise HTTPException(status_code=404, detail="Recording not found")

    info = _load_recording_meta(rec_dir)
    if info is None:
        raise HTTPException(status_code=404, detail="Recording metadata not found")
    return info


@router.delete("/{recording_id}", status_code=200)
async def delete_recording(
    recording_id: str,
    request: Request,
) -> dict[str, str]:
    """Delete a recording and all its files."""
    recs_dir = _get_recordings_dir(request)
    rec_dir = recs_dir / recording_id

    if not rec_dir.exists():
        raise HTTPException(status_code=404, detail="Recording not found")

    import shutil

    shutil.rmtree(rec_dir)
    logger.info("Recording deleted: %s", recording_id)
    return {"message": f"Recording {recording_id} deleted"}
