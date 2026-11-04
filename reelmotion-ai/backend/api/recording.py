"""Recording management API endpoints — Phase 1 stubs.

Recording creation is browser-side (Phase 2). These endpoints
provide listing, retrieval, and deletion of stored recordings.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
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


@router.post("/upload", response_model=RecordingInfo, status_code=201)
async def upload_recording(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form("Untitled Recording"),
    duration: float = Form(0.0),
    fps: int = Form(30),
    resolution: str = Form("1280x720"),
) -> RecordingInfo:
    """Upload recorded video blob with metadata."""
    recs_dir = _get_recordings_dir(request)
    recs_dir.mkdir(parents=True, exist_ok=True)

    rec_id = f"rec_{uuid.uuid4().hex[:10]}"
    rec_dir = recs_dir / rec_id
    rec_dir.mkdir(parents=True, exist_ok=True)

    # Determine file extension from uploaded file or default to mp4/webm
    ext = Path(file.filename or "recording.mp4").suffix or ".mp4"
    if ext not in [".mp4", ".webm", ".mov", ".mkv"]:
        ext = ".mp4"
    target_filename = f"video{ext}"
    target_path = rec_dir / target_filename

    content = await file.read()
    with open(target_path, "wb") as f:
        f.write(content)

    file_size_mb = round(len(content) / (1024 * 1024), 2)

    # Parse resolution (e.g., "720x1280" or "1080x1920")
    res_list = [1280, 720]
    if "x" in resolution:
        try:
            parts = [int(p.strip()) for p in resolution.split("x")]
            if len(parts) == 2:
                res_list = parts
        except Exception:
            pass

    info = RecordingInfo(
        id=rec_id,
        name=title,
        duration_seconds=duration,
        resolution=res_list,
        fps=fps,
        file_size_mb=file_size_mb,
        created_at=datetime.now(timezone.utc).isoformat(),
        format=ext.lstrip("."),
    )

    with open(rec_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(info.model_dump(), f, indent=2)

    logger.info("Recording uploaded successfully: %s (%s, %.2fs, %.2fMB)", rec_id, title, duration, file_size_mb)
    return info


@router.get("/{recording_id}/file")
async def get_recording_file(
    recording_id: str,
    request: Request,
):
    """Download/stream raw recording video file."""
    recs_dir = _get_recordings_dir(request)
    rec_dir = recs_dir / recording_id
    if not rec_dir.exists():
        raise HTTPException(status_code=404, detail="Recording not found")

    for candidate in [rec_dir / "video.mp4", rec_dir / "video.webm", rec_dir / "recording.mp4"]:
        if candidate.exists():
            return FileResponse(candidate, media_type="video/mp4", filename=f"{recording_id}.mp4")

    # Check any video file in directory
    for f in rec_dir.iterdir():
        if f.suffix in [".mp4", ".webm", ".mov"]:
            return FileResponse(f, media_type="video/mp4", filename=f.name)

    raise HTTPException(status_code=404, detail="Recording video file not found")


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
