"""Character management API endpoints — Phase 4.

Provides full REST endpoints for:
- Character profile creation and management
- Multi-view reference image uploads and automatic quality validation
- Character readiness assessment for AI generation
- Image serving and reference cleanup
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.character.character_profile import (
    VIEW_TYPES,
    CharacterProfile,
    CharacterStorageManager,
    CharacterViews,
)
from backend.character.identity_validation import (
    CharacterReadinessReport,
    ImageQualityReport,
)
from backend.character.reference_manager import (
    SUPPORTED_EXTENSIONS,
    CharacterReferenceManager,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/characters", tags=["characters"])


class CharacterCreateRequest(BaseModel):
    """Request payload for creating a character profile."""

    name: str = Field(..., min_length=1, max_length=100)
    tags: list[str] = Field(default_factory=list)


class CharacterListItem(BaseModel):
    """Summary item for characters list view."""

    id: str
    name: str
    created_at: str
    updated_at: str | None = None
    tags: list[str] = Field(default_factory=list)
    reference_count: int = 0
    thumbnail: str | None = None
    thumbnail_url: str | None = None
    readiness_score: float = 0.0
    is_ready: bool = False
    single_image_mode: bool = False


class CharacterDetailResponse(BaseModel):
    """Full character profile with validation report and view URLs."""

    id: str
    name: str
    created_at: str
    updated_at: str | None = None
    tags: list[str] = Field(default_factory=list)
    views: CharacterViews
    view_urls: dict[str, str] = Field(default_factory=dict)
    thumbnail: str | None = None
    thumbnail_url: str | None = None
    reference_count: int = 0
    readiness: CharacterReadinessReport


class ReferenceUploadResponse(BaseModel):
    """Response returned upon uploading a reference image."""

    character_id: str
    view_type: str
    filename: str
    image_url: str
    quality: ImageQualityReport
    readiness: CharacterReadinessReport


def _get_managers(request: Request) -> tuple[CharacterStorageManager, CharacterReferenceManager]:
    """Retrieve or construct CharacterStorageManager and CharacterReferenceManager."""
    try:
        settings = request.app.state.settings
        chars_dir = settings.resolved_paths()["characters_dir"]
    except Exception:
        from backend.core.config import get_settings
        chars_dir = get_settings().resolved_paths()["characters_dir"]

    storage = CharacterStorageManager(chars_dir)
    ref_mgr = CharacterReferenceManager(storage)
    return storage, ref_mgr


def _build_image_url(character_id: str, filename: str) -> str:
    """Build public image URL for a character's reference image."""
    return f"/api/characters/{character_id}/images/{filename}"


@router.get("", response_model=list[CharacterListItem])
async def list_characters(request: Request) -> list[CharacterListItem]:
    """List all character profiles with thumbnails and AI readiness metrics."""
    storage, ref_mgr = _get_managers(request)
    profiles = storage.list_profiles()

    items: list[CharacterListItem] = []
    for profile in profiles:
        char_dir = storage.get_character_dir(profile.id)
        readiness = ref_mgr.get_readiness_report(profile.id)

        thumb_url = None
        if profile.thumbnail:
            thumb_url = _build_image_url(profile.id, profile.thumbnail)

        readiness_score = readiness.readiness_score if readiness else 0.0
        is_ready = readiness.is_ready if readiness else False
        single_image_mode = readiness.single_image_mode if readiness else False

        items.append(
            CharacterListItem(
                id=profile.id,
                name=profile.name,
                created_at=profile.created_at,
                updated_at=profile.updated_at,
                tags=profile.tags,
                reference_count=profile.reference_count,
                thumbnail=profile.thumbnail,
                thumbnail_url=thumb_url,
                readiness_score=readiness_score,
                is_ready=is_ready,
                single_image_mode=single_image_mode,
            )
        )

    return items


@router.post("", response_model=CharacterDetailResponse, status_code=status.HTTP_201_CREATED)
async def create_character(
    request: Request,
    body: CharacterCreateRequest,
) -> CharacterDetailResponse:
    """Create a new character profile."""
    storage, ref_mgr = _get_managers(request)

    profile = storage.create_profile(
        name=body.name.strip(),
        tags=body.tags,
    )
    logger.info("Created character profile: %s (%s)", profile.name, profile.id)

    readiness = ref_mgr.get_readiness_report(profile.id)
    if readiness is None:
        raise HTTPException(status_code=500, detail="Failed to initialize readiness report.")

    return CharacterDetailResponse(
        id=profile.id,
        name=profile.name,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
        tags=profile.tags,
        views=profile.views,
        view_urls={},
        thumbnail=None,
        thumbnail_url=None,
        reference_count=0,
        readiness=readiness,
    )


@router.get("/{character_id}", response_model=CharacterDetailResponse)
async def get_character(
    character_id: str,
    request: Request,
) -> CharacterDetailResponse:
    """Get character details and validation readiness report."""
    storage, ref_mgr = _get_managers(request)
    profile = storage.load_profile(character_id)

    if profile is None:
        raise HTTPException(status_code=404, detail="Character not found")

    readiness = ref_mgr.get_readiness_report(character_id)
    if readiness is None:
        raise HTTPException(status_code=500, detail="Failed to generate readiness report")

    view_urls: dict[str, str] = {}
    for vt, fn in profile.views.active_views().items():
        view_urls[vt] = _build_image_url(character_id, fn)

    thumb_url = None
    if profile.thumbnail:
        thumb_url = _build_image_url(character_id, profile.thumbnail)

    return CharacterDetailResponse(
        id=profile.id,
        name=profile.name,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
        tags=profile.tags,
        views=profile.views,
        view_urls=view_urls,
        thumbnail=profile.thumbnail,
        thumbnail_url=thumb_url,
        reference_count=profile.reference_count,
        readiness=readiness,
    )


@router.post("/{character_id}/upload-reference", response_model=ReferenceUploadResponse)
async def upload_reference(
    character_id: str,
    request: Request,
    file: UploadFile = File(...),
    view_type: str = Form("front"),
) -> ReferenceUploadResponse:
    """Upload a reference image for a specific view angle and run validation."""
    storage, ref_mgr = _get_managers(request)

    profile = storage.load_profile(character_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Character not found")

    view_type_clean = view_type.strip().lower()
    if view_type_clean not in VIEW_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid view_type '{view_type}'. Allowed types: {', '.join(VIEW_TYPES)}",
        )

    filename = file.filename or f"{view_type_clean}.png"
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image extension '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        stored_filename, path, quality_report = ref_mgr.save_reference(
            character_id=character_id,
            content=content,
            original_filename=filename,
            view_type=view_type_clean,
        )

        readiness = ref_mgr.get_readiness_report(character_id)
        if readiness is None:
            raise HTTPException(status_code=500, detail="Could not compute readiness report.")

        image_url = _build_image_url(character_id, stored_filename)

        return ReferenceUploadResponse(
            character_id=character_id,
            view_type=view_type_clean,
            filename=stored_filename,
            image_url=image_url,
            quality=quality_report,
            readiness=readiness,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to upload reference image for character %s: %s", character_id, exc)
        raise HTTPException(status_code=500, detail=f"Image upload failed: {exc}")


@router.get("/{character_id}/images/{filename}")
async def get_character_image(
    character_id: str,
    filename: str,
    request: Request,
) -> FileResponse:
    """Serve a character reference image."""
    storage, ref_mgr = _get_managers(request)
    path = ref_mgr.get_reference_by_name(character_id, filename)

    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    mime_type, _ = mimetypes.guess_type(str(path))
    return FileResponse(path, media_type=mime_type or "image/png")


@router.delete("/{character_id}/references/{view_type}")
async def delete_reference(
    character_id: str,
    view_type: str,
    request: Request,
) -> dict[str, str]:
    """Delete a specific view reference for a character."""
    storage, ref_mgr = _get_managers(request)
    profile = storage.load_profile(character_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Character not found")

    if view_type not in VIEW_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid view_type '{view_type}'")

    success = ref_mgr.delete_reference(character_id, view_type)
    if not success:
        raise HTTPException(status_code=404, detail=f"No reference image found for view '{view_type}'")

    return {"message": f"Reference for {view_type} deleted"}


@router.get("/{character_id}/readiness", response_model=CharacterReadinessReport)
async def get_character_readiness(
    character_id: str,
    request: Request,
) -> CharacterReadinessReport:
    """Get on-demand AI readiness evaluation for a character."""
    storage, ref_mgr = _get_managers(request)
    report = ref_mgr.get_readiness_report(character_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Character not found")
    return report


@router.delete("/{character_id}", status_code=status.HTTP_200_OK)
async def delete_character(
    character_id: str,
    request: Request,
) -> dict[str, str]:
    """Delete a character profile and all associated reference images."""
    storage, _ = _get_managers(request)
    profile = storage.load_profile(character_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Character not found")

    success = storage.delete_profile(character_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete character directory")

    logger.info("Deleted character: %s", character_id)
    return {"message": f"Character {character_id} deleted", "id": character_id}
