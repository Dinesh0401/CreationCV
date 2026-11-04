"""Character management API endpoints — Phase 1 stubs.

Full implementation in Phase 4. Currently provides basic CRUD
stubs that work with the characters directory.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/characters", tags=["characters"])


class CharacterInfo(BaseModel):
    """Character metadata."""

    id: str
    name: str
    created_at: str
    reference_count: int = 0
    thumbnail: str | None = None


class CharacterCreateRequest(BaseModel):
    """Request body for creating a character."""

    name: str = Field(..., min_length=1, max_length=100)


class CharacterCreateResponse(BaseModel):
    """Response for character creation."""

    id: str
    name: str
    created_at: str
    message: str


def _get_characters_dir(request: Request) -> Path:
    """Get the resolved characters directory."""
    settings = request.app.state.settings
    paths = settings.resolved_paths()
    return paths["characters_dir"]


def _load_character_meta(char_dir: Path) -> CharacterInfo | None:
    """Load character metadata from a character directory."""
    meta_path = char_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CharacterInfo(**data)
    except Exception as exc:
        logger.warning("Failed to load character metadata from %s: %s", meta_path, exc)
        return None


@router.get("", response_model=list[CharacterInfo])
async def list_characters(request: Request) -> list[CharacterInfo]:
    """List all characters from the characters directory."""
    chars_dir = _get_characters_dir(request)
    if not chars_dir.exists():
        return []

    characters: list[CharacterInfo] = []
    for entry in sorted(chars_dir.iterdir()):
        if entry.is_dir():
            info = _load_character_meta(entry)
            if info:
                characters.append(info)
    return characters


@router.post("", response_model=CharacterCreateResponse, status_code=201)
async def create_character(
    request: Request,
    body: CharacterCreateRequest,
) -> CharacterCreateResponse:
    """Create a new character (stub — full upload in Phase 4).

    Creates the character directory and metadata file.
    Reference image upload will be implemented in Phase 4.
    """
    chars_dir = _get_characters_dir(request)
    chars_dir.mkdir(parents=True, exist_ok=True)

    char_id = str(uuid.uuid4())
    char_dir = chars_dir / char_id
    char_dir.mkdir(exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    meta = {
        "id": char_id,
        "name": body.name,
        "created_at": now,
        "reference_count": 0,
        "thumbnail": None,
    }

    with open(char_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Character created: %s (%s)", body.name, char_id)
    return CharacterCreateResponse(
        id=char_id,
        name=body.name,
        created_at=now,
        message="Character created. Upload reference images in Phase 4.",
    )


@router.get("/{character_id}", response_model=CharacterInfo)
async def get_character(
    character_id: str,
    request: Request,
) -> CharacterInfo:
    """Get character info by ID."""
    chars_dir = _get_characters_dir(request)
    char_dir = chars_dir / character_id

    if not char_dir.exists():
        raise HTTPException(status_code=404, detail="Character not found")

    info = _load_character_meta(char_dir)
    if info is None:
        raise HTTPException(status_code=404, detail="Character metadata not found")
    return info


@router.delete("/{character_id}", status_code=200)
async def delete_character(
    character_id: str,
    request: Request,
) -> dict[str, str]:
    """Delete a character and all its files."""
    chars_dir = _get_characters_dir(request)
    char_dir = chars_dir / character_id

    if not char_dir.exists():
        raise HTTPException(status_code=404, detail="Character not found")

    import shutil

    shutil.rmtree(char_dir)
    logger.info("Character deleted: %s", character_id)
    return {"message": f"Character {character_id} deleted"}
