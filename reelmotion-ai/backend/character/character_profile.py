"""Character profile data model and storage manager — Phase 4.

Defines the CharacterProfile schema with multi-view references
(front, side, back, face_closeup, outfit_details) and persistent storage
in characters/<character_id>/profile.json.
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from backend.core.config import PROJECT_ROOT, get_settings

logger = logging.getLogger(__name__)

VIEW_TYPES: list[str] = ["front", "side", "back", "face_closeup", "outfit_details"]


class CharacterViews(BaseModel):
    """Multi-view reference paths for a character."""

    front: str | None = None
    side: str | None = None
    back: str | None = None
    face_closeup: str | None = None
    outfit_details: str | None = None

    def get_view(self, view_type: str) -> str | None:
        """Get the reference filename/path for a specific view type."""
        return getattr(self, view_type, None)

    def set_view(self, view_type: str, filename: str | None) -> None:
        """Set the reference filename/path for a specific view type."""
        if hasattr(self, view_type):
            setattr(self, view_type, filename)
        else:
            raise ValueError(f"Unknown view type: {view_type}. Must be one of {VIEW_TYPES}")

    def active_views(self) -> dict[str, str]:
        """Return a mapping of all currently configured views."""
        result: dict[str, str] = {}
        for vt in VIEW_TYPES:
            val = getattr(self, vt)
            if val is not None:
                result[vt] = val
        return result

    def count(self) -> int:
        """Count how many views are set."""
        return len(self.active_views())


class CharacterProfile(BaseModel):
    """Complete character profile including multi-view references and metadata."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=1, max_length=100)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str | None = None
    tags: list[str] = Field(default_factory=list)
    views: CharacterViews = Field(default_factory=CharacterViews)
    thumbnail: str | None = None
    reference_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def refresh_reference_count(self) -> int:
        """Recalculate reference count based on active views."""
        self.reference_count = self.views.count()
        return self.reference_count


class CharacterStorageManager:
    """Manages character directories and profile persistence."""

    def __init__(self, base_dir: Path | str | None = None) -> None:
        if base_dir is not None:
            self.base_dir = Path(base_dir).resolve()
        else:
            try:
                settings = get_settings()
                self.base_dir = settings.resolved_paths()["characters_dir"]
            except Exception:
                self.base_dir = (PROJECT_ROOT / "characters").resolve()

        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_character_dir(self, character_id: str) -> Path:
        """Get absolute path to a character's directory."""
        return self.base_dir / character_id

    def get_references_dir(self, character_id: str) -> Path:
        """Get absolute path to a character's reference images directory."""
        ref_dir = self.get_character_dir(character_id) / "references"
        ref_dir.mkdir(parents=True, exist_ok=True)
        return ref_dir

    def create_profile(
        self,
        name: str,
        tags: list[str] | None = None,
        character_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CharacterProfile:
        """Create a new character profile on disk and return it."""
        char_id = character_id or str(uuid.uuid4())
        char_dir = self.get_character_dir(char_id)
        char_dir.mkdir(parents=True, exist_ok=True)
        self.get_references_dir(char_id)

        now = datetime.now(timezone.utc).isoformat()
        profile = CharacterProfile(
            id=char_id,
            name=name,
            created_at=now,
            updated_at=now,
            tags=tags or [],
            views=CharacterViews(),
            thumbnail=None,
            reference_count=0,
            metadata=metadata or {},
        )
        self.save_profile(profile)
        return profile

    def save_profile(self, profile: CharacterProfile) -> CharacterProfile:
        """Save a CharacterProfile to characters/<character_id>/profile.json and meta.json."""
        char_dir = self.get_character_dir(profile.id)
        char_dir.mkdir(parents=True, exist_ok=True)

        profile.updated_at = datetime.now(timezone.utc).isoformat()
        profile.refresh_reference_count()

        # If thumbnail is not set, use front view or first available view
        if not profile.thumbnail and profile.reference_count > 0:
            profile.thumbnail = profile.views.front or next(iter(profile.views.active_views().values()), None)

        profile_path = char_dir / "profile.json"
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile.model_dump(), f, indent=2)

        # Also write legacy meta.json for backwards compatibility
        meta_path = char_dir / "meta.json"
        meta_data = {
            "id": profile.id,
            "name": profile.name,
            "created_at": profile.created_at,
            "reference_count": profile.reference_count,
            "thumbnail": profile.thumbnail,
            "tags": profile.tags,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)

        logger.debug("Saved profile for character %s at %s", profile.id, profile_path)
        return profile

    def load_profile(self, character_id: str) -> CharacterProfile | None:
        """Load character profile from profile.json or legacy meta.json."""
        char_dir = self.get_character_dir(character_id)
        if not char_dir.exists() or not char_dir.is_dir():
            return None

        profile_path = char_dir / "profile.json"
        if profile_path.exists():
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return CharacterProfile(**data)
            except Exception as exc:
                logger.error("Failed to load profile.json for %s: %s", character_id, exc)

        # Fallback to meta.json
        meta_path = char_dir / "meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                profile = CharacterProfile(
                    id=meta.get("id", character_id),
                    name=meta.get("name", "Unnamed Character"),
                    created_at=meta.get("created_at", datetime.now(timezone.utc).isoformat()),
                    tags=meta.get("tags", []),
                    thumbnail=meta.get("thumbnail"),
                    reference_count=meta.get("reference_count", 0),
                )
                return profile
            except Exception as exc:
                logger.error("Failed to load meta.json for %s: %s", character_id, exc)

        return None

    def list_profiles(self) -> list[CharacterProfile]:
        """List all available character profiles, sorted by created_at descending."""
        if not self.base_dir.exists():
            return []

        profiles: list[CharacterProfile] = []
        for entry in self.base_dir.iterdir():
            if entry.is_dir():
                profile = self.load_profile(entry.name)
                if profile is not None:
                    profiles.append(profile)

        profiles.sort(key=lambda p: p.created_at, reverse=True)
        return profiles

    def delete_profile(self, character_id: str) -> bool:
        """Delete character directory and all associated files."""
        char_dir = self.get_character_dir(character_id)
        if not char_dir.exists():
            return False

        try:
            shutil.rmtree(char_dir)
            logger.info("Deleted character profile %s at %s", character_id, char_dir)
            return True
        except Exception as exc:
            logger.error("Failed to delete character %s: %s", character_id, exc)
            return False


# Alias for convenience
CharacterProfileManager = CharacterStorageManager
