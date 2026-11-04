"""Character reference image manager — Phase 4.

Handles storing, retrieving, deleting reference images for character profiles,
and managing multi-view references (front, side, back, face_closeup, outfit_details).
"""

from __future__ import annotations

import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from backend.character.character_profile import (
    VIEW_TYPES,
    CharacterProfile,
    CharacterStorageManager,
)
from backend.character.identity_validation import (
    CharacterReadinessReport,
    ImageQualityReport,
    generate_readiness_report,
    validate_image,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _sanitize_filename(name: str) -> str:
    """Sanitize filename to prevent directory traversal or invalid characters."""
    base = os.path.basename(name)
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "_", base)
    return sanitized or "reference.png"


class CharacterReferenceManager:
    """Manages storage and lifecycle of reference images for characters."""

    def __init__(self, storage_manager: CharacterStorageManager | None = None) -> None:
        self.storage_manager = storage_manager or CharacterStorageManager()

    def save_reference(
        self,
        character_id: str,
        content: bytes,
        original_filename: str,
        view_type: str = "front",
    ) -> tuple[str, Path, ImageQualityReport]:
        """Save a reference image for a specific view type and run quality validation.

        Returns:
            Tuple of (stored_filename, absolute_path, ImageQualityReport).
        """
        if view_type not in VIEW_TYPES:
            raise ValueError(
                f"Invalid view_type '{view_type}'. Supported: {', '.join(VIEW_TYPES)}"
            )

        profile = self.storage_manager.load_profile(character_id)
        if profile is None:
            raise FileNotFoundError(f"Character {character_id} not found.")

        ext = Path(original_filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        ref_dir = self.storage_manager.get_references_dir(character_id)

        # Standardized filename: e.g. front.png or side_ref.png
        safe_name = _sanitize_filename(original_filename)
        stored_filename = f"{view_type}_{safe_name}" if not safe_name.startswith(view_type) else safe_name
        dest_path = ref_dir / stored_filename

        # Remove previous file for this view if it had a different name
        old_file = profile.views.get_view(view_type)
        if old_file and old_file != stored_filename:
            old_path = ref_dir / old_file
            if old_path.exists():
                try:
                    old_path.unlink()
                except OSError as exc:
                    logger.warning("Could not delete previous reference file %s: %s", old_path, exc)

        with open(dest_path, "wb") as f:
            f.write(content)

        logger.info("Saved reference image %s for character %s (%s)", stored_filename, character_id, view_type)

        # Automatic quality validation
        quality_report = validate_image(dest_path, view_type=view_type)

        # Update profile
        profile.views.set_view(view_type, stored_filename)
        if not profile.thumbnail or view_type == "front":
            profile.thumbnail = stored_filename

        self.storage_manager.save_profile(profile)

        return stored_filename, dest_path, quality_report

    def get_reference_path(self, character_id: str, view_type: str) -> Path | None:
        """Resolve path to a reference image by view type."""
        profile = self.storage_manager.load_profile(character_id)
        if profile is None:
            return None

        filename = profile.views.get_view(view_type)
        if not filename:
            return None

        ref_dir = self.storage_manager.get_references_dir(character_id)
        path = ref_dir / filename
        if path.exists():
            return path

        # Check legacy location in character root dir
        char_dir = self.storage_manager.get_character_dir(character_id)
        legacy_path = char_dir / filename
        return legacy_path if legacy_path.exists() else None

    def get_reference_by_name(self, character_id: str, filename: str) -> Path | None:
        """Resolve path to a reference image by its filename."""
        safe_name = _sanitize_filename(filename)
        ref_dir = self.storage_manager.get_references_dir(character_id)
        path = ref_dir / safe_name
        if path.exists():
            return path

        char_dir = self.storage_manager.get_character_dir(character_id)
        legacy_path = char_dir / safe_name
        return legacy_path if legacy_path.exists() else None

    def list_references(self, character_id: str) -> dict[str, str]:
        """List active view references for a character."""
        profile = self.storage_manager.load_profile(character_id)
        if profile is None:
            return {}
        return profile.views.active_views()

    def delete_reference(self, character_id: str, view_type: str) -> bool:
        """Delete reference image for a specific view type."""
        if view_type not in VIEW_TYPES:
            raise ValueError(f"Invalid view_type '{view_type}'.")

        profile = self.storage_manager.load_profile(character_id)
        if profile is None:
            return False

        filename = profile.views.get_view(view_type)
        if not filename:
            return False

        ref_dir = self.storage_manager.get_references_dir(character_id)
        file_path = ref_dir / filename
        if file_path.exists():
            try:
                file_path.unlink()
            except OSError as exc:
                logger.warning("Could not delete file %s: %s", file_path, exc)

        profile.views.set_view(view_type, None)

        # Update thumbnail if needed
        if profile.thumbnail == filename:
            active = profile.views.active_views()
            profile.thumbnail = active.get("front") or next(iter(active.values()), None)

        self.storage_manager.save_profile(profile)
        logger.info("Deleted %s reference for character %s", view_type, character_id)
        return True

    def get_readiness_report(self, character_id: str) -> CharacterReadinessReport | None:
        """Generate full CharacterReadinessReport for a character."""
        profile = self.storage_manager.load_profile(character_id)
        if profile is None:
            return None

        char_dir = self.storage_manager.get_character_dir(character_id)
        return generate_readiness_report(profile, char_dir)
