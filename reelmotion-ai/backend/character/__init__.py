"""Character module — Phase 4.

Provides:
- Character profile schema and storage (profile.json & meta.json)
- Multi-view reference image management (front, side, back, face_closeup, outfit_details)
- AI image quality validation and character readiness reporting
"""

from backend.character.character_profile import (
    VIEW_TYPES,
    CharacterProfile,
    CharacterProfileManager,
    CharacterStorageManager,
    CharacterViews,
)
from backend.character.identity_validation import (
    CharacterReadinessReport,
    ImageQualityReport,
    compute_character_consistency,
    generate_readiness_report,
    validate_image,
)
from backend.character.reference_manager import (
    CharacterReferenceManager,
)

__all__ = [
    "VIEW_TYPES",
    "CharacterProfile",
    "CharacterViews",
    "CharacterStorageManager",
    "CharacterProfileManager",
    "ImageQualityReport",
    "CharacterReadinessReport",
    "validate_image",
    "compute_character_consistency",
    "generate_readiness_report",
    "CharacterReferenceManager",
]
