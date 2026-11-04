"""Generation module for ReelMotion AI.

Provides:
- GenerationModelManager, ModelOffloadConfig, VRAMStatus
- WanBackend, DenseMotionFieldWarper, GenerationConfig, GenerationResult
- LipSyncBackend, AudioFeatureExtractor, PhonemeArticulation, LipSyncConfig, LipSyncResult
"""

from backend.generation.lip_sync_backend import (
    AudioFeatureExtractor,
    LipSyncBackend,
    LipSyncConfig,
    LipSyncResult,
    PhonemeArticulation,
)
from backend.generation.model_manager import (
    GenerationModelManager,
    ModelOffloadConfig,
    VRAMStatus,
    default_model_manager,
)
from backend.generation.wan_backend import (
    DenseMotionFieldWarper,
    FrameLandmarks,
    GenerationConfig,
    GenerationResult,
    WanBackend,
)

__all__ = [
    "GenerationModelManager",
    "ModelOffloadConfig",
    "VRAMStatus",
    "default_model_manager",
    "WanBackend",
    "DenseMotionFieldWarper",
    "FrameLandmarks",
    "GenerationConfig",
    "GenerationResult",
    "LipSyncBackend",
    "LipSyncConfig",
    "LipSyncResult",
    "PhonemeArticulation",
    "AudioFeatureExtractor",
]
