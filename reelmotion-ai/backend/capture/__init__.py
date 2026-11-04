"""Capture module for ReelMotion AI — Phase 2.

Provides:
- Camera device discovery, enumeration, and resolution validation
- Video frame extraction and verification
- Audio format verification (WAV, MP3, AAC, WebM) and waveform analysis
- Audio-video synchronization and drift calculation
"""

from backend.capture.audio import (
    AudioAnalyzer,
    AudioFormat,
    AudioStreamInfo,
    AudioValidationResult,
    WaveformSummary,
)
from backend.capture.camera import (
    CameraInfo,
    CameraManager,
    FrameInfo,
    Resolution,
    ResolutionValidationResult,
    VideoFrameExtractor,
    VideoMetadata,
    VideoVerificationResult,
    verify_video,
)
from backend.capture.synchronizer import (
    FrameAudioWindow,
    MediaSynchronizer,
    SyncReport,
    SyncStatus,
)

__all__ = [
    "AudioAnalyzer",
    "AudioFormat",
    "AudioStreamInfo",
    "AudioValidationResult",
    "CameraInfo",
    "CameraManager",
    "FrameAudioWindow",
    "FrameInfo",
    "MediaSynchronizer",
    "Resolution",
    "ResolutionValidationResult",
    "SyncReport",
    "SyncStatus",
    "VideoFrameExtractor",
    "VideoMetadata",
    "VideoVerificationResult",
    "WaveformSummary",
    "verify_video",
]
