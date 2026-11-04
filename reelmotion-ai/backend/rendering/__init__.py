"""Rendering module for ReelMotion AI.

Provides:
- BackgroundCompositor, CompositorConfig
- AudioMixer, AudioSyncConfig
- VideoExporter, AspectRatioMode, TARGET_WIDTH, TARGET_HEIGHT
"""

from backend.rendering.audio_mixer import AudioMixer, AudioSyncConfig
from backend.rendering.background_compositor import (
    BackgroundCompositor,
    CompositorConfig,
)
from backend.rendering.export import (
    TARGET_HEIGHT,
    TARGET_WIDTH,
    AspectRatioMode,
    VideoExporter,
)

__all__ = [
    "BackgroundCompositor",
    "CompositorConfig",
    "AudioMixer",
    "AudioSyncConfig",
    "VideoExporter",
    "AspectRatioMode",
    "TARGET_WIDTH",
    "TARGET_HEIGHT",
]
