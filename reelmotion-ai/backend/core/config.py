"""Configuration system for ReelMotion AI.

Loads settings from configs/default.yaml and .env,
providing a validated Pydantic Settings model.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Project root: two levels up from backend/core/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "default.yaml"
ENV_FILE_PATH = PROJECT_ROOT / ".env"


def load_yaml_config(path: Path | None = None) -> dict[str, Any]:
    """Load YAML configuration file. Returns empty dict on failure."""
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        logger.warning("Config file not found: %s", config_path)
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.error("Failed to load config %s: %s", config_path, exc)
        return {}


class ServerSettings(BaseModel):
    """Server configuration."""

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    debug: bool = False
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
        ]
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid}")
        return upper


class PathSettings(BaseModel):
    """Directory path configuration."""

    models_dir: str = "./models"
    characters_dir: str = "./characters"
    recordings_dir: str = "./recordings"
    exports_dir: str = "./exports"
    logs_dir: str = "./logs"

    def resolve(self, base: Path | None = None) -> dict[str, Path]:
        """Resolve all paths relative to the given base directory."""
        root = base or PROJECT_ROOT
        return {
            name: (root / getattr(self, name)).resolve()
            for name in self.__class__.model_fields
        }


class CaptureSettings(BaseModel):
    """Camera capture configuration."""

    default_camera_index: int = 0
    default_resolution: list[int] = Field(default_factory=lambda: [1280, 720])
    default_fps: int = 30
    supported_resolutions: list[list[int]] = Field(
        default_factory=lambda: [[640, 480], [1280, 720], [1920, 1080]]
    )


class RecordingSettings(BaseModel):
    """Recording format configuration."""

    format: str = "mp4"
    video_codec: str = "h264"
    audio_codec: str = "aac"
    default_bitrate: str = "8M"


class ExportSettings(BaseModel):
    """Export configuration."""

    width: int = 1080
    height: int = 1920
    aspect_ratio: str = "9:16"
    fps: int = 30
    format: str = "mp4"


class HardwareSettings(BaseModel):
    """Hardware override configuration."""

    force_cpu: bool = False
    max_vram_mb: int | None = None
    max_ram_mb: int | None = None
    vram_safety_margin_mb: int = 512
    enable_memory_monitoring: bool = True
    monitoring_interval_seconds: int = 5


class CharacterSettings(BaseModel):
    """Character image constraints."""

    accepted_formats: list[str] = Field(
        default_factory=lambda: ["png", "jpg", "jpeg", "webp"]
    )
    max_file_size_mb: int = 50
    min_resolution: list[int] = Field(default_factory=lambda: [256, 256])
    max_resolution: list[int] = Field(default_factory=lambda: [4096, 4096])
    max_references_per_character: int = 10


class Settings(BaseSettings):
    """Root application settings, loaded from YAML + environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="REELMOTION_",
        env_file=str(ENV_FILE_PATH) if ENV_FILE_PATH.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Nested settings from YAML
    server: ServerSettings = Field(default_factory=ServerSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    capture: CaptureSettings = Field(default_factory=CaptureSettings)
    recording: RecordingSettings = Field(default_factory=RecordingSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)
    hardware: HardwareSettings = Field(default_factory=HardwareSettings)
    character: CharacterSettings = Field(default_factory=CharacterSettings)

    # Flat env overrides
    host: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    log_level: str | None = None
    debug: bool | None = None
    force_cpu: bool | None = None
    models_dir: str | None = None
    characters_dir: str | None = None
    recordings_dir: str | None = None
    exports_dir: str | None = None

    def model_post_init(self, __context: Any) -> None:
        """Apply flat env overrides to nested settings after loading."""
        if self.host is not None:
            self.server.host = self.host
        if self.port is not None:
            self.server.port = self.port
        if self.log_level is not None:
            self.server.log_level = self.log_level.upper()
        if self.debug is not None:
            self.server.debug = self.debug
        if self.force_cpu is not None:
            self.hardware.force_cpu = self.force_cpu
        if self.models_dir is not None:
            self.paths.models_dir = self.models_dir
        if self.characters_dir is not None:
            self.paths.characters_dir = self.characters_dir
        if self.recordings_dir is not None:
            self.paths.recordings_dir = self.recordings_dir
        if self.exports_dir is not None:
            self.paths.exports_dir = self.exports_dir

    def resolved_paths(self) -> dict[str, Path]:
        """Get all paths resolved to absolute paths."""
        return self.paths.resolve(PROJECT_ROOT)


def _build_settings(config_path: Path | None = None) -> Settings:
    """Build Settings by merging YAML config with env vars."""
    yaml_data = load_yaml_config(config_path)
    return Settings(**yaml_data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get the singleton Settings instance."""
    return _build_settings()


def get_fresh_settings(config_path: Path | None = None) -> Settings:
    """Create a fresh Settings instance (not cached). Useful for testing."""
    return _build_settings(config_path)
