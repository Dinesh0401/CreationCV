"""Tests for configuration system and validation."""

from pathlib import Path
import pytest
from pydantic import ValidationError
from backend.core.config import (
    Settings,
    ServerSettings,
    load_yaml_config,
    get_settings,
)


def test_config_loads_defaults():
    """Test default config loading."""
    settings = get_settings()
    assert settings.server.host == "127.0.0.1"
    assert settings.server.port == 8000
    assert "mp4" in settings.recording.format.lower()


def test_config_validation_invalid_port():
    """Failure test: Port out of range (e.g. 70000 or -1) must fail validation."""
    with pytest.raises(ValidationError):
        ServerSettings(port=70000)

    with pytest.raises(ValidationError):
        ServerSettings(port=-5)


def test_config_validation_invalid_log_level():
    """Failure test: Invalid log level must raise ValidationError."""
    with pytest.raises(ValidationError):
        ServerSettings(log_level="SUPER_VERBOSE")


def test_missing_yaml_file_fallback(tmp_path: Path):
    """Failure test: Non-existent YAML file should return empty dict and not crash."""
    non_existent = tmp_path / "does_not_exist.yaml"
    data = load_yaml_config(non_existent)
    assert data == {}
