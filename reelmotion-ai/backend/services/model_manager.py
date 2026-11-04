"""Model manifest manager for ReelMotion AI.

Loads the model manifest from configs/model_manifest.yaml,
provides model listing, compatibility checking, and file verification.
No actual downloads — just the interface and manifest reading.
"""

from __future__ import annotations

import hashlib
import logging
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from backend.services.hardware_manager import HardwareProfile, HardwareTier

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "configs" / "model_manifest.yaml"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


class ModelStatus(str, Enum):
    """Model availability status."""

    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    READY = "ready"
    INCOMPATIBLE = "incompatible"
    CORRUPTED = "corrupted"


class ModelFile(BaseModel):
    """A single file belonging to a model."""

    filename: str
    size_mb: int = 0
    sha256: str = ""


class ModelInfo(BaseModel):
    """Complete information about a model from the manifest."""

    id: str
    name: str
    purpose: str = ""
    phase: int = 1
    repository: str = ""
    license: str = ""
    license_url: str = ""
    files: list[ModelFile] = Field(default_factory=list)
    min_vram_mb: int = 0
    recommended_vram_mb: int = 0
    supports_cpu: bool = False
    supports_fp16: bool = False
    status: ModelStatus = ModelStatus.NOT_DOWNLOADED
    notes: str = ""


class HardwarePreset(BaseModel):
    """Hardware-specific preset configuration."""

    max_resolution: int = 512
    max_frames: int = 24
    preferred_models: list[str] = Field(default_factory=list)
    enable_cpu_offload: bool = True
    precision: str = "fp16"


class ModelCompatibility(BaseModel):
    """Compatibility report for a model vs. detected hardware."""

    model_id: str
    model_name: str
    compatible: bool
    reason: str = ""
    recommended: bool = False
    meets_minimum: bool = False
    meets_recommended: bool = False


class ModelManifest(BaseModel):
    """Parsed model manifest file."""

    models: list[ModelInfo] = Field(default_factory=list)
    hardware_presets: dict[str, HardwarePreset] = Field(default_factory=dict)


class ModelManager:
    """Manages the model manifest and compatibility checking.

    Loads from configs/model_manifest.yaml, checks model compatibility
    with detected hardware, and verifies model file integrity.
    """

    def __init__(
        self,
        manifest_path: Path | None = None,
        models_dir: Path | None = None,
    ) -> None:
        self._manifest_path = manifest_path or DEFAULT_MANIFEST_PATH
        self._models_dir = models_dir or DEFAULT_MODELS_DIR
        self._manifest: ModelManifest | None = None

    def load_manifest(self) -> ModelManifest:
        """Load and parse the model manifest YAML file."""
        if not self._manifest_path.exists():
            logger.warning("Model manifest not found: %s", self._manifest_path)
            self._manifest = ModelManifest(models=[], hardware_presets={})
            return self._manifest

        try:
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Malformed manifest YAML: {exc}") from exc

        if not isinstance(raw, dict):
            raise ValueError("Manifest must be a YAML mapping at the top level")

        models_raw = raw.get("models", [])
        if not isinstance(models_raw, list):
            raise ValueError("'models' key must be a list")

        models: list[ModelInfo] = []
        for item in models_raw:
            if not isinstance(item, dict):
                logger.warning("Skipping non-dict model entry: %s", item)
                continue
            try:
                models.append(ModelInfo(**item))
            except Exception as exc:
                logger.warning("Skipping invalid model entry %s: %s", item.get("id", "?"), exc)

        presets_raw = raw.get("hardware_presets", {})
        presets: dict[str, HardwarePreset] = {}
        if isinstance(presets_raw, dict):
            for key, val in presets_raw.items():
                if isinstance(val, dict):
                    presets[key] = HardwarePreset(**val)

        self._manifest = ModelManifest(models=models, hardware_presets=presets)
        logger.info("Loaded model manifest: %d models, %d presets",
                     len(models), len(presets))
        return self._manifest

    @property
    def manifest(self) -> ModelManifest:
        """Get the loaded manifest, loading it if necessary."""
        if self._manifest is None:
            self.load_manifest()
        assert self._manifest is not None
        return self._manifest

    def list_models(self) -> list[ModelInfo]:
        """List all models in the manifest."""
        return self.manifest.models

    def get_model(self, model_id: str) -> ModelInfo | None:
        """Get a specific model by ID."""
        for model in self.manifest.models:
            if model.id == model_id:
                return model
        return None

    def check_compatibility(
        self, model: ModelInfo, hardware: HardwareProfile
    ) -> ModelCompatibility:
        """Check if a model is compatible with the detected hardware.

        Args:
            model: The model to check.
            hardware: Detected hardware profile.

        Returns:
            Compatibility report.
        """
        gpu = hardware.gpu
        tier = hardware.tier

        # CPU-only system
        if tier == HardwareTier.CPU_ONLY:
            if model.supports_cpu:
                return ModelCompatibility(
                    model_id=model.id,
                    model_name=model.name,
                    compatible=True,
                    reason="Model supports CPU inference",
                    meets_minimum=True,
                    meets_recommended=False,
                )
            else:
                return ModelCompatibility(
                    model_id=model.id,
                    model_name=model.name,
                    compatible=False,
                    reason="Model requires GPU but none detected",
                )

        # GPU available
        vram = gpu.vram_total_mb
        meets_min = vram >= model.min_vram_mb
        meets_rec = vram >= model.recommended_vram_mb

        if not meets_min and not model.supports_cpu:
            return ModelCompatibility(
                model_id=model.id,
                model_name=model.name,
                compatible=False,
                reason=(
                    f"Requires {model.min_vram_mb} MB VRAM, "
                    f"but only {vram} MB available"
                ),
            )

        # Determine if this model is recommended for the tier
        preset = self.manifest.hardware_presets.get(tier.value)
        recommended = (
            model.id in preset.preferred_models if preset else False
        )

        return ModelCompatibility(
            model_id=model.id,
            model_name=model.name,
            compatible=True,
            reason="Compatible" if meets_min else "CPU fallback available",
            recommended=recommended,
            meets_minimum=meets_min,
            meets_recommended=meets_rec,
        )

    def get_compatible_models(
        self, hardware: HardwareProfile
    ) -> list[ModelCompatibility]:
        """Get compatibility reports for all models."""
        return [
            self.check_compatibility(model, hardware)
            for model in self.manifest.models
        ]

    def get_recommended_models(
        self, hardware: HardwareProfile
    ) -> list[ModelInfo]:
        """Get models recommended for the hardware tier."""
        preset = self.manifest.hardware_presets.get(hardware.tier.value)
        if not preset:
            # Fallback: return all CPU-capable models
            return [m for m in self.manifest.models if m.supports_cpu]

        return [
            m
            for m in self.manifest.models
            if m.id in preset.preferred_models
        ]

    def get_preset(self, tier: HardwareTier) -> HardwarePreset | None:
        """Get the hardware preset for a given tier."""
        return self.manifest.hardware_presets.get(tier.value)

    def check_model_status(self, model_id: str) -> ModelStatus:
        """Check the actual on-disk status of a model.

        Looks for model files in the models directory and
        optionally verifies checksums.
        """
        model = self.get_model(model_id)
        if model is None:
            return ModelStatus.NOT_DOWNLOADED

        model_dir = self._models_dir / model_id
        if not model_dir.exists():
            return ModelStatus.NOT_DOWNLOADED

        for mf in model.files:
            file_path = model_dir / mf.filename
            if not file_path.exists():
                return ModelStatus.NOT_DOWNLOADED

            # Verify checksum if provided
            if mf.sha256:
                actual_hash = self._compute_sha256(file_path)
                if actual_hash != mf.sha256:
                    logger.warning(
                        "Checksum mismatch for %s/%s: expected %s, got %s",
                        model_id,
                        mf.filename,
                        mf.sha256,
                        actual_hash,
                    )
                    return ModelStatus.CORRUPTED

        return ModelStatus.READY

    @staticmethod
    def _compute_sha256(file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()
