"""Hardware-aware model loading and memory management for ReelMotion AI.

Provides VRAM monitoring, safety margin reservation (512MB default),
CPU offloading configuration for 6GB RTX 4050, and dynamic model lifecycle
(loading, LRU eviction, and unloading).
"""

from __future__ import annotations

import gc
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator

import torch

from backend.services.hardware_manager import HardwareManager, HardwareTier
from backend.services.model_manager import ModelManager as ManifestModelManager

logger = logging.getLogger(__name__)

DEFAULT_SAFETY_MARGIN_MB = 512
RTX_4050_VRAM_MB = 6144


@dataclass
class ModelOffloadConfig:
    """CPU offloading and execution configuration tailored for constrained VRAM."""

    enable_cpu_offload: bool = True
    device: str = "cuda"
    offload_device: str = "cpu"
    precision: str = "fp16"  # "fp16", "bf16", "fp32"
    chunk_size_frames: int = 16
    sequential_cpu_offload: bool = True
    vram_safety_margin_mb: int = DEFAULT_SAFETY_MARGIN_MB

    @classmethod
    def rtx_4050_preset(cls) -> "ModelOffloadConfig":
        """Preconfigured offloading profile specifically tuned for 6GB RTX 4050."""
        return cls(
            enable_cpu_offload=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            offload_device="cpu",
            precision="fp16",
            chunk_size_frames=16,
            sequential_cpu_offload=True,
            vram_safety_margin_mb=DEFAULT_SAFETY_MARGIN_MB,
        )

    @classmethod
    def cpu_only_preset(cls) -> "ModelOffloadConfig":
        """Configuration for systems without an active GPU."""
        return cls(
            enable_cpu_offload=False,
            device="cpu",
            offload_device="cpu",
            precision="fp32",
            chunk_size_frames=8,
            sequential_cpu_offload=False,
            vram_safety_margin_mb=0,
        )


@dataclass
class LoadedModelEntry:
    """Metadata tracking an in-memory model."""

    model_id: str
    model: Any
    device: str
    estimated_vram_mb: int
    loaded_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)


@dataclass
class VRAMStatus:
    """Current VRAM memory status with safety margin."""

    cuda_available: bool
    total_mb: int = 0
    allocated_mb: int = 0
    reserved_mb: int = 0
    free_mb: int = 0
    safety_margin_mb: int = DEFAULT_SAFETY_MARGIN_MB
    usable_mb: int = 0

    def can_allocate(self, required_mb: int) -> bool:
        """Check if required MB can be accommodated within usable VRAM."""
        if not self.cuda_available:
            return False
        return self.usable_mb >= required_mb


class GenerationModelManager:
    """Hardware-aware model lifecycle manager for generation backends.

    Monitors available VRAM, reserves a 512MB safety buffer, enforces
    CPU offloading on 6GB RTX 4050 class GPUs, and handles model
    loading and eviction.
    """

    def __init__(
        self,
        safety_margin_mb: int = DEFAULT_SAFETY_MARGIN_MB,
        force_cpu: bool = False,
        default_offload_config: ModelOffloadConfig | None = None,
        manifest_manager: ManifestModelManager | None = None,
    ) -> None:
        self.safety_margin_mb = safety_margin_mb
        self.force_cpu = force_cpu
        self._loaded_models: dict[str, LoadedModelEntry] = {}
        self.manifest_manager = manifest_manager or ManifestModelManager()

        if default_offload_config is not None:
            self.offload_config = default_offload_config
        elif self.force_cpu or not torch.cuda.is_available():
            self.offload_config = ModelOffloadConfig.cpu_only_preset()
        else:
            self.offload_config = ModelOffloadConfig.rtx_4050_preset()
            self.offload_config.vram_safety_margin_mb = safety_margin_mb

    def get_vram_status(self) -> VRAMStatus:
        """Get current VRAM memory metrics from PyTorch."""
        if self.force_cpu or not torch.cuda.is_available():
            return VRAMStatus(
                cuda_available=False,
                total_mb=0,
                allocated_mb=0,
                reserved_mb=0,
                free_mb=0,
                safety_margin_mb=self.safety_margin_mb,
                usable_mb=0,
            )

        try:
            device = torch.cuda.current_device()
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            total_mb = int(total_bytes / (1024 * 1024))
            free_mb = int(free_bytes / (1024 * 1024))
            allocated_mb = int(torch.cuda.memory_allocated(device) / (1024 * 1024))
            reserved_mb = int(torch.cuda.memory_reserved(device) / (1024 * 1024))
            usable_mb = max(0, free_mb - self.safety_margin_mb)

            return VRAMStatus(
                cuda_available=True,
                total_mb=total_mb,
                allocated_mb=allocated_mb,
                reserved_mb=reserved_mb,
                free_mb=free_mb,
                safety_margin_mb=self.safety_margin_mb,
                usable_mb=usable_mb,
            )
        except Exception as exc:
            logger.warning("Failed to query CUDA memory: %s", exc)
            return VRAMStatus(
                cuda_available=True,
                total_mb=0,
                allocated_mb=0,
                reserved_mb=0,
                free_mb=0,
                safety_margin_mb=self.safety_margin_mb,
                usable_mb=0,
            )

    def resolve_target_device(self, required_vram_mb: int = 0) -> str:
        """Determine whether to place a model on CUDA or CPU based on VRAM headroom."""
        if self.force_cpu or not torch.cuda.is_available():
            return "cpu"

        status = self.get_vram_status()
        if status.can_allocate(required_vram_mb):
            return "cuda"

        # Try LRU eviction to make room
        freed = self.evict_lru(required_vram_mb)
        status = self.get_vram_status()
        if status.can_allocate(required_vram_mb):
            return "cuda"

        logger.info(
            "Insufficient VRAM for %d MB (usable: %d MB after freeing %d MB). Offloading to CPU.",
            required_vram_mb,
            status.usable_mb,
            freed,
        )
        return "cpu"

    def is_model_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded in memory."""
        return model_id in self._loaded_models

    def get_loaded_model(self, model_id: str) -> Any | None:
        """Get loaded model instance and update its last-used timestamp."""
        entry = self._loaded_models.get(model_id)
        if entry:
            entry.last_used_at = time.time()
            return entry.model
        return None

    def load_model(
        self,
        model_id: str,
        loader_fn: Callable[..., Any],
        required_vram_mb: int = 0,
        force_cpu: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Load a model with hardware-aware device selection and VRAM margin protection.

        Args:
            model_id: Identifier for the model.
            loader_fn: Callable that accepts device (str) and kwargs and returns the model.
            required_vram_mb: Estimated VRAM footprint in MB.
            force_cpu: Force CPU placement regardless of GPU headroom.
            **kwargs: Extra parameters passed to loader_fn.

        Returns:
            The loaded model instance.
        """
        if model_id in self._loaded_models:
            entry = self._loaded_models[model_id]
            entry.last_used_at = time.time()
            logger.debug("Reusing in-memory model '%s'", model_id)
            return entry.model

        if force_cpu:
            target_device = "cpu"
        else:
            target_device = self.resolve_target_device(required_vram_mb)

        logger.info(
            "Loading model '%s' on %s (estimated VRAM: %d MB, safety margin: %d MB)",
            model_id,
            target_device,
            required_vram_mb,
            self.safety_margin_mb,
        )

        model = loader_fn(device=target_device, **kwargs)

        self._loaded_models[model_id] = LoadedModelEntry(
            model_id=model_id,
            model=model,
            device=target_device,
            estimated_vram_mb=required_vram_mb if target_device == "cuda" else 0,
            loaded_at=time.time(),
            last_used_at=time.time(),
        )
        return model

    def register_loaded_model(
        self,
        model_id: str,
        model: Any,
        device: str = "cpu",
        estimated_vram_mb: int = 0,
    ) -> None:
        """Register an externally constructed model with the manager."""
        self._loaded_models[model_id] = LoadedModelEntry(
            model_id=model_id,
            model=model,
            device=device,
            estimated_vram_mb=estimated_vram_mb,
            loaded_at=time.time(),
            last_used_at=time.time(),
        )

    def unload_model(self, model_id: str) -> bool:
        """Unload a specific model and reclaim GPU/system memory."""
        entry = self._loaded_models.pop(model_id, None)
        if entry is None:
            return False

        logger.info("Unloading model '%s' from %s", model_id, entry.device)
        del entry.model
        del entry
        self._collect_garbage()
        return True

    def evict_lru(self, needed_mb: int) -> int:
        """Evict least recently used GPU models until needed_mb is freed.

        Returns:
            Total MB freed.
        """
        cuda_entries = [
            (mid, entry)
            for mid, entry in self._loaded_models.items()
            if entry.device == "cuda"
        ]
        if not cuda_entries:
            return 0

        # Sort by last_used_at ascending (oldest first)
        cuda_entries.sort(key=lambda item: item[1].last_used_at)

        freed_mb = 0
        for mid, entry in cuda_entries:
            if freed_mb >= needed_mb:
                break
            logger.info("Evicting LRU model '%s' (estimated %d MB)", mid, entry.estimated_vram_mb)
            freed_mb += entry.estimated_vram_mb
            self.unload_model(mid)

        return freed_mb

    def unload_all(self) -> int:
        """Unload all currently tracked models."""
        count = len(self._loaded_models)
        self._loaded_models.clear()
        self._collect_garbage()
        logger.info("Unloaded all %d models", count)
        return count

    @contextmanager
    def active_model(
        self,
        model_id: str,
        loader_fn: Callable[..., Any],
        required_vram_mb: int = 0,
        auto_unload: bool = False,
        **kwargs: Any,
    ) -> Generator[Any, None, None]:
        """Context manager to ensure a model is loaded for execution and optionally unloaded."""
        model = self.load_model(
            model_id=model_id,
            loader_fn=loader_fn,
            required_vram_mb=required_vram_mb,
            **kwargs,
        )
        try:
            yield model
        finally:
            if auto_unload:
                self.unload_model(model_id)

    def _collect_garbage(self) -> None:
        """Clean up Python garbage and empty CUDA memory cache."""
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as exc:
                logger.debug("CUDA empty_cache error: %s", exc)


# Global default instance
default_model_manager = GenerationModelManager()

# Alias for compatibility
ModelManager = GenerationModelManager

