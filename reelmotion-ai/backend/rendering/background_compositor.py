"""Background Compositor for ReelMotion AI.

Preserves the original room background, environmental lighting, and studio setting
while seamlessly compositing the generated character into the scene.
Features:
- Alpha matting from person segmentation masks
- Foreground edge feathering and erosion to eliminate harsh halos
- Ambient lighting / color temperature harmonization
- Frame-by-frame and batch sequence compositing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CompositorConfig:
    """Configuration for background compositing and edge feathering."""

    feather_radius: int = 5  # Gaussian blur kernel radius
    erosion_size: int = 2  # Pixel erosion to eliminate fringe halos
    ambient_match_strength: float = 0.12  # Subtle ambient lighting tint (0.0 to 1.0)
    enable_ambient_matching: bool = True
    blur_sigma: float = 2.0


class BackgroundCompositor:
    """Composites generated character frames into the original recorded room environment."""

    def __init__(self, config: Optional[CompositorConfig] = None) -> None:
        self.config = config or CompositorConfig()

    @staticmethod
    def _normalize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        """Convert mask to a 2D float32 array in range [0.0, 1.0] matching target shape."""
        h, w = target_shape[:2]

        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        if mask.dtype == np.uint8:
            norm_mask = mask.astype(np.float32) / 255.0
        else:
            norm_mask = mask.astype(np.float32)
            if norm_mask.max() > 1.0:
                norm_mask = norm_mask / norm_mask.max()

        return np.clip(norm_mask, 0.0, 1.0)

    @classmethod
    def synthesize_fallback_mask(cls, frame: np.ndarray) -> np.ndarray:
        """Synthesize an elliptical center-person mask if segmentation is unavailable."""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        center = (w // 2, int(h * 0.58))
        axes = (int(w * 0.32), int(h * 0.44))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        return cv2.GaussianBlur(mask, (31, 31), 10.0)

    def prepare_alpha_matte(self, mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        """Generate anti-aliased, de-haloed alpha matte from raw segmentation mask."""
        norm_mask = self._normalize_mask(mask, target_shape)

        # 1. Morphological erosion to strip original person edge pixels (prevents halo)
        if self.config.erosion_size > 0:
            kernel_size = self.config.erosion_size * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            eroded = cv2.erode(norm_mask, kernel, iterations=1)
        else:
            eroded = norm_mask

        # 2. Gaussian feathering for smooth edge transition
        if self.config.feather_radius > 0:
            ksize = self.config.feather_radius * 2 + 1
            feathered = cv2.GaussianBlur(eroded, (ksize, ksize), self.config.blur_sigma)
        else:
            feathered = eroded

        return np.clip(feathered, 0.0, 1.0)

    def harmonize_lighting(
        self,
        character_frame: np.ndarray,
        original_frame: np.ndarray,
        alpha_matte: np.ndarray,
    ) -> np.ndarray:
        """Gently adjust character foreground tones toward the ambient background room lighting."""
        if not self.config.enable_ambient_matching or self.config.ambient_match_strength <= 0:
            return character_frame

        bg_mask = alpha_matte < 0.15
        if not np.any(bg_mask):
            return character_frame

        # Compute ambient background mean color
        bg_mean = original_frame[bg_mask].mean(axis=0)
        fg_mask = alpha_matte > 0.6
        if not np.any(fg_mask):
            return character_frame

        fg_mean = character_frame[fg_mask].mean(axis=0)
        # Ratio for subtle tinting
        ratio = (bg_mean + 1.0) / (fg_mean + 1.0)
        clamped_ratio = np.clip(ratio, 0.8, 1.25)

        # Interpolate between original character color and tinted color
        tinted = character_frame.astype(np.float32) * clamped_ratio
        harmonized = (
            (1.0 - self.config.ambient_match_strength) * character_frame.astype(np.float32)
            + self.config.ambient_match_strength * tinted
        )

        return np.clip(harmonized, 0, 255).astype(np.uint8)

    def composite_frame(
        self,
        character_frame: np.ndarray,
        original_frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Composite a single character frame onto the original background frame."""
        h, w = original_frame.shape[:2]

        # Ensure matching spatial resolution
        if character_frame.shape[:2] != (h, w):
            character_frame = cv2.resize(character_frame, (w, h), interpolation=cv2.INTER_LINEAR)

        if mask is None:
            mask = self.synthesize_fallback_mask(original_frame)

        alpha = self.prepare_alpha_matte(mask, (h, w))
        char_lit = self.harmonize_lighting(character_frame, original_frame, alpha)

        alpha_3d = np.expand_dims(alpha, axis=-1)

        # Alpha composite: char * alpha + room_bg * (1 - alpha)
        composited = (
            char_lit.astype(np.float32) * alpha_3d
            + original_frame.astype(np.float32) * (1.0 - alpha_3d)
        )

        return np.clip(composited, 0, 255).astype(np.uint8)

    def composite_sequence(
        self,
        character_frames: Sequence[np.ndarray],
        original_frames: Sequence[np.ndarray],
        masks: Optional[Sequence[Optional[np.ndarray]]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> list[np.ndarray]:
        """Composite a sequence of character frames onto original background frames."""
        total = min(len(character_frames), len(original_frames))
        if total == 0:
            return []

        out_frames: list[np.ndarray] = []
        for i in range(total):
            c_frame = character_frames[i]
            o_frame = original_frames[i]
            m = masks[i] if (masks is not None and i < len(masks)) else None

            comp = self.composite_frame(c_frame, o_frame, m)
            out_frames.append(comp)

            if progress_callback and (i + 1) % 15 == 0:
                progress_callback(
                    float(i + 1) / total * 100.0,
                    f"Composited background on frame {i + 1}/{total}",
                )

        return out_frames
