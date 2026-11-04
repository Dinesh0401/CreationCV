"""Person segmentation and background isolation module for ReelMotion AI.

Generates binary and alpha masks of people, separates foreground person with RGBA alpha channel,
and isolates original backgrounds using MediaPipe or OpenCV GrabCut / morphological fallbacks.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class SegmentationMetadata(BaseModel):
    """Metadata regarding generated person segmentation masks."""

    width: int
    height: int
    person_coverage_ratio: float = 0.0
    has_person: bool = False
    backend: str = "opencv_fallback"  # 'mediapipe' or 'opencv_fallback'


class SegmentationResult(BaseModel):
    """Container for segmentation masks and isolated background/foreground."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: SegmentationMetadata
    binary_mask: Any = Field(default=None, exclude=True)  # uint8 (H, W) values {0, 255}
    alpha_mask: Any = Field(default=None, exclude=True)  # float32 (H, W) values [0.0, 1.0]
    foreground_rgba: Any = Field(default=None, exclude=True)  # uint8 (H, W, 4)
    background_isolated: Any = Field(default=None, exclude=True)  # uint8 (H, W, 3)

    def to_base64_masks(self) -> tuple[str, str]:
        """Encode binary mask and alpha mask as base64 PNG strings."""
        bin_b64 = ""
        alpha_b64 = ""

        if self.binary_mask is not None:
            ret, buf = cv2.imencode(".png", self.binary_mask)
            if ret:
                bin_b64 = base64.b64encode(buf).decode("ascii")

        if self.alpha_mask is not None:
            # Convert float32 [0.0, 1.0] to uint8 [0, 255]
            alpha_u8 = (self.alpha_mask * 255.0).astype(np.uint8)
            ret, buf = cv2.imencode(".png", alpha_u8)
            if ret:
                alpha_b64 = base64.b64encode(buf).decode("ascii")

        return bin_b64, alpha_b64


class PersonSegmenter:
    """Person segmentation mask generator and background isolator."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        force_fallback: bool = False,
    ) -> None:
        self.force_fallback = force_fallback
        self.model_path = Path(model_path) if model_path else None
        self._segmenter: Any = None
        self.backend = "opencv_fallback"

        if not self.force_fallback:
            self._init_mediapipe()

    def _init_mediapipe(self) -> None:
        """Attempt to load MediaPipe ImageSegmenter."""
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            target_path = self.model_path
            if target_path is None or not target_path.exists():
                candidates = [
                    Path("models/selfie_segmenter.task"),
                    Path("models/image_segmenter.task"),
                    Path(__file__).resolve().parent.parent.parent / "models" / "selfie_segmenter.task",
                ]
                for c in candidates:
                    if c.exists():
                        target_path = c
                        break

            if target_path and target_path.exists():
                base_opts = mp_python.BaseOptions(model_asset_path=str(target_path))
                options = mp_vision.ImageSegmenterOptions(
                    base_options=base_opts,
                    running_mode=mp_vision.RunningMode.IMAGE,
                    output_category_mask=True,
                    output_confidence_masks=True,
                )
                self._segmenter = mp_vision.ImageSegmenter.create_from_options(options)
                self.backend = "mediapipe"
                logger.info("MediaPipe ImageSegmenter loaded from %s", target_path)
            else:
                logger.info("MediaPipe segmenter task model not found. Using OpenCV segmentation fallback.")
                self.backend = "opencv_fallback"
        except Exception as exc:
            logger.info("MediaPipe segmenter initialization skipped (%s); using OpenCV fallback.", exc)
            self._segmenter = None
            self.backend = "opencv_fallback"

    def segment(
        self,
        image: np.ndarray,
        bounding_box: tuple[float, float, float, float] | None = None,
    ) -> SegmentationResult:
        """Generate binary mask, antialiased alpha mask, and foreground RGBA."""
        if image is None or image.size == 0:
            return self._build_empty_result(0, 0)

        if self.backend == "mediapipe" and self._segmenter is not None:
            try:
                res = self._segment_mediapipe(image)
                if res.metadata.has_person:
                    return res
                # Fall back to OpenCV GrabCut / morphological if neural segmenter finds no person
                cv_res = self._segment_opencv(image, bounding_box)
                if cv_res.metadata.has_person:
                    return cv_res
                return res
            except Exception as exc:
                logger.warning("MediaPipe segmentation failed (%s); falling back to OpenCV.", exc)
                return self._segment_opencv(image, bounding_box)

        return self._segment_opencv(image, bounding_box)

    def _segment_mediapipe(self, image: np.ndarray) -> SegmentationResult:
        """Execute MediaPipe selfie segmentation."""
        import mediapipe as mp

        h, w = image.shape[:2]
        bgr = image if len(image.shape) == 3 and image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._segmenter.segment(mp_image)

        if not result.confidence_masks:
            return self._segment_opencv(image)

        conf_mask = result.confidence_masks[0].numpy_view()
        if conf_mask.ndim == 3:
            conf_mask = conf_mask.squeeze(-1) if conf_mask.shape[-1] == 1 else conf_mask[:, :, 0]
        if conf_mask.shape[:2] != (h, w):
            conf_mask = cv2.resize(conf_mask, (w, h))

        alpha_mask = np.clip(conf_mask, 0.0, 1.0).astype(np.float32)
        binary_mask = (alpha_mask > 0.5).astype(np.uint8) * 255

        coverage = float(np.count_nonzero(binary_mask) / (h * w))
        has_person = coverage > 0.02

        fg_rgba = self.isolate_foreground(bgr, alpha_mask)
        bg_isolated = self.isolate_background(bgr, binary_mask)

        metadata = SegmentationMetadata(
            width=w,
            height=h,
            person_coverage_ratio=round(coverage, 4),
            has_person=has_person,
            backend="mediapipe",
        )

        return SegmentationResult(
            metadata=metadata,
            binary_mask=binary_mask,
            alpha_mask=alpha_mask,
            foreground_rgba=fg_rgba,
            background_isolated=bg_isolated,
        )

    def _segment_opencv(
        self,
        image: np.ndarray,
        bounding_box: tuple[float, float, float, float] | None = None,
    ) -> SegmentationResult:
        """OpenCV contour, GrabCut, and morphological person segmentation fallback."""
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return self._build_empty_result(0, 0)

        bgr = image if len(image.shape) == 3 and image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 1. Check if blank or solid image
        if float(np.std(gray)) < 5.0:
            return self._build_empty_result(w, h)

        # 2. Determine initial person region of interest (ROI)
        if bounding_box is not None:
            xmin, ymin, xmax, ymax = bounding_box
            rx = int(np.clip(xmin * w, 0, w - 1))
            ry = int(np.clip(ymin * h, 0, h - 1))
            rw = int(np.clip((xmax - xmin) * w, 1, w - rx))
            rh = int(np.clip((ymax - ymin) * h, 1, h - ry))
        else:
            # Detect candidate person via Otsu threshold + skin color
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if cv2.countNonZero(otsu) > (h * w * 0.75):
                otsu = cv2.bitwise_not(otsu)

            contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > (h * w * 0.02):
                    rx, ry, rw, rh = cv2.boundingRect(largest)
                else:
                    rx, ry, rw, rh = int(w * 0.15), int(h * 0.05), int(w * 0.70), int(h * 0.90)
            else:
                rx, ry, rw, rh = int(w * 0.15), int(h * 0.05), int(w * 0.70), int(h * 0.90)

        # Clamp ROI
        rx = max(1, min(rx, w - 2))
        ry = max(1, min(ry, h - 2))
        rw = max(10, min(rw, w - rx - 1))
        rh = max(10, min(rh, h - ry - 1))

        # 3. Fast GrabCut or threshold refinement
        mask_grab = np.zeros((h, w), dtype=np.uint8)
        mask_grab.fill(cv2.GC_BGD)
        # Probable foreground inside rect
        mask_grab[ry : ry + rh, rx : rx + rw] = cv2.GC_PR_FGD
        # Certain foreground in center of ROI
        center_y = ry + int(rh * 0.2)
        center_h = int(rh * 0.6)
        center_x = rx + int(rw * 0.25)
        center_w = int(rw * 0.5)
        mask_grab[center_y : center_y + center_h, center_x : center_x + center_w] = cv2.GC_FGD

        try:
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            rect = (rx, ry, rw, rh)
            cv2.grabCut(bgr, mask_grab, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_MASK)
            binary_mask = np.where((mask_grab == cv2.GC_FGD) | (mask_grab == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except Exception:
            # Fallback to thresholding if GrabCut fails on uniform regions
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            binary_mask[ry : ry + rh, rx : rx + rw] = 255

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 4. Generate antialiased alpha matte [0.0, 1.0] with Gaussian feathering on borders
        blurred_mask = cv2.GaussianBlur(binary_mask.astype(np.float32), (7, 7), 0)
        alpha_mask = np.clip(blurred_mask / 255.0, 0.0, 1.0)

        coverage = float(np.count_nonzero(binary_mask) / (h * w))
        has_person = coverage > 0.015

        fg_rgba = self.isolate_foreground(bgr, alpha_mask)
        bg_isolated = self.isolate_background(bgr, binary_mask)

        metadata = SegmentationMetadata(
            width=w,
            height=h,
            person_coverage_ratio=round(coverage, 4),
            has_person=has_person,
            backend="opencv_fallback",
        )

        return SegmentationResult(
            metadata=metadata,
            binary_mask=binary_mask,
            alpha_mask=alpha_mask,
            foreground_rgba=fg_rgba,
            background_isolated=bg_isolated,
        )

    def _build_empty_result(self, width: int, height: int) -> SegmentationResult:
        """Return zero-coverage segmentation result for empty/blank input."""
        bin_m = np.zeros((height, width), dtype=np.uint8) if width > 0 and height > 0 else np.zeros((1, 1), dtype=np.uint8)
        alpha_m = np.zeros((height, width), dtype=np.float32) if width > 0 and height > 0 else np.zeros((1, 1), dtype=np.float32)
        return SegmentationResult(
            metadata=SegmentationMetadata(
                width=width,
                height=height,
                person_coverage_ratio=0.0,
                has_person=False,
                backend=self.backend,
            ),
            binary_mask=bin_m,
            alpha_mask=alpha_m,
            foreground_rgba=None,
            background_isolated=None,
        )

    @classmethod
    def isolate_foreground(cls, image_bgr: np.ndarray, alpha_mask: np.ndarray) -> np.ndarray:
        """Extract foreground person as an RGBA image where alpha channel is the segmentation mask."""
        h, w = image_bgr.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = image_bgr

        if alpha_mask.dtype == np.float32 or alpha_mask.dtype == np.float64:
            rgba[:, :, 3] = (np.clip(alpha_mask, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            rgba[:, :, 3] = alpha_mask.astype(np.uint8)

        return rgba

    @classmethod
    def composite_background(
        cls,
        foreground_rgba: np.ndarray,
        background_bgr: np.ndarray,
    ) -> np.ndarray:
        """Blend foreground person onto a new background using alpha channel."""
        h, w = foreground_rgba.shape[:2]
        bg = cv2.resize(background_bgr, (w, h))

        fg_bgr = foreground_rgba[:, :, :3].astype(np.float32)
        alpha = (foreground_rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]

        composite = (fg_bgr * alpha + bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        return composite

    @classmethod
    def isolate_background(
        cls,
        image_bgr: np.ndarray,
        binary_mask: np.ndarray,
    ) -> np.ndarray:
        """Retain original background while isolating and masking out the foreground person.

        Uses inpainting to seamlessly fill the space where the person stood.
        """
        h, w = image_bgr.shape[:2]
        if binary_mask.shape[:2] != (h, w):
            binary_mask = cv2.resize(binary_mask, (w, h))

        # Dilate mask slightly to cover edge artifacts
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(binary_mask, kernel, iterations=2)

        # Inpaint foreground person cutout to reconstruct background
        inpainted = cv2.inpaint(image_bgr, dilated, 5, cv2.INPAINT_TELEA)
        return inpainted

