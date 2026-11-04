"""Camera capture and video validation module for ReelMotion AI.

Provides camera enumeration, resolution negotiation, video frame extraction,
and FPS / resolution verification using OpenCV.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# Standard resolutions to test for cameras
STANDARD_RESOLUTIONS: list[tuple[int, int]] = [
    (640, 480),
    (1280, 720),
    (1920, 1080),
]


class Resolution(BaseModel):
    """Image / video frame resolution."""

    width: int
    height: int

    @property
    def as_tuple(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def aspect_ratio_str(self) -> str:
        gcd = np.gcd(self.width, self.height)
        if gcd == 0:
            return "unknown"
        return f"{self.width // gcd}:{self.height // gcd}"


class ResolutionValidationResult(BaseModel):
    """Result of attempting to set a specific camera resolution."""

    camera_index: int
    requested_width: int
    requested_height: int
    actual_width: int
    actual_height: int
    is_exact_match: bool
    supported: bool


class CameraInfo(BaseModel):
    """Information about an enumerated system camera."""

    index: int
    name: str
    is_available: bool
    current_resolution: tuple[int, int] | None = None
    current_fps: float = 0.0
    supported_resolutions: list[tuple[int, int]] = Field(default_factory=list)


class VideoMetadata(BaseModel):
    """Video stream / file technical metadata."""

    source: str | None = None
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    fourcc: str = ""
    is_valid: bool = True
    error_message: str | None = None


class VideoVerificationResult(BaseModel):
    """Verification results comparing video properties against expectations."""

    is_valid: bool
    metadata: VideoMetadata
    fps_match: bool
    resolution_match: bool
    expected_fps: float | None = None
    expected_resolution: tuple[int, int] | None = None
    fps_difference: float = 0.0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FrameInfo(BaseModel):
    """Individual extracted video frame with timing and dimensions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_index: int
    timestamp_sec: float
    timestamp_ms: float
    width: int
    height: int
    image: Any = Field(default=None, exclude=True)

    def to_rgb(self) -> np.ndarray:
        """Return image converted from BGR to RGB."""
        if self.image is None:
            raise ValueError("Frame image is None")
        if len(self.image.shape) == 2:
            return cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def to_bgr(self) -> np.ndarray:
        """Return image in BGR format."""
        if self.image is None:
            raise ValueError("Frame image is None")
        return self.image


class CameraManager:
    """Manages system camera discovery, testing, and configuration."""

    def __init__(self, preferred_resolutions: list[tuple[int, int]] | None = None) -> None:
        self.preferred_resolutions = preferred_resolutions or STANDARD_RESOLUTIONS

    def enumerate_cameras(
        self,
        max_devices: int = 4,
        check_resolutions: bool = True,
    ) -> list[CameraInfo]:
        """Scan system for available video capture devices."""
        cameras: list[CameraInfo] = []

        # Backend flags to try (DirectShow first on Windows, fallback to ANY)
        backend_flags = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]

        for idx in range(max_devices):
            opened = False
            cap = None

            for backend in backend_flags:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                    if cap.isOpened():
                        opened = True
                        break
                    cap.release()
                except Exception as exc:
                    logger.debug("Failed opening camera %d with backend %s: %s", idx, backend, exc)

            if not opened or cap is None:
                continue

            try:
                # Test reading a frame to verify it actually works
                ret, _ = cap.read()
                if not ret:
                    cap.release()
                    continue

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                if fps <= 0 or fps > 240:
                    fps = 30.0  # sensible default when camera driver reports 0

                supported: list[tuple[int, int]] = []
                if check_resolutions:
                    for req_w, req_h in self.preferred_resolutions:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
                        act_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        act_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        if act_w == req_w and act_h == req_h:
                            supported.append((act_w, act_h))
                        elif (act_w, act_h) not in supported and act_w > 0 and act_h > 0:
                            supported.append((act_w, act_h))

                cameras.append(
                    CameraInfo(
                        index=idx,
                        name=f"Camera {idx}",
                        is_available=True,
                        current_resolution=(width, height),
                        current_fps=fps,
                        supported_resolutions=supported,
                    )
                )
            except Exception as exc:
                logger.warning("Error inspecting camera %d: %s", idx, exc)
            finally:
                cap.release()

        return cameras

    def validate_resolution(
        self,
        camera_index: int,
        target_resolution: tuple[int, int],
    ) -> ResolutionValidationResult:
        """Test whether a specific camera index supports a given resolution."""
        req_w, req_h = target_resolution
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return ResolutionValidationResult(
                camera_index=camera_index,
                requested_width=req_w,
                requested_height=req_h,
                actual_width=0,
                actual_height=0,
                is_exact_match=False,
                supported=False,
            )

        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
            act_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            act_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            exact = (act_w == req_w and act_h == req_h)
            return ResolutionValidationResult(
                camera_index=camera_index,
                requested_width=req_w,
                requested_height=req_h,
                actual_width=act_w,
                actual_height=act_h,
                is_exact_match=exact,
                supported=exact or (act_w > 0 and act_h > 0),
            )
        finally:
            cap.release()


class VideoFrameExtractor:
    """Extracts frames, timestamps, and metadata from video files or buffers."""

    def __init__(self, source: str | Path | bytes) -> None:
        self.source = source
        self._temp_path: Path | None = None
        self._video_path: str = ""

        if isinstance(source, bytes):
            # Write to a temporary file for OpenCV to open
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_file.write(source)
            temp_file.close()
            self._temp_path = Path(temp_file.name)
            self._video_path = str(self._temp_path)
        else:
            self._video_path = str(source)

    def __enter__(self) -> VideoFrameExtractor:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """Remove any temporary file created for byte buffers."""
        if self._temp_path and self._temp_path.exists():
            try:
                self._temp_path.unlink()
            except Exception as exc:
                logger.debug("Failed cleaning up temp file %s: %s", self._temp_path, exc)
            self._temp_path = None

    def get_metadata(self) -> VideoMetadata:
        """Inspect video stream properties."""
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            return VideoMetadata(
                source=self._video_path,
                width=0,
                height=0,
                fps=0.0,
                frame_count=0,
                duration_seconds=0.0,
                is_valid=False,
                error_message="Could not open video file or stream",
            )

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_chars = [chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)]
            fourcc_str = "".join(fourcc_chars).strip()

            duration = (frame_count / fps) if (fps > 0 and frame_count > 0) else 0.0

            # Double check if frame_count is reported accurately
            if frame_count <= 0:
                # Count frames manually if header was incomplete
                count = 0
                while True:
                    ret = cap.grab()
                    if not ret:
                        break
                    count += 1
                frame_count = count
                duration = (frame_count / fps) if fps > 0 else 0.0

            return VideoMetadata(
                source=self._video_path,
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
                duration_seconds=round(duration, 4),
                fourcc=fourcc_str,
                is_valid=width > 0 and height > 0 and frame_count > 0,
            )
        finally:
            cap.release()

    def extract_generator(
        self,
        target_fps: float | None = None,
        max_frames: int | None = None,
    ) -> Iterator[FrameInfo]:
        """Yield frames sequentially with calculated timestamps.

        If `target_fps` is lower than the source video fps, downsamples frames evenly.
        """
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            logger.error("Unable to open video: %s", self._video_path)
            return

        try:
            source_fps = float(cap.get(cv2.CAP_PROP_FPS))
            if source_fps <= 0.0:
                source_fps = 30.0

            sample_interval = 1
            if target_fps and target_fps < source_fps:
                sample_interval = max(1, round(source_fps / target_fps))

            frame_idx = 0
            yielded_count = 0

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if frame_idx % sample_interval == 0:
                    time_sec = frame_idx / source_fps
                    time_ms = time_sec * 1000.0
                    h, w = frame.shape[:2]

                    yield FrameInfo(
                        frame_index=frame_idx,
                        timestamp_sec=round(time_sec, 4),
                        timestamp_ms=round(time_ms, 2),
                        width=w,
                        height=h,
                        image=frame,
                    )
                    yielded_count += 1
                    if max_frames and yielded_count >= max_frames:
                        break

                frame_idx += 1
        finally:
            cap.release()

    def extract_all(
        self,
        target_fps: float | None = None,
        max_frames: int | None = None,
    ) -> list[FrameInfo]:
        """Extract all matching frames into a list."""
        return list(self.extract_generator(target_fps=target_fps, max_frames=max_frames))

    def extract_frame_at_time(self, timestamp_seconds: float) -> FrameInfo | None:
        """Extract a single frame closest to the specified timestamp in seconds."""
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            return None

        try:
            source_fps = float(cap.get(cv2.CAP_PROP_FPS))
            if source_fps <= 0:
                source_fps = 30.0

            target_msec = timestamp_seconds * 1000.0
            cap.set(cv2.CAP_PROP_POS_MSEC, target_msec)
            ret, frame = cap.read()
            if not ret or frame is None:
                return None

            actual_pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            h, w = frame.shape[:2]

            return FrameInfo(
                frame_index=max(0, frame_idx),
                timestamp_sec=round(actual_pos_msec / 1000.0, 4),
                timestamp_ms=round(actual_pos_msec, 2),
                width=w,
                height=h,
                image=frame,
            )
        finally:
            cap.release()


def verify_video(
    source: str | Path | bytes,
    expected_fps: float | None = None,
    expected_resolution: tuple[int, int] | None = None,
    fps_tolerance: float = 1.5,
) -> VideoVerificationResult:
    """Validate video file integrity, resolution, and frame rate against expected targets."""
    errors: list[str] = []
    warnings: list[str] = []

    extractor = VideoFrameExtractor(source)
    try:
        metadata = extractor.get_metadata()
    finally:
        extractor.cleanup()

    if not metadata.is_valid:
        errors.append(metadata.error_message or "Video could not be decoded or has 0 frames")
        return VideoVerificationResult(
            is_valid=False,
            metadata=metadata,
            fps_match=False,
            resolution_match=False,
            expected_fps=expected_fps,
            expected_resolution=expected_resolution,
            fps_difference=0.0,
            errors=errors,
            warnings=warnings,
        )

    # Check resolution
    res_match = True
    if expected_resolution is not None:
        exp_w, exp_h = expected_resolution
        if metadata.width != exp_w or metadata.height != exp_h:
            res_match = False
            errors.append(
                f"Resolution mismatch: expected {exp_w}x{exp_h}, got {metadata.width}x{metadata.height}"
            )

    # Check FPS
    fps_match = True
    diff = 0.0
    if expected_fps is not None:
        diff = abs(metadata.fps - expected_fps)
        if diff > fps_tolerance:
            fps_match = False
            errors.append(
                f"FPS mismatch: expected {expected_fps:.2f}, got {metadata.fps:.2f} (tolerance +/-{fps_tolerance})"
            )

    is_valid = len(errors) == 0

    return VideoVerificationResult(
        is_valid=is_valid,
        metadata=metadata,
        fps_match=fps_match,
        resolution_match=res_match,
        expected_fps=expected_fps,
        expected_resolution=expected_resolution,
        fps_difference=round(diff, 2),
        errors=errors,
        warnings=warnings,
    )
