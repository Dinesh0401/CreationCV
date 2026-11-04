"""High-Quality 1080x1920 (9:16) MP4 H.264 Video Export Pipeline for ReelMotion AI.

Provides:
- Aspect ratio preservation with intelligent crop/padding (center crop, blurred background fit, solid pad).
- 1080x1920 MP4 video encoding with H.264 codec compatibility.
- Seamless integration with AudioMixer for original audio preservation.
- Granular progress tracking callbacks for job queue integration.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import cv2
import numpy as np

from backend.rendering.audio_mixer import AudioMixer

logger = logging.getLogger(__name__)

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920


class AspectRatioMode(str, Enum):
    """Aspect ratio adaptation mode for 9:16 (1080x1920) reels export."""

    CENTER_CROP = "center_crop"
    FIT_BLUR_PAD = "fit_blur_pad"
    FIT_PAD = "fit_pad"


class VideoExporter:
    """Encodes and exports video frames to standard 1080x1920 MP4 format."""

    def __init__(
        self,
        target_width: int = TARGET_WIDTH,
        target_height: int = TARGET_HEIGHT,
        fps: int = 30,
        mode: AspectRatioMode = AspectRatioMode.FIT_BLUR_PAD,
        audio_mixer: Optional[AudioMixer] = None,
    ) -> None:
        self.target_width = target_width
        self.target_height = target_height
        self.fps = fps
        self.mode = mode
        self.audio_mixer = audio_mixer or AudioMixer()

    def format_frame_9_16(
        self,
        frame: np.ndarray,
        mode: Optional[AspectRatioMode] = None,
    ) -> np.ndarray:
        """Format an input frame (any aspect ratio) into 1080x1920 (9:16)."""
        target_w, target_h = self.target_width, self.target_height
        fh, fw = frame.shape[:2]

        if (fw, fh) == (target_w, target_h):
            return frame

        use_mode = mode or self.mode

        if use_mode == AspectRatioMode.CENTER_CROP:
            # Scale so image covers the canvas completely
            scale = max(target_w / fw, target_h / fh)
            scaled_w = int(round(fw * scale))
            scaled_h = int(round(fh * scale))
            scaled = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)

            # Center crop
            x1 = (scaled_w - target_w) // 2
            y1 = (scaled_h - target_h) // 2
            return scaled[y1 : y1 + target_h, x1 : x1 + target_w]

        elif use_mode == AspectRatioMode.FIT_BLUR_PAD:
            # Fit inside canvas
            scale = min(target_w / fw, target_h / fh)
            scaled_w = max(1, int(round(fw * scale)))
            scaled_h = max(1, int(round(fh * scale)))
            fg = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)

            # Background: scaled to fill + heavy blur + darken
            bg_scale = max(target_w / fw, target_h / fh)
            bg_w = int(round(fw * bg_scale))
            bg_h = int(round(fh * bg_scale))
            bg = cv2.resize(frame, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)

            bx1 = (bg_w - target_w) // 2
            by1 = (bg_h - target_h) // 2
            bg_cropped = bg[by1 : by1 + target_h, bx1 : bx1 + target_w]
            bg_blurred = cv2.GaussianBlur(bg_cropped, (75, 75), 25.0)
            # Darken background slightly to emphasize foreground subject without large float32 allocations
            bg_darkened = cv2.convertScaleAbs(bg_blurred, alpha=0.45)

            # Paste foreground in center
            start_x = (target_w - scaled_w) // 2
            start_y = (target_h - scaled_h) // 2
            bg_darkened[start_y : start_y + scaled_h, start_x : start_x + scaled_w] = fg
            return bg_darkened

        else:  # FIT_PAD
            scale = min(target_w / fw, target_h / fh)
            scaled_w = max(1, int(round(fw * scale)))
            scaled_h = max(1, int(round(fh * scale)))
            fg = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)

            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            start_x = (target_w - scaled_w) // 2
            start_y = (target_h - scaled_h) // 2
            canvas[start_y : start_y + scaled_h, start_x : start_x + scaled_w] = fg
            return canvas

    def export(
        self,
        frames: Sequence[np.ndarray],
        output_path: Union[str, Path],
        audio_source: Optional[Union[str, Path, np.ndarray]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> dict[str, Any]:
        """Export frames to a 1080x1920 MP4 file with synchronized audio.

        Args:
            frames: Sequence of RGB numpy images.
            output_path: Destination path for the MP4 video.
            audio_source: Optional audio path or waveform.
            progress_callback: Optional status callback (percent: float, message: str).

        Returns:
            Export metadata dictionary.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        total_frames = len(frames)
        if total_frames == 0:
            raise ValueError("No frames provided for export")

        duration_sec = total_frames / float(self.fps)

        if progress_callback:
            progress_callback(5.0, "Formatting frames to 1080x1920 (9:16)...")

        # Step 1: Format frames to 9:16 portrait
        formatted_frames: list[np.ndarray] = []
        for i, frame in enumerate(frames):
            formatted_frames.append(self.format_frame_9_16(frame))
            if progress_callback and (i + 1) % 15 == 0:
                pct = 5.0 + (float(i + 1) / total_frames) * 35.0
                progress_callback(pct, f"Formatted frame {i + 1}/{total_frames} for 9:16 reels")

        # Step 2: Write temporary raw video via OpenCV VideoWriter
        temp_video_path = out_p.with_suffix(".temp_raw.mp4")
        if progress_callback:
            progress_callback(45.0, "Encoding video stream...")

        # FourCC codecs: try avc1, mp4v
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(temp_video_path),
            fourcc,
            float(self.fps),
            (self.target_width, self.target_height),
        )

        if not writer.isOpened():
            # Fallback codec
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(
                str(temp_video_path),
                fourcc,
                float(self.fps),
                (self.target_width, self.target_height),
            )

        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {temp_video_path}")

        try:
            for i, f in enumerate(formatted_frames):
                # OpenCV expects BGR
                bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
                if progress_callback and (i + 1) % 15 == 0:
                    pct = 45.0 + (float(i + 1) / total_frames) * 35.0
                    progress_callback(pct, f"Encoded video frame {i + 1}/{total_frames}")
        finally:
            writer.release()

        # Step 3: Audio mixing and multiplexing
        if progress_callback:
            progress_callback(85.0, "Multiplexing synchronized audio track...")

        if audio_source is not None:
            self.audio_mixer.mux(
                video_path=temp_video_path,
                audio_source=audio_source,
                output_path=out_p,
                target_duration_sec=duration_sec,
            )
        else:
            # No audio provided, move temp video to target
            if temp_video_path.exists():
                shutil.move(str(temp_video_path), str(out_p))

        # Cleanup temp file if it still exists
        temp_video_path.unlink(missing_ok=True)

        file_size_mb = out_p.stat().st_size / (1024 * 1024) if out_p.exists() else 0.0

        if progress_callback:
            progress_callback(100.0, "Video export complete!")

        metadata = {
            "output_path": str(out_p),
            "width": self.target_width,
            "height": self.target_height,
            "aspect_ratio": "9:16",
            "fps": self.fps,
            "frame_count": total_frames,
            "duration_seconds": round(duration_sec, 2),
            "file_size_mb": round(file_size_mb, 2),
            "format": "mp4",
            "mode": self.mode.value,
        }
        logger.info("Export completed successfully: %s (%dx%d, %.1fs, %.2fMB)",
                    out_p.name, self.target_width, self.target_height, duration_sec, file_size_mb)
        return metadata
