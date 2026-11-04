"""Video and audio synchronization module for ReelMotion AI.

Aligns video frame timestamps with audio track samples, calculates drift,
and checks synchronization compliance for lip sync and video generation.
"""

from __future__ import annotations

import logging
import math
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SyncStatus(str, Enum):
    """Synchronization status state."""

    IN_SYNC = "in_sync"
    AUDIO_AHEAD = "audio_ahead"
    VIDEO_AHEAD = "video_ahead"
    DESYNCHRONIZED = "desynchronized"
    DURATION_MISMATCH = "duration_mismatch"


class FrameAudioWindow(BaseModel):
    """Alignment between an individual video frame and its corresponding audio slice."""

    frame_index: int
    frame_time_sec: float
    audio_start_sec: float
    audio_end_sec: float
    audio_start_sample: int
    audio_end_sample: int
    audio_rms: float | None = None
    is_aligned: bool = True


class SyncReport(BaseModel):
    """Comprehensive video + audio synchronization assessment."""

    status: SyncStatus
    is_synchronized: bool
    video_duration_sec: float
    audio_duration_sec: float
    duration_diff_sec: float
    duration_diff_ms: float
    max_drift_ms: float = 0.0
    total_frames: int
    video_fps: float
    audio_sample_rate: int
    alignments: list[FrameAudioWindow] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class MediaSynchronizer:
    """Calculates frame-audio alignment windows and detects AV synchronization drift."""

    def __init__(
        self,
        sync_tolerance_ms: float = 50.0,
        mismatch_threshold_ms: float = 500.0,
    ) -> None:
        self.sync_tolerance_ms = sync_tolerance_ms
        self.mismatch_threshold_ms = mismatch_threshold_ms

    def align_frames_to_audio(
        self,
        video_fps: float,
        total_frames: int,
        audio_sample_rate: int,
        audio_samples: np.ndarray | None = None,
        offset_ms: float = 0.0,
    ) -> list[FrameAudioWindow]:
        """Map every video frame index to its corresponding audio sample window.

        Args:
            video_fps: Video frames per second (e.g. 30.0).
            total_frames: Total number of video frames.
            audio_sample_rate: Audio sampling frequency in Hz (e.g. 44100).
            audio_samples: Optional audio samples array to calculate per-frame RMS.
            offset_ms: Optional audio start offset in milliseconds (positive = audio delayed).

        Returns:
            List of FrameAudioWindow objects.
        """
        if video_fps <= 0:
            raise ValueError(f"video_fps must be positive, got {video_fps}")
        if audio_sample_rate <= 0:
            raise ValueError(f"audio_sample_rate must be positive, got {audio_sample_rate}")

        offset_sec = offset_ms / 1000.0
        frame_duration = 1.0 / video_fps
        total_audio_samples = len(audio_samples) if audio_samples is not None else int(
            (total_frames * frame_duration + abs(offset_sec)) * audio_sample_rate
        )

        alignments: list[FrameAudioWindow] = []

        for idx in range(total_frames):
            frame_time = idx * frame_duration
            audio_start_sec = max(0.0, frame_time - offset_sec)
            audio_end_sec = max(0.0, (idx + 1) * frame_duration - offset_sec)

            start_sample = int(round(audio_start_sec * audio_sample_rate))
            end_sample = int(round(audio_end_sec * audio_sample_rate))

            # Clamp to available audio bounds
            clamped_start = max(0, min(start_sample, total_audio_samples))
            clamped_end = max(clamped_start, min(end_sample, total_audio_samples))

            rms_val = None
            if audio_samples is not None and clamped_end > clamped_start:
                slice_data = audio_samples[clamped_start:clamped_end]
                if slice_data.size > 0:
                    rms_val = round(float(np.sqrt(np.mean(slice_data**2))), 4)

            is_aligned = (clamped_end - clamped_start) > 0

            alignments.append(
                FrameAudioWindow(
                    frame_index=idx,
                    frame_time_sec=round(frame_time, 4),
                    audio_start_sec=round(audio_start_sec, 4),
                    audio_end_sec=round(audio_end_sec, 4),
                    audio_start_sample=clamped_start,
                    audio_end_sample=clamped_end,
                    audio_rms=rms_val,
                    is_aligned=is_aligned,
                )
            )

        return alignments

    def check_synchronization(
        self,
        video_duration_sec: float,
        audio_duration_sec: float,
        video_fps: float = 30.0,
        audio_sample_rate: int = 44100,
        offset_ms: float = 0.0,
        audio_samples: np.ndarray | None = None,
    ) -> SyncReport:
        """Evaluate overall synchronization between video and audio tracks."""
        diff_sec = video_duration_sec - audio_duration_sec
        diff_ms = diff_sec * 1000.0
        abs_diff_ms = abs(diff_ms)

        total_frames = max(1, int(round(video_duration_sec * video_fps)))
        recommendations: list[str] = []

        if abs_diff_ms > self.mismatch_threshold_ms:
            status = SyncStatus.DURATION_MISMATCH
            is_sync = False
            recommendations.append(
                f"Severe duration disparity: {abs_diff_ms:.1f}ms difference. "
                "Trim the longer track or pad the shorter track before processing."
            )
        elif abs_diff_ms > self.sync_tolerance_ms:
            if diff_ms > 0:
                status = SyncStatus.VIDEO_AHEAD
                recommendations.append(
                    f"Video track is {abs_diff_ms:.1f}ms longer than audio. "
                    "Apply audio padding or trim trailing video frames."
                )
            else:
                status = SyncStatus.AUDIO_AHEAD
                recommendations.append(
                    f"Audio track is {abs_diff_ms:.1f}ms longer than video. "
                    "Apply video loop/hold or trim trailing audio samples."
                )
            is_sync = False
        else:
            status = SyncStatus.IN_SYNC
            is_sync = True
            recommendations.append("Video and audio are synchronized within acceptable threshold.")

        alignments = self.align_frames_to_audio(
            video_fps=video_fps,
            total_frames=total_frames,
            audio_sample_rate=audio_sample_rate,
            audio_samples=audio_samples,
            offset_ms=offset_ms,
        )

        return SyncReport(
            status=status,
            is_synchronized=is_sync,
            video_duration_sec=round(video_duration_sec, 4),
            audio_duration_sec=round(audio_duration_sec, 4),
            duration_diff_sec=round(diff_sec, 4),
            duration_diff_ms=round(diff_ms, 2),
            max_drift_ms=round(abs_diff_ms, 2),
            total_frames=total_frames,
            video_fps=video_fps,
            audio_sample_rate=audio_sample_rate,
            alignments=alignments,
            recommendations=recommendations,
        )

    def slice_audio_for_frame(
        self,
        frame_index: int,
        video_fps: float,
        audio_samples: np.ndarray,
        audio_sample_rate: int,
        padding_ms: float = 0.0,
    ) -> np.ndarray:
        """Extract exact audio slice corresponding to a given video frame."""
        frame_duration = 1.0 / video_fps
        pad_sec = padding_ms / 1000.0

        start_sec = max(0.0, frame_index * frame_duration - pad_sec)
        end_sec = (frame_index + 1) * frame_duration + pad_sec

        start_idx = int(round(start_sec * audio_sample_rate))
        end_idx = min(len(audio_samples), int(round(end_sec * audio_sample_rate)))

        if start_idx >= len(audio_samples):
            return np.array([], dtype=audio_samples.dtype)

        return audio_samples[start_idx:end_idx]

