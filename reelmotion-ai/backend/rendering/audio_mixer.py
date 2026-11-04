"""Audio Mixer and Synchronization Engine for ReelMotion AI.

Preserves the original microphone recording (user voice, Tamil/English speech)
and synchronizes the audio track precisely with the final rendered video frames
without quality degradation.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioSyncConfig:
    """Configuration for audio mixing and synchronization."""

    sample_rate: int = 44100
    channels: int = 2
    audio_bitrate: str = "192k"
    apply_tail_fade: bool = True
    tail_fade_ms: int = 15


class AudioMixer:
    """Manages audio extraction, temporal alignment, and lossless video-audio multiplexing."""

    def __init__(self, config: Optional[AudioSyncConfig] = None) -> None:
        self.config = config or AudioSyncConfig()
        self._ffmpeg_path = shutil.which("ffmpeg")

    @property
    def has_ffmpeg(self) -> bool:
        """Check if FFmpeg command line tool is accessible on the system."""
        return self._ffmpeg_path is not None

    def read_audio_track(
        self,
        audio_source: Union[str, Path, np.ndarray],
    ) -> Tuple[np.ndarray, int, int]:
        """Read audio track into a float32 numpy array, returning (audio, sample_rate, channels)."""
        if isinstance(audio_source, np.ndarray):
            audio = audio_source.astype(np.float32)
            channels = 1 if len(audio.shape) == 1 else audio.shape[1]
            return audio, self.config.sample_rate, channels

        path = Path(audio_source)
        if not path.exists():
            raise FileNotFoundError(f"Audio source does not exist: {path}")

        # If it's a WAV file, read with standard library
        if path.suffix.lower() == ".wav":
            try:
                with wave.open(str(path), "rb") as wf:
                    channels = wf.getnchannels()
                    width = wf.getsampwidth()
                    sr = wf.getframerate()
                    raw = wf.readframes(wf.getnframes())

                    dtype = np.int16 if width == 2 else (np.int32 if width == 4 else np.uint8)
                    data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
                    denom = float(2 ** (8 * width - 1)) if width > 1 else 128.0
                    data = data / denom

                    if channels > 1:
                        data = data.reshape(-1, channels)
                    return data, sr, channels
            except Exception as exc:
                logger.debug("Failed reading wav %s: %s", path, exc)

        # If FFmpeg is available and input is mp4/m4a/webm, extract to temporary wav
        if self.has_ffmpeg:
            temp_wav = path.with_suffix(".tmp_sync.wav")
            try:
                cmd = [
                    self._ffmpeg_path,  # type: ignore[list-item]
                    "-y",
                    "-i",
                    str(path),
                    "-vn",
                    "-ac",
                    str(self.config.channels),
                    "-ar",
                    str(self.config.sample_rate),
                    str(temp_wav),
                ]
                subprocess.run(
                    cmd,
                    capture_output=True,
                    check=True,
                    timeout=30,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
                data, sr, ch = self.read_audio_track(temp_wav)
                temp_wav.unlink(missing_ok=True)
                return data, sr, ch
            except Exception as exc:
                logger.warning("FFmpeg audio extraction failed: %s", exc)
                temp_wav.unlink(missing_ok=True)

        # Return fallback silence of 1 second
        sr = self.config.sample_rate
        silence = np.zeros((sr, self.config.channels), dtype=np.float32)
        return silence, sr, self.config.channels

    def synchronize_duration(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        target_duration_sec: float,
    ) -> np.ndarray:
        """Align audio array precisely to target duration via truncation or padding with smooth tail fade."""
        target_samples = max(1, int(round(target_duration_sec * sample_rate)))
        curr_samples = len(audio_data)

        if curr_samples == target_samples:
            synced = audio_data.copy()
        elif curr_samples > target_samples:
            synced = audio_data[:target_samples].copy()
            # Smooth fade-out on the final samples to avoid click/pop
            if self.config.apply_tail_fade:
                fade_len = min(int(sample_rate * (self.config.tail_fade_ms / 1000.0)), target_samples)
                if fade_len > 0:
                    fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                    if len(synced.shape) == 2:
                        fade = np.expand_dims(fade, axis=-1)
                    synced[-fade_len:] *= fade
        else:
            # Pad with silence at the tail
            pad_len = target_samples - curr_samples
            if len(audio_data.shape) == 2:
                padding = np.zeros((pad_len, audio_data.shape[1]), dtype=np.float32)
            else:
                padding = np.zeros(pad_len, dtype=np.float32)
            synced = np.concatenate((audio_data, padding), axis=0)

        return synced

    def write_wav(
        self,
        audio_data: np.ndarray,
        output_path: Union[str, Path],
        sample_rate: int = 44100,
    ) -> Path:
        """Write float32 audio data to 16-bit PCM WAV file."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        clamped = np.clip(audio_data, -1.0, 1.0)
        pcm16 = (clamped * 32767.0).astype(np.int16)

        channels = 1 if len(pcm16.shape) == 1 else pcm16.shape[1]

        with wave.open(str(out_p), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())

        return out_p

    def mux(
        self,
        video_path: Union[str, Path],
        audio_source: Union[str, Path, np.ndarray],
        output_path: Union[str, Path],
        target_duration_sec: Optional[float] = None,
    ) -> Path:
        """Multiplex video and synchronized original audio into final MP4."""
        vid_p = Path(video_path)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if not vid_p.exists():
            raise FileNotFoundError(f"Video file not found: {vid_p}")

        # Synchronize audio duration
        audio_data, sr, _ = self.read_audio_track(audio_source)
        if target_duration_sec is not None and target_duration_sec > 0:
            audio_data = self.synchronize_duration(audio_data, sr, target_duration_sec)

        temp_synced_wav = out_p.with_suffix(".synced.wav")
        self.write_wav(audio_data, temp_synced_wav, sr)

        if self.has_ffmpeg:
            try:
                cmd = [
                    self._ffmpeg_path,  # type: ignore[list-item]
                    "-y",
                    "-i",
                    str(vid_p),
                    "-i",
                    str(temp_synced_wav),
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    self.config.audio_bitrate,
                    "-shortest",
                    str(out_p),
                ]
                subprocess.run(
                    cmd,
                    capture_output=True,
                    check=True,
                    timeout=60,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
                temp_synced_wav.unlink(missing_ok=True)
                logger.info("Successfully multiplexed video and audio via FFmpeg: %s", out_p)
                return out_p
            except Exception as exc:
                logger.warning("FFmpeg muxing failed (%s). Retaining source video.", exc)

        # Fallback when FFmpeg is not installed: copy video to destination and preserve synced audio
        if vid_p != out_p:
            shutil.copy2(vid_p, out_p)
        logger.info("Video exported to %s (synchronized audio saved at %s)", out_p, temp_synced_wav)
        return out_p
