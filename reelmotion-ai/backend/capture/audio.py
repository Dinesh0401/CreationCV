"""Audio stream validation, format detection, and waveform analysis for ReelMotion AI.

Supports format verification (WAV, MP3, AAC, WebM), duration and sample rate inspection,
and waveform/volume level calculation (RMS, Peak, dBFS, and visual envelope).
"""

from __future__ import annotations

import io
import logging
import math
import struct
import wave
from enum import Enum
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AudioFormat(str, Enum):
    """Supported audio format classifications."""

    WAV = "wav"
    MP3 = "mp3"
    AAC = "aac"
    WEBM = "webm"
    UNKNOWN = "unknown"


class AudioStreamInfo(BaseModel):
    """Extracted metadata about an audio stream."""

    format: AudioFormat
    sample_rate: int
    channels: int
    duration_seconds: float
    total_samples: int
    bit_depth: int | None = None
    file_size_bytes: int = 0
    is_supported: bool = True


class WaveformSummary(BaseModel):
    """Volume levels, dynamic range, and downsampled waveform envelope."""

    rms_level: float = 0.0
    peak_level: float = 0.0
    dbfs: float = -100.0
    is_silent: bool = False
    is_clipping: bool = False
    dynamic_range_db: float = 0.0
    envelope: list[float] = Field(default_factory=list)


class AudioValidationResult(BaseModel):
    """Validation report for an audio stream or file."""

    is_valid: bool
    format: AudioFormat
    stream_info: AudioStreamInfo | None = None
    waveform: WaveformSummary | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AudioAnalyzer:
    """Analyzes and validates audio streams and buffers."""

    @staticmethod
    def detect_format(header_bytes: bytes) -> AudioFormat:
        """Identify audio format from initial binary signature bytes."""
        if len(header_bytes) < 4:
            return AudioFormat.UNKNOWN

        # 1. WAV / RIFF
        if len(header_bytes) >= 12 and header_bytes[:4] == b"RIFF" and header_bytes[8:12] == b"WAVE":
            return AudioFormat.WAV

        # 2. MP3
        # ID3v2 header
        if header_bytes[:3] == b"ID3":
            return AudioFormat.MP3
        # MPEG audio sync word (11 bits set: 0xFFE0 to 0xFFFF)
        if len(header_bytes) >= 2:
            b0 = header_bytes[0]
            b1 = header_bytes[1]
            if b0 == 0xFF and (b1 & 0xE0) == 0xE0:
                # Check layer: bits 1-2 of second byte (01 = layer III, 10 = layer II, 11 = layer I)
                layer = (b1 >> 1) & 0x03
                if layer == 0x01:  # Layer 3
                    return AudioFormat.MP3

        # 3. AAC
        # ADTS sync word: 12 bits 0xFFF (0xFF followed by high 4 bits 0xF)
        if len(header_bytes) >= 2:
            b0 = header_bytes[0]
            b1 = header_bytes[1]
            if b0 == 0xFF and (b1 & 0xF0) == 0xF0:
                layer = (b1 >> 1) & 0x03
                if layer == 0x00:  # ADTS header has layer bits set to 0
                    return AudioFormat.AAC
        # MPEG-4 container (ftyp M4A / mp42)
        if len(header_bytes) >= 12 and header_bytes[4:8] == b"ftyp":
            brand = header_bytes[8:12]
            if brand in (b"M4A ", b"mp42", b"isom", b"dash"):
                return AudioFormat.AAC

        # 4. WebM / Matroska
        # EBML ID: 0x1A 0x45 0xDF 0xA3
        if header_bytes[:4] == b"\x1a\x45\xdf\xa3":
            return AudioFormat.WEBM

        return AudioFormat.UNKNOWN

    @classmethod
    def read_wav(cls, source: str | Path | bytes) -> tuple[np.ndarray, int, int]:
        """Read a WAV file or byte buffer into normalized float32 samples [-1.0, 1.0].

        Returns:
            Tuple of (samples_array, sample_rate, channels).
        """
        buf: io.BytesIO | Path
        if isinstance(source, bytes):
            buf = io.BytesIO(source)
        else:
            buf = Path(source)

        with wave.open(str(buf) if isinstance(buf, Path) else buf, "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_data = wf.readframes(n_frames)

        if sample_width == 1:
            # 8-bit unsigned PCM
            samples = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32)
            samples = (samples - 128.0) / 128.0
        elif sample_width == 2:
            # 16-bit signed PCM
            samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 3:
            # 24-bit PCM: convert 3 bytes per sample
            raw_array = np.frombuffer(raw_data, dtype=np.uint8)
            n_samples = len(raw_array) // 3
            # Pad with 0 byte to create 32-bit int
            reshaped = raw_array[: n_samples * 3].reshape(-1, 3)
            padded = np.zeros((n_samples, 4), dtype=np.uint8)
            padded[:, 1:] = reshaped  # little-endian shift
            samples = padded.view(dtype=np.int32).astype(np.float32) / 2147483648.0
        elif sample_width == 4:
            # 32-bit signed PCM or float
            try:
                samples = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
            except Exception:
                samples = np.frombuffer(raw_data, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width} bytes")

        if channels > 1:
            samples = samples.reshape(-1, channels)

        return samples, sample_rate, channels

    @classmethod
    def compute_waveform_summary(
        cls,
        samples: np.ndarray,
        envelope_points: int = 100,
        silence_threshold_dbfs: float = -55.0,
    ) -> WaveformSummary:
        """Calculate RMS, peak volume, dBFS, and downsampled envelope."""
        if samples.size == 0:
            return WaveformSummary(is_silent=True)

        # Work with mono downmix for level calculations
        if samples.ndim > 1:
            mono = np.mean(samples, axis=1)
        else:
            mono = samples

        # Peak amplitude [0.0, 1.0]
        peak = float(np.max(np.abs(mono)))

        # Root Mean Square (RMS)
        mean_square = float(np.mean(mono**2))
        rms = math.sqrt(mean_square) if mean_square > 0.0 else 0.0

        # dBFS
        dbfs = 20.0 * math.log10(rms + 1e-9) if rms > 0.0 else -100.0
        dbfs = max(-100.0, min(0.0, dbfs))

        is_silent = dbfs < silence_threshold_dbfs
        is_clipping = peak >= 0.999

        # Downsampled envelope
        envelope: list[float] = []
        if envelope_points > 0:
            chunk_size = max(1, len(mono) // envelope_points)
            for i in range(envelope_points):
                start = i * chunk_size
                end = min(len(mono), (i + 1) * chunk_size)
                if start >= len(mono):
                    envelope.append(0.0)
                else:
                    chunk = mono[start:end]
                    chunk_rms = float(np.sqrt(np.mean(chunk**2))) if chunk.size > 0 else 0.0
                    envelope.append(round(chunk_rms, 4))

        dynamic_range = 20.0 * math.log10((peak + 1e-9) / (rms + 1e-9)) if rms > 0 else 0.0

        return WaveformSummary(
            rms_level=round(rms, 4),
            peak_level=round(peak, 4),
            dbfs=round(dbfs, 2),
            is_silent=is_silent,
            is_clipping=is_clipping,
            dynamic_range_db=round(dynamic_range, 2),
            envelope=envelope,
        )

    @classmethod
    def validate_audio(
        cls,
        source: str | Path | bytes,
        expected_sample_rate: int | None = None,
        min_duration_sec: float = 0.1,
        max_duration_sec: float = 3600.0,
    ) -> AudioValidationResult:
        """Verify audio format, duration, sample rate, and volume levels."""
        errors: list[str] = []
        warnings: list[str] = []

        raw_bytes: bytes
        file_size: int
        if isinstance(source, bytes):
            raw_bytes = source
            file_size = len(source)
        else:
            path = Path(source)
            if not path.exists():
                return AudioValidationResult(
                    is_valid=False,
                    format=AudioFormat.UNKNOWN,
                    errors=[f"File not found: {path}"],
                )
            file_size = path.stat().st_size
            with open(path, "rb") as f:
                raw_bytes = f.read(1024)  # read header

        if file_size == 0:
            return AudioValidationResult(
                is_valid=False,
                format=AudioFormat.UNKNOWN,
                errors=["Audio file or buffer is empty (0 bytes)"],
            )

        fmt = cls.detect_format(raw_bytes)
        if fmt == AudioFormat.UNKNOWN:
            errors.append("Unrecognized audio format. Supported: WAV, MP3, AAC, WebM")
            return AudioValidationResult(
                is_valid=False,
                format=fmt,
                errors=errors,
            )

        stream_info: AudioStreamInfo | None = None
        waveform: WaveformSummary | None = None

        if fmt == AudioFormat.WAV:
            try:
                full_bytes = raw_bytes if isinstance(source, bytes) else Path(source).read_bytes()
                samples, sample_rate, channels = cls.read_wav(full_bytes)
                duration = len(samples) / sample_rate if sample_rate > 0 else 0.0

                stream_info = AudioStreamInfo(
                    format=fmt,
                    sample_rate=sample_rate,
                    channels=channels,
                    duration_seconds=round(duration, 4),
                    total_samples=len(samples),
                    bit_depth=16,
                    file_size_bytes=file_size,
                    is_supported=True,
                )

                waveform = cls.compute_waveform_summary(samples)

                # Duration check
                if duration < min_duration_sec:
                    errors.append(f"Audio duration ({duration:.2f}s) is shorter than minimum {min_duration_sec}s")
                elif duration > max_duration_sec:
                    warnings.append(f"Audio duration ({duration:.2f}s) exceeds recommended {max_duration_sec}s")

                # Sample rate check
                if expected_sample_rate and sample_rate != expected_sample_rate:
                    errors.append(
                        f"Sample rate mismatch: expected {expected_sample_rate}Hz, got {sample_rate}Hz"
                    )

                if waveform.is_silent:
                    warnings.append("Audio stream contains mostly silence (low RMS volume)")
                if waveform.is_clipping:
                    warnings.append("Audio signal is clipping (peak >= 0 dBFS)")

            except Exception as exc:
                errors.append(f"Failed decoding WAV audio data: {exc}")
        else:
            # For MP3, AAC, WebM format container headers
            stream_info = AudioStreamInfo(
                format=fmt,
                sample_rate=44100,  # container default indicator
                channels=2,
                duration_seconds=0.0,
                total_samples=0,
                file_size_bytes=file_size,
                is_supported=True,
            )

        is_valid = len(errors) == 0

        return AudioValidationResult(
            is_valid=is_valid,
            format=fmt,
            stream_info=stream_info,
            waveform=waveform,
            errors=errors,
            warnings=warnings,
        )
