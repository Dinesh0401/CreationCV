"""Unit tests for Phase 2 capture module: camera, audio, and synchronizer."""

from __future__ import annotations

import io
import math
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.capture.audio import (
    AudioAnalyzer,
    AudioFormat,
    AudioStreamInfo,
    AudioValidationResult,
    WaveformSummary,
)
from backend.capture.camera import (
    CameraInfo,
    CameraManager,
    FrameInfo,
    Resolution,
    ResolutionValidationResult,
    VideoFrameExtractor,
    VideoMetadata,
    VideoVerificationResult,
    verify_video,
)
from backend.capture.synchronizer import (
    FrameAudioWindow,
    MediaSynchronizer,
    SyncReport,
    SyncStatus,
)


# Helper: create synthetic test video file
def _create_test_video(path: Path, frame_count: int = 15, fps: float = 30.0, width: int = 320, height: int = 240) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(frame_count):
        # Create distinct frames with moving rectangle
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x = int((i / max(1, frame_count - 1)) * (width - 40))
        cv2.rectangle(frame, (x, 50), (x + 30, 100), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()


# Helper: create synthetic test WAV bytes
def _create_test_wav_bytes(duration_sec: float = 0.5, sample_rate: int = 44100, freq: float = 440.0) -> bytes:
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    samples = (np.sin(2 * np.pi * freq * t) * 16384).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Camera Module Tests
# ---------------------------------------------------------------------------

def test_resolution_model():
    """Test Resolution model helper properties."""
    res = Resolution(width=1920, height=1080)
    assert res.as_tuple == (1920, 1080)
    assert res.aspect_ratio_str == "16:9"

    res_vga = Resolution(width=640, height=480)
    assert res_vga.aspect_ratio_str == "4:3"


def test_camera_manager_enumeration():
    """Camera enumeration should complete without exceptions even in headless environments."""
    manager = CameraManager()
    devices = manager.enumerate_cameras(max_devices=2, check_resolutions=False)
    assert isinstance(devices, list)
    for dev in devices:
        assert isinstance(dev, CameraInfo)
        assert dev.index >= 0
        assert dev.is_available is True


def test_camera_resolution_validation_invalid_index():
    """Validating non-existent camera should return supported=False."""
    manager = CameraManager()
    result = manager.validate_resolution(camera_index=999, target_resolution=(1280, 720))
    assert isinstance(result, ResolutionValidationResult)
    assert result.camera_index == 999
    assert result.supported is False
    assert result.is_exact_match is False


def test_video_frame_extractor_and_metadata(tmp_path: Path):
    """Test extracting video metadata, individual frames, and timestamps."""
    video_path = tmp_path / "test_sample.mp4"
    _create_test_video(video_path, frame_count=10, fps=25.0, width=320, height=240)

    extractor = VideoFrameExtractor(video_path)
    try:
        meta = extractor.get_metadata()
        assert meta.is_valid is True
        assert meta.width == 320
        assert meta.height == 240
        assert meta.fps == pytest.approx(25.0, abs=1.0)
        assert meta.frame_count == 10
        assert meta.duration_seconds == pytest.approx(0.4, abs=0.05)

        # Test extract_all
        frames = extractor.extract_all()
        assert len(frames) == 10
        f0 = frames[0]
        assert isinstance(f0, FrameInfo)
        assert f0.frame_index == 0
        assert f0.timestamp_sec == 0.0
        assert f0.width == 320
        assert f0.height == 240

        # Frame conversions
        rgb = f0.to_rgb()
        assert rgb.shape == (240, 320, 3)
        bgr = f0.to_bgr()
        assert bgr.shape == (240, 320, 3)

        # Test extract_frame_at_time
        f_mid = extractor.extract_frame_at_time(0.2)
        assert f_mid is not None
        assert f_mid.timestamp_sec >= 0.0
    finally:
        extractor.cleanup()


def test_video_frame_extractor_from_bytes(tmp_path: Path):
    """Test extractor initialized directly with raw bytes."""
    video_path = tmp_path / "bytes_src.mp4"
    _create_test_video(video_path, frame_count=6, fps=30.0, width=160, height=120)
    raw_bytes = video_path.read_bytes()

    with VideoFrameExtractor(raw_bytes) as extractor:
        meta = extractor.get_metadata()
        assert meta.is_valid is True
        assert meta.width == 160
        assert meta.height == 120
        frames = extractor.extract_all(max_frames=3)
        assert len(frames) == 3


def test_verify_video_matches_and_mismatches(tmp_path: Path):
    """Test verify_video with matching and failing targets."""
    video_path = tmp_path / "verify.mp4"
    _create_test_video(video_path, frame_count=15, fps=30.0, width=640, height=480)

    # 1. Matching verification
    res_ok = verify_video(video_path, expected_fps=30.0, expected_resolution=(640, 480))
    assert res_ok.is_valid is True
    assert res_ok.fps_match is True
    assert res_ok.resolution_match is True
    assert len(res_ok.errors) == 0

    # 2. Mismatched resolution
    res_bad_res = verify_video(video_path, expected_resolution=(1280, 720))
    assert res_bad_res.is_valid is False
    assert res_bad_res.resolution_match is False
    assert any("Resolution mismatch" in err for err in res_bad_res.errors)

    # 3. Mismatched FPS
    res_bad_fps = verify_video(video_path, expected_fps=60.0)
    assert res_bad_fps.is_valid is False
    assert res_bad_fps.fps_match is False
    assert any("FPS mismatch" in err for err in res_bad_fps.errors)


# ---------------------------------------------------------------------------
# Audio Module Tests
# ---------------------------------------------------------------------------

def test_audio_format_detection():
    """Verify magic bytes detection for WAV, MP3, AAC, WebM, and unknown."""
    # WAV
    wav_hdr = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
    assert AudioAnalyzer.detect_format(wav_hdr) == AudioFormat.WAV

    # MP3 (ID3 tag)
    mp3_id3 = b"ID3\x04\x00\x00\x00\x00\x00\x00"
    assert AudioAnalyzer.detect_format(mp3_id3) == AudioFormat.MP3

    # MP3 (MPEG-1 Layer III frame sync: 0xFF, 0xFB)
    mp3_sync = b"\xFF\xFB\x90\x64"
    assert AudioAnalyzer.detect_format(mp3_sync) == AudioFormat.MP3

    # AAC (ADTS header 0xFF, 0xF1)
    aac_adts = b"\xFF\xF1\x50\x80"
    assert AudioAnalyzer.detect_format(aac_adts) == AudioFormat.AAC

    # AAC (M4A container)
    m4a_hdr = b"\x00\x00\x00\x20ftypM4A \x00\x00\x00\x00"
    assert AudioAnalyzer.detect_format(m4a_hdr) == AudioFormat.AAC

    # WebM
    webm_hdr = b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81\x01\x42\xf2\x81\x04\x42\xf3\x81\x08"
    assert AudioAnalyzer.detect_format(webm_hdr) == AudioFormat.WEBM

    # Unknown
    assert AudioAnalyzer.detect_format(b"RANDOM_BYTES_DATA") == AudioFormat.UNKNOWN


def test_audio_wav_reading_and_waveform():
    """Test reading PCM WAV data and computing RMS, peak, dBFS, and envelope."""
    wav_data = _create_test_wav_bytes(duration_sec=0.25, sample_rate=44100, freq=440.0)
    samples, sample_rate, channels = AudioAnalyzer.read_wav(wav_data)

    assert sample_rate == 44100
    assert channels == 1
    assert len(samples) == int(44100 * 0.25)
    assert np.max(np.abs(samples)) <= 1.0

    summary = AudioAnalyzer.compute_waveform_summary(samples, envelope_points=25)
    assert isinstance(summary, WaveformSummary)
    assert summary.peak_level > 0.4
    assert summary.rms_level > 0.2
    assert -30.0 < summary.dbfs < -3.0
    assert summary.is_silent is False
    assert len(summary.envelope) == 25


def test_audio_validation_pipeline(tmp_path: Path):
    """Test complete AudioAnalyzer.validate_audio flow on file and bytes."""
    wav_bytes = _create_test_wav_bytes(duration_sec=0.5, sample_rate=44100)
    wav_file = tmp_path / "valid.wav"
    wav_file.write_bytes(wav_bytes)

    # Valid check on file
    result = AudioAnalyzer.validate_audio(wav_file, expected_sample_rate=44100)
    assert result.is_valid is True
    assert result.format == AudioFormat.WAV
    assert result.stream_info is not None
    assert result.stream_info.sample_rate == 44100
    assert result.stream_info.duration_seconds == pytest.approx(0.5, abs=0.01)
    assert result.waveform is not None
    assert len(result.errors) == 0

    # Sample rate mismatch check
    res_mismatch = AudioAnalyzer.validate_audio(wav_bytes, expected_sample_rate=16000)
    assert res_mismatch.is_valid is False
    assert any("Sample rate mismatch" in err for err in res_mismatch.errors)

    # Empty buffer check
    res_empty = AudioAnalyzer.validate_audio(b"")
    assert res_empty.is_valid is False
    assert any("empty" in err for err in res_empty.errors)


# ---------------------------------------------------------------------------
# Synchronizer Module Tests
# ---------------------------------------------------------------------------

def test_synchronizer_alignment():
    """Test frame-to-audio sample range mapping."""
    syncer = MediaSynchronizer()
    fps = 30.0
    total_frames = 15
    sample_rate = 48000

    # 15 frames at 30 fps = 0.5s = 24000 audio samples
    synthetic_audio = np.sin(np.linspace(0, 100, 24000, dtype=np.float32))
    alignments = syncer.align_frames_to_audio(
        video_fps=fps,
        total_frames=total_frames,
        audio_sample_rate=sample_rate,
        audio_samples=synthetic_audio,
    )

    assert len(alignments) == total_frames
    for i, w in enumerate(alignments):
        assert isinstance(w, FrameAudioWindow)
        assert w.frame_index == i
        assert w.audio_start_sample <= w.audio_end_sample
        assert w.is_aligned is True
        assert w.audio_rms is not None

    # Verify first window bounds
    assert alignments[0].audio_start_sample == 0
    assert alignments[0].audio_end_sample == int(round(1.0 / 30.0 * 48000))


def test_synchronizer_status_checks():
    """Verify in-sync, video-ahead, audio-ahead, and duration-mismatch detection."""
    syncer = MediaSynchronizer(sync_tolerance_ms=50.0, mismatch_threshold_ms=500.0)

    # 1. Perfectly in sync (< 50ms)
    rep_sync = syncer.check_synchronization(video_duration_sec=1.0, audio_duration_sec=1.02)
    assert rep_sync.status == SyncStatus.IN_SYNC
    assert rep_sync.is_synchronized is True
    assert rep_sync.max_drift_ms == pytest.approx(20.0, abs=1.0)

    # 2. Video ahead (video longer by 150ms)
    rep_vid = syncer.check_synchronization(video_duration_sec=1.15, audio_duration_sec=1.00)
    assert rep_vid.status == SyncStatus.VIDEO_AHEAD
    assert rep_vid.is_synchronized is False

    # 3. Audio ahead (audio longer by 150ms)
    rep_aud = syncer.check_synchronization(video_duration_sec=1.00, audio_duration_sec=1.15)
    assert rep_aud.status == SyncStatus.AUDIO_AHEAD
    assert rep_aud.is_synchronized is False

    # 4. Severe duration mismatch (> 500ms)
    rep_mismatch = syncer.check_synchronization(video_duration_sec=3.0, audio_duration_sec=1.5)
    assert rep_mismatch.status == SyncStatus.DURATION_MISMATCH
    assert rep_mismatch.is_synchronized is False


def test_synchronizer_slice_audio():
    """Test extracting exact audio slice for frame index."""
    syncer = MediaSynchronizer()
    sample_rate = 1000
    fps = 10.0  # 100 samples per frame
    samples = np.arange(1000, dtype=np.float32)

    slice_f0 = syncer.slice_audio_for_frame(0, video_fps=fps, audio_samples=samples, audio_sample_rate=sample_rate)
    assert len(slice_f0) == 100
    assert slice_f0[0] == 0
    assert slice_f0[-1] == 99

    slice_f2 = syncer.slice_audio_for_frame(2, video_fps=fps, audio_samples=samples, audio_sample_rate=sample_rate)
    assert len(slice_f2) == 100
    assert slice_f2[0] == 200
    assert slice_f2[-1] == 299

    # Out of range frame
    slice_oob = syncer.slice_audio_for_frame(50, video_fps=fps, audio_samples=samples, audio_sample_rate=sample_rate)
    assert len(slice_oob) == 0

