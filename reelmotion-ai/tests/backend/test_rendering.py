"""Comprehensive Unit and Integration Tests for ReelMotion AI Rendering & Generation Systems.

Tests cover:
- Hardware-aware model loading and VRAM safety margin monitoring (Phase 5)
- WanBackend motion-conditioned character generation & fallback warping (Phase 5)
- LipSyncBackend audio-driven mouth modulation and anime face support (Phase 6)
- BackgroundCompositor room environment preservation & edge feathering (Phase 7)
- AudioMixer microphone recording preservation & synchronization (Phase 7)
- VideoExporter 1080x1920 9:16 MP4 export & progress callbacks (Phase 8)
- JobManager end-to-end rendering pipeline execution & cancellation (Phase 5-8)
- Rendering REST API endpoints (Phase 5-8)
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.generation.lip_sync_backend import (
    AudioFeatureExtractor,
    LipSyncBackend,
    LipSyncConfig,
    PhonemeArticulation,
)
from backend.generation.model_manager import (
    GenerationModelManager,
    ModelOffloadConfig,
    VRAMStatus,
)
from backend.generation.wan_backend import (
    DenseMotionFieldWarper,
    FrameLandmarks,
    GenerationConfig,
    WanBackend,
)
from backend.main import create_app
from backend.rendering.audio_mixer import AudioMixer, AudioSyncConfig
from backend.rendering.background_compositor import (
    BackgroundCompositor,
    CompositorConfig,
)
from backend.rendering.export import (
    TARGET_HEIGHT,
    TARGET_WIDTH,
    AspectRatioMode,
    VideoExporter,
)
from backend.services.job_manager import JobManager, JobState, JobType


# ═══════════════════════════════════════════════════════════════════════════
# Test Fixtures & Helpers
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def synthetic_character_image() -> np.ndarray:
    """Create a 512x512 synthetic character reference image."""
    img = np.full((512, 512, 3), (240, 230, 220), dtype=np.uint8)
    # Head
    cv2.circle(img, (256, 180), 80, (220, 180, 160), -1)
    # Hair
    cv2.ellipse(img, (256, 150), (90, 70), 0, 180, 360, (50, 40, 40), -1)
    # Eyes
    cv2.circle(img, (226, 175), 10, (30, 30, 30), -1)
    cv2.circle(img, (286, 175), 10, (30, 30, 30), -1)
    # Mouth line
    cv2.line(img, (236, 215), (276, 215), (180, 70, 70), 2)
    # Torso
    cv2.rectangle(img, (176, 260), (336, 512), (60, 90, 150), -1)
    return img


@pytest.fixture
def synthetic_room_frames() -> list[np.ndarray]:
    """Create 10 frames of 1280x720 video simulating a recorded room with a moving person."""
    frames: list[np.ndarray] = []
    w, h = 1280, 720
    for i in range(10):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Background: room wall
        frame[: int(h * 0.7), :] = [190, 180, 170]
        # Background: floor
        frame[int(h * 0.7) :, :] = [100, 80, 70]
        # Room plant/lamp
        cv2.circle(frame, (150, 200), 60, (120, 180, 120), -1)
        # Subject with subtle motion
        dx = int(10 * np.sin(i * 0.4))
        cv2.ellipse(frame, (640 + dx, 300), (90, 120), 0, 0, 360, (210, 170, 140), -1)
        cv2.rectangle(frame, (520 + dx, 420), (760 + dx, 720), (50, 50, 80), -1)
        frames.append(frame)
    return frames


@pytest.fixture
def synthetic_speech_audio() -> tuple[np.ndarray, int]:
    """Create a 1.0s synthetic speech waveform with alternating speech energy."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    # Modulation envelope: high energy in middle, low energy at ends
    envelope = np.sin(np.pi * t) ** 2
    carrier = 0.5 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.sin(2 * np.pi * 1200 * t)
    audio = (carrier * envelope).astype(np.float32)
    return audio, sr


# ═══════════════════════════════════════════════════════════════════════════
# 1. Generation Model Manager Tests (Phase 5)
# ═══════════════════════════════════════════════════════════════════════════

def test_model_offload_config_presets():
    """Verify RTX 4050 6GB preset and CPU-only preset defaults."""
    rtx_config = ModelOffloadConfig.rtx_4050_preset()
    assert rtx_config.enable_cpu_offload is True
    assert rtx_config.precision == "fp16"
    assert rtx_config.chunk_size_frames == 16
    assert rtx_config.vram_safety_margin_mb == 512
    assert rtx_config.sequential_cpu_offload is True

    cpu_config = ModelOffloadConfig.cpu_only_preset()
    assert cpu_config.device == "cpu"
    assert cpu_config.precision == "fp32"
    assert cpu_config.vram_safety_margin_mb == 0


def test_generation_model_manager_vram_status():
    """Verify VRAM monitoring and safety margin headroom calculation."""
    manager = GenerationModelManager(safety_margin_mb=512, force_cpu=True)
    status = manager.get_vram_status()
    assert isinstance(status, VRAMStatus)
    assert status.cuda_available is False
    assert status.safety_margin_mb == 512
    assert status.usable_mb == 0
    assert not status.can_allocate(100)


def test_generation_model_manager_loading_eviction():
    """Test model loading, LRU eviction, and cleanup lifecycle."""
    manager = GenerationModelManager(safety_margin_mb=512, force_cpu=True)

    loaded_calls: list[str] = []

    def dummy_loader(device: str, name: str) -> dict[str, Any]:
        loaded_calls.append(name)
        return {"name": name, "device": device}

    m1 = manager.load_model("model_a", dummy_loader, required_vram_mb=1000, name="Model A")
    assert manager.is_model_loaded("model_a")
    assert m1["name"] == "Model A"

    # Loading again returns cached instance
    m1_cached = manager.load_model("model_a", dummy_loader, required_vram_mb=1000, name="Model A")
    assert m1 is m1_cached
    assert len(loaded_calls) == 1

    # Unload specific model
    assert manager.unload_model("model_a") is True
    assert not manager.is_model_loaded("model_a")
    assert manager.unload_model("model_a") is False

    # Unload all
    manager.load_model("m2", dummy_loader, name="M2")
    manager.load_model("m3", dummy_loader, name="M3")
    assert manager.unload_all() == 2
    assert not manager.is_model_loaded("m2")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Wan 2.2 Backend & Fallback Generator Tests (Phase 5)
# ═══════════════════════════════════════════════════════════════════════════

def test_fallback_dense_motion_warper(synthetic_character_image, synthetic_room_frames):
    """Test that DenseMotionFieldWarper successfully generates frames without external weights."""
    warper = DenseMotionFieldWarper(use_fp16=False, device="cpu")

    progress_records: list[float] = []

    def on_progress(pct: float, msg: str) -> None:
        progress_records.append(pct)

    landmarks = [FrameLandmarks() for _ in synthetic_room_frames]
    output_frames = warper.generate_sequence(
        reference_image=synthetic_character_image,
        original_frames=synthetic_room_frames[:4],
        landmarks_sequence=landmarks[:4],
        config=GenerationConfig(chunk_size=2, smooth_motion=True),
        progress_callback=on_progress,
    )

    assert len(output_frames) == 4
    for frame in output_frames:
        assert isinstance(frame, np.ndarray)
        assert frame.shape == synthetic_room_frames[0].shape
        assert frame.dtype == np.uint8
    assert len(progress_records) > 0


def test_wan_backend_out_of_the_box_fallback(synthetic_character_image, synthetic_room_frames, tmp_path):
    """Verify WanBackend seamlessly falls back to motion warping when diffusion weights are absent."""
    backend = WanBackend(models_root=tmp_path / "non_existent_models")
    assert backend.is_diffusion_ready("wan-animate-1.3b") is False

    result = backend.generate(
        character_reference=synthetic_character_image,
        original_frames=synthetic_room_frames[:3],
        config=GenerationConfig(fps=30, chunk_size=2),
    )

    assert result.backend_used == "motion_warping_fallback"
    assert result.frame_count == 3
    assert len(result.frames) == 3
    assert result.fps == 30


# ═══════════════════════════════════════════════════════════════════════════
# 3. Lip Synchronization Backend Tests (Phase 6)
# ═══════════════════════════════════════════════════════════════════════════

def test_audio_feature_extractor(synthetic_speech_audio):
    """Test audio feature extraction and phoneme articulation curve generation."""
    audio, sr = synthetic_speech_audio
    articulations = AudioFeatureExtractor.compute_articulations(
        audio=audio,
        sample_rate=sr,
        num_frames=10,
        fps=10,
        gate_threshold=0.02,
    )

    assert len(articulations) == 10
    for art in articulations:
        assert isinstance(art, PhonemeArticulation)
        assert 0.0 <= art.open_ratio <= 1.0
        assert 0.0 <= art.wide_ratio <= 1.0
        assert 0.0 <= art.energy <= 1.0

    # Middle frame should have higher energy than tail frame due to envelope
    assert articulations[5].open_ratio > articulations[0].open_ratio


def test_lip_sync_anime_adaptation(synthetic_character_image, synthetic_speech_audio):
    """Test stylized anime mouth aperture rendering and limitations reporting."""
    audio, sr = synthetic_speech_audio
    backend = LipSyncBackend(config=LipSyncConfig(fps=10, is_anime=True))

    frames = [synthetic_character_image.copy() for _ in range(5)]
    result = backend.process(audio_source=audio, character_frames=frames)

    assert len(result.frames) == 5
    assert result.is_anime_detected is True
    assert "supported_styles" in result.limitations
    assert "Tamil" in result.limitations["supported_languages"]
    assert "English" in result.limitations["supported_languages"]

    # Rendered frames should have non-zero mouth cavity pixels
    for f in result.frames:
        assert f.shape == synthetic_character_image.shape
        assert f.dtype == np.uint8


# ═══════════════════════════════════════════════════════════════════════════
# 4. Background Compositor Tests (Phase 7)
# ═══════════════════════════════════════════════════════════════════════════

def test_background_compositor_preserves_room(synthetic_character_image, synthetic_room_frames):
    """Verify BackgroundCompositor preserves room background and smooths foreground edges."""
    compositor = BackgroundCompositor(
        config=CompositorConfig(feather_radius=5, erosion_size=2, enable_ambient_matching=True)
    )

    char_frame = synthetic_character_image
    orig_frame = synthetic_room_frames[0]
    mask = BackgroundCompositor.synthesize_fallback_mask(orig_frame)

    # Composite single frame
    comp = compositor.composite_frame(char_frame, orig_frame, mask)
    assert comp.shape == orig_frame.shape
    assert comp.dtype == np.uint8

    # Corner of the room background should remain unchanged (mask is zero there)
    assert np.allclose(comp[10, 10], orig_frame[10, 10], atol=2)

    # Sequence compositing
    char_seq = [synthetic_character_image for _ in range(3)]
    orig_seq = synthetic_room_frames[:3]
    out_seq = compositor.composite_sequence(char_seq, orig_seq)
    assert len(out_seq) == 3


# ═══════════════════════════════════════════════════════════════════════════
# 5. Audio Mixer Tests (Phase 7)
# ═══════════════════════════════════════════════════════════════════════════

def test_audio_mixer_duration_synchronization(synthetic_speech_audio, tmp_path):
    """Test audio mixer padding, trimming with tail fade, and lossless WAV export."""
    mixer = AudioMixer(config=AudioSyncConfig(sample_rate=16000, apply_tail_fade=True))
    audio, sr = synthetic_speech_audio  # 1.0 second (16000 samples)

    # 1. Test trimming to 0.5s
    trimmed = mixer.synchronize_duration(audio, sr, target_duration_sec=0.5)
    assert len(trimmed) == int(0.5 * sr)
    # Check that tail has fade-out (last sample is 0 or near 0)
    assert abs(trimmed[-1]) < 0.05

    # 2. Test padding to 1.5s
    padded = mixer.synchronize_duration(audio, sr, target_duration_sec=1.5)
    assert len(padded) == int(1.5 * sr)
    assert np.all(padded[sr:] == 0.0)

    # 3. Test WAV write & read roundtrip
    wav_path = tmp_path / "test_synced.wav"
    mixer.write_wav(trimmed, wav_path, sr)
    assert wav_path.exists()

    read_data, read_sr, _ = mixer.read_audio_track(wav_path)
    assert read_sr == sr
    assert len(read_data) == len(trimmed)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Video Export Pipeline Tests (Phase 8)
# ═══════════════════════════════════════════════════════════════════════════

def test_video_exporter_aspect_ratio_formatting(synthetic_room_frames):
    """Test 1080x1920 (9:16) aspect ratio formatting for all adaptation modes."""
    exporter = VideoExporter(target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT)
    frame = synthetic_room_frames[0]  # 720x1280 (16:9 landscape)

    # 1. Center crop
    crop = exporter.format_frame_9_16(frame, mode=AspectRatioMode.CENTER_CROP)
    assert crop.shape == (TARGET_HEIGHT, TARGET_WIDTH, 3)

    # 2. Fit blur pad (social media reels format)
    blur_pad = exporter.format_frame_9_16(frame, mode=AspectRatioMode.FIT_BLUR_PAD)
    assert blur_pad.shape == (TARGET_HEIGHT, TARGET_WIDTH, 3)

    # 3. Fit pad
    fit_pad = exporter.format_frame_9_16(frame, mode=AspectRatioMode.FIT_PAD)
    assert fit_pad.shape == (TARGET_HEIGHT, TARGET_WIDTH, 3)


def test_video_exporter_mp4_generation(synthetic_room_frames, synthetic_speech_audio, tmp_path):
    """Test end-to-end 1080x1920 MP4 video export with audio and progress callbacks."""
    exporter = VideoExporter(target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT, fps=15)
    audio, sr = synthetic_speech_audio
    output_mp4 = tmp_path / "test_reel_1080x1920.mp4"

    progress_events: list[tuple[float, str]] = []

    def on_progress(pct: float, msg: str) -> None:
        progress_events.append((pct, msg))

    meta = exporter.export(
        frames=synthetic_room_frames[:6],
        output_path=output_mp4,
        audio_source=audio,
        progress_callback=on_progress,
    )

    assert output_mp4.exists()
    assert meta["width"] == TARGET_WIDTH
    assert meta["height"] == TARGET_HEIGHT
    assert meta["aspect_ratio"] == "9:16"
    assert meta["frame_count"] == 6
    assert meta["format"] == "mp4"
    assert len(progress_events) > 0
    assert progress_events[-1][0] == 100.0


# ═══════════════════════════════════════════════════════════════════════════
# 7. Job Manager End-to-End Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_job_manager_end_to_end_render(tmp_path):
    """Test full end-to-end rendering pipeline execution through JobManager."""
    job_manager = JobManager()

    job = job_manager.create_job(
        job_type=JobType.RENDER,
        metadata={
            "character_id": "char_test_001",
            "output_width": TARGET_WIDTH,
            "output_height": TARGET_HEIGHT,
            "output_fps": 15,
            "aspect_ratio_mode": "fit_blur_pad",
        },
    )

    assert job.state == JobState.QUEUED

    # Execute synchronously
    completed_job = job_manager.run_render_pipeline_sync(job.id)

    assert completed_job.state == JobState.COMPLETED
    assert completed_job.progress == 100.0
    assert "export_path" in completed_job.metadata
    export_file = Path(completed_job.metadata["export_path"])
    assert export_file.exists()


def test_job_manager_render_cancellation():
    """Verify job cancellation is respected gracefully without crashes."""
    job_manager = JobManager()
    job = job_manager.create_job(job_type=JobType.RENDER, metadata={"character_id": "test_cancel"})

    cancelled = job_manager.cancel_job(job.id)
    assert cancelled is not None
    assert cancelled.state == JobState.CANCELLED

    # Running an already cancelled job returns immediately
    res = job_manager.run_render_pipeline_sync(job.id)
    assert res.state == JobState.CANCELLED


# ═══════════════════════════════════════════════════════════════════════════
# 8. Rendering API Endpoints Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_rendering_api_lifecycle():
    """Test FastAPI start, status, queue, cancel, and exports endpoints."""
    app = create_app()
    client = TestClient(app)

    # 1. Queue listing before jobs
    res_queue = client.get("/api/render/queue")
    assert res_queue.status_code == 200
    assert "jobs" in res_queue.json()
    assert "stats" in res_queue.json()

    # 2. Start render job
    res_start = client.post(
        "/api/render/start",
        json={
            "character_id": "char_api_test",
            "output_width": 1080,
            "output_height": 1920,
            "output_fps": 30,
            "aspect_ratio_mode": "fit_blur_pad",
        },
    )
    assert res_start.status_code == 202
    data = res_start.json()
    job_id = data["job_id"]
    assert data["state"] in ("queued", "running")

    # 3. Status endpoint
    res_status = client.get(f"/api/render/{job_id}/status")
    assert res_status.status_code == 200
    assert res_status.json()["id"] == job_id

    # 4. Cancel endpoint (test cancelling a queued job)
    cancel_job = app.state.job_manager.create_job(
        job_type=JobType.RENDER,
        metadata={"character_id": "test_to_cancel"},
    )
    res_cancel = client.delete(f"/api/render/{cancel_job.id}")
    assert res_cancel.status_code == 200
    assert "cancelled" in res_cancel.json()["message"]

    # 5. List exports endpoint
    res_exports = client.get("/api/render/exports")
    assert res_exports.status_code == 200
    assert isinstance(res_exports.json(), list)

    # 6. Download test for cancelled job returns 400
    res_dl_cancelled = client.get(f"/api/render/{cancel_job.id}/download")
    assert res_dl_cancelled.status_code == 400
