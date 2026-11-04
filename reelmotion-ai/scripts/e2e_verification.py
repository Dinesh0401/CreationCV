"""End-to-End Pipeline Verification Script for ReelMotion AI.

Executes the complete pipeline:
1. Ingest character reference
2. Extract pose landmarks & person segmentation
3. Run character replacement (WanBackend / fallback warper)
4. Run lip synchronization (LipSyncBackend)
5. Preserve background & composite
6. Preserve original audio
7. Export 1080x1920 9:16 MP4 video
8. Validate and report technical metrics
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import psutil
import torch

from backend.capture.audio import AudioAnalyzer
from backend.capture.camera import VideoFrameExtractor
from backend.generation.lip_sync_backend import LipSyncBackend, LipSyncConfig
from backend.generation.wan_backend import FrameLandmarks, GenerationConfig, WanBackend
from backend.rendering.audio_mixer import AudioMixer, AudioSyncConfig
from backend.rendering.background_compositor import BackgroundCompositor, CompositorConfig
from backend.rendering.export import AspectRatioMode, VideoExporter
from backend.vision.pose import PoseEstimator
from backend.vision.segmentation import PersonSegmenter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("e2e_verification")


def get_memory_info() -> dict:
    ram_mb = psutil.virtual_memory().used / (1024 * 1024)
    info = {"ram_used_mb": round(ram_mb, 1)}
    if torch.cuda.is_available():
        info["vram_used_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
        info["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 1)
    else:
        info["vram_used_mb"] = 0.0
        info["vram_reserved_mb"] = 0.0
    return info


def run_e2e_pipeline() -> dict:
    results: dict = {
        "status": "started",
        "stages": {},
        "hardware": {},
        "output_verification": {},
    }

    # 1. Hardware Detection
    results["hardware"] = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    logger.info("Hardware detected: %s", results["hardware"])

    char_ref_path = PROJECT_ROOT / "characters" / "test_anime_character.png"
    video_path = PROJECT_ROOT / "recordings" / "test_human_recording.mp4"
    audio_path = PROJECT_ROOT / "recordings" / "test_human_audio.wav"
    output_dir = PROJECT_ROOT / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    export_path = output_dir / "e2e_character_replacement.mp4"

    # Stage 1: Load Character Reference
    t0 = time.time()
    ref_bgr = cv2.imread(str(char_ref_path))
    if ref_bgr is None:
        raise FileNotFoundError(f"Character reference not found at {char_ref_path}")
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    results["stages"]["1_character_ingestion"] = {
        "status": "success",
        "resolution": [ref_rgb.shape[1], ref_rgb.shape[0]],
        "elapsed_sec": round(time.time() - t0, 3),
        "memory": get_memory_info(),
    }
    logger.info("Stage 1 (Character Ingestion) complete: %s", results["stages"]["1_character_ingestion"])

    # Stage 2: Load Video Frames & Audio
    t0 = time.time()
    extractor = VideoFrameExtractor(video_path)
    video_meta = extractor.get_metadata()
    raw_frames_info = extractor.extract_all()
    extractor.cleanup()
    original_frames_rgb = [f.to_rgb() for f in raw_frames_info]
    original_frames_bgr = [f.to_bgr() for f in raw_frames_info]

    audio_samples, sample_rate, channels = AudioAnalyzer.read_wav(audio_path)
    results["stages"]["2_capture_ingestion"] = {
        "status": "success",
        "frame_count": len(original_frames_rgb),
        "video_fps": video_meta.fps,
        "video_resolution": [video_meta.width, video_meta.height],
        "audio_sample_rate": sample_rate,
        "audio_samples": len(audio_samples),
        "elapsed_sec": round(time.time() - t0, 3),
        "memory": get_memory_info(),
    }
    logger.info("Stage 2 (Capture Ingestion) complete: %s", results["stages"]["2_capture_ingestion"])

    # Stage 3: Vision Extraction (Pose + Segmentation)
    t0 = time.time()
    pose_estimator = PoseEstimator()
    segmenter = PersonSegmenter()

    landmarks_seq: list[FrameLandmarks] = []
    masks_seq: list[np.ndarray] = []
    backgrounds_seq: list[np.ndarray] = []

    for i, frame_bgr in enumerate(original_frames_bgr):
        p_res = pose_estimator.estimate(frame_bgr)
        body_lms = [{"x": lm.x, "y": lm.y, "visibility": lm.visibility} for lm in p_res.landmarks]
        flm = FrameLandmarks(body_landmarks=body_lms, bbox=p_res.bounding_box)
        landmarks_seq.append(flm)

        s_res = segmenter.segment(frame_bgr, bounding_box=p_res.bounding_box)
        masks_seq.append(s_res.binary_mask)
        backgrounds_seq.append(s_res.background_isolated)

    results["stages"]["3_vision_tracking"] = {
        "status": "success",
        "pose_backend": pose_estimator.backend,
        "segmentation_backend": segmenter.backend,
        "frames_processed": len(landmarks_seq),
        "elapsed_sec": round(time.time() - t0, 3),
        "memory": get_memory_info(),
    }
    logger.info("Stage 3 (Vision Tracking) complete: %s", results["stages"]["3_vision_tracking"])

    # Stage 4: Character Replacement (WanBackend)
    t0 = time.time()
    wan = WanBackend()
    gen_cfg = GenerationConfig(
        fps=int(video_meta.fps),
        use_fp16=torch.cuda.is_available(),
        chunk_size=16,
    )
    gen_result = wan.generate(
        character_reference=ref_rgb,
        original_frames=original_frames_rgb,
        landmarks_sequence=landmarks_seq,
        config=gen_cfg,
    )
    results["stages"]["4_character_replacement"] = {
        "status": "success",
        "backend_used": gen_result.backend_used,
        "device_used": gen_result.device,
        "frames_generated": len(gen_result.frames),
        "elapsed_sec": round(time.time() - t0, 3),
        "memory": get_memory_info(),
    }
    logger.info("Stage 4 (Character Replacement) complete: %s", results["stages"]["4_character_replacement"])

    # Stage 5: Lip Synchronization
    t0 = time.time()
    ls_cfg = LipSyncConfig(fps=int(video_meta.fps), sample_rate=sample_rate, is_anime=True)
    lip_sync = LipSyncBackend(config=ls_cfg)
    ls_result = lip_sync.process(
        audio_source=audio_samples,
        character_frames=gen_result.frames,
    )
    results["stages"]["5_lip_synchronization"] = {
        "status": "success",
        "is_anime_detected": ls_result.is_anime_detected,
        "articulations_count": len(ls_result.articulations),
        "elapsed_sec": round(time.time() - t0, 3),
        "memory": get_memory_info(),
    }
    logger.info("Stage 5 (Lip Synchronization) complete: %s", results["stages"]["5_lip_synchronization"])

    # Stage 6: Background Preservation & Compositing
    t0 = time.time()
    compositor = BackgroundCompositor(CompositorConfig(feather_radius=5))
    composited_frames_bgr: list[np.ndarray] = []

    for i in range(len(ls_result.frames)):
        char_frame_bgr = cv2.cvtColor(ls_result.frames[i], cv2.COLOR_RGB2BGR)
        bg_frame = backgrounds_seq[i]
        mask = masks_seq[i]
        comp = compositor.composite_frame(
            character_frame=char_frame_bgr,
            original_frame=bg_frame,
            mask=mask,
        )
        composited_frames_bgr.append(comp)

    results["stages"]["6_background_compositing"] = {
        "status": "success",
        "composited_frames": len(composited_frames_bgr),
        "elapsed_sec": round(time.time() - t0, 3),
        "memory": get_memory_info(),
    }
    logger.info("Stage 6 (Background Compositing) complete: %s", results["stages"]["6_background_compositing"])

    # Stage 7: Audio Preservation & Mixing
    t0 = time.time()
    audio_mixer = AudioMixer(AudioSyncConfig(sample_rate=sample_rate))
    audio_data, sr, ch = audio_mixer.read_audio_track(audio_path)
    synced_audio = audio_mixer.synchronize_duration(
        audio_data=audio_data,
        sample_rate=sr,
        target_duration_sec=video_meta.duration_seconds,
    )
    results["stages"]["7_audio_mixing"] = {
        "status": "success",
        "audio_samples_out": len(synced_audio),
        "target_duration_sec": video_meta.duration_seconds,
        "elapsed_sec": round(time.time() - t0, 3),
        "memory": get_memory_info(),
    }
    logger.info("Stage 7 (Audio Mixing) complete: %s", results["stages"]["7_audio_mixing"])

    # Stage 8: 9:16 Vertical 1080x1920 MP4 Export
    t0 = time.time()
    exporter = VideoExporter(mode=AspectRatioMode.FIT_PAD, fps=int(video_meta.fps))
    export_result = exporter.export(
        frames=composited_frames_bgr,
        output_path=export_path,
        audio_source=audio_path,
    )
    results["stages"]["8_mp4_export"] = {
        "status": "success" if export_path.exists() else "failed",
        "output_path": str(export_path),
        "file_size_bytes": export_result.get("file_size_bytes", 0),
        "width": export_result.get("width", 1080),
        "height": export_result.get("height", 1920),
        "fps": export_result.get("fps", int(video_meta.fps)),
        "frame_count": export_result.get("frame_count", len(composited_frames_bgr)),
        "duration_sec": export_result.get("duration_seconds", 0.0),
        "elapsed_sec": round(time.time() - t0, 3),
        "memory": get_memory_info(),
    }
    logger.info("Stage 8 (MP4 Export) complete: %s", results["stages"]["8_mp4_export"])

    # Stage 9: Video Playback & Quality Inspection
    t0 = time.time()
    cap = cv2.VideoCapture(str(export_path))
    read_frames = 0
    mean_brightness = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        read_frames += 1
        mean_brightness.append(float(np.mean(frame)))
    cap.release()

    results["output_verification"] = {
        "file_exists": export_path.exists(),
        "file_size_kb": round(export_path.stat().st_size / 1024, 1),
        "frames_readable": read_frames,
        "mean_frame_brightness": round(float(np.mean(mean_brightness)), 2) if mean_brightness else 0.0,
        "brightness_variance": round(float(np.var(mean_brightness)), 2) if mean_brightness else 0.0,
        "is_vertical_1080x1920": (export_result.get("width") == 1080 and export_result.get("height") == 1920),
    }

    results["status"] = "completed"
    return results


if __name__ == "__main__":
    report = run_e2e_pipeline()
    report_json_path = PROJECT_ROOT / "exports" / "e2e_verification_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\n================ E2E VERIFICATION REPORT ================")
    print(json.dumps(report, indent=2))
