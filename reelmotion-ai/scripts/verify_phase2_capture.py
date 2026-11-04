"""Phase 2 Physical Capture Verification Script for ReelMotion AI.

Tests physical camera capture, physical microphone recording, and A/V synchronization.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import sounddevice as sd
import wave

from backend.capture.audio import AudioAnalyzer, AudioFormat
from backend.capture.camera import CameraManager, VideoFrameExtractor, verify_video
from backend.capture.synchronizer import MediaSynchronizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_capture")


def record_physical_camera_and_mic(duration_sec: float = 5.0) -> dict:
    recordings_dir = PROJECT_ROOT / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    video_path = recordings_dir / "physical_webcam_5s.mp4"
    audio_path = recordings_dir / "physical_mic_5s.wav"
    preview_path = recordings_dir / "physical_camera_preview.png"

    report: dict = {
        "status": "started",
        "camera_enumeration": [],
        "preview_capture": {},
        "video_recording": {},
        "audio_recording": {},
        "av_synchronization": {},
    }

    # 1. Enumerate Cameras
    camera_mgr = CameraManager()
    cameras = camera_mgr.enumerate_cameras(max_devices=3, check_resolutions=True)
    report["camera_enumeration"] = [c.model_dump() for c in cameras]
    logger.info("Enumerated %d camera(s): %s", len(cameras), report["camera_enumeration"])

    # 2. Camera Preview and recording
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        report["preview_capture"] = {"error": "Camera 0 could not be opened"}
        logger.error("Camera 0 could not be opened.")
        return report

    # Negotiate resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, preview_frame = cap.read()
    if ret and preview_frame is not None:
        cv2.imwrite(str(preview_path), preview_frame)
        mean_brightness = float(np.mean(preview_frame))
        report["preview_capture"] = {
            "status": "success",
            "path": str(preview_path),
            "width": actual_w,
            "height": actual_h,
            "mean_brightness": round(mean_brightness, 2),
            "is_privacy_screen": (mean_brightness > 130 and mean_brightness < 145),
        }
        logger.info("Saved camera preview to %s (mean brightness: %.2f)", preview_path, mean_brightness)
    else:
        report["preview_capture"] = {"error": "Failed reading preview frame"}

    # 3. Audio Recording Setup (using sounddevice)
    sample_rate = 44100
    channels = 1
    total_audio_samples = int(duration_sec * sample_rate)
    logger.info("Recording %d audio samples at %d Hz from microphone...", total_audio_samples, sample_rate)

    audio_buffer = []

    def audio_callback(indata, frames, time_info, status):
        if status:
            logger.warning("Sounddevice status: %s", status)
        audio_buffer.append(indata.copy())

    # Start audio stream
    audio_stream = sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        callback=audio_callback,
    )
    audio_stream.start()

    # 4. Record 5 Seconds Video
    target_fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, target_fps, (actual_w, actual_h))

    start_time = time.time()
    frames_written = 0
    frame_timestamps = []

    logger.info("Recording 5 seconds of video from physical camera...")
    while True:
        elapsed = time.time() - start_time
        if elapsed >= duration_sec:
            break

        ret, frame = cap.read()
        if ret and frame is not None:
            writer.write(frame)
            frame_timestamps.append(elapsed)
            frames_written += 1
        time.sleep(1.0 / (target_fps * 1.5))

    cap.release()
    writer.release()
    actual_video_duration = time.time() - start_time

    # Stop audio stream
    audio_stream.stop()
    audio_stream.close()

    # Concatenate audio
    if audio_buffer:
        recorded_audio = np.concatenate(audio_buffer, axis=0)
    else:
        recorded_audio = np.zeros((total_audio_samples, channels), dtype=np.float32)

    actual_audio_duration = len(recorded_audio) / sample_rate

    # Write WAV file
    audio_int16 = (np.clip(recorded_audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(audio_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    logger.info("Saved %d frames (%.2fs) to %s", frames_written, actual_video_duration, video_path)
    logger.info("Saved %d audio samples (%.2fs) to %s", len(recorded_audio), actual_audio_duration, audio_path)

    # 5. Verify Video Output
    v_verify = verify_video(video_path, expected_fps=target_fps, expected_resolution=(actual_w, actual_h))
    report["video_recording"] = {
        "status": "success",
        "path": str(video_path),
        "frames_written": frames_written,
        "duration_seconds": round(actual_video_duration, 4),
        "measured_fps": round(frames_written / actual_video_duration, 2),
        "resolution": [actual_w, actual_h],
        "is_valid": v_verify.is_valid,
        "verification_errors": v_verify.errors,
        "verification_warnings": v_verify.warnings,
    }

    # 6. Verify Audio Output
    a_verify = AudioAnalyzer.validate_audio(audio_path, expected_sample_rate=sample_rate)
    report["audio_recording"] = {
        "status": "success",
        "path": str(audio_path),
        "format": a_verify.format.value,
        "sample_rate": a_verify.stream_info.sample_rate if a_verify.stream_info else sample_rate,
        "channels": a_verify.stream_info.channels if a_verify.stream_info else channels,
        "duration_seconds": a_verify.stream_info.duration_seconds if a_verify.stream_info else round(actual_audio_duration, 4),
        "rms_level": a_verify.waveform.rms_level if a_verify.waveform else 0.0,
        "peak_level": a_verify.waveform.peak_level if a_verify.waveform else 0.0,
        "dbfs": a_verify.waveform.dbfs if a_verify.waveform else -100.0,
        "is_silent": a_verify.waveform.is_silent if a_verify.waveform else False,
        "is_clipping": a_verify.waveform.is_clipping if a_verify.waveform else False,
        "is_valid": a_verify.is_valid,
        "errors": a_verify.errors,
        "warnings": a_verify.warnings,
    }

    # 7. Verify A/V Synchronization
    syncer = MediaSynchronizer(sync_tolerance_ms=80.0, mismatch_threshold_ms=500.0)
    sync_report = syncer.check_synchronization(
        video_duration_sec=actual_video_duration,
        audio_duration_sec=actual_audio_duration,
        video_fps=target_fps,
        audio_sample_rate=sample_rate,
        audio_samples=recorded_audio.flatten(),
    )
    report["av_synchronization"] = {
        "status": sync_report.status.value,
        "is_synchronized": sync_report.is_synchronized,
        "video_duration_sec": sync_report.video_duration_sec,
        "audio_duration_sec": sync_report.audio_duration_sec,
        "duration_diff_ms": sync_report.duration_diff_ms,
        "max_drift_ms": sync_report.max_drift_ms,
        "total_alignments_calculated": len(sync_report.alignments),
        "recommendations": sync_report.recommendations,
    }

    report["status"] = "completed"

    report_path = recordings_dir / "capture_verification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Verification report saved to %s", report_path)

    return report


if __name__ == "__main__":
    rep = record_physical_camera_and_mic(5.0)
    print("\n=== PHYSICAL CAPTURE VERIFICATION RESULT ===")
    print(json.dumps(rep, indent=2))
