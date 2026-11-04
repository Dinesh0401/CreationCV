"""Unit and integration tests for Phase 3 vision module: pose, face, hands, segmentation, and API."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.capture.camera import VideoFrameExtractor
from backend.vision.face_tracking import (
    FaceTracker,
    FaceTrackingResult,
)
from backend.vision.hand_tracking import (
    HandLandmarkIndex,
    HandTracker,
    HandTrackingResult,
)
from backend.vision.pose import (
    PoseEstimator,
    PoseLandmarkIndex,
    PoseResult,
)
from backend.vision.segmentation import (
    PersonSegmenter,
    SegmentationResult,
)


# ---------------------------------------------------------------------------
# Synthetic Image Helpers
# ---------------------------------------------------------------------------

def _create_synthetic_person_image(w: int = 400, h: int = 500) -> np.ndarray:
    """Create a synthetic image containing a stylized human figure."""
    img = np.full((h, w, 3), 40, dtype=np.uint8)  # dark background

    # Skin tone BGR: approx (140, 160, 220)
    skin_color = (140, 160, 220)
    shirt_color = (200, 100, 50)
    pants_color = (80, 50, 30)

    # Head
    cv2.circle(img, (w // 2, int(h * 0.16)), int(h * 0.08), skin_color, -1)
    # Eyes
    cv2.circle(img, (int(w * 0.44), int(h * 0.15)), 4, (30, 30, 30), -1)
    cv2.circle(img, (int(w * 0.56), int(h * 0.15)), 4, (30, 30, 30), -1)
    # Mouth
    cv2.ellipse(img, (w // 2, int(h * 0.20)), (14, 6), 0, 0, 180, (50, 50, 180), -1)

    # Torso / shirt
    cv2.rectangle(img, (int(w * 0.30), int(h * 0.26)), (int(w * 0.70), int(h * 0.60)), shirt_color, -1)

    # Arms
    cv2.line(img, (int(w * 0.30), int(h * 0.28)), (int(w * 0.16), int(h * 0.52)), shirt_color, 24)
    cv2.line(img, (int(w * 0.70), int(h * 0.28)), (int(w * 0.84), int(h * 0.52)), shirt_color, 24)
    # Hands
    cv2.circle(img, (int(w * 0.16), int(h * 0.54)), 16, skin_color, -1)
    cv2.circle(img, (int(w * 0.84), int(h * 0.54)), 16, skin_color, -1)

    # Legs / pants
    cv2.rectangle(img, (int(w * 0.32), int(h * 0.60)), (int(w * 0.48), int(h * 0.94)), pants_color, -1)
    cv2.rectangle(img, (int(w * 0.52), int(h * 0.60)), (int(w * 0.68), int(h * 0.94)), pants_color, -1)

    return img


def _create_blank_image(w: int = 200, h: int = 200) -> np.ndarray:
    """Create a completely black image with no features."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _encode_png_bytes(img: np.ndarray) -> bytes:
    ret, buf = cv2.imencode(".png", img)
    assert ret
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Pose Estimation Tests
# ---------------------------------------------------------------------------

def test_pose_estimation_person():
    """Verify 33 landmarks returned for synthetic human figure."""
    estimator = PoseEstimator(force_fallback=True)
    img = _create_synthetic_person_image()
    result = estimator.estimate(img)

    assert isinstance(result, PoseResult)
    assert result.has_pose is True
    assert len(result.landmarks) == 33
    assert result.confidence > 0.5
    assert result.bounding_box is not None

    # Check normalization range [0.0, 1.0]
    for lm in result.landmarks:
        assert 0.0 <= lm.x <= 1.0
        assert 0.0 <= lm.y <= 1.0
        assert 0.0 <= lm.visibility <= 1.0
        assert lm.name == PoseLandmarkIndex(lm.index).name

    # Check key topological ordering (head is above hips, hips above feet)
    nose = result.landmarks[PoseLandmarkIndex.NOSE]
    left_hip = result.landmarks[PoseLandmarkIndex.LEFT_HIP]
    left_foot = result.landmarks[PoseLandmarkIndex.LEFT_FOOT_INDEX]
    assert nose.y < left_hip.y
    assert left_hip.y < left_foot.y


def test_pose_estimation_blank_image():
    """Blank/black image should report has_pose=False with 0 visibility."""
    estimator = PoseEstimator(force_fallback=True)
    blank = _create_blank_image()
    result = estimator.estimate(blank)

    assert isinstance(result, PoseResult)
    assert result.has_pose is False
    assert len(result.landmarks) == 33
    for lm in result.landmarks:
        assert lm.visibility == 0.0


def test_pose_drawing():
    """Verify skeleton visualization draws lines and returns valid image."""
    estimator = PoseEstimator(force_fallback=True)
    img = _create_synthetic_person_image()
    result = estimator.estimate(img)
    drawn = PoseEstimator.draw_pose(img, result)

    assert drawn.shape == img.shape
    # Image should be modified with skeleton pixels
    assert not np.array_equal(drawn, img)


# ---------------------------------------------------------------------------
# Face Tracking Tests
# ---------------------------------------------------------------------------

def test_face_tracking_landmarks_and_metrics():
    """Verify facial tracking detects landmarks, EAR blink, and MAR mouth."""
    tracker = FaceTracker(force_fallback=True)
    img = _create_synthetic_person_image()
    result = tracker.track(img)

    assert isinstance(result, FaceTrackingResult)
    assert result.has_face is True
    assert result.landmark_count in (68, 478)
    assert len(result.landmarks) == result.landmark_count
    assert result.bounding_box is not None

    # Verify EAR blink metrics
    assert result.blink.average_ear >= 0.0
    assert isinstance(result.blink.is_blinking, bool)

    # Verify MAR mouth metrics
    assert result.mouth.mouth_aspect_ratio >= 0.0
    assert isinstance(result.mouth.is_mouth_open, bool)


def test_face_tracking_blank_image():
    """Blank frame should produce has_face=False."""
    tracker = FaceTracker(force_fallback=True)
    blank = _create_blank_image()
    result = tracker.track(blank)

    assert result.has_face is False
    assert len(result.landmarks) == 0


# ---------------------------------------------------------------------------
# Hand Tracking Tests
# ---------------------------------------------------------------------------

def test_hand_tracking_landmarks():
    """Verify hand tracker detects 21 landmarks and handedness."""
    tracker = HandTracker(force_fallback=True)
    img = _create_synthetic_person_image()
    result = tracker.track(img)

    assert isinstance(result, HandTrackingResult)
    assert result.has_hands is True
    assert len(result.hands) > 0

    for hand in result.hands:
        assert len(hand.landmarks) == 21
        assert hand.handedness in ("Left", "Right")
        assert hand.bounding_box is not None
        assert isinstance(hand.is_open, bool)
        assert isinstance(hand.is_fist, bool)

        # Check landmark normalization
        for lm in hand.landmarks:
            assert 0.0 <= lm.x <= 1.0
            assert 0.0 <= lm.y <= 1.0
            assert lm.name == HandLandmarkIndex(lm.index).name


def test_hand_tracking_blank_image():
    """Blank frame should return has_hands=False."""
    tracker = HandTracker(force_fallback=True)
    blank = _create_blank_image()
    result = tracker.track(blank)

    assert result.has_hands is False
    assert len(result.hands) == 0


# ---------------------------------------------------------------------------
# Segmentation Tests
# ---------------------------------------------------------------------------

def test_person_segmentation_masks():
    """Verify person segmentation generates binary mask, alpha matte, and RGBA cutout."""
    segmenter = PersonSegmenter(force_fallback=True)
    img = _create_synthetic_person_image()
    result = segmenter.segment(img)

    assert isinstance(result, SegmentationResult)
    assert result.metadata.has_person is True
    assert result.metadata.person_coverage_ratio > 0.05
    assert result.metadata.width == img.shape[1]
    assert result.metadata.height == img.shape[0]

    # Binary mask
    assert result.binary_mask is not None
    assert result.binary_mask.shape == (img.shape[0], img.shape[1])
    assert set(np.unique(result.binary_mask)).issubset({0, 255})

    # Alpha mask
    assert result.alpha_mask is not None
    assert result.alpha_mask.dtype == np.float32
    assert 0.0 <= np.min(result.alpha_mask)
    assert np.max(result.alpha_mask) <= 1.0

    # Foreground RGBA
    assert result.foreground_rgba is not None
    assert result.foreground_rgba.shape == (img.shape[0], img.shape[1], 4)

    # Base64 masks
    bin_b64, alpha_b64 = result.to_base64_masks()
    assert len(bin_b64) > 100
    assert len(alpha_b64) > 100


def test_background_isolation_and_compositing():
    """Test compositing foreground onto a new background and isolating background."""
    img = _create_synthetic_person_image(w=300, h=300)
    segmenter = PersonSegmenter(force_fallback=True)
    res = segmenter.segment(img)

    # Composite onto solid red background
    red_bg = np.zeros((300, 300, 3), dtype=np.uint8)
    red_bg[:, :, 2] = 255  # Red channel in BGR
    composite = PersonSegmenter.composite_background(res.foreground_rgba, red_bg)

    assert composite.shape == (300, 300, 3)
    # The composite should differ from original and from pure red
    assert not np.array_equal(composite, img)
    assert not np.array_equal(composite, red_bg)

    # Isolated background
    isolated_bg = PersonSegmenter.isolate_background(img, res.binary_mask)
    assert isolated_bg.shape == (300, 300, 3)


# ---------------------------------------------------------------------------
# Vision API Router Endpoints Tests
# ---------------------------------------------------------------------------

def test_api_vision_status(client: TestClient):
    """GET /api/vision/status returns 200 with backends and capabilities."""
    response = client.get("/api/vision/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "backends" in data
    assert "pose" in data["backends"]
    assert "face" in data["backends"]
    assert "hands" in data["backends"]
    assert "segmentation" in data["backends"]
    assert len(data["capabilities"]) >= 5


def test_api_vision_extract_image(client: TestClient):
    """POST /api/vision/extract with an uploaded image frame."""
    img = _create_synthetic_person_image()
    png_bytes = _encode_png_bytes(img)

    files = {"file": ("frame.png", png_bytes, "image/png")}
    response = client.post("/api/vision/extract", files=files)
    assert response.status_code == 200

    data = response.json()
    assert data["width"] == img.shape[1]
    assert data["height"] == img.shape[0]
    assert data["is_video"] is False
    assert data["processing_time_ms"] > 0.0

    # Pose
    assert data["pose"] is not None
    assert data["pose"]["has_pose"] is True
    assert len(data["pose"]["landmarks"]) == 33

    # Face
    assert data["face"] is not None
    assert data["face"]["has_face"] is True
    assert "blink" in data["face"]
    assert "mouth" in data["face"]

    # Hands
    assert data["hands"] is not None
    assert data["hands"]["has_hands"] is True

    # Segmentation
    assert data["segmentation"] is not None
    assert len(data["segmentation"]["binary_mask_base64"]) > 0
    assert len(data["segmentation"]["alpha_mask_base64"]) > 0


def test_api_vision_extract_video(client: TestClient, tmp_path: Path):
    """POST /api/vision/extract with an uploaded video file."""
    video_path = tmp_path / "stream.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (300, 300))
    person_frame = _create_synthetic_person_image(w=300, h=300)
    for _ in range(5):
        writer.write(person_frame)
    writer.release()

    video_bytes = video_path.read_bytes()
    files = {"file": ("stream.mp4", video_bytes, "video/mp4")}
    response = client.post("/api/vision/extract?frame_index=0", files=files)
    assert response.status_code == 200

    data = response.json()
    assert data["is_video"] is True
    assert data["width"] == 300
    assert data["height"] == 300
    assert data["pose"]["has_pose"] is True


def test_api_vision_individual_endpoints(client: TestClient):
    """Test dedicated /pose, /face, /hands, and /segment endpoints."""
    img = _create_synthetic_person_image()
    png_bytes = _encode_png_bytes(img)
    files = {"file": ("frame.png", png_bytes, "image/png")}

    # 1. /api/vision/pose
    resp_pose = client.post("/api/vision/pose", files=files)
    assert resp_pose.status_code == 200
    assert resp_pose.json()["has_pose"] is True

    # 2. /api/vision/face
    resp_face = client.post("/api/vision/face", files=files)
    assert resp_face.status_code == 200
    assert resp_face.json()["has_face"] is True

    # 3. /api/vision/hands
    resp_hands = client.post("/api/vision/hands", files=files)
    assert resp_hands.status_code == 200
    assert resp_hands.json()["has_hands"] is True

    # 4. /api/vision/segment
    resp_seg = client.post("/api/vision/segment", files=files)
    assert resp_seg.status_code == 200
    assert len(resp_seg.json()["binary_mask_base64"]) > 0


def test_api_vision_empty_file_error(client: TestClient):
    """Uploading an empty file should return 400 Bad Request."""
    files = {"file": ("empty.png", b"", "image/png")}
    response = client.post("/api/vision/extract", files=files)
    assert response.status_code == 400

