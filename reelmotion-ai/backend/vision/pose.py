"""Pose estimation module for ReelMotion AI.

Implements MediaPipe PoseLandmarker (33 body landmarks) with an OpenCV contour
and morphological skeleton fallback when MediaPipe is unavailable, loading,
or in CPU-only test environments.
"""

from __future__ import annotations

import logging
from enum import IntEnum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PoseLandmarkIndex(IntEnum):
    """MediaPipe standard 33 pose landmark indices."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


POSE_CONNECTIONS: list[tuple[int, int]] = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left Arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right Arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Left Leg
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    # Right Leg
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]


class NormalizedLandmark(BaseModel):
    """Normalized 3D landmark with visibility confidence."""

    index: int
    name: str = ""
    x: float = Field(..., description="Normalized X [0.0, 1.0]")
    y: float = Field(..., description="Normalized Y [0.0, 1.0]")
    z: float = Field(default=0.0, description="Normalized depth")
    visibility: float = Field(default=1.0, description="Confidence [0.0, 1.0]")
    presence: float = Field(default=1.0, description="Presence score [0.0, 1.0]")


class PoseResult(BaseModel):
    """Result of pose estimation on a single frame."""

    has_pose: bool = False
    confidence: float = 0.0
    backend: str = "opencv_fallback"  # 'mediapipe' or 'opencv_fallback'
    landmarks: list[NormalizedLandmark] = Field(default_factory=list)
    world_landmarks: list[NormalizedLandmark] = Field(default_factory=list)
    bounding_box: tuple[float, float, float, float] | None = None  # (xmin, ymin, xmax, ymax)


class PoseEstimator:
    """Body pose estimator supporting MediaPipe with OpenCV contour skeleton fallback."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        min_detection_confidence: float = 0.5,
        force_fallback: bool = False,
    ) -> None:
        self.min_detection_confidence = min_detection_confidence
        self.force_fallback = force_fallback
        self.model_path = Path(model_path) if model_path else None
        self._landmarker: Any = None
        self.backend = "opencv_fallback"

        if not self.force_fallback:
            self._init_mediapipe()

    def _init_mediapipe(self) -> None:
        """Attempt to initialize MediaPipe PoseLandmarker."""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            # Look for model path
            target_path = self.model_path
            if target_path is None or not target_path.exists():
                # Check standard locations
                candidates = [
                    Path("models/pose_landmarker_heavy.task"),
                    Path("models/pose_landmarker.task"),
                    Path(__file__).resolve().parent.parent.parent / "models" / "pose_landmarker_heavy.task",
                    Path(__file__).resolve().parent.parent.parent / "models" / "pose_landmarker.task",
                ]
                for c in candidates:
                    if c.exists():
                        target_path = c
                        break

            if target_path and target_path.exists():
                base_opts = mp_python.BaseOptions(model_asset_path=str(target_path))
                options = mp_vision.PoseLandmarkerOptions(
                    base_options=base_opts,
                    running_mode=mp_vision.RunningMode.IMAGE,
                    min_pose_detection_confidence=self.min_detection_confidence,
                    min_pose_presence_confidence=self.min_detection_confidence,
                )
                self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
                self.backend = "mediapipe"
                logger.info("MediaPipe PoseLandmarker initialized successfully from %s", target_path)
            else:
                logger.info("MediaPipe pose task model not found. Using OpenCV skeleton fallback.")
                self.backend = "opencv_fallback"
        except Exception as exc:
            logger.info("MediaPipe Pose initialization failed (%s). Using OpenCV skeleton fallback.", exc)
            self._landmarker = None
            self.backend = "opencv_fallback"

    def estimate(self, image: np.ndarray) -> PoseResult:
        """Estimate 33 body pose landmarks on an input image (BGR or RGB)."""
        if image is None or image.size == 0:
            return PoseResult(has_pose=False, backend=self.backend)

        if self.backend == "mediapipe" and self._landmarker is not None:
            try:
                return self._estimate_mediapipe(image)
            except Exception as exc:
                logger.warning("MediaPipe inference error (%s); falling back to OpenCV.", exc)
                return self._estimate_opencv(image)

        return self._estimate_opencv(image)

    def _estimate_mediapipe(self, image: np.ndarray) -> PoseResult:
        """Perform pose estimation using MediaPipe Tasks."""
        import mediapipe as mp

        if len(image.shape) == 2:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detection_result = self._landmarker.detect(mp_image)

        if not detection_result.pose_landmarks:
            return PoseResult(has_pose=False, backend="mediapipe")

        landmarks_list: list[NormalizedLandmark] = []
        raw_landmarks = detection_result.pose_landmarks[0]

        xs = [lm.x for lm in raw_landmarks]
        ys = [lm.y for lm in raw_landmarks]
        bbox = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))

        for idx, lm in enumerate(raw_landmarks):
            name = PoseLandmarkIndex(idx).name if idx < len(PoseLandmarkIndex) else f"LANDMARK_{idx}"
            landmarks_list.append(
                NormalizedLandmark(
                    index=idx,
                    name=name,
                    x=float(lm.x),
                    y=float(lm.y),
                    z=float(lm.z) if hasattr(lm, "z") else 0.0,
                    visibility=float(lm.visibility) if hasattr(lm, "visibility") else 1.0,
                    presence=float(lm.presence) if hasattr(lm, "presence") else 1.0,
                )
            )

        return PoseResult(
            has_pose=True,
            confidence=0.95,
            backend="mediapipe",
            landmarks=landmarks_list,
            bounding_box=bbox,
        )

    def _estimate_opencv(self, image: np.ndarray) -> PoseResult:
        """OpenCV contour and morphological skeleton fallback pose extraction."""
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return PoseResult(has_pose=False, backend="opencv_fallback")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Otsu threshold + morphological cleanup to find foreground person
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if the foreground is darker than background
        white_pixels = cv2.countNonZero(thresh)
        if white_pixels > (h * w * 0.75):
            thresh = cv2.bitwise_not(thresh)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return self._build_empty_pose("opencv_fallback")

        # Select largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < (h * w * 0.015):  # Too small to be a person
            return self._build_empty_pose("opencv_fallback")

        # Bounding box & moments
        bx, by, bw, bh = cv2.boundingRect(largest)
        M = cv2.moments(largest)
        cx = int(M["m10"] / (M["m00"] + 1e-5))
        cy = int(M["m01"] / (M["m00"] + 1e-5))

        # Extreme contour points
        leftmost = tuple(largest[largest[:, :, 0].argmin()][0])
        rightmost = tuple(largest[largest[:, :, 0].argmax()][0])
        topmost = tuple(largest[largest[:, :, 1].argmin()][0])
        bottommost = tuple(largest[largest[:, :, 1].argmax()][0])

        # Normalized coordinates helper
        def norm(px: float, py: float) -> tuple[float, float]:
            return (
                round(float(np.clip(px / w, 0.0, 1.0)), 4),
                round(float(np.clip(py / h, 0.0, 1.0)), 4),
            )

        # Synthesize 33 landmarks based on human morphological anatomy
        head_cx, head_cy = topmost[0], by + int(bh * 0.12)
        nose_x, nose_y = norm(head_cx, head_cy)

        # Eyes & Ears
        eye_offset_x = int(bw * 0.04)
        eye_y = head_cy - int(bh * 0.02)
        le_x, le_y = norm(head_cx - eye_offset_x, eye_y)
        re_x, re_y = norm(head_cx + eye_offset_x, eye_y)
        ear_offset_x = int(bw * 0.08)
        lear_x, lear_y = norm(head_cx - ear_offset_x, head_cy)
        rear_x, rear_y = norm(head_cx + ear_offset_x, head_cy)

        # Mouth
        mouth_y = head_cy + int(bh * 0.04)
        lm_x, lm_y = norm(head_cx - int(bw * 0.03), mouth_y)
        rm_x, rm_y = norm(head_cx + int(bw * 0.03), mouth_y)

        # Shoulders
        shoulder_y = by + int(bh * 0.24)
        ls_x, ls_y = norm(bx + int(bw * 0.25), shoulder_y)
        rs_x, rs_y = norm(bx + int(bw * 0.75), shoulder_y)

        # Elbows & Wrists (adjusted towards extreme sides)
        elbow_y = by + int(bh * 0.40)
        wrist_y = by + int(bh * 0.55)
        lw_pt = leftmost if leftmost[1] > shoulder_y else (bx + int(bw * 0.12), wrist_y)
        rw_pt = rightmost if rightmost[1] > shoulder_y else (bx + int(bw * 0.88), wrist_y)

        lelb_x, lelb_y = norm(bx + int(bw * 0.18), elbow_y)
        relb_x, relb_y = norm(bx + int(bw * 0.82), elbow_y)
        lw_x, lw_y = norm(lw_pt[0], lw_pt[1])
        rw_x, rw_y = norm(rw_pt[0], rw_pt[1])

        # Hips
        hip_y = by + int(bh * 0.52)
        lhip_x, lhip_y = norm(bx + int(bw * 0.35), hip_y)
        rhip_x, rhip_y = norm(bx + int(bw * 0.65), hip_y)

        # Knees
        knee_y = by + int(bh * 0.74)
        lknee_x, lknee_y = norm(bx + int(bw * 0.33), knee_y)
        rknee_x, rknee_y = norm(bx + int(bw * 0.67), knee_y)

        # Ankles & Feet
        ankle_y = by + int(bh * 0.92)
        foot_y = min(h - 1, by + bh)
        lank_x, lank_y = norm(bx + int(bw * 0.30), ankle_y)
        rank_x, rank_y = norm(bx + int(bw * 0.70), ankle_y)
        lfoot_x, lfoot_y = norm(bx + int(bw * 0.28), foot_y)
        rfoot_x, rfoot_y = norm(bx + int(bw * 0.72), foot_y)

        coords: list[tuple[float, float]] = [
            (nose_x, nose_y),      # 0 NOSE
            (le_x, le_y),          # 1 LEFT_EYE_INNER
            (le_x, le_y),          # 2 LEFT_EYE
            (le_x, le_y),          # 3 LEFT_EYE_OUTER
            (re_x, re_y),          # 4 RIGHT_EYE_INNER
            (re_x, re_y),          # 5 RIGHT_EYE
            (re_x, re_y),          # 6 RIGHT_EYE_OUTER
            (lear_x, lear_y),      # 7 LEFT_EAR
            (rear_x, rear_y),      # 8 RIGHT_EAR
            (lm_x, lm_y),          # 9 MOUTH_LEFT
            (rm_x, rm_y),          # 10 MOUTH_RIGHT
            (ls_x, ls_y),          # 11 LEFT_SHOULDER
            (rs_x, rs_y),          # 12 RIGHT_SHOULDER
            (lelb_x, lelb_y),      # 13 LEFT_ELBOW
            (relb_x, relb_y),      # 14 RIGHT_ELBOW
            (lw_x, lw_y),          # 15 LEFT_WRIST
            (rw_x, rw_y),          # 16 RIGHT_WRIST
            (lw_x, min(1.0, lw_y + 0.02)),  # 17 LEFT_PINKY
            (rw_x, min(1.0, rw_y + 0.02)),  # 18 RIGHT_PINKY
            (lw_x, min(1.0, lw_y + 0.03)),  # 19 LEFT_INDEX
            (rw_x, min(1.0, rw_y + 0.03)),  # 20 RIGHT_INDEX
            (lw_x, min(1.0, lw_y + 0.01)),  # 21 LEFT_THUMB
            (rw_x, min(1.0, rw_y + 0.01)),  # 22 RIGHT_THUMB
            (lhip_x, lhip_y),      # 23 LEFT_HIP
            (rhip_x, rhip_y),      # 24 RIGHT_HIP
            (lknee_x, lknee_y),    # 25 LEFT_KNEE
            (rknee_x, rknee_y),    # 26 RIGHT_KNEE
            (lank_x, lank_y),      # 27 LEFT_ANKLE
            (rank_x, rank_y),      # 28 RIGHT_ANKLE
            (lank_x, min(1.0, lank_y + 0.02)),  # 29 LEFT_HEEL
            (rank_x, min(1.0, rank_y + 0.02)),  # 30 RIGHT_HEEL
            (lfoot_x, lfoot_y),    # 31 LEFT_FOOT_INDEX
            (rfoot_x, rfoot_y),    # 32 RIGHT_FOOT_INDEX
        ]

        landmarks = [
            NormalizedLandmark(
                index=i,
                name=PoseLandmarkIndex(i).name,
                x=c[0],
                y=c[1],
                z=0.0,
                visibility=0.85,
                presence=0.90,
            )
            for i, c in enumerate(coords)
        ]

        bbox = (
            round(bx / w, 4),
            round(by / h, 4),
            round((bx + bw) / w, 4),
            round((by + bh) / h, 4),
        )

        return PoseResult(
            has_pose=True,
            confidence=0.78,
            backend="opencv_fallback",
            landmarks=landmarks,
            bounding_box=bbox,
        )

    def _build_empty_pose(self, backend: str) -> PoseResult:
        """Return 33 landmarks with visibility=0 when no pose is detectable."""
        landmarks = [
            NormalizedLandmark(
                index=i,
                name=PoseLandmarkIndex(i).name,
                x=0.0,
                y=0.0,
                z=0.0,
                visibility=0.0,
                presence=0.0,
            )
            for i in range(33)
        ]
        return PoseResult(
            has_pose=False,
            confidence=0.0,
            backend=backend,
            landmarks=landmarks,
        )

    @classmethod
    def draw_pose(
        cls,
        image: np.ndarray,
        result: PoseResult,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Render skeleton lines and landmark points on an image."""
        vis = image.copy()
        if not result.has_pose or not result.landmarks:
            return vis

        h, w = vis.shape[:2]
        pts: dict[int, tuple[int, int]] = {}

        for lm in result.landmarks:
            if lm.visibility > 0.3:
                px = int(lm.x * w)
                py = int(lm.y * h)
                pts[lm.index] = (px, py)
                cv2.circle(vis, (px, py), 4, (0, 0, 255), -1)

        for p1, p2 in POSE_CONNECTIONS:
            if p1 in pts and p2 in pts:
                cv2.line(vis, pts[p1], pts[p2], color, thickness)

        return vis

