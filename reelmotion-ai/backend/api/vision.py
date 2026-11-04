"""Vision API router for ReelMotion AI.

Exposes endpoints for pose estimation, face landmark tracking (blink + mouth),
hand tracking, and person segmentation / background isolation.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from backend.capture.camera import VideoFrameExtractor
from backend.vision.face_tracking import FaceTracker, FaceTrackingResult
from backend.vision.hand_tracking import HandTracker, HandTrackingResult
from backend.vision.pose import PoseEstimator, PoseResult
from backend.vision.segmentation import PersonSegmenter, SegmentationMetadata

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/vision", tags=["vision"])

# Lazy singleton instances
_pose_estimator: PoseEstimator | None = None
_face_tracker: FaceTracker | None = None
_hand_tracker: HandTracker | None = None
_segmenter: PersonSegmenter | None = None


def get_pose_estimator() -> PoseEstimator:
    global _pose_estimator
    if _pose_estimator is None:
        _pose_estimator = PoseEstimator()
    return _pose_estimator


def get_face_tracker() -> FaceTracker:
    global _face_tracker
    if _face_tracker is None:
        _face_tracker = FaceTracker()
    return _face_tracker


def get_hand_tracker() -> HandTracker:
    global _hand_tracker
    if _hand_tracker is None:
        _hand_tracker = HandTracker()
    return _hand_tracker


def get_person_segmenter() -> PersonSegmenter:
    global _segmenter
    if _segmenter is None:
        _segmenter = PersonSegmenter()
    return _segmenter


class SegmentationResponse(BaseModel):
    """Segmentation output with base64 PNG masks."""

    metadata: SegmentationMetadata
    binary_mask_base64: str = ""
    alpha_mask_base64: str = ""


class VisionExtractResponse(BaseModel):
    """Complete vision extraction response for a frame or video."""

    width: int
    height: int
    processing_time_ms: float
    is_video: bool = False
    frames_processed: int = 1
    pose: PoseResult | None = None
    face: FaceTrackingResult | None = None
    hands: HandTrackingResult | None = None
    segmentation: SegmentationResponse | None = None


class VisionStatusResponse(BaseModel):
    """Vision subsystems status and active backends."""

    status: str = "ready"
    backends: dict[str, str] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)


def _decode_image_bytes(raw_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR numpy array."""
    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty image data provided",
        )
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode image. Ensure format is JPEG, PNG, or WEBP.",
        )
    return image


def _extract_from_frame(
    frame_bgr: np.ndarray,
    include_pose: bool = True,
    include_face: bool = True,
    include_hands: bool = True,
    include_segmentation: bool = True,
) -> tuple[PoseResult | None, FaceTrackingResult | None, HandTrackingResult | None, SegmentationResponse | None]:
    """Run requested vision models on a single frame."""
    pose_res: PoseResult | None = None
    face_res: FaceTrackingResult | None = None
    hands_res: HandTrackingResult | None = None
    seg_resp: SegmentationResponse | None = None

    if include_pose:
        pose_res = get_pose_estimator().estimate(frame_bgr)

    if include_face:
        face_res = get_face_tracker().track(frame_bgr)

    if include_hands:
        hands_res = get_hand_tracker().track(frame_bgr)

    if include_segmentation:
        bbox = pose_res.bounding_box if (pose_res and pose_res.has_pose) else None
        seg_res = get_person_segmenter().segment(frame_bgr, bounding_box=bbox)
        bin_b64, alpha_b64 = seg_res.to_base64_masks()
        seg_resp = SegmentationResponse(
            metadata=seg_res.metadata,
            binary_mask_base64=bin_b64,
            alpha_mask_base64=alpha_b64,
        )

    return pose_res, face_res, hands_res, seg_resp


@router.get("/status", response_model=VisionStatusResponse)
async def get_vision_status() -> VisionStatusResponse:
    """Return status of vision models and whether MediaPipe or OpenCV fallback is active."""
    pose = get_pose_estimator()
    face = get_face_tracker()
    hands = get_hand_tracker()
    seg = get_person_segmenter()

    backends = {
        "pose": pose.backend,
        "face": face.backend,
        "hands": hands.backend,
        "segmentation": seg.backend,
    }

    capabilities = [
        "33_landmark_body_pose",
        "facial_landmarks_tracking",
        "eye_blink_ear_analysis",
        "mouth_open_mar_analysis",
        "21_landmark_bimanual_hand_tracking",
        "person_binary_and_alpha_segmentation",
        "background_isolation",
    ]

    return VisionStatusResponse(
        status="ready",
        backends=backends,
        capabilities=capabilities,
    )


@router.post("/extract", response_model=VisionExtractResponse)
async def extract_vision(
    file: UploadFile = File(...),
    include_pose: Annotated[bool, Query()] = True,
    include_face: Annotated[bool, Query()] = True,
    include_hands: Annotated[bool, Query()] = True,
    include_segmentation: Annotated[bool, Query()] = True,
    frame_index: Annotated[int, Query()] = 0,
) -> VisionExtractResponse:
    """Extract pose, face, hand landmarks, and person segmentation mask from an image or video."""
    t0 = time.time()
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    filename = (file.filename or "").lower()
    is_video = any(filename.endswith(ext) for ext in [".mp4", ".webm", ".avi", ".mov", ".mkv"])

    frame_bgr: np.ndarray

    if is_video:
        extractor = VideoFrameExtractor(raw_bytes)
        try:
            frames = extractor.extract_all(max_frames=frame_index + 1)
            if not frames:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Could not extract any valid video frames",
                )
            target_frame = frames[min(frame_index, len(frames) - 1)]
            frame_bgr = target_frame.to_bgr()
        finally:
            extractor.cleanup()
    else:
        frame_bgr = _decode_image_bytes(raw_bytes)

    h, w = frame_bgr.shape[:2]
    pose_res, face_res, hands_res, seg_resp = _extract_from_frame(
        frame_bgr=frame_bgr,
        include_pose=include_pose,
        include_face=include_face,
        include_hands=include_hands,
        include_segmentation=include_segmentation,
    )

    t1 = time.time()
    elapsed_ms = round((t1 - t0) * 1000.0, 2)

    return VisionExtractResponse(
        width=w,
        height=h,
        processing_time_ms=elapsed_ms,
        is_video=is_video,
        frames_processed=1,
        pose=pose_res,
        face=face_res,
        hands=hands_res,
        segmentation=seg_resp,
    )


@router.post("/pose", response_model=PoseResult)
async def extract_pose(file: UploadFile = File(...)) -> PoseResult:
    """Extract 33 body landmarks from an uploaded image."""
    raw_bytes = await file.read()
    frame = _decode_image_bytes(raw_bytes)
    return get_pose_estimator().estimate(frame)


@router.post("/face", response_model=FaceTrackingResult)
async def extract_face(file: UploadFile = File(...)) -> FaceTrackingResult:
    """Extract facial landmarks and blink/mouth ratios from an uploaded image."""
    raw_bytes = await file.read()
    frame = _decode_image_bytes(raw_bytes)
    return get_face_tracker().track(frame)


@router.post("/hands", response_model=HandTrackingResult)
async def extract_hands(file: UploadFile = File(...)) -> HandTrackingResult:
    """Extract left and right hand 21 landmarks from an uploaded image."""
    raw_bytes = await file.read()
    frame = _decode_image_bytes(raw_bytes)
    return get_hand_tracker().track(frame)


@router.post("/segment", response_model=SegmentationResponse)
async def extract_segmentation(file: UploadFile = File(...)) -> SegmentationResponse:
    """Extract binary and alpha segmentation masks from an uploaded image."""
    raw_bytes = await file.read()
    frame = _decode_image_bytes(raw_bytes)
    res = get_person_segmenter().segment(frame)
    bin_b64, alpha_b64 = res.to_base64_masks()
    return SegmentationResponse(
        metadata=res.metadata,
        binary_mask_base64=bin_b64,
        alpha_mask_base64=alpha_b64,
    )
