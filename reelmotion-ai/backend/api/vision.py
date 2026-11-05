"""Vision API router for ReelMotion AI.

Exposes endpoints for pose estimation, face landmark tracking (blink + mouth),
hand tracking, and person segmentation / background isolation.
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.capture.camera import VideoFrameExtractor
from backend.vision.face_tracking import FaceTracker, FaceTrackingResult
from backend.vision.hand_tracking import HandTracker, HandTrackingResult
from backend.vision.pose import POSE_CONNECTIONS, PoseEstimator, PoseResult
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


def _overlay_rgba(
    background: np.ndarray,
    foreground_rgba: np.ndarray,
    x: int,
    y: int,
    opacity: float = 1.0,
) -> np.ndarray:
    """Alpha-composite an RGBA foreground image onto a BGR background image."""
    bh, bw = background.shape[:2]
    fh, fw = foreground_rgba.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bw, x + fw), min(bh, y + fh)

    fx1, fy1 = max(0, -x), max(0, -y)
    fx2, fy2 = fx1 + (x2 - x1), fy1 + (y2 - y1)

    if x2 <= x1 or y2 <= y1 or fx2 <= fx1 or fy2 <= fy1:
        return background

    roi = background[y1:y2, x1:x2]
    fg_crop = foreground_rgba[fy1:fy2, fx1:fx2]

    fg_rgb = fg_crop[:, :, :3]
    alpha = (fg_crop[:, :, 3].astype(np.float32) / 255.0) * opacity
    alpha_3c = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

    background[y1:y2, x1:x2] = (fg_rgb * alpha_3c + roi * (1.0 - alpha_3c)).astype(np.uint8)
    return background


def _generate_live_frames(
    camera_idx: int = 0,
    show_char: bool = True,
    char_mode: str = "FULL",
    char_opacity: float = 1.0,
    show_pose: bool = False,
    show_face: bool = False,
    show_hands: bool = False,
    show_mask: bool = False,
    blur_bg: bool = False,
):
    """Generator yielding MJPEG multipart frames from live webcam with AI character overlays."""
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, f"Could not open camera {camera_idx}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        _, jpeg = cv2.imencode(".jpg", blank)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        return

    # Load character RGBA
    chars_dir = Path(__file__).resolve().parent.parent.parent / "characters"
    char_path = chars_dir / "anime_girl_rgba.png"
    if not char_path.exists():
        char_path = chars_dir / "anime_girl_reference.png"

    char_rgba = None
    if char_path.exists():
        loaded = cv2.imread(str(char_path), cv2.IMREAD_UNCHANGED)
        if loaded is not None:
            if len(loaded.shape) == 2:
                char_rgba = cv2.cvtColor(loaded, cv2.COLOR_GRAY2BGRA)
            elif loaded.shape[2] == 3:
                gray = cv2.cvtColor(loaded, cv2.COLOR_BGR2GRAY)
                alpha = np.where(gray > 15, 255, 0).astype(np.uint8)
                char_rgba = cv2.cvtColor(loaded, cv2.COLOR_BGR2BGRA)
                char_rgba[:, :, 3] = alpha
            else:
                char_rgba = loaded

    pose_est = get_pose_estimator()
    face_trk = get_face_tracker()
    hand_trk = get_hand_tracker()
    seg_mod = get_person_segmenter()

    last_seg_time = 0.0
    cached_seg = None
    fps = 0.0
    start_t = time.time()
    frames = 0

    # Smoothing filters
    sm_x, sm_y, sm_w, sm_h = None, None, None, None
    alpha_filter = 0.45

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            display = frame.copy()

            # Vision Tracking
            p_res = pose_est.estimate(frame)
            f_res = face_trk.track(frame)
            h_res = hand_trk.track(frame) if show_hands else None

            # Segmentation (sampled for background blur or mask)
            now = time.time()
            if (show_mask or blur_bg) and (cached_seg is None or (now - last_seg_time) > 0.09):
                cached_seg = seg_mod.segment(frame)
                last_seg_time = now

            # Background blur
            if blur_bg and cached_seg and cached_seg.alpha_mask is not None:
                alpha = cached_seg.alpha_mask
                if alpha.shape[:2] != (h, w):
                    alpha = cv2.resize(alpha, (w, h))
                a3 = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
                bg = cv2.GaussianBlur(frame, (45, 45), 0)
                display = (frame * a3 + bg * (1.0 - a3)).astype(np.uint8)

            # Character Overlay Calculation
            if show_char and char_rgba is not None:
                target_bbox = None
                if char_mode in ["FULL", "HOLOGRAM"]:
                    if p_res and p_res.has_pose:
                        cy = [lm.y * h for lm in p_res.landmarks if lm.visibility > 0.25]
                        cx = [lm.x * w for lm in p_res.landmarks if lm.visibility > 0.25]
                        if cx and cy:
                            min_x, max_x = min(cx), max(cx)
                            min_y, max_y = min(cy), max(cy)
                            user_cx = (min_x + max_x) / 2.0
                            pad_h = (max_y - min_y) * 0.12
                            user_h = max(100.0, (max_y - min_y) + pad_h * 2)
                            ch_h, ch_w = char_rgba.shape[:2]
                            user_w = user_h * (ch_w / max(1, ch_h))
                            target_bbox = (user_cx - user_w / 2.0, max(0.0, min_y - pad_h), user_w, user_h)

                    elif f_res and f_res.has_face and f_res.bounding_box:
                        bx1, by1, bx2, by2 = f_res.bounding_box
                        face_w = (bx2 - bx1) * w
                        face_h = (by2 - by1) * h
                        face_cx = (bx1 + bx2) / 2.0 * w
                        ch_h, ch_w = char_rgba.shape[:2]
                        user_h = face_h * 3.8
                        user_w = user_h * (ch_w / max(1, ch_h))
                        target_bbox = (face_cx - user_w / 2.0, max(0.0, by1 * h - face_h * 0.25), user_w, user_h)

                elif char_mode == "HEAD":
                    if f_res and f_res.has_face and f_res.bounding_box:
                        bx1, by1, bx2, by2 = f_res.bounding_box
                        face_w = (bx2 - bx1) * w * 1.5
                        face_h = (by2 - by1) * h * 1.5
                        face_cx = (bx1 + bx2) / 2.0 * w
                        face_cy = (by1 + by2) / 2.0 * h
                        target_bbox = (face_cx - face_w / 2.0, face_cy - face_h / 2.0, face_w, face_h)

                if target_bbox is not None:
                    bx, by, bw, bh = target_bbox
                    if sm_x is None:
                        sm_x, sm_y, sm_w, sm_h = bx, by, bw, bh
                    else:
                        sm_x = (1.0 - alpha_filter) * sm_x + alpha_filter * bx
                        sm_y = (1.0 - alpha_filter) * sm_y + alpha_filter * by
                        sm_w = (1.0 - alpha_filter) * sm_w + alpha_filter * bw
                        sm_h = (1.0 - alpha_filter) * sm_h + alpha_filter * bh

                    tw, th = max(10, int(sm_w)), max(10, int(sm_h))
                    tx, ty = int(sm_x), int(sm_y)

                    scaled = cv2.resize(char_rgba, (tw, th), interpolation=cv2.INTER_LINEAR)
                    eff_op = 0.55 if char_mode == "HOLOGRAM" else char_opacity
                    _overlay_rgba(display, scaled, tx, ty, opacity=eff_op)

            # Person mask overlay
            if show_mask and not blur_bg and cached_seg and cached_seg.binary_mask is not None:
                mask = cached_seg.binary_mask
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h))
                tint = np.zeros_like(display)
                tint[:, :] = (255, 120, 0)
                display = np.where(mask[:, :, np.newaxis] > 128, cv2.addWeighted(display, 0.7, tint, 0.3, 0), display)

            # Skeleton
            if show_pose and p_res and p_res.has_pose:
                coords = {}
                for lm in p_res.landmarks:
                    if lm.visibility > 0.3:
                        px, py = int(lm.x * w), int(lm.y * h)
                        coords[lm.index] = (px, py)
                        cv2.circle(display, (px, py), 4, (0, 255, 120), -1)
                for c1, c2 in POSE_CONNECTIONS:
                    if c1 in coords and c2 in coords:
                        cv2.line(display, coords[c1], coords[c2], (0, 240, 255), 2)

            # Face mesh
            if show_face and f_res and f_res.has_face:
                step = 2 if len(f_res.landmarks) > 100 else 1
                for i in range(0, len(f_res.landmarks), step):
                    lm = f_res.landmarks[i]
                    px, py = int(lm.x * w), int(lm.y * h)
                    cv2.circle(display, (px, py), 1, (255, 180, 50), -1)
                if f_res.bounding_box:
                    bx1, by1 = int(f_res.bounding_box[0] * w), int(f_res.bounding_box[1] * h)
                    info = []
                    if f_res.blink.is_blinking:
                        info.append("BLINK")
                    if f_res.mouth.is_mouth_open:
                        info.append("TALKING")
                    if info:
                        cv2.putText(display, " - ".join(info), (bx1, max(20, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Hands
            if show_hands and h_res:
                hand_conns = [
                    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
                ]
                for hand in [h_res.left_hand, h_res.right_hand]:
                    if not hand:
                        continue
                    b_col = (255, 200, 0) if hand.handedness == "Left" else (0, 220, 255)
                    h_pts = {}
                    for lm in hand.landmarks:
                        px, py = int(lm.x * w), int(lm.y * h)
                        h_pts[lm.index] = (px, py)
                        cv2.circle(display, (px, py), 3, (0, 255, 120), -1)
                    for p1, p2 in hand_conns:
                        if p1 in h_pts and p2 in h_pts:
                            cv2.line(display, h_pts[p1], h_pts[p2], b_col, 2)

            # FPS
            frames += 1
            el = time.time() - start_t
            if el >= 0.5:
                fps = frames / el
                frames = 0
                start_t = time.time()

            cv2.putText(display, f"REELMOTION AI | {fps:.1f} FPS | CHAR: {'ON' if show_char else 'OFF'}", (16, 32), cv2.FONT_HERSHEY_DUPLEX, 0.60, (0, 240, 255), 1)

            ret_enc, jpeg = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret_enc:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
    finally:
        cap.release()


@router.get("/stream")
def live_stream(
    camera: int = Query(default=0, ge=0),
    char: bool = Query(default=True),
    mode: str = Query(default="FULL"),
    opacity: float = Query(default=1.0, ge=0.1, le=1.0),
    pose: bool = Query(default=False),
    face: bool = Query(default=False),
    hands: bool = Query(default=False),
    mask: bool = Query(default=False),
    blur: bool = Query(default=False),
) -> StreamingResponse:
    """Stream live webcam video with real-time AI character and vision tracking overlays."""
    return StreamingResponse(
        _generate_live_frames(
            camera_idx=camera,
            show_char=char,
            char_mode=mode,
            char_opacity=opacity,
            show_pose=pose,
            show_face=face,
            show_hands=hands,
            show_mask=mask,
            blur_bg=blur,
        ),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/live", response_class=HTMLResponse)
def live_viewer_page() -> str:
    """Return dark-mode interactive Live Character Studio HTML web interface."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ReelMotion AI — Live Character Studio</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0d0f17;
      color: #e2e8f0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }
    header {
      background: #151824;
      border-bottom: 1px solid #242938;
      padding: 12px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .badge {
      background: linear-gradient(135deg, #00f0ff, #7000ff);
      color: #fff;
      font-weight: 700;
      font-size: 11px;
      padding: 4px 8px;
      border-radius: 4px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    h1 { font-size: 18px; font-weight: 600; color: #f8fafc; }
    .controls {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .btn {
      background: #1e2436;
      border: 1px solid #333b54;
      color: #cbd5e1;
      padding: 7px 14px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .btn:hover { background: #2a334d; color: #fff; border-color: #00f0ff; }
    .btn.active {
      background: rgba(0, 240, 255, 0.18);
      border-color: #00f0ff;
      color: #00f0ff;
      box-shadow: 0 0 10px rgba(0, 240, 255, 0.25);
    }
    .btn.primary {
      background: linear-gradient(135deg, #00f0ff, #0077ff);
      color: #000;
      font-weight: 700;
      border: none;
    }
    .main-view {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #090b10;
      padding: 16px;
      position: relative;
    }
    .video-container {
      position: relative;
      max-width: 100%;
      max-height: 100%;
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 10px 40px rgba(0,0,0,0.8), 0 0 24px rgba(0, 240, 255, 0.25);
      border: 1px solid #22293d;
    }
    img.live-stream {
      display: block;
      width: 100%;
      max-height: 82vh;
      object-fit: contain;
      border-radius: 12px;
    }
    footer {
      background: #151824;
      border-top: 1px solid #242938;
      padding: 10px 24px;
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      color: #94a3b8;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span class="badge">LIVE AR</span>
      <h1>ReelMotion AI — Character Studio</h1>
    </div>
    <div class="controls">
      <button class="btn active" id="btn-char" onclick="toggleParam('char')">🎭 Character: ON</button>
      <button class="btn active" id="btn-mode" onclick="cycleMode()">Mode: FULL</button>
      <button class="btn" id="btn-opacity" onclick="cycleOpacity()">Opacity: 100%</button>
      <button class="btn" id="btn-blur" onclick="toggleParam('blur')">Background Blur</button>
      <button class="btn" id="btn-pose" onclick="toggleParam('pose')">Pose Skeleton</button>
      <button class="btn" id="btn-face" onclick="toggleParam('face')">Face Mesh</button>
    </div>
  </header>

  <main class="main-view">
    <div class="video-container">
      <img id="stream" class="live-stream" src="/api/vision/stream?char=1&mode=FULL&opacity=1.0&pose=0&face=0&hands=0&mask=0&blur=0" alt="Live Character Video Stream" />
    </div>
  </main>

  <footer>
    <div>Model: Anime Girl Reference | Live Motion Tracking & AR Overlay (30 FPS)</div>
    <div>Local GPU Inference Engine</div>
  </footer>

  <script>
    const state = {
      char: true,
      mode: 'FULL',
      opacity: 1.0,
      pose: false,
      face: false,
      hands: false,
      mask: false,
      blur: false
    };

    const modes = ['FULL', 'HOLOGRAM', 'HEAD'];
    let modeIdx = 0;
    const opacities = [1.0, 0.8, 0.6, 0.4];
    let opIdx = 0;

    function updateStream() {
      const img = document.getElementById('stream');
      const q = new URLSearchParams({
        char: state.char ? '1' : '0',
        mode: state.mode,
        opacity: state.opacity.toFixed(1),
        pose: state.pose ? '1' : '0',
        face: state.face ? '1' : '0',
        hands: state.hands ? '1' : '0',
        mask: state.mask ? '1' : '0',
        blur: state.blur ? '1' : '0',
        t: Date.now()
      });
      img.src = '/api/vision/stream?' + q.toString();
    }

    function toggleParam(name) {
      state[name] = !state[name];
      const btn = document.getElementById('btn-' + name);
      if (btn) {
        if (state[name]) btn.classList.add('active');
        else btn.classList.remove('active');
        if (name === 'char') {
          btn.textContent = '🎭 Character: ' + (state.char ? 'ON' : 'OFF');
        }
      }
      updateStream();
    }

    function cycleMode() {
      modeIdx = (modeIdx + 1) % modes.length;
      state.mode = modes[modeIdx];
      const btn = document.getElementById('btn-mode');
      if (btn) btn.textContent = 'Mode: ' + state.mode;
      updateStream();
    }

    function cycleOpacity() {
      opIdx = (opIdx + 1) % opacities.length;
      state.opacity = opacities[opIdx];
      const btn = document.getElementById('btn-opacity');
      if (btn) btn.textContent = 'Opacity: ' + Math.round(state.opacity * 100) + '%';
      updateStream();
    }
  </script>
</body>
</html>
"""


