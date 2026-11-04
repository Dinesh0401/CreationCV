"""MediaPipe Neural Tracking and Segmentation Verification Script.

Executes genuine Google MediaPipe neural inference on real human imagery:
- PoseLandmarker (33 landmarks)
- FaceLandmarker (478 mesh points + EAR / MAR)
- HandLandmarker (21 landmarks per hand)
- ImageSegmenter (Selfie neural segmentation)

Saves annotated verification images and produces full metrics.
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
import mediapipe as mp
import numpy as np

from backend.vision.face_tracking import FaceTracker
from backend.vision.hand_tracking import HandTracker
from backend.vision.pose import POSE_CONNECTIONS, PoseEstimator
from backend.vision.segmentation import PersonSegmenter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_mediapipe")


def run_mediapipe_verification() -> dict:
    assets_dir = PROJECT_ROOT / "recordings" / "real_human"
    models_dir = PROJECT_ROOT / "models"
    output_dir = assets_dir

    report: dict = {
        "status": "started",
        "mediapipe_version": mp.__version__,
        "model_files": {},
        "pose_verification": {},
        "face_verification": {},
        "hand_verification": {},
        "segmentation_verification": {},
    }

    # 1. Model Files Audit
    models_to_check = [
        "pose_landmarker_heavy.task",
        "face_landmarker.task",
        "hand_landmarker.task",
        "selfie_segmenter.task",
    ]
    for m in models_to_check:
        p = models_dir / m
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        report["model_files"][m] = {
            "exists": exists,
            "size_bytes": size,
            "path": str(p),
        }
        logger.info("Model %s: exists=%s, size=%d bytes", m, exists, size)

    # 2. Real Pose Verification
    pose_img_path = assets_dir / "real_pose.jpg"
    pose_bgr = cv2.imread(str(pose_img_path))
    if pose_bgr is None:
        raise FileNotFoundError(f"Missing {pose_img_path}")

    pose_estimator = PoseEstimator()
    t0 = time.time()
    pose_result = pose_estimator.estimate(pose_bgr)
    pose_time_ms = round((time.time() - t0) * 1000.0, 2)

    # Annotate pose
    pose_annotated = pose_bgr.copy()
    h, w = pose_annotated.shape[:2]
    pts = {}
    for lm in pose_result.landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        pts[lm.index] = (px, py)
        cv2.circle(pose_annotated, (px, py), 5, (0, 0, 255), -1)
        cv2.circle(pose_annotated, (px, py), 2, (255, 255, 255), -1)

    for p1, p2 in POSE_CONNECTIONS:
        if p1 in pts and p2 in pts:
            cv2.line(pose_annotated, pts[p1], pts[p2], (0, 255, 0), 2)

    pose_out_path = output_dir / "verified_mediapipe_pose.png"
    cv2.imwrite(str(pose_out_path), pose_annotated)

    avg_vis = float(np.mean([lm.visibility for lm in pose_result.landmarks])) if pose_result.landmarks else 0.0

    report["pose_verification"] = {
        "backend": pose_result.backend,
        "is_genuine_mediapipe": (pose_result.backend == "mediapipe"),
        "has_pose": pose_result.has_pose,
        "landmark_count": len(pose_result.landmarks),
        "average_visibility": round(avg_vis, 4),
        "confidence": pose_result.confidence,
        "bounding_box": pose_result.bounding_box,
        "inference_time_ms": pose_time_ms,
        "device": "CPU (TensorFlow Lite XNNPACK delegate)",
        "annotated_preview": str(pose_out_path),
    }
    logger.info("Pose verification: %s", report["pose_verification"])

    # 3. Real Face Verification
    face_img_path = assets_dir / "real_portrait.jpg"
    face_bgr = cv2.imread(str(face_img_path))
    if face_bgr is None:
        raise FileNotFoundError(f"Missing {face_img_path}")

    face_tracker = FaceTracker()
    t0 = time.time()
    face_result = face_tracker.track(face_bgr)
    face_time_ms = round((time.time() - t0) * 1000.0, 2)

    # Annotate face mesh
    face_annotated = face_bgr.copy()
    fh, fw = face_annotated.shape[:2]
    for lm in face_result.landmarks:
        px, py = int(lm.x * fw), int(lm.y * fh)
        cv2.circle(face_annotated, (px, py), 1, (0, 255, 255), -1)

    face_out_path = output_dir / "verified_mediapipe_face.png"
    cv2.imwrite(str(face_out_path), face_annotated)

    report["face_verification"] = {
        "backend": face_result.backend,
        "is_genuine_mediapipe": (face_result.backend == "mediapipe"),
        "has_face": face_result.has_face,
        "landmark_count": face_result.landmark_count,
        "confidence": face_result.confidence,
        "ear_left": face_result.blink.left_eye_ear,
        "ear_right": face_result.blink.right_eye_ear,
        "is_blinking": face_result.blink.is_blinking,
        "mar": face_result.mouth.mouth_aspect_ratio,
        "is_mouth_open": face_result.mouth.is_mouth_open,
        "bounding_box": face_result.bounding_box,
        "inference_time_ms": face_time_ms,
        "device": "CPU (TensorFlow Lite XNNPACK delegate)",
        "annotated_preview": str(face_out_path),
    }
    logger.info("Face verification: %s", report["face_verification"])

    # 4. Real Hand Verification
    hand_img_path = assets_dir / "real_hand_thumbsup.jpg"
    hand_bgr = cv2.imread(str(hand_img_path))
    if hand_bgr is None:
        raise FileNotFoundError(f"Missing {hand_img_path}")

    hand_tracker = HandTracker()
    t0 = time.time()
    hand_result = hand_tracker.track(hand_bgr)
    hand_time_ms = round((time.time() - t0) * 1000.0, 2)

    # Annotate hand
    hand_annotated = hand_bgr.copy()
    hh, hw = hand_annotated.shape[:2]
    for h_obj in hand_result.hands:
        for lm in h_obj.landmarks:
            px, py = int(lm.x * hw), int(lm.y * hh)
            cv2.circle(hand_annotated, (px, py), 4, (0, 165, 255), -1)
            cv2.circle(hand_annotated, (px, py), 2, (255, 255, 255), -1)

    hand_out_path = output_dir / "verified_mediapipe_hand.png"
    cv2.imwrite(str(hand_out_path), hand_annotated)

    detected_hands_summary = []
    for h_obj in hand_result.hands:
        detected_hands_summary.append({
            "handedness": h_obj.handedness,
            "confidence": h_obj.confidence,
            "landmarks_count": len(h_obj.landmarks),
            "is_open": h_obj.is_open,
            "is_fist": h_obj.is_fist,
        })

    report["hand_verification"] = {
        "backend": hand_result.backend,
        "is_genuine_mediapipe": (hand_result.backend == "mediapipe"),
        "has_hands": hand_result.has_hands,
        "hands_detected_count": len(hand_result.hands),
        "hands": detected_hands_summary,
        "inference_time_ms": hand_time_ms,
        "device": "CPU (TensorFlow Lite XNNPACK delegate)",
        "annotated_preview": str(hand_out_path),
    }
    logger.info("Hand verification: %s", report["hand_verification"])

    # 5. Real Segmentation Verification
    seg_img_path = assets_dir / "real_person.png"
    seg_bgr = cv2.imread(str(seg_img_path))
    if seg_bgr is None:
        raise FileNotFoundError(f"Missing {seg_img_path}")

    segmenter = PersonSegmenter()
    t0 = time.time()
    seg_result = segmenter.segment(seg_bgr)
    seg_time_ms = round((time.time() - t0) * 1000.0, 2)

    # Save segmentation outputs
    orig_path = output_dir / "verified_segmentation_original.png"
    bin_path = output_dir / "verified_segmentation_binary_mask.png"
    alpha_path = output_dir / "verified_segmentation_alpha_mask.png"
    fg_path = output_dir / "verified_segmentation_foreground_rgba.png"
    bg_path = output_dir / "verified_segmentation_isolated_bg.png"

    cv2.imwrite(str(orig_path), seg_bgr)
    cv2.imwrite(str(bin_path), seg_result.binary_mask)
    cv2.imwrite(str(alpha_path), (seg_result.alpha_mask * 255.0).astype(np.uint8))
    cv2.imwrite(str(fg_path), seg_result.foreground_rgba)
    cv2.imwrite(str(bg_path), seg_result.background_isolated)

    report["segmentation_verification"] = {
        "backend": seg_result.metadata.backend,
        "is_genuine_mediapipe": (seg_result.metadata.backend == "mediapipe"),
        "has_person": seg_result.metadata.has_person,
        "person_coverage_ratio": seg_result.metadata.person_coverage_ratio,
        "resolution": [seg_result.metadata.width, seg_result.metadata.height],
        "inference_time_ms": seg_time_ms,
        "device": "CPU (TensorFlow Lite XNNPACK delegate)",
        "saved_files": {
            "original": str(orig_path),
            "binary_mask": str(bin_path),
            "alpha_mask": str(alpha_path),
            "foreground_rgba": str(fg_path),
            "isolated_background": str(bg_path),
        },
    }
    logger.info("Segmentation verification: %s", report["segmentation_verification"])

    report["status"] = "completed"

    report_path = output_dir / "mediapipe_verification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    res = run_mediapipe_verification()
    print("\n=== MEDIAPIPE NEURAL VERIFICATION RESULT ===")
    print(json.dumps(res, indent=2))

