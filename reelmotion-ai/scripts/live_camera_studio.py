"""ReelMotion AI — Live Vision Camera Studio.

Interactive real-time webcam viewer with live MediaPipe/OpenCV tracking:
- 33-point Pose Landmark Skeleton
- 478-point Face Mesh with Eye Blink (EAR) and Mouth Openness (MAR) detection
- 21-point Left and Right Hand Landmark Tracking
- Person Segmentation Mask and Background Isolation / Blur
- Side-by-Side split view and Snapshot capture
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from backend.vision.face_tracking import FaceTracker
from backend.vision.hand_tracking import HandTracker
from backend.vision.pose import POSE_CONNECTIONS, PoseEstimator
from backend.vision.segmentation import PersonSegmenter


def draw_hud(
    canvas: np.ndarray,
    fps: float,
    has_pose: bool,
    has_face: bool,
    blink_state: bool,
    mouth_state: bool,
    has_hands: tuple[bool, bool],
    has_mask: bool,
    mask_coverage: float,
    mode_name: str,
    show_pose: bool,
    show_face: bool,
    show_hands: bool,
    show_mask: bool,
) -> None:
    """Render a modern studio HUD overlay on the canvas."""
    h, w = canvas.shape[:2]

    # Top header banner (semi-transparent dark bar)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 54), (15, 17, 26), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    # Title & FPS
    cv2.putText(
        canvas,
        "REELMOTION AI - LIVE VISION STUDIO",
        (16, 26),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        (0, 240, 255),  # Cyan
        1,
        cv2.LINE_AA,
    )
    fps_color = (0, 255, 120) if fps >= 20 else (0, 200, 255) if fps >= 10 else (0, 70, 255)
    cv2.putText(
        canvas,
        f"{fps:.1f} FPS",
        (16, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        fps_color,
        1,
        cv2.LINE_AA,
    )

    # Status pills
    x_pos = 380
    badges = [
        (f"POSE: {'ON' if show_pose else 'OFF'}", (0, 255, 120) if has_pose and show_pose else (120, 120, 120)),
        (f"FACE: {'ON' if show_face else 'OFF'}", (0, 255, 120) if has_face and show_face else (120, 120, 120)),
        (f"BLINK: {'YES' if blink_state else 'NO'}", (0, 200, 255) if blink_state else (160, 160, 160)),
        (f"MOUTH: {'OPEN' if mouth_state else 'CLOSED'}", (255, 100, 255) if mouth_state else (160, 160, 160)),
        (f"HANDS: {'L' if has_hands[0] else '-'}/{'R' if has_hands[1] else '-'}", (0, 255, 200) if (has_hands[0] or has_hands[1]) and show_hands else (120, 120, 120)),
        (f"MASK: {mask_coverage*100:.0f}%", (255, 180, 50) if has_mask and show_mask else (120, 120, 120)),
    ]

    for label, col in badges:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(canvas, (x_pos, 14), (x_pos + tw + 14, 38), (30, 32, 45), -1)
        cv2.rectangle(canvas, (x_pos, 14), (x_pos + tw + 14, 38), col, 1)
        cv2.putText(canvas, label, (x_pos + 7, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        x_pos += tw + 20

    # Bottom helper bar
    cv2.rectangle(overlay, (0, h - 34), (w, h), (15, 17, 26), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
    help_text = "[1] Pose | [2] Face | [3] Hands | [4] Mask | [B] Background Blur | [TAB] Split View | [SPACE] Snap | [Q] Quit"
    cv2.putText(
        canvas,
        help_text,
        (16, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (200, 205, 215),
        1,
        cv2.LINE_AA,
    )


def draw_face_overlay(canvas: np.ndarray, face_res: any) -> None:
    """Draw facial mesh landmarks and bounding box."""
    if not face_res or not face_res.has_face:
        return

    h, w = canvas.shape[:2]

    # Draw bounding box if present
    if face_res.bounding_box:
        xmin, ymin, xmax, ymax = face_res.bounding_box
        bx1, by1 = int(xmin * w), int(ymin * h)
        bx2, by2 = int(xmax * w), int(ymax * h)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 220, 255), 1, cv2.LINE_AA)

    # Sample landmarks to avoid rendering clutter (every 3rd or key points)
    step = 2 if len(face_res.landmarks) > 100 else 1
    for i in range(0, len(face_res.landmarks), step):
        lm = face_res.landmarks[i]
        px = int(lm.x * w)
        py = int(lm.y * h)
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(canvas, (px, py), 1, (255, 180, 50), -1, cv2.LINE_AA)

    # Highlight eye and mouth state near face
    if face_res.bounding_box:
        bx1, by1 = int(face_res.bounding_box[0] * w), int(face_res.bounding_box[1] * h)
        status_line = []
        if face_res.blink.is_blinking:
            status_line.append("BLINK")
        if face_res.mouth.is_mouth_open:
            status_line.append("TALKING")
        if status_line:
            cv2.putText(
                canvas,
                " • ".join(status_line),
                (bx1, max(20, by1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )


def draw_hands_overlay(canvas: np.ndarray, hands_res: any) -> None:
    """Draw hand skeletons and landmarks."""
    if not hands_res:
        return

    h, w = canvas.shape[:2]
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),  # Pinky + Palm
    ]

    for hand in [hands_res.left_hand, hands_res.right_hand]:
        if not hand:
            continue
        # Color based on handedness: cyan for left, yellow for right
        bone_color = (255, 200, 0) if hand.handedness == "Left" else (0, 220, 255)
        joint_color = (0, 255, 120)

        coords = {}
        for lm in hand.landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            coords[lm.index] = (px, py)
            cv2.circle(canvas, (px, py), 3, joint_color, -1, cv2.LINE_AA)

        for p1, p2 in hand_connections:
            if p1 in coords and p2 in coords:
                cv2.line(canvas, coords[p1], coords[p2], bone_color, 2, cv2.LINE_AA)

        if 0 in coords:
            cv2.putText(
                canvas,
                hand.handedness.upper(),
                (coords[0][0] - 10, coords[0][1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                bone_color,
                1,
                cv2.LINE_AA,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="ReelMotion AI Live Camera Studio")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Camera width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera height (default: 720)")
    args = parser.parse_args()

    print("=====================================================")
    print("  ReelMotion AI — Live Vision Camera Studio")
    print("=====================================================")
    print(f"Opening camera index {args.camera} ({args.width}x{args.height})...")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"CAP_DSHOW failed, falling back to default API...")
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera {args.camera}!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera active: {actual_w}x{actual_h}")
    print("Initializing MediaPipe neural models...")

    pose_est = PoseEstimator()
    face_trk = FaceTracker()
    hand_trk = HandTracker()
    seg_mod = PersonSegmenter()

    print("Vision models ready. Starting live stream...")
    print("Controls: [1] Pose, [2] Face, [3] Hands, [4] Mask, [B] Background Blur, [TAB] Split View, [SPACE] Snapshot, [Q] Quit")

    window_name = "ReelMotion AI - Live Vision Studio"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1280, actual_w), min(720, actual_h))

    # Toggles
    show_pose = True
    show_face = True
    show_hands = True
    show_mask = True
    bg_mode = False  # False = Normal/Mask, True = Blurred/Isolated Background
    split_view = False

    fps = 0.0
    frame_count = 0
    start_time = time.time()
    last_seg_time = 0.0
    cached_seg_res = None

    snapshots_dir = PROJECT_ROOT / "recordings" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: Failed to capture camera frame. Retrying...")
                time.sleep(0.05)
                continue

            # Mirror frame horizontally for natural selfie view
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            h, w = frame.shape[:2]

            # 1. Pose estimation
            pose_res = pose_est.estimate(frame) if show_pose else None

            # 2. Face tracking
            face_res = face_trk.track(frame) if show_face else None

            # 3. Hand tracking
            hands_res = hand_trk.track(frame) if show_hands else None

            # 4. Person segmentation (update every ~100ms or 3 frames to maintain 30 FPS)
            now = time.time()
            if show_mask or bg_mode:
                if cached_seg_res is None or (now - last_seg_time) > 0.08:
                    cached_seg_res = seg_mod.segment(frame)
                    last_seg_time = now

            # Apply Background Blur / Isolation if active
            if bg_mode and cached_seg_res and cached_seg_res.alpha_mask is not None:
                alpha = cached_seg_res.alpha_mask
                if alpha.shape[:2] != (h, w):
                    alpha = cv2.resize(alpha, (w, h))
                alpha_3c = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

                # Create blurred background
                bg_blurred = cv2.GaussianBlur(frame, (45, 45), 0)
                # Composite: foreground + blurred background
                display_frame = (frame * alpha_3c + bg_blurred * (1.0 - alpha_3c)).astype(np.uint8)

            # Apply translucent person segmentation mask overlay
            if show_mask and not bg_mode and cached_seg_res and cached_seg_res.binary_mask is not None:
                mask = cached_seg_res.binary_mask
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h))
                # Create glowing neon cyan mask tint
                tint = np.zeros_like(display_frame)
                tint[:, :] = (255, 120, 0)  # Neon cyan/azure in BGR
                display_frame = np.where(mask[:, :, np.newaxis] > 128, cv2.addWeighted(display_frame, 0.7, tint, 0.3, 0), display_frame)
                # Draw subtle silhouette contour
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display_frame, contours, -1, (0, 240, 255), 2)

            # Draw Pose Skeleton
            if show_pose and pose_res and pose_res.has_pose:
                coords = {}
                for lm in pose_res.landmarks:
                    if lm.visibility > 0.3:
                        px = int(lm.x * w)
                        py = int(lm.y * h)
                        coords[lm.index] = (px, py)
                        cv2.circle(display_frame, (px, py), 4, (0, 255, 120), -1, cv2.LINE_AA)

                for p1, p2 in POSE_CONNECTIONS:
                    if p1 in coords and p2 in coords:
                        cv2.line(display_frame, coords[p1], coords[p2], (0, 240, 255), 2, cv2.LINE_AA)

            # Draw Face Mesh & Expressions
            if show_face and face_res:
                draw_face_overlay(display_frame, face_res)

            # Draw Hand Landmarks
            if show_hands and hands_res:
                draw_hands_overlay(display_frame, hands_res)

            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            # Split view mode (Left: Raw Camera, Right: Vision Tracking)
            if split_view:
                half_w = w // 2
                raw_half = cv2.resize(frame, (half_w, h))
                tracked_half = cv2.resize(display_frame, (half_w, h))
                # Add dividing line
                cv2.line(tracked_half, (0, 0), (0, h), (0, 240, 255), 2)
                display_frame = np.hstack([raw_half, tracked_half])
                cv2.putText(display_frame, "RAW WEBCAM", (16, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(display_frame, "AI VISION TRACKING", (half_w + 16, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 255), 1, cv2.LINE_AA)

            # Draw HUD
            has_p = bool(pose_res and pose_res.has_pose)
            has_f = bool(face_res and face_res.has_face)
            blink = bool(face_res and face_res.blink.is_blinking)
            mouth = bool(face_res and face_res.mouth.is_mouth_open)
            has_h = (bool(hands_res and hands_res.left_hand), bool(hands_res and hands_res.right_hand))
            has_m = bool(cached_seg_res and cached_seg_res.metadata.has_person)
            cov = float(cached_seg_res.metadata.person_coverage_ratio) if cached_seg_res else 0.0

            draw_hud(
                display_frame,
                fps=fps,
                has_pose=has_p,
                has_face=has_f,
                blink_state=blink,
                mouth_state=mouth,
                has_hands=has_h,
                has_mask=has_m,
                mask_coverage=cov,
                mode_name="SPLIT" if split_view else "BLUR" if bg_mode else "STUDIO",
                show_pose=show_pose,
                show_face=show_face,
                show_hands=show_hands,
                show_mask=show_mask,
            )

            cv2.imshow(window_name, display_frame)

            # Process key events
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), ord("Q"), 27]:  # Q or ESC
                print("Exiting Live Vision Studio...")
                break
            elif key == ord("1"):
                show_pose = not show_pose
                print(f"[Toggle] Pose: {show_pose}")
            elif key == ord("2"):
                show_face = not show_face
                print(f"[Toggle] Face: {show_face}")
            elif key == ord("3"):
                show_hands = not show_hands
                print(f"[Toggle] Hands: {show_hands}")
            elif key == ord("4"):
                show_mask = not show_mask
                print(f"[Toggle] Mask: {show_mask}")
            elif key in [ord("b"), ord("B")]:
                bg_mode = not bg_mode
                print(f"[Toggle] Background Mode: {'Blur/Isolation' if bg_mode else 'Normal'}")
            elif key == 9:  # TAB key
                split_view = not split_view
                print(f"[Toggle] Split View: {split_view}")
            elif key == 32:  # SPACE key
                stamp = int(time.time())
                snap_path = snapshots_dir / f"live_snapshot_{stamp}.png"
                cv2.imwrite(str(snap_path), display_frame)
                print(f"Snapshot saved: {snap_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Studio closed.")


if __name__ == "__main__":
    main()

