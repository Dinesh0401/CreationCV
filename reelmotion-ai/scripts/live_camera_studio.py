"""ReelMotion AI — Live Vision & Character Replacement Studio.

Real-time webcam studio with:
- Live AI Character Overlay directly on user's video (Full Body Puppet, Head Avatar, Hologram)
- Multi-character switching ([N] key: Anime Girl, Neural SD 1.5 Anime, etc.)
- 33-point Pose Landmark Skeleton
- 478-point Face Mesh with Eye Blink (EAR) & Mouth Openness (MAR) detection
- 21-point Hand Landmark Tracking (Left & Right)
- Person Segmentation Mask & Background Isolation / Blur
- Side-by-side Split View & Instant Snapshot capture
"""

from __future__ import annotations

import argparse
import glob
import math
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


def overlay_rgba(
    background: np.ndarray,
    foreground_rgba: np.ndarray,
    x: int,
    y: int,
    opacity: float = 1.0,
) -> np.ndarray:
    """Alpha-composite an RGBA foreground image onto a BGR background image."""
    bh, bw = background.shape[:2]
    fh, fw = foreground_rgba.shape[:2]

    # Clip coordinates within background bounds
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


class CharacterManager:
    """Loads, prepares, and manages character assets for real-time overlay."""

    def __init__(self, characters_dir: Path) -> None:
        self.characters_dir = characters_dir
        self.character_list: list[dict[str, any]] = []
        self.current_idx = 0
        self._load_all_characters()

    def _load_all_characters(self) -> None:
        candidates = [
            (self.characters_dir / "anime_girl_rgba.png", "Anime Girl (Original)"),
            (self.characters_dir / "neural_sd15_anime_rgba.png", "Neural SD1.5 Character"),
            (self.characters_dir / "anime_girl_reference.png", "Anime Reference"),
        ]

        # Look for extra PNG cutouts
        extra_pngs = sorted(list(self.characters_dir.glob("*.png")))
        for p in extra_pngs:
            if not any(c[0] == p for c in candidates):
                candidates.append((p, p.stem.replace("_", " ").title()))

        for path, name in candidates:
            if not path.exists():
                continue
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            # Ensure RGBA
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                # Add alpha channel from luminance/threshold if missing
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                alpha = np.where(gray > 15, 255, 0).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                img[:, :, 3] = alpha

            # Pre-extract head region (upper 42% of character)
            ch, cw = img.shape[:2]
            head_crop = img[0 : int(ch * 0.44), int(cw * 0.15) : int(cw * 0.85)].copy()

            self.character_list.append({
                "name": name,
                "path": str(path),
                "rgba": img,
                "head_rgba": head_crop,
            })

        print(f"Loaded {len(self.character_list)} anime character profile(s) for live overlay.")

    @property
    def current(self) -> dict[str, any] | None:
        if not self.character_list:
            return None
        return self.character_list[self.current_idx]

    def next_character(self) -> str:
        if not self.character_list:
            return "None"
        self.current_idx = (self.current_idx + 1) % len(self.character_list)
        return self.character_list[self.current_idx]["name"]


def draw_hud(
    canvas: np.ndarray,
    fps: float,
    has_pose: bool,
    has_face: bool,
    blink_state: bool,
    mouth_state: bool,
    char_mode: str,
    char_name: str,
    char_opacity: float,
    split_view: bool,
    bg_mode: bool,
    show_pose: bool,
    show_face: bool,
    show_hands: bool,
    show_mask: bool,
) -> None:
    """Render a modern studio HUD overlay on the canvas."""
    h, w = canvas.shape[:2]

    # Top header bar
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 54), (15, 17, 26), -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)

    # Title & FPS
    cv2.putText(
        canvas,
        "REELMOTION AI - LIVE CHARACTER STUDIO",
        (16, 26),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        (0, 240, 255),
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

    # Status badges
    x_pos = 380
    badges = [
        (f"CHAR: {char_mode}", (0, 255, 120) if char_mode != "OFF" else (120, 120, 120)),
        (f"MODEL: {char_name[:16]}", (0, 220, 255) if char_mode != "OFF" else (120, 120, 120)),
        (f"OPACITY: {int(char_opacity*100)}%", (255, 200, 50) if char_mode != "OFF" else (120, 120, 120)),
        (f"BLINK: {'YES' if blink_state else 'NO'}", (0, 200, 255) if blink_state else (160, 160, 160)),
        (f"TALK: {'YES' if mouth_state else 'NO'}", (255, 100, 255) if mouth_state else (160, 160, 160)),
    ]

    for label, col in badges:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.rectangle(canvas, (x_pos, 14), (x_pos + tw + 12, 38), (30, 32, 45), -1)
        cv2.rectangle(canvas, (x_pos, 14), (x_pos + tw + 12, 38), col, 1)
        cv2.putText(canvas, label, (x_pos + 6, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)
        x_pos += tw + 18

    # Bottom helper bar
    cv2.rectangle(overlay, (0, h - 34), (w, h), (15, 17, 26), -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
    help_text = "[C] Mode: Full/Holo/Head/Off | [N] Next Char | [O] Opacity | [B] Blur BG | [TAB] Split | [1-4] Skeleton | [SPACE] Snap | [Q] Quit"
    cv2.putText(
        canvas,
        help_text,
        (16, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (210, 220, 235),
        1,
        cv2.LINE_AA,
    )


def draw_face_overlay(canvas: np.ndarray, face_res: any) -> None:
    """Draw facial mesh landmarks and bounding box."""
    if not face_res or not face_res.has_face:
        return

    h, w = canvas.shape[:2]
    step = 2 if len(face_res.landmarks) > 100 else 1
    for i in range(0, len(face_res.landmarks), step):
        lm = face_res.landmarks[i]
        px = int(lm.x * w)
        py = int(lm.y * h)
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(canvas, (px, py), 1, (255, 180, 50), -1, cv2.LINE_AA)


def draw_hands_overlay(canvas: np.ndarray, hands_res: any) -> None:
    """Draw hand skeletons and landmarks."""
    if not hands_res:
        return

    h, w = canvas.shape[:2]
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
    ]

    for hand in [hands_res.left_hand, hands_res.right_hand]:
        if not hand:
            continue
        bone_color = (255, 200, 0) if hand.handedness == "Left" else (0, 220, 255)
        coords = {}
        for lm in hand.landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            coords[lm.index] = (px, py)
            cv2.circle(canvas, (px, py), 3, (0, 255, 120), -1, cv2.LINE_AA)

        for p1, p2 in hand_connections:
            if p1 in coords and p2 in coords:
                cv2.line(canvas, coords[p1], coords[p2], bone_color, 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="ReelMotion AI Live Vision & Character Studio")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Camera width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera height (default: 720)")
    args = parser.parse_args()

    print("=====================================================")
    print("  ReelMotion AI — Live Character Replacement Studio")
    print("=====================================================")

    characters_dir = PROJECT_ROOT / "characters"
    char_mgr = CharacterManager(characters_dir)

    print(f"Opening camera {args.camera} ({args.width}x{args.height})...")
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera {args.camera}!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera active: {actual_w}x{actual_h}")
    print("Initializing neural tracking models...")

    pose_est = PoseEstimator()
    face_trk = FaceTracker()
    hand_trk = HandTracker()
    seg_mod = PersonSegmenter()

    window_name = "ReelMotion AI - Live Vision Studio"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1280, actual_w), min(720, actual_h))

    # Character Overlay Modes: 'FULL', 'HOLOGRAM', 'HEAD', 'OFF'
    char_modes = ["FULL", "HOLOGRAM", "HEAD", "OFF"]
    char_mode_idx = 0  # Starts ON in FULL mode!
    char_opacity = 1.0

    # Vision overlay toggles
    show_pose = False     # False by default so character stands out cleanly!
    show_face = False
    show_hands = False
    show_mask = False
    bg_mode = False       # Background blur mode
    split_view = False

    fps = 0.0
    frame_count = 0
    start_time = time.time()
    last_seg_time = 0.0
    cached_seg_res = None

    # EMA smoothing state for character bounding box
    smooth_x = None
    smooth_y = None
    smooth_w = None
    smooth_h = None
    ema_alpha = 0.40  # Responsiveness vs smoothness (0.40 = fast and stable)

    snapshots_dir = PROJECT_ROOT / "recordings" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    print("\nLive Studio Ready!")
    print("Controls:")
    print("  [C] Cycle Character Overlay: FULL -> HOLOGRAM -> HEAD -> OFF")
    print("  [N] Next Character Model")
    print("  [O] Adjust Character Opacity (100% -> 80% -> 60% -> 40%)")
    print("  [B] Toggle Background Blur / Isolation")
    print("  [TAB] Split View (Raw Camera vs Character Overlay)")
    print("  [1-4] Toggle Skeletons (Pose, Face, Hands, Mask)")
    print("  [SPACE] Save Snapshot")
    print("  [Q] or [ESC] Exit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)  # Natural selfie mirror view
            h, w = frame.shape[:2]
            display_frame = frame.copy()

            # Track pose & face
            pose_res = pose_est.estimate(frame)
            face_res = face_trk.track(frame)
            hands_res = hand_trk.track(frame) if show_hands else None

            # Person segmentation (for background blur or mask)
            now = time.time()
            if (bg_mode or show_mask) and (cached_seg_res is None or (now - last_seg_time) > 0.09):
                cached_seg_res = seg_mod.segment(frame)
                last_seg_time = now

            # 1. Apply Background Blur / Isolation if active
            if bg_mode and cached_seg_res and cached_seg_res.alpha_mask is not None:
                alpha = cached_seg_res.alpha_mask
                if alpha.shape[:2] != (h, w):
                    alpha = cv2.resize(alpha, (w, h))
                alpha_3c = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
                bg_blurred = cv2.GaussianBlur(frame, (45, 45), 0)
                display_frame = (frame * alpha_3c + bg_blurred * (1.0 - alpha_3c)).astype(np.uint8)

            # 2. Determine User Body Target Bounding Box
            target_bbox = None  # (x, y, w, h)
            current_mode = char_modes[char_mode_idx]

            if current_mode in ["FULL", "HOLOGRAM"]:
                if pose_res and pose_res.has_pose:
                    # Find torso / upper-body bounds
                    coords_y = [lm.y * h for lm in pose_res.landmarks if lm.visibility > 0.25]
                    coords_x = [lm.x * w for lm in pose_res.landmarks if lm.visibility > 0.25]
                    if coords_x and coords_y:
                        min_x, max_x = min(coords_x), max(coords_x)
                        min_y, max_y = min(coords_y), max(coords_y)

                        # Pad body
                        pad_w = (max_x - min_x) * 0.15
                        pad_h = (max_y - min_y) * 0.10
                        user_cx = (min_x + max_x) / 2.0
                        user_h = max(80.0, (max_y - min_y) + pad_h * 2)

                        # Match character aspect ratio
                        char_data = char_mgr.current
                        if char_data:
                            ch_h, ch_w = char_data["rgba"].shape[:2]
                            aspect = ch_w / max(1, ch_h)
                            user_w = user_h * aspect

                            bx = user_cx - (user_w / 2.0)
                            by = max(0.0, min_y - pad_h)
                            target_bbox = (bx, by, user_w, user_h)

                elif face_res and face_res.has_face and face_res.bounding_box:
                    # Fallback to face-based body estimation
                    bx1, by1, bx2, by2 = face_res.bounding_box
                    face_w = (bx2 - bx1) * w
                    face_h = (by2 - by1) * h
                    face_cx = (bx1 + bx2) / 2.0 * w

                    char_data = char_mgr.current
                    if char_data:
                        ch_h, ch_w = char_data["rgba"].shape[:2]
                        user_h = face_h * 3.8
                        user_w = user_h * (ch_w / max(1, ch_h))
                        bx = face_cx - (user_w / 2.0)
                        by = max(0.0, by1 * h - face_h * 0.25)
                        target_bbox = (bx, by, user_w, user_h)

            elif current_mode == "HEAD":
                if face_res and face_res.has_face and face_res.bounding_box:
                    bx1, by1, bx2, by2 = face_res.bounding_box
                    face_w = (bx2 - bx1) * w * 1.5
                    face_h = (by2 - by1) * h * 1.5
                    face_cx = (bx1 + bx2) / 2.0 * w
                    face_cy = (by1 + by2) / 2.0 * h
                    target_bbox = (face_cx - face_w / 2.0, face_cy - face_h / 2.0, face_w, face_h)

            # Smooth target bbox with EMA filter
            if target_bbox is not None:
                bx, by, bw, bh = target_bbox
                if smooth_x is None:
                    smooth_x, smooth_y, smooth_w, smooth_h = bx, by, bw, bh
                else:
                    smooth_x = (1.0 - ema_alpha) * smooth_x + ema_alpha * bx
                    smooth_y = (1.0 - ema_alpha) * smooth_y + ema_alpha * by
                    smooth_w = (1.0 - ema_alpha) * smooth_w + ema_alpha * bw
                    smooth_h = (1.0 - ema_alpha) * smooth_h + ema_alpha * bh

                # 3. Render Anime Character Overlay
                char_data = char_mgr.current
                if char_data:
                    overlay_img = char_data["head_rgba"] if current_mode == "HEAD" else char_data["rgba"]
                    tw = max(10, int(smooth_w))
                    th = max(10, int(smooth_h))
                    tx = int(smooth_x)
                    ty = int(smooth_y)

                    scaled_char = cv2.resize(overlay_img, (tw, th), interpolation=cv2.INTER_LINEAR)
                    eff_opacity = 0.55 if current_mode == "HOLOGRAM" else char_opacity

                    # If talking or blinking, modulate slightly for life-like animation
                    if face_res and face_res.mouth.is_mouth_open:
                        # Slight scale pulse on mouth talking
                        pass

                    overlay_rgba(display_frame, scaled_char, tx, ty, opacity=eff_opacity)

            # 4. Draw optional skeleton overlays if enabled by keys
            if show_pose and pose_res and pose_res.has_pose:
                coords = {}
                for lm in pose_res.landmarks:
                    if lm.visibility > 0.3:
                        px, py = int(lm.x * w), int(lm.y * h)
                        coords[lm.index] = (px, py)
                        cv2.circle(display_frame, (px, py), 4, (0, 255, 120), -1, cv2.LINE_AA)
                for p1, p2 in POSE_CONNECTIONS:
                    if p1 in coords and p2 in coords:
                        cv2.line(display_frame, coords[p1], coords[p2], (0, 240, 255), 2, cv2.LINE_AA)

            if show_face and face_res:
                draw_face_overlay(display_frame, face_res)

            if show_hands and hands_res:
                draw_hands_overlay(display_frame, hands_res)

            if show_mask and not bg_mode and cached_seg_res and cached_seg_res.binary_mask is not None:
                mask = cached_seg_res.binary_mask
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h))
                tint = np.zeros_like(display_frame)
                tint[:, :] = (255, 120, 0)
                display_frame = np.where(mask[:, :, np.newaxis] > 128, cv2.addWeighted(display_frame, 0.7, tint, 0.3, 0), display_frame)

            # FPS measurement
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            # Split view mode (Left: Raw Camera, Right: Character Overlaid Video)
            if split_view:
                half_w = w // 2
                raw_half = cv2.resize(frame, (half_w, h))
                tracked_half = cv2.resize(display_frame, (half_w, h))
                cv2.line(tracked_half, (0, 0), (0, h), (0, 240, 255), 2)
                display_frame = np.hstack([raw_half, tracked_half])
                cv2.putText(display_frame, "RAW WEBCAM", (16, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(display_frame, f"AI CHARACTER OVERLAY ({current_mode})", (half_w + 16, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 255), 1, cv2.LINE_AA)

            # Draw HUD
            has_p = bool(pose_res and pose_res.has_pose)
            has_f = bool(face_res and face_res.has_face)
            blink = bool(face_res and face_res.blink.is_blinking)
            mouth = bool(face_res and face_res.mouth.is_mouth_open)
            cur_char_name = char_mgr.current["name"] if char_mgr.current else "None"

            draw_hud(
                display_frame,
                fps=fps,
                has_pose=has_p,
                has_face=has_f,
                blink_state=blink,
                mouth_state=mouth,
                char_mode=current_mode,
                char_name=cur_char_name,
                char_opacity=char_opacity,
                split_view=split_view,
                bg_mode=bg_mode,
                show_pose=show_pose,
                show_face=show_face,
                show_hands=show_hands,
                show_mask=show_mask,
            )

            cv2.imshow(window_name, display_frame)

            # Key controls
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), ord("Q"), 27]:
                print("Exiting Character Studio...")
                break
            elif key in [ord("c"), ord("C")]:
                char_mode_idx = (char_mode_idx + 1) % len(char_modes)
                print(f"[Toggle] Character Overlay Mode: {char_modes[char_mode_idx]}")
            elif key in [ord("n"), ord("N")]:
                name = char_mgr.next_character()
                print(f"[Switch] Character Model: {name}")
            elif key in [ord("o"), ord("O")]:
                # Cycle opacity: 1.0 -> 0.8 -> 0.6 -> 0.4
                char_opacity = 1.0 if char_opacity <= 0.45 else round(char_opacity - 0.2, 1)
                print(f"[Opacity] Character Opacity: {int(char_opacity * 100)}%")
            elif key in [ord("b"), ord("B")]:
                bg_mode = not bg_mode
                print(f"[Toggle] Background Blur: {bg_mode}")
            elif key == 9:  # TAB
                split_view = not split_view
                print(f"[Toggle] Split View: {split_view}")
            elif key == ord("1"):
                show_pose = not show_pose
            elif key == ord("2"):
                show_face = not show_face
            elif key == ord("3"):
                show_hands = not show_hands
            elif key == ord("4"):
                show_mask = not show_mask
            elif key == 32:  # SPACE
                stamp = int(time.time())
                snap_path = snapshots_dir / f"character_snapshot_{stamp}.png"
                cv2.imwrite(str(snap_path), display_frame)
                print(f"Snapshot saved: {snap_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Studio closed.")


if __name__ == "__main__":
    main()
