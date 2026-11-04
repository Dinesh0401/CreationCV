"""Identity and Image Quality Validation for ReelMotion AI — Phase 4.

Validates reference images for:
- Resolution (minimum 256x256, optimal >= 512x512)
- Face visibility & sharpness (Laplacian variance)
- Full-body completeness (head, torso, legs, hands)
- Background complexity (edge density and border variance)
- Multi-image character consistency (Lab color histogram & perceptual hash)

Produces ImageQualityReport and CharacterReadinessReport.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from backend.character.character_profile import CharacterProfile, VIEW_TYPES

logger = logging.getLogger(__name__)

# Minimum and optimal resolution requirements
MIN_DIMENSION = 256
OPTIMAL_DIMENSION = 512

# Sharpness threshold (Laplacian variance)
SHARPNESS_BLUR_THRESHOLD = 35.0
SHARPNESS_OPTIMAL_THRESHOLD = 90.0

# Background complexity threshold (0 = clean, 100 = cluttered)
BACKGROUND_COMPLEX_THRESHOLD = 45.0


class BodyPartsReport(BaseModel):
    """Detection results for body landmarks / parts."""

    head: bool = False
    torso: bool = False
    legs: bool = False
    hands: bool = False


class ImageQualityReport(BaseModel):
    """Detailed quality validation report for an individual reference image."""

    view_type: str | None = None
    width: int
    height: int
    resolution_ok: bool
    is_optimal_resolution: bool
    sharpness_score: float = Field(..., description="0-100 sharpness rating")
    sharpness_raw_variance: float
    is_sharp: bool
    face_detected: bool
    face_confidence: float = 0.0
    face_box: tuple[int, int, int, int] | None = None
    is_full_body: bool
    body_parts: BodyPartsReport
    background_complexity: float = Field(..., description="0-100 complexity rating; <45 is clean")
    is_clean_background: bool
    issues: list[str] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, description="Overall image quality score 0-100")


class CharacterReadinessReport(BaseModel):
    """Character studio readiness report determining eligibility for AI replacement."""

    character_id: str
    readiness_score: float = Field(..., ge=0.0, le=100.0, description="0-100% readiness score")
    is_ready: bool
    single_image_mode: bool
    consistency_score: float | None = None
    recommendations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    missing_attributes: list[str] = Field(default_factory=list)
    view_reports: dict[str, ImageQualityReport] = Field(default_factory=dict)


def _load_image_as_bgr(
    image_input: str | Path | bytes | np.ndarray | Image.Image,
) -> np.ndarray:
    """Normalize input into a uint8 BGR numpy array."""
    if isinstance(image_input, np.ndarray):
        if image_input.dtype != np.uint8:
            image_input = np.clip(image_input, 0, 255).astype(np.uint8)
        if len(image_input.shape) == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        if image_input.shape[2] == 4:
            # Handle RGBA by blending over white background
            alpha = image_input[:, :, 3] / 255.0
            bgr = image_input[:, :, :3]
            white = np.ones_like(bgr) * 255
            blended = (bgr * alpha[:, :, None] + white * (1.0 - alpha[:, :, None])).astype(np.uint8)
            return blended
        return image_input

    if isinstance(image_input, Image.Image):
        rgba = image_input.convert("RGBA")
        np_img = np.array(rgba)
        return _load_image_as_bgr(np_img)

    if isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Failed to decode image from bytes.")
        return _load_image_as_bgr(img)

    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image from path: {path}")
        return _load_image_as_bgr(img)

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def check_resolution(width: int, height: int) -> tuple[bool, bool, list[str]]:
    """Validate image resolution."""
    resolution_ok = width >= MIN_DIMENSION and height >= MIN_DIMENSION
    is_optimal = width >= OPTIMAL_DIMENSION and height >= OPTIMAL_DIMENSION

    issues: list[str] = []
    if not resolution_ok:
        issues.append(
            f"Image resolution ({width}x{height}) is below minimum requirement ({MIN_DIMENSION}x{MIN_DIMENSION})."
        )
    elif not is_optimal:
        issues.append(
            f"Image resolution ({width}x{height}) is sub-optimal; >= {OPTIMAL_DIMENSION}x{OPTIMAL_DIMENSION} recommended."
        )

    return resolution_ok, is_optimal, issues


def check_sharpness(gray: np.ndarray) -> tuple[float, float, bool, list[str]]:
    """Compute sharpness using Laplacian variance."""
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())

    # Map variance to 0-100 score with soft saturation
    # ~35 maps to ~45, ~90 maps to ~80, >= 150 maps to 100
    sharpness_score = float(np.clip((variance / 150.0) * 100.0, 0.0, 100.0))
    sharpness_score = round(sharpness_score, 1)

    is_sharp = variance >= SHARPNESS_BLUR_THRESHOLD
    issues: list[str] = []
    if not is_sharp:
        issues.append(
            f"Image appears blurry (sharpness metric {variance:.1f} < {SHARPNESS_BLUR_THRESHOLD})."
        )

    return sharpness_score, round(variance, 1), is_sharp, issues


def check_face_visibility(
    bgr: np.ndarray,
    view_type: str | None = None,
) -> tuple[bool, float, tuple[int, int, int, int] | None, list[str]]:
    """Detect presence and quality of character face using skin tone and geometry."""
    h, w = bgr.shape[:2]
    issues: list[str] = []

    # 1. Skin tone extraction via YCrCb
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    # Standard human skin range in YCrCb
    skin_mask = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8),
    )

    # Focus face detection primarily on upper 65% of the frame
    upper_limit = int(h * 0.65)
    upper_mask = np.zeros_like(skin_mask)
    upper_mask[:upper_limit, :] = skin_mask[:upper_limit, :]

    # Clean noise with morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(upper_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box: tuple[int, int, int, int] | None = None
    best_conf = 0.0
    total_area = float(h * w)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        rel_area = area / total_area
        if rel_area < 0.005 or rel_area > 0.60:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bh / max(1, bw)

        # Faces typically have aspect ratio 0.8 to 2.2
        if 0.75 <= aspect <= 2.5:
            # Score candidate based on size and vertical centrality
            center_dist = abs((x + bw / 2.0) - (w / 2.0)) / (w / 2.0)
            score = float(np.clip(1.0 - (center_dist * 0.4) + min(0.3, rel_area * 2.0), 0.1, 0.95))
            if score > best_conf:
                best_conf = score
                best_box = (int(x), int(y), int(bw), int(bh))

    face_detected = best_box is not None and best_conf >= 0.35

    # If view is face_closeup, require clear face
    if view_type == "face_closeup" and not face_detected:
        issues.append("Face could not be clearly identified in face closeup view.")
    elif view_type == "front" and not face_detected:
        issues.append("Face is not clearly visible in front view. Ensure good lighting and facing camera.")

    return face_detected, round(best_conf, 2), best_box, issues


def check_full_body(
    bgr: np.ndarray,
    view_type: str | None = None,
) -> tuple[bool, BodyPartsReport, list[str]]:
    """Determine full-body completeness: head, torso, legs, hands."""
    h, w = bgr.shape[:2]
    issues: list[str] = []

    # Estimate background from borders (outer 8%)
    bw_margin = max(4, int(w * 0.08))
    bh_margin = max(4, int(h * 0.08))

    border_samples = np.concatenate([
        bgr[:bh_margin, :].reshape(-1, 3),
        bgr[-bh_margin:, :].reshape(-1, 3),
        bgr[:, :bw_margin].reshape(-1, 3),
        bgr[:, -bw_margin:].reshape(-1, 3),
    ])
    bg_median = np.median(border_samples, axis=0)

    # Difference from background
    diff = np.linalg.norm(bgr.astype(float) - bg_median, axis=2)
    # Grayscale edge fallback
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)

    fg_mask = (diff > 25.0) | (edges > 0)

    # Define vertical zones
    top_end = int(h * 0.28)      # Head zone: 0% - 28%
    mid_end = int(h * 0.65)      # Torso/Hands: 28% - 65%
    # Lower zone: Legs: 65% - 100%

    top_cov = float(np.mean(fg_mask[:top_end, :]))
    mid_cov = float(np.mean(fg_mask[top_end:mid_end, :]))
    bot_cov = float(np.mean(fg_mask[mid_end:, :]))

    # Check lateral expansion in middle zone for arms/hands
    mid_fg = fg_mask[top_end:mid_end, :]
    left_hand_cov = float(np.mean(mid_fg[:, : int(w * 0.25)]))
    right_hand_cov = float(np.mean(mid_fg[:, int(w * 0.75) :]))
    hands_cov = left_hand_cov + right_hand_cov

    has_head = top_cov > 0.03
    has_torso = mid_cov > 0.06
    has_legs = bot_cov > 0.04
    has_hands = hands_cov > 0.01 or (mid_cov > 0.15)  # Arms close to body or extended

    is_full_body = has_head and has_torso and has_legs

    body_parts = BodyPartsReport(
        head=has_head,
        torso=has_torso,
        legs=has_legs,
        hands=has_hands,
    )

    if view_type in ["front", "side", "back"] and not is_full_body:
        missing = []
        if not has_head:
            missing.append("head")
        if not has_torso:
            missing.append("torso")
        if not has_legs:
            missing.append("legs")
        issues.append(f"Incomplete full-body coverage in {view_type} view (missing {', '.join(missing)}).")

    return is_full_body, body_parts, issues


def check_background_complexity(bgr: np.ndarray) -> tuple[float, bool, list[str]]:
    """Assess background clutter from image borders."""
    h, w = bgr.shape[:2]
    tb = max(6, int(h * 0.12))
    lr = max(6, int(w * 0.12))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:tb, :] = True
    border_mask[-tb:, :] = True
    border_mask[:, :lr] = True
    border_mask[:, -lr:] = True

    border_pixels = gray[border_mask]
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges[border_mask] > 0)) * 100.0
    color_std = float(np.std(border_pixels))

    complexity = float(np.clip(edge_density * 2.5 + (color_std / 64.0) * 50.0, 0.0, 100.0))
    complexity = round(complexity, 1)

    is_clean = complexity < BACKGROUND_COMPLEX_THRESHOLD
    issues: list[str] = []
    if not is_clean:
        issues.append(
            f"Background complexity is elevated ({complexity:.1f}/100). Plain or solid background is recommended."
        )

    return complexity, is_clean, issues


def validate_image(
    image_input: str | Path | bytes | np.ndarray | Image.Image,
    view_type: str | None = None,
) -> ImageQualityReport:
    """Run full image quality validation pipeline on a single reference image."""
    bgr = _load_image_as_bgr(image_input)
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    all_issues: list[str] = []

    # 1. Resolution
    res_ok, opt_res, res_issues = check_resolution(w, h)
    all_issues.extend(res_issues)

    # 2. Sharpness
    sharp_score, sharp_var, is_sharp, sharp_issues = check_sharpness(gray)
    all_issues.extend(sharp_issues)

    # 3. Face
    face_detected, face_conf, face_box, face_issues = check_face_visibility(bgr, view_type)
    all_issues.extend(face_issues)

    # 4. Full body
    is_full_body, body_parts, body_issues = check_full_body(bgr, view_type)
    all_issues.extend(body_issues)

    # 5. Background complexity
    bg_comp, is_clean_bg, bg_issues = check_background_complexity(bgr)
    all_issues.extend(bg_issues)

    # Overall image quality score (0-100)
    score = 0.0
    if res_ok:
        score += 25.0
    if opt_res:
        score += 15.0
    score += (sharp_score / 100.0) * 25.0
    if is_clean_bg:
        score += 15.0
    if face_detected or view_type in ["back", "outfit_details"]:
        score += 10.0
    if is_full_body or view_type in ["face_closeup", "outfit_details"]:
        score += 10.0

    score = float(np.clip(score, 0.0, 100.0))

    return ImageQualityReport(
        view_type=view_type,
        width=w,
        height=h,
        resolution_ok=res_ok,
        is_optimal_resolution=opt_res,
        sharpness_score=sharp_score,
        sharpness_raw_variance=sharp_var,
        is_sharp=is_sharp,
        face_detected=face_detected,
        face_confidence=face_conf,
        face_box=face_box,
        is_full_body=is_full_body,
        body_parts=body_parts,
        background_complexity=bg_comp,
        is_clean_background=is_clean_bg,
        issues=all_issues,
        quality_score=round(score, 1),
    )


def compute_perceptual_hash(bgr: np.ndarray) -> np.ndarray:
    """Compute 64-bit difference hash (dHash)."""
    resized = cv2.resize(bgr, (9, 8), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    diff = gray[:, 1:] > gray[:, :-1]
    return diff.flatten()


def compute_color_histogram(bgr: np.ndarray) -> np.ndarray:
    """Compute Lab chromaticity (a* and b*) color histogram."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    # Focus on center 70% of image to emphasize character over peripheral background
    h, w = lab.shape[:2]
    crop = lab[int(h * 0.15) : int(h * 0.85), int(w * 0.15) : int(w * 0.85)]
    hist = cv2.calcHist([crop], [1, 2], None, [16, 16], [0, 256, 0, 256])
    cv2.normalize(hist, hist, 1, 0, cv2.NORM_L1)
    return hist


def compute_character_consistency(
    images: list[np.ndarray | str | Path | bytes],
) -> float:
    """Calculate multi-image character consistency score across reference images (0-100)."""
    if len(images) < 2:
        return 100.0

    bgrs = [_load_image_as_bgr(img) for img in images]
    hashes = [compute_perceptual_hash(img) for img in bgrs]
    hists = [compute_color_histogram(img) for img in bgrs]

    scores: list[float] = []
    n = len(bgrs)

    for i in range(n):
        for j in range(i + 1, n):
            # Perceptual hash similarity (dHash Hamming distance)
            hash_sim = float(np.sum(hashes[i] == hashes[j])) / 64.0

            # Color histogram similarity via Bhattacharyya distance (0 = identical, 1 = disjoint)
            bhat_dist = cv2.compareHist(hists[i], hists[j], cv2.HISTCMP_BHATTACHARYYA)
            color_sim = float(np.clip(1.0 - bhat_dist, 0.0, 1.0))

            # Pair score: color matters most for character outfit/skin identity across different poses
            pair_score = (0.65 * color_sim + 0.35 * hash_sim) * 100.0
            scores.append(pair_score)

    avg_score = float(np.mean(scores)) if scores else 100.0
    return round(float(np.clip(avg_score, 0.0, 100.0)), 1)


def generate_readiness_report(
    profile: CharacterProfile,
    character_dir: Path,
) -> CharacterReadinessReport:
    """Generate comprehensive character readiness report and recommendations."""
    view_reports: dict[str, ImageQualityReport] = {}
    loaded_images: list[np.ndarray] = []
    warnings: list[str] = []
    recommendations: list[str] = []
    missing_attributes: list[str] = []

    # Check each standard view
    for vt in VIEW_TYPES:
        filename = profile.views.get_view(vt)
        if not filename:
            missing_attributes.append(vt)
            continue

        # Look in references subfolder or character dir root
        file_path = character_dir / "references" / filename
        if not file_path.exists():
            file_path = character_dir / filename

        if not file_path.exists():
            missing_attributes.append(vt)
            warnings.append(f"Referenced image for {vt} ({filename}) is missing on disk.")
            continue

        try:
            report = validate_image(file_path, view_type=vt)
            view_reports[vt] = report
            bgr = _load_image_as_bgr(file_path)
            loaded_images.append(bgr)
            if report.issues:
                for issue in report.issues:
                    warnings.append(f"[{vt.capitalize()}] {issue}")
        except Exception as exc:
            logger.error("Failed to validate image %s for %s: %s", file_path, vt, exc)
            warnings.append(f"Failed to process {vt} image: {exc}")

    active_count = len(view_reports)

    # 0 references
    if active_count == 0:
        return CharacterReadinessReport(
            character_id=profile.id,
            readiness_score=0.0,
            is_ready=False,
            single_image_mode=False,
            consistency_score=None,
            recommendations=[
                "Upload at least one clear reference image of your character to get started.",
                "A front full-body image and side reference will improve character consistency across reel motions.",
            ],
            warnings=warnings or ["No reference images uploaded."],
            missing_attributes=missing_attributes,
            view_reports={},
        )

    # Consistency score for multiple images
    consistency: float | None = None
    if len(loaded_images) >= 2:
        consistency = compute_character_consistency(loaded_images)
        if consistency < 50.0:
            warnings.append(
                f"Multi-image consistency is low ({consistency:.1f}%). Verify lighting, hair, and costume match across views."
            )

    # 1 reference -> Single Image Mode
    if active_count == 1:
        single_mode = True
        only_view, only_report = next(iter(view_reports.items()))

        base = 35.0
        if only_report.is_optimal_resolution:
            base += 5.0
        elif only_report.resolution_ok:
            base += 3.0

        if only_report.is_sharp:
            base += 5.0
        if only_report.is_clean_background:
            base += 5.0
        if only_report.is_full_body:
            base += 5.0
        if only_report.face_detected:
            base += 5.0

        if not only_report.resolution_ok:
            base = min(base, 30.0)

        # Single image mode capped at 60% max (higher requires multi-angle)
        score = float(np.clip(base, 0.0, 60.0))
        is_ready = score >= 50.0 and only_report.resolution_ok

        recommendations.append("A front full-body image and side reference will improve character consistency")
        if "side" in missing_attributes:
            recommendations.append("Upload a side reference to improve 3D body rotation fidelity.")
        if "face_closeup" in missing_attributes:
            recommendations.append("Upload a face closeup for crisper facial expression tracking.")
        if "back" in missing_attributes:
            recommendations.append("Upload a back view for 360-degree reel motions.")

        return CharacterReadinessReport(
            character_id=profile.id,
            readiness_score=round(score, 1),
            is_ready=is_ready,
            single_image_mode=single_mode,
            consistency_score=None,
            recommendations=recommendations,
            warnings=warnings,
            missing_attributes=missing_attributes,
            view_reports=view_reports,
        )

    # Multi-reference mode (>= 2 images)
    single_mode = False
    # View contribution weights
    weights = {
        "front": 40.0,
        "side": 30.0,
        "back": 15.0,
        "face_closeup": 10.0,
        "outfit_details": 5.0,
    }

    raw_score = sum(weights.get(vt, 10.0) for vt in view_reports)

    # Quality deductions/bonuses
    avg_quality = float(np.mean([r.quality_score for r in view_reports.values()]))
    quality_factor = (avg_quality - 70.0) * 0.15  # -10 to +4.5
    raw_score += quality_factor

    # Consistency modifier
    if consistency is not None:
        if consistency >= 75.0:
            raw_score += 5.0
        elif consistency < 55.0:
            raw_score -= 12.0

    final_score = float(np.clip(round(raw_score, 1), 0.0, 100.0))
    has_front = "front" in view_reports
    has_side = "side" in view_reports

    is_ready = final_score >= 50.0 and (has_front or active_count >= 2)

    # Recommendations
    if not has_front:
        recommendations.append("A front full-body image is essential for character pose alignment.")
    if not has_side:
        recommendations.append("A side reference will improve character consistency during turns.")
    if "back" in missing_attributes:
        recommendations.append("Add a back reference for full 360-degree reel transitions.")
    if "face_closeup" in missing_attributes:
        recommendations.append("Add a face closeup reference for sharper facial animations.")

    return CharacterReadinessReport(
        character_id=profile.id,
        readiness_score=final_score,
        is_ready=is_ready,
        single_image_mode=single_mode,
        consistency_score=consistency,
        recommendations=recommendations,
        warnings=warnings,
        missing_attributes=missing_attributes,
        view_reports=view_reports,
    )
