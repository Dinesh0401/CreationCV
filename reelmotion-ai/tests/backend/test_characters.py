"""Unit and integration tests for Phase 4 Character Studio, Validation, and APIs."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.character.character_profile import (
    VIEW_TYPES,
    CharacterProfile,
    CharacterStorageManager,
    CharacterViews,
)
from backend.character.identity_validation import (
    CharacterReadinessReport,
    ImageQualityReport,
    check_background_complexity,
    check_face_visibility,
    check_full_body,
    check_resolution,
    check_sharpness,
    compute_character_consistency,
    generate_readiness_report,
    validate_image,
)
from backend.character.reference_manager import CharacterReferenceManager


def _create_synthetic_character_image(
    width: int = 512,
    height: int = 512,
    skin_color: tuple[int, int, int] = (180, 200, 230),  # BGR skin-tone
    torso_color: tuple[int, int, int] = (200, 50, 40),   # BGR torso
    legs_color: tuple[int, int, int] = (30, 30, 80),     # BGR legs
    clean_bg: bool = True,
    full_body: bool = True,
    blurry: bool = False,
) -> np.ndarray:
    """Generate a synthetic test image with controllable character features."""
    if clean_bg:
        img = np.ones((height, width, 3), dtype=np.uint8) * 245
    else:
        # High-frequency textured background
        img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    center_x = width // 2

    # Head / Face
    head_radius = int(min(width, height) * 0.08)
    head_y = int(height * 0.16)
    cv2.circle(img, (center_x, head_y), head_radius, skin_color, -1)

    # Torso
    torso_w = int(width * 0.28)
    torso_top = int(height * 0.26)
    torso_bot = int(height * 0.60)
    cv2.rectangle(
        img,
        (center_x - torso_w // 2, torso_top),
        (center_x + torso_w // 2, torso_bot),
        torso_color,
        -1,
    )

    # Hands / Arms
    hand_y = int(height * 0.42)
    arm_span = int(width * 0.20)
    cv2.circle(img, (center_x - arm_span, hand_y), int(width * 0.04), skin_color, -1)
    cv2.circle(img, (center_x + arm_span, hand_y), int(width * 0.04), skin_color, -1)

    # Legs (only if full_body requested)
    if full_body:
        leg_w = int(width * 0.08)
        leg_top = torso_bot
        leg_bot = int(height * 0.92)
        cv2.rectangle(
            img,
            (center_x - leg_w - 6, leg_top),
            (center_x - 6, leg_bot),
            legs_color,
            -1,
        )
        cv2.rectangle(
            img,
            (center_x + 6, leg_top),
            (center_x + leg_w + 6, leg_bot),
            legs_color,
            -1,
        )

    if blurry:
        img = cv2.GaussianBlur(img, (45, 45), 0)

    return img


def _image_to_png_bytes(img: np.ndarray) -> bytes:
    """Encode numpy BGR image to PNG bytes."""
    success, buf = cv2.imencode(".png", img)
    assert success
    return buf.tobytes()


# ═══════════════════════════════════════════════════════════════════════
# 1. Profile & Storage Manager Tests
# ═══════════════════════════════════════════════════════════════════════

def test_character_views_model():
    """Test CharacterViews manipulation and active views counting."""
    views = CharacterViews()
    assert views.count() == 0
    assert views.front is None

    views.set_view("front", "front_test.png")
    views.set_view("side", "side_test.png")
    assert views.count() == 2
    assert views.get_view("front") == "front_test.png"
    assert views.get_view("side") == "side_test.png"
    assert views.active_views() == {
        "front": "front_test.png",
        "side": "side_test.png",
    }

    with pytest.raises(ValueError):
        views.set_view("invalid_view", "foo.png")


def test_character_profile_model():
    """Test CharacterProfile serialization and reference counting."""
    profile = CharacterProfile(name="Cyber Sam")
    assert profile.name == "Cyber Sam"
    assert profile.reference_count == 0
    assert len(profile.id) > 0

    profile.views.front = "front.png"
    profile.views.side = "side.png"
    assert profile.refresh_reference_count() == 2

    dump = profile.model_dump()
    assert dump["name"] == "Cyber Sam"
    assert dump["views"]["front"] == "front.png"

    reloaded = CharacterProfile(**dump)
    assert reloaded.id == profile.id
    assert reloaded.views.front == "front.png"


def test_storage_manager_crud(tmp_path: Path):
    """Test CharacterStorageManager profile creation, saving, loading, listing, deleting."""
    mgr = CharacterStorageManager(base_dir=tmp_path)

    profile = mgr.create_profile(name="Test Char", tags=["hero", "action"])
    assert profile.id is not None
    assert profile.name == "Test Char"
    assert profile.tags == ["hero", "action"]

    # Verify profile.json exists
    char_dir = mgr.get_character_dir(profile.id)
    assert (char_dir / "profile.json").exists()
    assert (char_dir / "meta.json").exists()

    # Load profile
    loaded = mgr.load_profile(profile.id)
    assert loaded is not None
    assert loaded.name == "Test Char"
    assert loaded.tags == ["hero", "action"]

    # Update and save
    loaded.views.front = "front_01.png"
    mgr.save_profile(loaded)

    reloaded = mgr.load_profile(profile.id)
    assert reloaded.views.front == "front_01.png"
    assert reloaded.reference_count == 1

    # List profiles
    all_chars = mgr.list_profiles()
    assert len(all_chars) == 1
    assert all_chars[0].id == profile.id

    # Delete profile
    deleted = mgr.delete_profile(profile.id)
    assert deleted is True
    assert mgr.load_profile(profile.id) is None
    assert len(mgr.list_profiles()) == 0


# ═══════════════════════════════════════════════════════════════════════
# 2. Identity & Image Quality Validation Tests
# ═══════════════════════════════════════════════════════════════════════

def test_check_resolution():
    """Test image resolution constraints."""
    # Sub-minimum
    ok, opt, issues = check_resolution(200, 200)
    assert ok is False
    assert opt is False
    assert len(issues) > 0

    # Minimum acceptable
    ok, opt, issues = check_resolution(300, 300)
    assert ok is True
    assert opt is False

    # Optimal
    ok, opt, issues = check_resolution(512, 512)
    assert ok is True
    assert opt is True
    assert len(issues) == 0


def test_check_sharpness():
    """Test sharpness detection on sharp vs blurred images."""
    sharp_img = _create_synthetic_character_image(512, 512, blurry=False)
    gray_sharp = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2GRAY)
    score_s, var_s, is_sharp_s, issues_s = check_sharpness(gray_sharp)
    assert is_sharp_s is True
    assert score_s > 50.0

    blur_img = _create_synthetic_character_image(512, 512, blurry=True)
    gray_blur = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
    score_b, var_b, is_sharp_b, issues_b = check_sharpness(gray_blur)
    assert is_sharp_b is False
    assert len(issues_b) > 0


def test_check_face_visibility():
    """Test skin-tone and geometry face detection."""
    img_with_face = _create_synthetic_character_image(512, 512)
    detected, conf, box, issues = check_face_visibility(img_with_face, view_type="front")
    assert detected is True
    assert conf > 0.3
    assert box is not None

    # Blank image with no face
    blank = np.ones((512, 512, 3), dtype=np.uint8) * 128
    detected_b, conf_b, box_b, issues_b = check_face_visibility(blank, view_type="front")
    assert detected_b is False


def test_check_full_body():
    """Test full body completeness detection."""
    full_img = _create_synthetic_character_image(512, 512, full_body=True)
    is_full, parts, issues = check_full_body(full_img, view_type="front")
    assert is_full is True
    assert parts.head is True
    assert parts.torso is True
    assert parts.legs is True

    # Crop to head and torso only
    crop_img = _create_synthetic_character_image(512, 512, full_body=False)
    is_full_c, parts_c, issues_c = check_full_body(crop_img, view_type="front")
    assert is_full_c is False
    assert parts_c.legs is False
    assert len(issues_c) > 0


def test_check_background_complexity():
    """Test background complexity analysis."""
    clean_img = _create_synthetic_character_image(512, 512, clean_bg=True)
    comp_clean, is_clean, issues = check_background_complexity(clean_img)
    assert is_clean is True
    assert comp_clean < 45.0

    noisy_img = _create_synthetic_character_image(512, 512, clean_bg=False)
    comp_noisy, is_clean_n, issues_n = check_background_complexity(noisy_img)
    assert is_clean_n is False
    assert comp_noisy >= 45.0
    assert len(issues_n) > 0


def test_validate_image_pipeline():
    """Test complete validate_image pipeline."""
    img = _create_synthetic_character_image(512, 512)
    png_bytes = _image_to_png_bytes(img)

    report = validate_image(png_bytes, view_type="front")
    assert isinstance(report, ImageQualityReport)
    assert report.width == 512
    assert report.height == 512
    assert report.resolution_ok is True
    assert report.is_optimal_resolution is True
    assert report.is_sharp is True
    assert report.face_detected is True
    assert report.is_full_body is True
    assert report.quality_score >= 70.0


def test_compute_character_consistency():
    """Test multi-image character consistency score."""
    img1 = _create_synthetic_character_image(512, 512, torso_color=(200, 50, 40))
    img2 = _create_synthetic_character_image(512, 512, torso_color=(210, 55, 45))
    diff_char = _create_synthetic_character_image(512, 512, torso_color=(10, 200, 10))

    sim_score = compute_character_consistency([img1, img2])
    diff_score = compute_character_consistency([img1, diff_char])

    assert sim_score >= 60.0
    assert sim_score > diff_score


def test_character_readiness_report_generation(tmp_path: Path):
    """Test readiness report for 0, 1, and multiple images."""
    storage = CharacterStorageManager(base_dir=tmp_path)
    profile = storage.create_profile("Readiness Test")

    # 1. Zero images
    char_dir = storage.get_character_dir(profile.id)
    rep0 = generate_readiness_report(profile, char_dir)
    assert rep0.readiness_score == 0.0
    assert rep0.is_ready is False
    assert rep0.single_image_mode is False
    assert len(rep0.missing_attributes) == len(VIEW_TYPES)

    # 2. One optimal image (Single image mode)
    ref_mgr = CharacterReferenceManager(storage)
    front_img = _create_synthetic_character_image(512, 512)
    ref_mgr.save_reference(profile.id, _image_to_png_bytes(front_img), "front.png", "front")

    profile = storage.load_profile(profile.id)
    rep1 = generate_readiness_report(profile, char_dir)
    assert rep1.single_image_mode is True
    assert rep1.is_ready is True
    assert rep1.readiness_score >= 50.0
    assert any("A front full-body image and side reference will improve character consistency" in r for r in rep1.recommendations)

    # 3. Two images (Multi-angle mode)
    side_img = _create_synthetic_character_image(512, 512)
    ref_mgr.save_reference(profile.id, _image_to_png_bytes(side_img), "side.png", "side")

    profile = storage.load_profile(profile.id)
    rep2 = generate_readiness_report(profile, char_dir)
    assert rep2.single_image_mode is False
    assert rep2.is_ready is True
    assert rep2.readiness_score > rep1.readiness_score
    assert rep2.consistency_score is not None


# ═══════════════════════════════════════════════════════════════════════
# 3. Reference Manager Tests
# ═══════════════════════════════════════════════════════════════════════

def test_reference_manager_operations(tmp_path: Path):
    """Test storing, retrieving, and deleting references."""
    storage = CharacterStorageManager(base_dir=tmp_path)
    ref_mgr = CharacterReferenceManager(storage)
    profile = storage.create_profile("Reference Char")

    img_bytes = _image_to_png_bytes(_create_synthetic_character_image(512, 512))

    # Save front reference
    fname, path, q_report = ref_mgr.save_reference(profile.id, img_bytes, "hero_front.png", "front")
    assert path.exists()
    assert q_report.resolution_ok is True

    # Retrieve path
    resolved = ref_mgr.get_reference_path(profile.id, "front")
    assert resolved == path

    # Check listing
    active = ref_mgr.list_references(profile.id)
    assert "front" in active

    # Delete reference
    deleted = ref_mgr.delete_reference(profile.id, "front")
    assert deleted is True
    assert not path.exists()
    assert ref_mgr.get_reference_path(profile.id, "front") is None


# ═══════════════════════════════════════════════════════════════════════
# 4. REST API Endpoint Tests
# ═══════════════════════════════════════════════════════════════════════

def test_api_create_and_list_characters(client: TestClient):
    """Test POST /api/characters and GET /api/characters."""
    # Create character
    resp = client.post("/api/characters", json={"name": "Cyber Warrior", "tags": ["cyber", "neon"]})
    assert resp.status_code == 201
    data = resp.json()
    char_id = data["id"]
    assert data["name"] == "Cyber Warrior"
    assert data["tags"] == ["cyber", "neon"]
    assert data["readiness"]["readiness_score"] == 0.0

    # List characters
    list_resp = client.get("/api/characters")
    assert list_resp.status_code == 200
    characters = list_resp.json()
    assert any(c["id"] == char_id for c in characters)


def test_api_upload_reference_and_validation(client: TestClient):
    """Test POST /api/characters/{id}/upload-reference and GET /api/characters/{id}."""
    # 1. Create character
    c_resp = client.post("/api/characters", json={"name": "Angle Test Character"})
    char_id = c_resp.json()["id"]

    # 2. Upload front reference image
    img = _create_synthetic_character_image(512, 512)
    png_bytes = _image_to_png_bytes(img)

    upload_resp = client.post(
        f"/api/characters/{char_id}/upload-reference",
        data={"view_type": "front"},
        files={"file": ("front_test.png", png_bytes, "image/png")},
    )
    assert upload_resp.status_code == 200
    upload_data = upload_resp.json()
    assert upload_data["character_id"] == char_id
    assert upload_data["view_type"] == "front"
    assert upload_data["quality"]["resolution_ok"] is True
    assert upload_data["readiness"]["is_ready"] is True
    assert upload_data["readiness"]["single_image_mode"] is True

    # 3. Retrieve character details
    get_resp = client.get(f"/api/characters/{char_id}")
    assert get_resp.status_code == 200
    detail = get_resp.json()
    assert detail["id"] == char_id
    assert detail["views"]["front"] is not None
    assert detail["view_urls"]["front"] is not None
    assert detail["thumbnail_url"] is not None
    assert detail["readiness"]["single_image_mode"] is True

    # 4. Fetch the served image
    img_resp = client.get(detail["view_urls"]["front"])
    assert img_resp.status_code == 200
    assert img_resp.headers["content-type"].startswith("image/")

    # 5. Delete the reference
    del_ref_resp = client.delete(f"/api/characters/{char_id}/references/front")
    assert del_ref_resp.status_code == 200

    # 6. Delete the character
    del_char_resp = client.delete(f"/api/characters/{char_id}")
    assert del_char_resp.status_code == 200

    # 7. Confirm 404 after deletion
    assert client.get(f"/api/characters/{char_id}").status_code == 404


def test_api_validation_errors(client: TestClient):
    """Test API error handling for invalid payloads and non-existent resources."""
    # 404 for non-existent character
    resp = client.get("/api/characters/non-existent-uuid")
    assert resp.status_code == 404

    # Create dummy character
    c_resp = client.post("/api/characters", json={"name": "Err Test"})
    char_id = c_resp.json()["id"]

    # Invalid view type
    img_bytes = _image_to_png_bytes(_create_synthetic_character_image(256, 256))
    bad_view_resp = client.post(
        f"/api/characters/{char_id}/upload-reference",
        data={"view_type": "invalid_angle"},
        files={"file": ("test.png", img_bytes, "image/png")},
    )
    assert bad_view_resp.status_code == 400

    # Unsupported format
    bad_fmt_resp = client.post(
        f"/api/characters/{char_id}/upload-reference",
        data={"view_type": "front"},
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert bad_fmt_resp.status_code == 400
