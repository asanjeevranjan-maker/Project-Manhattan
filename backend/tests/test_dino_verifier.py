"""
Unit tests for Secondary Detection Verification using SigLIP/CLIP.

Verifies:
1. crop_detection_box: 15% contextual padding, image boundary clamping, and < 10px min size rejection.
2. get_alternatives_for_class: Tailored negative distractor generation.
3. verify_detection: Single crop verification interface and fallbacks.
4. verify_detections:
   - Selective class verification (vehicle/boat/building verified; water/vegetation bypassed).
   - Filtering of false-positive detections when score < threshold.
   - Retention of valid detections when score >= threshold.
   - Preservation of original Grounding DINO scores (dino_score).
   - Empty detections and invalid image handling.
5. Graceful fallback when verifier is unavailable (never crashes, skipped: True).
"""

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from services.detection.verifier import (
    VERIFIER_AVAILABLE,
    ENABLE_VERIFICATION,
    VERIFICATION_THRESHOLD,
    DEFAULT_CROP_PADDING,
    MIN_CROP_SIZE,
    DEFAULT_VERIFICATION_CLASSES,
    CLASS_NEGATIVE_ALTERNATIVES,
    crop_detection_box,
    get_alternatives_for_class,
    verify_detection,
    verify_detections,
    SiglipDetectionVerifier,
)


# =====================================================================
# 1. BOUNDING BOX CROPPING & CONTEXTUAL PADDING
# =====================================================================

def test_crop_detection_box_standard_padding():
    """Bbox is expanded by 15% on each side and cropped correctly."""
    img = Image.new("RGB", (400, 400), color=(100, 100, 100))
    # Box width = 100, height = 100 -> padding = 15px on each side
    bbox = [100.0, 100.0, 200.0, 200.0]
    crop, is_tiny = crop_detection_box(img, bbox, padding=0.15, min_size=10)

    assert is_tiny is False
    assert crop is not None
    # 100 - 15 = 85, 200 + 15 = 215 -> width 130, height 130
    assert crop.width == 130
    assert crop.height == 130


def test_crop_detection_box_boundary_clamping():
    """Bbox near borders clamps to [0, img_dim] without out-of-bounds error."""
    img = Image.new("RGB", (200, 200), color=(50, 50, 50))
    # Top-left box: 0 - pad is negative -> clamps to 0
    bbox = [5.0, 5.0, 50.0, 50.0]
    crop, is_tiny = crop_detection_box(img, bbox, padding=0.20, min_size=10)

    assert is_tiny is False
    assert crop is not None
    assert crop.width <= 200
    assert crop.height <= 200


def test_crop_detection_box_rejects_tiny_crops():
    """Crops with width or height < min_size (10px) are marked as tiny and rejected."""
    img = Image.new("RGB", (200, 200), color=(0, 0, 0))
    tiny_bbox = [50.0, 50.0, 56.0, 56.0]  # width = 6px < 10px
    crop, is_tiny = crop_detection_box(img, tiny_bbox, min_size=10)

    assert is_tiny is True
    assert crop is None


def test_crop_detection_box_invalid_inputs():
    """Handles None image or corrupt bbox gracefully."""
    img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    crop, is_tiny = crop_detection_box(img, [])
    assert is_tiny is True
    assert crop is None

    crop2, is_tiny2 = crop_detection_box(None, [10, 10, 50, 50])
    assert is_tiny2 is True
    assert crop2 is None


# =====================================================================
# 2. CLASS ALTERNATIVES LOOKUP
# =====================================================================

def test_get_alternatives_for_class():
    """Returns specialized negative distractors for known classes."""
    vehicle_alts = get_alternatives_for_class("vehicle")
    assert "building roof" in vehicle_alts
    assert "road pavement" in vehicle_alts

    boat_alts = get_alternatives_for_class("boat")
    assert "water wave crest" in boat_alts
    assert "dock pier" in boat_alts

    building_alts = get_alternatives_for_class("building")
    assert "bare ground" in building_alts

    # Unknown class falls back to default distractors
    unknown_alts = get_alternatives_for_class("unknown_object_xyz")
    assert len(unknown_alts) > 0
    assert "natural terrain" in unknown_alts


# =====================================================================
# 3. VERIFY SINGLE DETECTION INTERFACE
# =====================================================================

def test_verify_detection_tiny_crop_rejection():
    """None crop (below min resolution) is rejected with clear explanation."""
    result = verify_detection(None, "vehicle", threshold=0.35, min_crop_size=10)
    assert result["verified"] is False
    assert result["passed"] is False
    assert result["verification_score"] == 0.0
    assert "below minimum resolution" in result["reason"]


def test_verify_detection_unavailable_model():
    """When verifier is unavailable in the environment, passes gracefully without error."""
    img_crop = Image.new("RGB", (50, 50), color=(128, 128, 128))
    with patch("services.detection.verifier.VERIFIER_AVAILABLE", False):
        result = verify_detection(img_crop, "vehicle")
        assert result["verified"] is False
        assert result["passed"] is True
        assert result.get("skipped") is True


# =====================================================================
# 4. SELECTIVE VERIFICATION & FALSE-POSITIVE ELIMINATION
# =====================================================================

def test_verify_detections_selective_class_bypass():
    """
    Classes prone to false positives (vehicle, boat) are targeted.
    Natural classes like water and vegetation bypass verifier.
    """
    img = Image.new("RGB", (500, 500), color=(100, 150, 100))
    detections = [
        {"label": "water", "box": [50, 50, 300, 300], "score": 0.88},
        {"label": "vegetation", "box": [10, 10, 200, 200], "score": 0.91},
        {"label": "vehicle", "box": [100, 100, 150, 150], "score": 0.75},
    ]

    # Mock verifier predict_batch returning a passing score for vehicle
    mock_verifier = MagicMock()
    mock_verifier.predict_batch.return_value = [
        {"score": 0.82, "best_alternative": "road pavement", "best_alt_score": 0.12}
    ]

    with patch("services.detection.verifier.VERIFIER_AVAILABLE", True), \
         patch("services.detection.verifier.get_verifier", return_value=mock_verifier):
        verified_dets, meta = verify_detections(img, detections, threshold=0.35)

        assert len(verified_dets) == 3
        # Water bypassed
        water_det = next(d for d in verified_dets if d["label"] == "water")
        assert water_det["verification"]["skipped"] is True
        assert water_det["verification"]["passed"] is True

        # Vegetation bypassed
        veg_det = next(d for d in verified_dets if d["label"] == "vegetation")
        assert veg_det["verification"]["skipped"] is True

        # Vehicle verified
        veh_det = next(d for d in verified_dets if d["label"] == "vehicle")
        assert veh_det["verification"]["verified"] is True
        assert veh_det["verification"]["passed"] is True
        assert veh_det["verification"]["score"] == 0.82
        assert veh_det["dino_score"] == 0.75


def test_verify_detections_eliminates_false_positives():
    """Candidates scoring below threshold against negative distractors are eliminated."""
    img = Image.new("RGB", (500, 500), color=(100, 100, 100))
    detections = [
        # Candidate 1: real vehicle
        {"label": "vehicle", "box": [100, 100, 160, 160], "score": 0.72},
        # Candidate 2: false-positive vehicle (actually road shadow)
        {"label": "vehicle", "box": [250, 250, 310, 310], "score": 0.61},
    ]

    mock_verifier = MagicMock()
    # Candidate 1 scores 0.78 (pass), Candidate 2 scores 0.15 (fail, distractor wins)
    mock_verifier.predict_batch.return_value = [
        {"score": 0.78, "best_alternative": "road pavement", "best_alt_score": 0.15},
        {"score": 0.15, "best_alternative": "tree shadow", "best_alt_score": 0.81},
    ]

    with patch("services.detection.verifier.VERIFIER_AVAILABLE", True), \
         patch("services.detection.verifier.get_verifier", return_value=mock_verifier):
        verified_dets, meta = verify_detections(img, detections, threshold=0.35)

        # Only real vehicle should survive
        assert len(verified_dets) == 1
        assert verified_dets[0]["box"] == [100, 100, 160, 160]
        assert verified_dets[0]["verification"]["passed"] is True
        assert verified_dets[0]["verification"]["score"] == 0.78

        # Metadata records the rejected false positive
        assert meta["verified_count"] == 1
        assert meta["rejected_count"] == 1
        assert meta["total_candidates"] == 2


def test_verify_detections_rejects_tiny_candidate_box():
    """Candidates with bounding boxes below MIN_CROP_SIZE (<10px) are eliminated."""
    img = Image.new("RGB", (300, 300), color=(80, 80, 80))
    detections = [
        {"label": "vehicle", "box": [20, 20, 25, 25], "score": 0.65},  # 5x5px
    ]

    with patch("services.detection.verifier.VERIFIER_AVAILABLE", True):
        verified_dets, meta = verify_detections(img, detections, min_crop_size=10)

        assert len(verified_dets) == 0
        assert meta["rejected_count"] == 1
        assert meta["verified_count"] == 0


def test_verify_detections_disabled_toggle():
    """When enable_verification=False, detections pass through untouched."""
    img = Image.new("RGB", (200, 200), color=(50, 50, 50))
    detections = [
        {"label": "vehicle", "box": [10, 10, 50, 50], "score": 0.70},
    ]
    verified_dets, meta = verify_detections(img, detections, enable_verification=False)

    assert len(verified_dets) == 1
    assert meta["enabled"] is False
    assert meta["verified_count"] == 0


def test_verify_detections_empty_detections():
    """Handles empty detections input without error."""
    img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    verified_dets, meta = verify_detections(img, [])

    assert verified_dets == []
    assert meta["total_candidates"] == 0
    assert meta["verified_count"] == 0
    assert meta["rejected_count"] == 0


def test_verify_detections_environment_fallback_when_torch_missing():
    """If ML libraries are missing (serverless/Vercel), candidates pass through safely."""
    img = Image.new("RGB", (200, 200), color=(100, 100, 100))
    detections = [
        {"label": "vehicle", "box": [20, 20, 80, 80], "score": 0.85},
        {"label": "building", "box": [90, 90, 160, 160], "score": 0.88},
    ]

    with patch("services.detection.verifier.VERIFIER_AVAILABLE", False):
        verified_dets, meta = verify_detections(img, detections)

        assert len(verified_dets) == 2
        assert meta["verification_available"] is False
        assert verified_dets[0]["verification"]["skipped"] is True
        assert verified_dets[0]["verification"]["passed"] is True
        assert verified_dets[0]["dino_score"] == 0.85
