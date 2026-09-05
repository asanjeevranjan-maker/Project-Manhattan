"""
Unit tests for Grounding DINO satellite vocabulary, class presets, class-specific thresholds,
label normalization, geometry validation, location calculation, and output formatting.
"""

import pytest
from services.detection.vocabulary import (
    SATELLITE_CLASSES,
    ANALYSIS_PRESETS,
    ABSTRACT_CONCEPTS,
    DEFAULT_CLASS_THRESHOLDS,
    ClassThreshold,
    get_class_threshold,
    normalize_label,
    sanitize_prompt,
    map_score_to_confidence_level,
    compute_relative_location,
    format_detection,
    filter_and_format_detections,
    remove_duplicate_detections,
    box_iou,
)


# =====================================================================
# 1. CLASS VOCABULARY & OBSERVABLE CLASSES
# =====================================================================
def test_satellite_classes_vocabulary():
    """Verifies all required observable classes exist and distinct classes are not merged."""
    required_classes = [
        "building", "vehicle", "road", "bridge", "river", "water body",
        "vegetation", "forest", "field", "bare soil", "construction area",
        "ship", "aircraft", "railway", "tower"
    ]
    for req in required_classes:
        assert req in SATELLITE_CLASSES, f"Missing required class: {req}"

    # Verify river and water body are distinct
    assert "river" in SATELLITE_CLASSES
    assert "water body" in SATELLITE_CLASSES
    assert "river" not in SATELLITE_CLASSES["water body"]
    assert "water body" not in SATELLITE_CLASSES["river"]

    # Verify road and railway are distinct
    assert "railway" not in SATELLITE_CLASSES["road"]
    assert "road" not in SATELLITE_CLASSES["railway"]

    # Verify vehicle, ship, aircraft are distinct
    assert "ship" not in SATELLITE_CLASSES["vehicle"]
    assert "aircraft" not in SATELLITE_CLASSES["vehicle"]


# =====================================================================
# 2. LABEL NORMALIZATION & JUNK SUPPRESSION
# =====================================================================
def test_label_normalization_synonyms():
    """Tests normalization of synonyms to canonical observable classes."""
    # Vehicles
    assert normalize_label("car") == ("vehicle", "car")
    assert normalize_label("truck") == ("vehicle", "truck")
    assert normalize_label("van") == ("vehicle", "van")
    assert normalize_label("automobile") == ("vehicle", "automobile")

    # Buildings
    assert normalize_label("house") == ("building", "house")
    assert normalize_label("roof") == ("building", "roof")
    assert normalize_label("warehouse") == ("building", "warehouse")
    assert normalize_label("structure") == ("building", "structure")

    # Hydrology
    assert normalize_label("stream") == ("river", "stream")
    assert normalize_label("canal") == ("river", "canal")
    assert normalize_label("lake") == ("water body", "lake")
    assert normalize_label("reservoir") == ("water body", "reservoir")

    # Transport & Infrastructure
    assert normalize_label("highway") == ("road", "highway")
    assert normalize_label("paved road") == ("road", "paved road")
    assert normalize_label("airplane") == ("aircraft", "airplane")
    assert normalize_label("cargo ship") == ("ship", "cargo ship")
    assert normalize_label("train tracks") == ("railway", "train tracks")
    assert normalize_label("pylon") == ("tower", "pylon")


def test_label_normalization_truncation_and_stopwords():
    """Verifies that truncated single-character labels and stopwords are rejected."""
    # Truncated single letters ('A', 'L', etc.)
    assert normalize_label("A")[0] is None
    assert normalize_label("L")[0] is None
    assert normalize_label("a")[0] is None
    assert normalize_label("x")[0] is None

    # Stopwords ('ALL', 'the', etc.)
    assert normalize_label("all")[0] is None
    assert normalize_label("ALL")[0] is None
    assert normalize_label("the")[0] is None
    assert normalize_label("and")[0] is None
    assert normalize_label("of")[0] is None
    assert normalize_label("")[0] is None
    assert normalize_label(None)[0] is None


# =====================================================================
# 3. CLASS-SPECIFIC THRESHOLDING
# =====================================================================
def test_class_specific_thresholds():
    """Tests that score, aspect ratio, and area ratio constraints are evaluated per class."""
    # Building requires min_score 0.35 by default
    b_pass = format_detection("1", "building", score=0.40, box=[10, 10, 50, 50], width=500, height=500)
    assert b_pass is not None
    assert b_pass["label"] == "building"

    b_fail = format_detection("2", "building", score=0.30, box=[10, 10, 50, 50], width=500, height=500)
    assert b_fail is None  # Dropped because 0.30 < 0.35

    # River has lower score threshold (0.28)
    r_pass = format_detection("3", "river", score=0.29, box=[10, 10, 200, 30], width=500, height=500)
    assert r_pass is not None
    assert r_pass["label"] == "river"

    # Vehicle area ratio constraint (max 0.025)
    # A box of 200x200 in a 500x500 image = 40,000 / 250,000 = 0.16 (way too big for a vehicle)
    v_huge = format_detection("4", "car", score=0.80, box=[50, 50, 250, 250], width=500, height=500)
    assert v_huge is None

    # A compact vehicle 20x15 in 500x500 = 300 / 250,000 = 0.0012 -> should pass
    v_compact = format_detection("5", "car", score=0.80, box=[50, 50, 70, 65], width=500, height=500)
    assert v_compact is not None
    assert v_compact["label"] == "vehicle"
    assert v_compact["raw_label"] == "car"


def test_class_specific_aspect_ratio():
    """Tests that elongated shapes for roads/rivers pass but fail for buildings."""
    # Road with aspect ratio 10:1 (200x20)
    road_det = format_detection("1", "road", score=0.75, box=[10, 10, 210, 30], width=1000, height=1000)
    assert road_det is not None

    # Building with extreme aspect ratio 10:1 (200x20) fails building aspect limit (max 5.0)
    building_det = format_detection("2", "building", score=0.75, box=[10, 10, 210, 30], width=1000, height=1000)
    assert building_det is None


# =====================================================================
# 4. VALID BBOX CONVERSION & LOCATION CALCULATION
# =====================================================================
def test_bbox_validation():
    """Verifies that invalid or zero-area boxes are rejected."""
    # Inverted box (x2 < x1)
    inv_x = format_detection("1", "building", score=0.8, box=[100, 10, 50, 50], width=500, height=500)
    assert inv_x is None

    # Inverted box (y2 < y1)
    inv_y = format_detection("2", "building", score=0.8, box=[10, 100, 50, 50], width=500, height=500)
    assert inv_y is None

    # Zero area box
    zero_box = format_detection("3", "building", score=0.8, box=[50, 50, 50, 50], width=500, height=500)
    assert zero_box is None


def test_relative_location_calculation():
    """Verifies quadrant location calculation based on center point."""
    width, height = 1000, 1000

    # Center box
    loc_center = compute_relative_location([450, 450, 550, 550], width, height)
    assert loc_center == "center"

    # Upper-left
    loc_ul = compute_relative_location([50, 50, 150, 150], width, height)
    assert loc_ul == "upper-left"

    # Upper-right
    loc_ur = compute_relative_location([850, 50, 950, 150], width, height)
    assert loc_ur == "upper-right"

    # Lower-left
    loc_ll = compute_relative_location([50, 850, 150, 950], width, height)
    assert loc_ll == "lower-left"

    # Lower-right
    loc_lr = compute_relative_location([850, 850, 950, 950], width, height)
    assert loc_lr == "lower-right"


def test_clean_output_format():
    """Tests the structured output dictionary matches requirements."""
    det = format_detection(
        det_id="det-1",
        raw_label="truck",
        score=0.842,
        box=[100.0, 100.0, 125.0, 115.0],
        width=1000,
        height=1000,
    )
    assert det is not None
    assert det["id"] == "det-1"
    assert det["label"] == "vehicle"
    assert det["raw_label"] == "truck"
    assert det["score"] == 0.842
    assert det["confidence"] == 0.842
    assert det["confidence_level"] == "high"
    assert det["bbox"] == [100.0, 100.0, 125.0, 115.0]
    assert det["box"] == [100.0, 100.0, 125.0, 115.0]
    assert det["center"] == [112.5, 107.5]
    assert det["relative_location"] == "upper-left"


# =====================================================================
# 5. EMPTY DETECTIONS & DUPLICATE REMOVAL (NMS)
# =====================================================================
def test_empty_detections():
    """Verifies empty detection inputs return empty lists without crashing."""
    assert filter_and_format_detections([], width=500, height=500) == []
    assert remove_duplicate_detections([]) == []


def test_duplicate_removal_same_class():
    """Tests that overlapping detections of the same class are deduplicated via NMS."""
    dets = [
        {"label": "building", "score": 0.85, "bbox": [100, 100, 200, 200]},
        {"label": "building", "score": 0.70, "bbox": [105, 105, 205, 205]},  # High overlap with above
        {"label": "building", "score": 0.80, "bbox": [400, 400, 500, 500]},  # Separate building
    ]
    kept = remove_duplicate_detections(dets, iou_threshold=0.25)
    assert len(kept) == 2
    # Kept highest scoring box of the overlapping pair (0.85) and the separate one (0.80)
    scores = [d["score"] for d in kept]
    assert 0.85 in scores
    assert 0.80 in scores
    assert 0.70 not in scores


def test_duplicate_removal_different_classes_preserved():
    """Tests that overlapping detections of different classes (e.g. vehicle on road) are NOT removed."""
    dets = [
        {"label": "road", "score": 0.85, "bbox": [50, 100, 500, 150]},
        {"label": "vehicle", "score": 0.78, "bbox": [100, 110, 130, 135]},  # Inside road bbox
    ]
    kept = remove_duplicate_detections(dets, iou_threshold=0.25)
    assert len(kept) == 2


# =====================================================================
# 6. PROMPT SANITIZATION & PRESETS
# =====================================================================
def test_prompt_sanitization_abstract_translation():
    """Tests that abstract hazard/danger queries are translated to concrete observable classes."""
    prompt = "identify flood danger area and vulnerable settlement"
    clean = sanitize_prompt(prompt)
    # Must NOT contain abstract terms
    assert "danger" not in clean
    assert "vulnerable" not in clean
    # Must contain observable classes separated by periods
    assert "river" in clean or "water body" in clean
    assert "building" in clean
    assert clean.endswith(" .")


def test_prompt_presets():
    """Verifies all presets format into clean period-separated prompts."""
    for preset_name, expected_classes in ANALYSIS_PRESETS.items():
        preset_prompt = sanitize_prompt(preset=preset_name)
        assert preset_prompt.endswith(" .")
        for cls in expected_classes:
            assert cls in preset_prompt


def test_confidence_level_mapping():
    """Verifies score mapping to qualitative confidence levels."""
    assert map_score_to_confidence_level(0.90) == "high"
    assert map_score_to_confidence_level(0.65) == "high"
    assert map_score_to_confidence_level(0.64) == "medium"
    assert map_score_to_confidence_level(0.40) == "medium"
    assert map_score_to_confidence_level(0.39) == "low"
    assert map_score_to_confidence_level(0.15) == "low"


def test_fastapi_discovery_endpoints():
    """Tests /classes, /presets, and /thresholds endpoints on the FastAPI backend."""
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    # /classes
    res_classes = client.get("/classes")
    assert res_classes.status_code == 200
    data_classes = res_classes.json()
    assert "classes" in data_classes
    assert "building" in data_classes["classes"]
    assert "vehicle" in data_classes["classes"]

    # /presets
    res_presets = client.get("/presets")
    assert res_presets.status_code == 200
    data_presets = res_presets.json()
    assert "presets" in data_presets
    assert "urban" in data_presets["presets"]
    assert "water" in data_presets["presets"]

    # /thresholds
    res_thresh = client.get("/thresholds")
    assert res_thresh.status_code == 200
    data_thresh = res_thresh.json()
    assert "thresholds" in data_thresh
    assert "building" in data_thresh["thresholds"]
    assert data_thresh["thresholds"]["building"]["min_score"] == 0.35
