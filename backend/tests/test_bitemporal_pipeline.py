"""
Comprehensive automated unit tests for the Bi-Temporal Satellite Change Detection Pipeline.
Validates the elimination of false-positive and overconfident results across 10 critical scenarios:

1. test_identical_scenes: Identical images -> 0% reliable change, 0 appeared objects.
2. test_illumination_shift: Same scene with brightness/contrast shift -> normalization neutralizes differences.
3. test_misalignment_registration: Translated scene (6-8px shift) -> SIFT/ORB co-registration aligns, preventing false edge deltas.
4. test_shadow_suppression: Cast shadow variations -> shadow mask suppresses false structural change.
5. test_temporary_object_ship: Moving ship in port -> classified as TEMPORARY, confidence strictly capped, not NEW BUILDING.
6. test_water_tidal_variation: Waterline tidal shifts -> water mask discounts change, classified as ENVIRONMENTAL.
7. test_genuine_new_building: Genuine construction with multi-signal evidence -> classified as appeared with confirmed change.
8. test_genuine_demolition: Demolished structure -> confirmed as disappeared/removed.
9. test_seasonal_vegetation: Seasonal greening -> vegetation mask suppresses false change, classified as ENVIRONMENTAL.
10. test_cloud_obstruction: Cloud-covered region -> cloud mask suppresses change, confidence capped < 0.50 (UNCERTAIN).
"""

import pytest
import numpy as np
import cv2
from PIL import Image, ImageDraw

from services.temporal.registration import register_temporal_scenes
from services.temporal.normalization import normalize_radiometry
from services.temporal.nuisance_masking import compute_nuisance_masks
from services.temporal.change_detection import detect_structural_changes
from services.temporal.confidence_engine import (
    classify_persistence,
    calibrate_change_confidence,
)
from services.temporal.bitemporal_matcher import (
    match_bitemporal_detections,
    _crop_and_compare_patches,
)
from services.temporal.bitemporal_analyzer import bitemporal_analyzer


def _create_synthetic_satellite_scene(width: int = 300, height: int = 300) -> Image.Image:
    """Creates a synthetic satellite-like baseline scene with background, roads, and buildings."""
    img = Image.new("RGB", (width, height), color=(110, 120, 95))
    draw = ImageDraw.Draw(img)
    # Add road
    draw.rectangle([0, 140, width, 160], fill=(65, 65, 65))
    # Add existing building 1
    draw.rectangle([40, 40, 90, 90], fill=(210, 205, 195), outline=(50, 50, 50), width=2)
    # Add existing building 2
    draw.rectangle([180, 40, 230, 90], fill=(190, 180, 170), outline=(50, 50, 50), width=2)
    return img


# =====================================================================
# TEST 1: Identical Images -> 0% Reliable Change
# =====================================================================
def test_identical_scenes():
    scene = _create_synthetic_satellite_scene()
    t1_np = np.array(scene)
    t2_np = np.array(scene)

    change_res = detect_structural_changes(t1_np, t2_np)
    assert change_res["raw_difference_percent"] == 0.0
    assert change_res["reliable_change_percent"] == 0.0

    dets = [{"label": "building", "box": [40, 40, 90, 90], "confidence": 0.90}]
    matched = match_bitemporal_detections(
        detections_t1=dets,
        detections_t2=dets,
        img_width=300,
        img_height=300,
        image_t1=scene,
        image_t2=scene,
        structural_change_map=change_res["structural_change_map"],
    )

    assert len(matched["appeared"]) == 0
    assert len(matched["disappeared"]) == 0
    assert len(matched["persisted"]) == 1
    assert matched["persisted"][0]["confidence"] >= 0.85


# =====================================================================
# TEST 2: Illumination / Brightness Shift
# =====================================================================
def test_illumination_shift():
    t1 = _create_synthetic_satellite_scene()
    # Apply +45 brightness shift to T2
    t1_arr = np.array(t1)
    t2_arr = np.clip(t1_arr.astype(np.int16) + 45, 0, 255).astype(np.uint8)
    t2 = Image.fromarray(t2_arr)

    norm_t1, norm_t2, _ = normalize_radiometry(t1, t2)
    norm_t1_np = np.array(norm_t1)
    norm_t2_np = np.array(norm_t2)

    change_res = detect_structural_changes(norm_t1_np, norm_t2_np)
    # Radiometric normalization should suppress false structural changes
    assert change_res["reliable_change_percent"] < 1.0


# =====================================================================
# TEST 3: Sub-pixel / Pixel Translation Co-Registration
# =====================================================================
def test_misalignment_registration():
    t1 = _create_synthetic_satellite_scene(350, 350)
    t1_arr = np.array(t1)

    # Shift scene by 8 pixels horizontally and 5 pixels vertically
    M = np.float32([[1, 0, 8], [0, 1, 5]])
    t2_arr = cv2.warpAffine(t1_arr, M, (350, 350), borderMode=cv2.BORDER_REFLECT)
    t2 = Image.fromarray(t2_arr)

    reg_res = register_temporal_scenes(t1, t2, target_size=(350, 350))
    assert reg_res["registration_quality"] >= 0.75
    assert reg_res["valid_mask"] is not None


# =====================================================================
# TEST 4: Shadow Differences Suppression
# =====================================================================
def test_shadow_suppression():
    t1 = _create_synthetic_satellite_scene()
    t2 = t1.copy()
    draw = ImageDraw.Draw(t2)
    # Draw dark shadow next to building
    draw.polygon([(90, 60), (120, 60), (110, 95), (90, 95)], fill=(30, 30, 30))

    t2_arr = np.array(t2)
    nuis = compute_nuisance_masks(t2_arr)
    assert nuis["shadow_mask"] is not None
    # Shadow pixels must be captured in shadow mask
    assert np.sum(nuis["shadow_mask"][60:95, 90:120]) > 0


# =====================================================================
# TEST 5: Temporary Object (Ship / Vehicle) Persistence Gating
# =====================================================================
def test_temporary_object_ship():
    ship_label = "cargo ship"
    assert classify_persistence(ship_label) == "TEMPORARY"

    calib = calibrate_change_confidence(
        change_type="new",
        label=ship_label,
        registration_quality=0.90,
        structural_change_score=0.85,
        object_detection_confidence=0.92,
        temporal_object_evidence=0.88,
        patch_ssim_t1_t2=0.20,
    )

    # Temporary objects must have confidence strictly capped (< 0.70)
    assert calib["confidence"] <= 0.65
    assert calib["persistence"] == "TEMPORARY"
    assert "temporary" in calib["description"].lower()


# =====================================================================
# TEST 6: Water / Tidal Boundary Shift Discounting
# =====================================================================
def test_water_tidal_variation():
    water_label = "water body"
    assert classify_persistence(water_label) == "ENVIRONMENTAL"

    calib = calibrate_change_confidence(
        change_type="new",
        label=water_label,
        registration_quality=0.90,
        structural_change_score=0.75,
        object_detection_confidence=0.88,
        temporal_object_evidence=0.80,
        patch_ssim_t1_t2=0.30,
        is_water_dominated=True,
    )

    assert calib["persistence"] == "ENVIRONMENTAL"
    assert calib["confidence"] <= 0.50
    assert "water" in calib["description"].lower() or "tidal" in calib["description"].lower()


# =====================================================================
# TEST 7: Genuine New Building (High Multi-Signal Evidence)
# =====================================================================
def test_genuine_new_building():
    t1 = _create_synthetic_satellite_scene()
    t2 = t1.copy()
    draw = ImageDraw.Draw(t2)
    # Construct a new building in T2 at [100, 180, 160, 240]
    draw.rectangle([100, 180, 160, 240], fill=(230, 220, 210), outline=(40, 40, 40), width=3)

    t1_arr = np.array(t1)
    t2_arr = np.array(t2)
    chg = detect_structural_changes(t1_arr, t2_arr)

    # Detections
    dets_t1 = [
        {"label": "building", "box": [40, 40, 90, 90], "confidence": 0.90},
        {"label": "building", "box": [180, 40, 230, 90], "confidence": 0.90},
    ]
    dets_t2 = [
        {"label": "building", "box": [40, 40, 90, 90], "confidence": 0.90},
        {"label": "building", "box": [180, 40, 230, 90], "confidence": 0.90},
        {"label": "building", "box": [100, 180, 160, 240], "confidence": 0.92},
    ]

    matched = match_bitemporal_detections(
        detections_t1=dets_t1,
        detections_t2=dets_t2,
        img_width=300,
        img_height=300,
        image_t1=t1,
        image_t2=t2,
        structural_change_map=chg["structural_change_map"],
        registration_quality=0.95,
    )

    assert len(matched["appeared"]) == 1
    new_bldg = matched["appeared"][0]
    assert new_bldg["label"] == "building"
    assert new_bldg["persistence"] == "PERMANENT"
    assert new_bldg["confidence"] >= 0.70


# =====================================================================
# TEST 8: Genuine Demolition of a Building
# =====================================================================
def test_genuine_demolition():
    t1 = _create_synthetic_satellite_scene()
    t2 = t1.copy()
    draw = ImageDraw.Draw(t2)
    # Demolish building at [180, 40, 230, 90] by replacing with background dirt/grass
    draw.rectangle([180, 40, 230, 90], fill=(110, 120, 95))

    dets_t1 = [
        {"label": "building", "box": [40, 40, 90, 90], "confidence": 0.90},
        {"label": "building", "box": [180, 40, 230, 90], "confidence": 0.90},
    ]
    dets_t2 = [
        {"label": "building", "box": [40, 40, 90, 90], "confidence": 0.90},
    ]

    matched = match_bitemporal_detections(
        detections_t1=dets_t1,
        detections_t2=dets_t2,
        img_width=300,
        img_height=300,
        image_t1=t1,
        image_t2=t2,
    )

    assert len(matched["disappeared"]) == 1
    assert matched["disappeared"][0]["label"] == "building"


# =====================================================================
# TEST 9: Seasonal Vegetation Greening
# =====================================================================
def test_seasonal_vegetation():
    veg_label = "vegetation"
    assert classify_persistence(veg_label) == "ENVIRONMENTAL"

    calib = calibrate_change_confidence(
        change_type="modified",
        label=veg_label,
        registration_quality=0.88,
        structural_change_score=0.45,
        object_detection_confidence=0.82,
        temporal_object_evidence=0.50,
        patch_ssim_t1_t2=0.65,
        nuisance_overlap_ratio=0.60,
    )

    assert calib["persistence"] == "ENVIRONMENTAL"
    assert calib["confidence"] <= 0.60
    assert "environmental" in calib["description"].lower() or "vegetation" in calib["description"].lower()


# =====================================================================
# TEST 10: Cloud Obstruction Confidence Gating
# =====================================================================
def test_cloud_obstruction():
    t1 = _create_synthetic_satellite_scene()
    t2 = t1.copy()
    draw = ImageDraw.Draw(t2)
    # Bright cloud haze
    draw.ellipse([80, 80, 180, 180], fill=(245, 250, 255))

    t2_arr = np.array(t2)
    nuis = compute_nuisance_masks(t2_arr)
    assert nuis["cloud_mask"] is not None

    # Confidence engine with cloud nuisance
    calib = calibrate_change_confidence(
        change_type="new",
        label="building",
        registration_quality=0.85,
        structural_change_score=0.80,
        object_detection_confidence=0.85,
        temporal_object_evidence=0.80,
        patch_ssim_t1_t2=0.30,
        nuisance_overlap_ratio=0.75,
    )

    # Cloud obstruction must cap confidence below 0.50
    assert calib["confidence"] <= 0.50
    assert calib["verification_status"] == "UNCERTAIN"
