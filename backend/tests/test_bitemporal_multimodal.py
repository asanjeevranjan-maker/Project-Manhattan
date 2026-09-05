"""
Comprehensive unit tests for Bi-Temporal Multimodal Analysis.

Verifies:
1. Optical + SAR fusion (IHS / backscatter injection, single modality fallback).
2. Spatial co-registration (normalization, quality metric, SAR transformation).
3. Bipartite object matching (appeared, disappeared, persisted, possibly_changed, IoU, centroid distance).
4. Land-cover mask deltas (+5.2% water, -4.6% vegetation, +2.1% built-up) and change region clustering.
5. Standalone SAR differential scattering analysis (bright corner scatterers vs dark water specular absorption).
6. Visual overlays generation (before, after, change overlays).
7. End-to-end BiTemporalMultimodalAnalyzer orchestration.
8. VLM context builder and prompt directives enforcing "Observed Change" vs "Possible Cause" separation.
9. FastAPI /temporal/multimodal endpoint execution.
"""

import pytest
import io
import base64
from PIL import Image
from fastapi.testclient import TestClient

from services.temporal.fusion import fuse_optical_and_sar, normalize_sar_backscatter
from services.temporal.registration import register_temporal_scenes
from services.temporal.bitemporal_matcher import (
    match_bitemporal_detections,
    box_iou,
    box_center_distance,
    get_quadrant_location,
)
from services.temporal.land_cover_change import calculate_land_cover_deltas
from services.temporal.sar_change import analyze_sar_differential_scattering
from services.temporal.overlay_generator import generate_bitemporal_overlays, _image_to_data_url
from services.temporal.bitemporal_analyzer import bitemporal_analyzer
from services.vision.context_builder import build_vision_context
from services.vision.prompt_builder import build_satellite_analysis_prompt
from services.vision.response_parser import parse_structured_response
from main import app


# =====================================================================
# 1. OPTICAL + SAR FUSION
# =====================================================================

def test_fusion_optical_only():
    opt = Image.new("RGB", (200, 200), color=(120, 80, 40))
    fused, meta = fuse_optical_and_sar(optical_image=opt, sar_image=None)
    assert meta["modality"] == "optical_only"
    assert meta["has_optical"] is True
    assert meta["has_sar"] is False
    assert fused.size == (200, 200)


def test_fusion_sar_only():
    sar = Image.new("L", (150, 150), color=180)
    fused, meta = fuse_optical_and_sar(optical_image=None, sar_image=sar)
    assert meta["modality"] == "sar_only"
    assert meta["has_optical"] is False
    assert meta["has_sar"] is True
    assert fused.mode == "RGB"
    assert fused.size == (150, 150)


def test_fusion_multimodal_optical_and_sar():
    opt = Image.new("RGB", (300, 300), color=(50, 150, 50))
    sar = Image.new("L", (250, 250), color=210)
    fused, meta = fuse_optical_and_sar(optical_image=opt, sar_image=sar, target_size=(300, 300))
    assert meta["modality"] == "optical+sar"
    assert meta["has_optical"] is True
    assert meta["has_sar"] is True
    assert meta["fusion_method"] == "ihs_backscatter_injection"
    assert fused.size == (300, 300)
    assert fused.mode == "RGB"


def test_fusion_invalid_both_none():
    with pytest.raises(ValueError):
        fuse_optical_and_sar(None, None)


# =====================================================================
# 2. SPATIAL CO-REGISTRATION
# =====================================================================

def test_registration_normalizes_dimensions():
    t1 = Image.new("RGB", (400, 400), color=(100, 100, 100))
    t2 = Image.new("RGB", (300, 500), color=(110, 110, 110))
    sar_t2 = Image.new("L", (300, 500), color=150)

    res = register_temporal_scenes(t1, t2, target_size=(400, 400), sar_t2=sar_t2)
    assert res["aligned_t1"].size == (400, 400)
    assert res["aligned_t2"].size == (400, 400)
    assert res["aligned_sar_t2"].size == (400, 400)
    assert 0.0 <= res["registration_quality"] <= 1.0


# =====================================================================
# 3. SPATIAL OBJECT MATCHING (APPEARED, DISAPPEARED, PERSISTED, CHANGED)
# =====================================================================

def test_bitemporal_object_matching_all_classes():
    dets_t1 = [
        # 1. Stable building (will persist)
        {"label": "building", "box": [100.0, 100.0, 180.0, 180.0], "confidence": 0.88},
        # 2. Modified structure (will be possibly_changed: area expands)
        {"label": "structure", "box": [300.0, 300.0, 350.0, 350.0], "confidence": 0.82},
        # 3. Demolished building (will disappear)
        {"label": "building", "box": [50.0, 50.0, 90.0, 90.0], "confidence": 0.79},
    ]

    dets_t2 = [
        # 1. Matches stable building
        {"label": "building", "box": [102.0, 101.0, 181.0, 179.0], "confidence": 0.90},
        # 2. Matches structure with 2x area expansion
        {"label": "structure", "box": [300.0, 300.0, 400.0, 400.0], "confidence": 0.85},
        # 4. Newly constructed vehicle in T2 (will appear)
        {"label": "vehicle", "box": [500.0, 500.0, 540.0, 540.0], "confidence": 0.84},
    ]

    matched = match_bitemporal_detections(
        detections_t1=dets_t1,
        detections_t2=dets_t2,
        img_width=600,
        img_height=600,
    )

    objs = matched["objects"]
    summary = matched["summary"]

    # Check counts
    assert summary["persisted_count"] == 1
    assert summary["possibly_changed_count"] == 1
    assert summary["appeared_count"] == 1
    assert summary["disappeared_count"] == 1

    # Verify persisted
    assert objs["persisted"][0]["label"] == "building"
    assert objs["persisted"][0]["iou"] >= 0.85

    # Verify changed
    assert objs["possibly_changed"][0]["label"] == "structure"
    assert objs["possibly_changed"][0]["area_ratio"] >= 1.50

    # Verify appeared
    assert objs["appeared"][0]["label"] == "vehicle"
    assert "lower-right" in objs["appeared"][0]["location"]

    # Verify disappeared
    assert objs["disappeared"][0]["label"] == "building"
    assert "upper-left" in objs["disappeared"][0]["location"]


# =====================================================================
# 4. LAND-COVER DELTAS & CHANGE REGIONS
# =====================================================================

def test_calculate_land_cover_deltas():
    lc_t1 = {
        "coverage": {
            "water": {"percentage": 10.0},
            "built_up": {"percentage": 25.0},
            "vegetation": {"percentage": 50.0},
            "bare_soil": {"percentage": 10.0},
            "other": {"percentage": 5.0},
        }
    }

    lc_t2 = {
        "coverage": {
            "water": {"percentage": 15.2},
            "built_up": {"percentage": 27.1},
            "vegetation": {"percentage": 45.4},
            "bare_soil": {"percentage": 10.8},
            "other": {"percentage": 1.5},
        }
    }

    # Detections indicating water in lower-right
    dets_t2 = [
        {"label": "water", "box": [700, 700, 950, 950]},
    ]

    delta_res = calculate_land_cover_deltas(
        land_cover_t1=lc_t1,
        land_cover_t2=lc_t2,
        detections_t2=dets_t2,
        image_size=(1000, 1000),
    )

    deltas = delta_res["land_cover_change"]
    assert deltas["water"] == 5.2
    assert deltas["built_up"] == 2.1
    assert deltas["vegetation"] == -4.6
    assert deltas["bare_soil"] == 0.8

    # Verify change regions
    regions = delta_res["change_regions"]
    assert len(regions) >= 3

    water_reg = next(r for r in regions if r["category"] == "water")
    assert water_reg["delta"] == 5.2
    assert water_reg["action"] == "increased"
    assert "lower-right" in water_reg["location"]


# =====================================================================
# 5. STANDALONE SAR DIFFERENTIAL SCATTERING ANALYSIS
# =====================================================================

def test_sar_differential_scattering_bright_and_dark_shifts():
    # 100x100 SAR T1: uniform medium backscatter (128)
    sar1 = Image.new("L", (100, 100), color=128)

    # SAR T2: Add bright corner scatterers (metal/structures) and dark water patch
    sar2 = Image.new("L", (100, 100), color=128)
    # Bright region: 20x20 at value 240 (diff = +112 > 40) -> 400 pixels = 4%
    for x in range(10, 30):
        for y in range(10, 30):
            sar2.putpixel((x, y), 240)

    # Dark specular region: 20x20 at value 20 (diff = -108 < -40) -> 400 pixels = 4%
    for x in range(60, 80):
        for y in range(60, 80):
            sar2.putpixel((x, y), 20)

    sar_diff = analyze_sar_differential_scattering(sar1, sar2, threshold=40)

    assert sar_diff["available"] is True
    assert sar_diff["bright_scatterers_percentage"] == 4.0
    assert sar_diff["dark_scatterers_percentage"] == 4.0
    assert sar_diff["radar_detected_new_structures"] is True
    assert sar_diff["radar_detected_flood_inundation"] is True
    assert "Radar backscatter surge" in sar_diff["radar_summary"]
    assert "Radar backscatter drop" in sar_diff["radar_summary"]


def test_sar_differential_scattering_missing_sar():
    res = analyze_sar_differential_scattering(None, None)
    assert res["available"] is False
    assert res["bright_scatterers_percentage"] == 0.0


# =====================================================================
# 6. VISUAL OVERLAYS GENERATOR
# =====================================================================

def test_generate_bitemporal_overlays():
    t1 = Image.new("RGB", (200, 200), color=(100, 100, 100))
    t2 = Image.new("RGB", (200, 200), color=(110, 110, 110))

    matched_objs = {
        "appeared": [{"box_t2": [10, 10, 50, 50]}],
        "disappeared": [{"box_t1": [60, 60, 90, 90]}],
        "persisted": [{"box_t1": [120, 120, 150, 150], "box_t2": [120, 120, 150, 150]}],
        "possibly_changed": [{"box_t1": [160, 160, 190, 190], "box_t2": [160, 160, 195, 195]}],
    }

    overlays = generate_bitemporal_overlays(t1, t2, matched_objs)
    assert "before_overlay_url" in overlays
    assert "after_overlay_url" in overlays
    assert "change_overlay_url" in overlays
    assert overlays["before_overlay_url"].startswith("data:image/png;base64,")
    assert overlays["change_overlay_url"].startswith("data:image/png;base64,")


# =====================================================================
# 7. END-TO-END BITEMPORAL MULTIMODAL ANALYZER
# =====================================================================

def test_bitemporal_analyzer_orchestration():
    opt1 = Image.new("RGB", (250, 250), color=(80, 140, 80))
    sar1 = Image.new("L", (250, 250), color=130)
    opt2 = Image.new("RGB", (250, 250), color=(85, 135, 85))
    sar2 = Image.new("L", (250, 250), color=135)

    dets_t1 = [{"label": "building", "box": [30, 30, 80, 80], "confidence": 0.85}]
    dets_t2 = [
        {"label": "building", "box": [31, 30, 81, 79], "confidence": 0.87},
        {"label": "vehicle", "box": [150, 150, 190, 190], "confidence": 0.78},
    ]

    lc_t1 = {"coverage": {"water": 10.0, "built_up": 20.0, "vegetation": 50.0, "bare_soil": 20.0}}
    lc_t2 = {"coverage": {"water": 15.2, "built_up": 22.1, "vegetation": 45.4, "bare_soil": 17.3}}

    analysis = bitemporal_analyzer.analyze(
        t1_optical=opt1,
        t1_sar=sar1,
        t2_optical=opt2,
        t2_sar=sar2,
        prompt="building, vehicle",
        date_t1="2024-01-01",
        date_t2="2024-06-01",
        detections_t1=dets_t1,
        detections_t2=dets_t2,
        land_cover_t1=lc_t1,
        land_cover_t2=lc_t2,
    )

    assert analysis["success"] is True
    assert analysis["dates"] == {"t1": "2024-01-01", "t2": "2024-06-01"}
    assert analysis["modalities"]["t1"]["modality"] == "optical+sar"
    assert analysis["modalities"]["t2"]["modality"] == "optical+sar"

    # Objects checked
    objs = analysis["objects"]
    assert len(objs["persisted"]) == 1
    assert len(objs["appeared"]) == 1

    # Land-cover deltas checked
    deltas = analysis["land_cover_change"]
    assert deltas["water"] == 5.2
    assert deltas["vegetation"] == -4.6
    assert deltas["built_up"] == 2.1

    # SAR checked
    assert analysis["sar_analysis"]["available"] is True

    # Overlays checked
    assert "change_overlay_url" in analysis["overlays"]


# =====================================================================
# 8. VLM CONTEXT & PROMPT DIRECTIVES (OBSERVED CHANGE VS CAUSE)
# =====================================================================

def test_vlm_context_builder_with_multimodal_change():
    change_payload = {
        "land_cover_change": {
            "water": 5.2,
            "built_up": 2.1,
            "vegetation": -4.6,
        },
        "objects_summary": {
            "appeared_count": 3,
            "disappeared_count": 1,
            "possibly_changed_count": 2,
            "persisted_count": 14,
        },
        "objects": {
            "appeared": [{"label": "building", "location": "lower-right region"}],
            "disappeared": [{"label": "structure", "location": "upper-left region"}],
        },
        "sar_analysis": {
            "available": True,
            "radar_summary": "Radar backscatter drop (-4.1%): Specular microwave absorption confirms standing water.",
        },
        "primary_shift": "Water coverage increased by 5.2 percentage points in lower-right region.",
    }

    ctx = build_vision_context(change_detection=change_payload)
    summary_text = ctx["summary_text"]

    assert "water: +5.2 percentage points" in summary_text
    assert "vegetation: -4.6 percentage points" in summary_text
    assert "built_up: +2.1 percentage points" in summary_text
    assert "3 appeared, 1 disappeared" in summary_text
    assert "Specular microwave absorption confirms standing water" in summary_text
    assert "Primary shift: Water coverage increased by 5.2 percentage points" in summary_text
    assert ctx["evidence_flags"]["change_detection_used"] is True


def test_prompt_builder_changes_mode_separates_observation_and_cause():
    system_prompt, user_prompt = build_satellite_analysis_prompt(
        user_query="What happened between these two dates?",
        analysis_mode="changes",
    )
    full_prompt = f"{system_prompt}\n{user_prompt}"
    # Check that the mandatory distinction rule is present
    assert "OBSERVED CHANGE" in full_prompt
    assert "POSSIBLE CAUSE" in full_prompt
    assert "percentage points" in full_prompt


def test_parse_structured_response_preserves_observed_and_causes():
    mock_json = """
    {
      "answer": "Water coverage expanded significantly in the lower-right.",
      "summary": "Water increased by 5.2 percentage points.",
      "observed_changes": [
        "Water coverage increased by 5.2 percentage points in the lower-right.",
        "3 new buildings appeared along the road."
      ],
      "possible_causes": [
        "Likely caused by seasonal river surge or upstream reservoir release."
      ],
      "evidence": {
        "detections_used": true,
        "segmentation_used": true,
        "land_cover_used": true,
        "change_detection_used": true
      },
      "observations": [
        {"finding": "Water expansion", "location": "lower-right", "confidence": "high"}
      ]
    }
    """
    parsed = parse_structured_response(mock_json, query="changes", change_used=True)
    assert len(parsed.observed_changes) == 2
    assert "5.2 percentage points" in parsed.observed_changes[0]
    assert len(parsed.possible_causes) == 1
    assert "seasonal river surge" in parsed.possible_causes[0]


# =====================================================================
# 9. FASTAPI ENDPOINT (/temporal/multimodal)
# =====================================================================

def test_fastapi_temporal_multimodal_endpoint():
    client = TestClient(app)

    # Create small test images
    img1 = Image.new("RGB", (100, 100), color=(70, 120, 70))
    img2 = Image.new("RGB", (100, 100), color=(75, 115, 75))

    url1 = _image_to_data_url(img1)
    url2 = _image_to_data_url(img2)

    payload = {
        "t1_optical": url1,
        "t2_optical": url2,
        "prompt": "building, water",
        "date_t1": "2024-01-01",
        "date_t2": "2024-06-01",
        "user_query": "Identify all structural and water shifts.",
        "enable_vlm_interpretation": False,  # Bypass live external LLM API
        "detections_t1": [{"label": "building", "box": [10, 10, 40, 40], "confidence": 0.85}],
        "detections_t2": [
            {"label": "building", "box": [10, 10, 40, 40], "confidence": 0.85},
            {"label": "water", "box": [60, 60, 95, 95], "confidence": 0.90},
        ],
        "land_cover_t1": {"coverage": {"water": 5.0, "built_up": 15.0, "vegetation": 80.0}},
        "land_cover_t2": {"coverage": {"water": 12.5, "built_up": 15.0, "vegetation": 72.5}},
    }

    response = client.post("/temporal/multimodal", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert data["land_cover_change"]["water"] == 7.5
    assert data["land_cover_change"]["vegetation"] == -7.5
    assert len(data["objects"]["appeared"]) == 1
    assert data["objects"]["appeared"][0]["label"] == "water"
    assert len(data["objects"]["persisted"]) == 1
    assert "change_overlay_url" in data["overlays"]
