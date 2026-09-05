"""
Unit tests for LandCoverAnalyzer and Objective Land-Cover Coverage Calculation.

Verifies:
1. Exact non-overlapping masks (exact pixel counts and percentages).
2. Overlapping masks (class priority hierarchy eliminates double counting).
3. No masks / segmentation unavailable (never fabricates fake numbers).
4. Percentages formula (class_pixels / valid_image_pixels * 100).
5. Strict sum tolerance (percentages always total 100.00%).
6. Overlay visualization preview generation (base64 data URL).
7. Legacy response mapping (replaces hallucinated 35%/45%/20% with real measurements or unavailable flag).
"""

import pytest
from PIL import Image
from services.detection.land_cover import (
    LandCoverAnalyzer,
    calculate_land_cover,
    DEFAULT_CATEGORY_PRIORITY,
    LABEL_TO_CATEGORY,
    map_label_to_category,
)
from services.vision.response_parser import to_legacy_analysis_result, SatelliteAnalysisStructured, ObservationItem


# =====================================================================
# 1. EXACT MASKS (KNOWN PIXEL COUNTS)
# =====================================================================
def test_exact_non_overlapping_masks():
    """
    On a 100x100 canvas (10,000 pixels total):
    - Water mask: 2,000 pixels -> 20.0%
    - Building mask: 3,000 pixels -> 30.0%
    - Vegetation mask: 4,000 pixels -> 40.0%
    - Unassigned ('other'): 1,000 pixels -> 10.0%
    """
    image_size = (100, 100)
    detections = [
        {
            "id": "det-1",
            "label": "water",
            "bbox": [0, 0, 100, 20],  # 100 * 20 = 2000 pixels
            "mask": {
                "format": "polygon",
                "polygon": [[0, 0], [99, 0], [99, 19], [0, 19]],
            },
        },
        {
            "id": "det-2",
            "label": "building",
            "bbox": [0, 20, 100, 50],  # 100 * 30 = 3000 pixels
            "mask": {
                "format": "polygon",
                "polygon": [[0, 20], [99, 20], [99, 49], [0, 49]],
            },
        },
        {
            "id": "det-3",
            "label": "vegetation",
            "bbox": [0, 50, 100, 90],  # 100 * 40 = 4000 pixels
            "mask": {
                "format": "polygon",
                "polygon": [[0, 50], [99, 50], [99, 89], [0, 89]],
            },
        },
    ]

    result = calculate_land_cover(
        image_size=image_size,
        detections=detections,
        segmentation_available=True,
    )

    assert result["available"] is True
    assert result["measured_from_masks"] is True
    assert result["estimated"] is False
    assert result["total_pixels"] == 10000

    lc = result["land_cover"]
    assert lc["water"]["pixels"] == 2000
    assert lc["water"]["percentage"] == 20.0

    assert lc["built_up"]["pixels"] == 3000
    assert lc["built_up"]["percentage"] == 30.0

    assert lc["vegetation"]["pixels"] == 4000
    assert lc["vegetation"]["percentage"] == 40.0

    assert lc["other"]["pixels"] == 1000
    assert lc["other"]["percentage"] == 10.0

    # Strict sum verification
    total_pct = sum(c["percentage"] for c in lc.values())
    assert abs(total_pct - 100.0) < 0.01


# =====================================================================
# 2. OVERLAPPING MASKS & CLASS PRIORITY
# =====================================================================
def test_overlapping_masks_priority_prevents_double_counting():
    """
    Tests priority hierarchy on a 100x100 canvas (10,000 pixels):
    Water mask: [0, 0, 49, 49] (50x50 = 2,500 pixels).
    Building mask: [25, 0, 74, 49] (50x50 = 2,500 pixels).
    Overlap area: [25, 0, 49, 49] (25x50 = 1,250 pixels).

    Under default priority (water -> built_up -> vegetation -> other):
    - Water claims all its 2,500 pixels.
    - Building claims only the unallocated 1,250 pixels (2,500 - 1,250).
    - Other claims remaining 6,250 pixels (10,000 - 3,750).
    Total counted pixels = 10,000 (NOT 11,250).
    """
    image_size = (100, 100)
    detections = [
        {
            "id": "det-water",
            "label": "water",
            "mask": {
                "format": "polygon",
                "polygon": [[0, 0], [49, 0], [49, 49], [0, 49]],
            },
        },
        {
            "id": "det-building",
            "label": "building",
            "mask": {
                "format": "polygon",
                "polygon": [[25, 0], [74, 0], [74, 49], [25, 49]],
            },
        },
    ]

    result = calculate_land_cover(
        image_size=image_size,
        detections=detections,
        segmentation_available=True,
    )

    lc = result["land_cover"]
    assert lc["water"]["pixels"] == 2500
    assert lc["water"]["percentage"] == 25.0

    # Building only claims non-overlapping pixels
    assert lc["built_up"]["pixels"] == 1250
    assert lc["built_up"]["percentage"] == 12.5

    assert lc["vegetation"]["pixels"] == 0
    assert lc["vegetation"]["percentage"] == 0.0

    assert lc["other"]["pixels"] == 6250
    assert lc["other"]["percentage"] == 62.5

    # Sum of occupied pixels must exactly equal 10,000
    total_pixels_counted = (
        lc["water"]["pixels"]
        + lc["built_up"]["pixels"]
        + lc["vegetation"]["pixels"]
        + lc["other"]["pixels"]
    )
    assert total_pixels_counted == 10000


# =====================================================================
# 3. NO MASKS & SEGMENTATION UNAVAILABLE (TRUTHFUL REPORTING)
# =====================================================================
def test_segmentation_unavailable_does_not_fabricate():
    """
    When segmentation is unavailable, the analyzer must NEVER fabricate
    hallucinated percentages like 35%/45%/20%.
    It must return available=False with a clear reason.
    """
    result = calculate_land_cover(
        image_size=(500, 500),
        detections=[],
        segmentation_available=False,
    )

    assert result["available"] is False
    assert result["reason"] == "Segmentation unavailable"
    assert result["measured_from_masks"] is False
    assert result["estimated"] is False
    assert "land_cover" not in result


def test_empty_detections_with_segmentation_available():
    """When segmentation is active but 0 masks were detected, 100% is classified as 'other'."""
    result = calculate_land_cover(
        image_size=(200, 200),
        detections=[],
        segmentation_available=True,
    )

    assert result["available"] is True
    assert result["measured_from_masks"] is True
    lc = result["land_cover"]
    assert lc["other"]["pixels"] == 40000
    assert lc["other"]["percentage"] == 100.0
    assert lc["built_up"]["pixels"] == 0
    assert lc["water"]["pixels"] == 0
    assert lc["vegetation"]["pixels"] == 0


# =====================================================================
# 4. PERCENTAGES FORMULA & SUM TOLERANCE
# =====================================================================
def test_percentages_formula_and_sum_tolerance():
    """
    Tests arbitrary pixel distribution to verify exact percentage math
    and guarantee total is 100.00% within 0.01% floating-point tolerance.
    """
    # 250 x 200 = 50,000 pixels
    image_size = (250, 200)
    total_pixels = 50000

    detections = [
        # 12,345 pixels water
        {
            "id": "det-1",
            "label": "river",
            "mask": {
                "format": "polygon",
                "polygon": [[0, 0], [123.45, 0], [123.45, 100], [0, 100]],
            },
        },
        # 15,000 pixels built-up
        {
            "id": "det-2",
            "label": "structure",
            "mask": {
                "format": "polygon",
                "polygon": [[0, 100], [150, 100], [150, 200], [0, 200]],
            },
        },
    ]

    result = calculate_land_cover(
        image_size=image_size,
        detections=detections,
        segmentation_available=True,
    )

    lc = result["land_cover"]
    sum_pct = sum(item["percentage"] for item in lc.values())
    assert abs(sum_pct - 100.0) <= 0.01

    sum_px = sum(item["pixels"] for item in lc.values())
    assert sum_px == total_pixels


# =====================================================================
# 5. OVERLAY VISUALIZATION PREVIEW
# =====================================================================
def test_overlay_visualization_generation():
    """Verifies that overlay visualization generates a valid base64 PNG data URL."""
    base_img = Image.new("RGB", (400, 300), color=(50, 70, 90))
    detections = [
        {
            "label": "water",
            "mask": {
                "format": "polygon",
                "polygon": [[10, 10], [100, 10], [100, 100], [10, 100]],
            },
        },
        {
            "label": "building",
            "mask": {
                "format": "polygon",
                "polygon": [[150, 10], [250, 10], [250, 100], [150, 100]],
            },
        },
    ]

    result = calculate_land_cover(
        image_size=(400, 300),
        detections=detections,
        segmentation_available=True,
        generate_visualization=True,
        base_image=base_img,
    )

    assert result["overlay_visualization"] is not None
    assert result["overlay_visualization"].startswith("data:image/png;base64,")


# =====================================================================
# 6. LEGACY RESPONSE MAPPING INTEGRATION
# =====================================================================
def test_legacy_response_mapping_with_real_land_cover():
    """
    Verifies that to_legacy_analysis_result replaces hardcoded values
    with real mask-measured coverage when available, or marks available=False.
    """
    mock_obs = [
        ObservationItem(finding="Harbor berth", location="center", confidence="high", evidence="Quay wall"),
    ]
    structured = SatelliteAnalysisStructured(
        summary="Harbor scene",
        answer_to_query="Active harbor",
        observations=mock_obs,
        uncertainties=[],
        model_notes={"recommended_action": "Inspect vessels"},
    )

    # Case A: Real land cover passed
    real_lc = {
        "available": True,
        "measured_from_masks": True,
        "coverage": [
            {"class": "water", "coverage": 0.65, "color": "#06b6d4"},
            {"class": "built-up", "coverage": 0.25, "color": "#f97316"},
            {"class": "other", "coverage": 0.10, "color": "#6b7280"},
        ],
        "land_cover": {
            "water": {"pixels": 6500, "percentage": 65.0},
            "built_up": {"pixels": 2500, "percentage": 25.0},
            "vegetation": {"pixels": 0, "percentage": 0.0},
            "other": {"pixels": 1000, "percentage": 10.0},
        },
    }
    legacy_a = to_legacy_analysis_result(structured, land_cover_result=real_lc)
    assert legacy_a["measured_from_masks"] is True
    assert legacy_a["coverage"] == real_lc["coverage"]
    assert legacy_a["land_cover"]["available"] is True

    # Case B: No land cover / segmentation unavailable
    legacy_b = to_legacy_analysis_result(structured, land_cover_result=None)
    assert legacy_b["measured_from_masks"] is False
    assert legacy_b["coverage"] == []  # NO fabricated 35%/45%/20%!
    assert legacy_b["land_cover"]["available"] is False
    assert legacy_b["land_cover"]["reason"] == "Segmentation unavailable"
