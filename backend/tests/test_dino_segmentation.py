"""
Unit tests for SAM2-based Object Segmentation for Grounding DINO.

Verifies:
1. Graceful fallback when SAM2 is unavailable (no crashes, segmentation_available=False).
2. Toggling via enable_segmentation=True/False.
3. Mocked SAM2 predictor integration (passes bounding boxes, produces polygon/RLE masks).
4. Association of each mask with its source detection and area calculation.
5. Class filtering (segments concrete classes like building/water, skips non-segmentable).
6. Transparent overlay preview generation (data URL formatting, downsampling).
7. Mask serialization (RLE encoding, polygon approximation, bounds).
8. Empty detections handling.
"""

import pytest
from PIL import Image
from services.detection.segmentation import (
    SAM2_AVAILABLE,
    SAM2_BACKEND,
    ENABLE_SEGMENTATION,
    DEFAULT_SEGMENTABLE_CLASSES,
    CLASS_OVERLAY_COLORS,
    is_class_segmentable,
    mask_to_rle,
    mask_to_polygon,
    compute_mask_bounds,
    compute_mask_area,
    generate_overlay_preview,
    segment_detections,
)


# =====================================================================
# MOCK PREDICTOR FIXTURE
# =====================================================================
class MockSAM2Predictor:
    """Simulates SAM2 predictor returning rectangular or polygon binary masks for a given box prompt."""
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.backend = "mock_sam2"

    def is_available(self) -> bool:
        return not self.should_fail

    def predict_mask(self, image_np, box_xyxy):
        if self.should_fail:
            return None, 0.0

        # Generate a simulated 2D mask matching the prompt box
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        # Invert if inverted
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)

        # Create a simple 2D boolean grid (e.g. 500x500 or matching image)
        h = max(bottom + 10, 300)
        w = max(right + 10, 300)
        grid = [[False] * w for _ in range(h)]
        for r in range(top, bottom):
            for c in range(left, right):
                grid[r][c] = True

        score = 0.94
        return grid, score


# =====================================================================
# 1. SAM2 UNAVAILABLE FALLBACK & SAFETY
# =====================================================================
def test_sam2_unavailable_fallback_does_not_crash():
    """
    When SAM2 is unavailable (weights missing or package uninstalled),
    the segmentation service must return Grounding DINO detections normally
    with 'segmentation_available': False.
    """
    img = Image.new("RGB", (400, 400), color=(120, 150, 180))
    detections = [
        {"id": "det-1", "label": "building", "score": 0.88, "bbox": [50.0, 50.0, 150.0, 150.0]},
        {"id": "det-2", "label": "water", "score": 0.92, "bbox": [200.0, 200.0, 350.0, 350.0]},
    ]

    mock_unavail = MockSAM2Predictor(should_fail=True)
    res_dets, meta = segment_detections(
        image=img,
        detections=detections,
        enable_segmentation=True,
        predictor_override=mock_unavail,
    )

    # Detections are preserved
    assert len(res_dets) == 2
    assert res_dets[0]["id"] == "det-1"
    assert res_dets[0]["label"] == "building"
    assert res_dets[0]["mask"] is None
    assert res_dets[0]["mask_area_pixels"] == 0

    # Metadata indicates graceful fallback
    assert meta["segmentation_available"] is False
    assert meta["segmented_count"] == 0
    assert meta["overlay_preview"] is None


# =====================================================================
# 2. ENABLE_SEGMENTATION TOGGLE
# =====================================================================
def test_enable_segmentation_toggle_off():
    """When enable_segmentation=False, bypasses SAM2 even if predictor is available."""
    img = Image.new("RGB", (300, 300))
    detections = [
        {"id": "det-1", "label": "building", "score": 0.85, "bbox": [20, 20, 100, 100]}
    ]
    mock_pred = MockSAM2Predictor()

    res_dets, meta = segment_detections(
        image=img,
        detections=detections,
        enable_segmentation=False,
        predictor_override=mock_pred,
    )

    assert meta["enabled"] is False
    assert meta["segmented_count"] == 0
    assert res_dets[0]["mask"] is None


# =====================================================================
# 3. MOCKED SAM2 SEGMENTATION & BOX PROMPTING
# =====================================================================
def test_mocked_sam2_segmentation_success():
    """
    Verifies that SAM2 box prompts produce valid masks,
    associated with their source detections, with polygon, bounds, and area.
    """
    img = Image.new("RGB", (500, 500), color=(80, 120, 90))
    detections = [
        {
            "id": "det-1",
            "label": "building",
            "score": 0.84,
            "bbox": [100.0, 100.0, 180.0, 180.0],
            "box": [100.0, 100.0, 180.0, 180.0],
        },
        {
            "id": "det-2",
            "label": "water",
            "score": 0.91,
            "bbox": [200.0, 200.0, 300.0, 300.0],
            "box": [200.0, 200.0, 300.0, 300.0],
        },
    ]

    mock_pred = MockSAM2Predictor()
    res_dets, meta = segment_detections(
        image=img,
        detections=detections,
        enable_segmentation=True,
        predictor_override=mock_pred,
    )

    assert meta["segmentation_available"] is True
    assert meta["enabled"] is True
    assert meta["segmented_count"] == 2
    assert meta["total_detections"] == 2
    assert meta["backend"] == "mock_sam2"

    # Verify building detection mask
    det1 = res_dets[0]
    assert det1["id"] == "det-1"
    assert det1["label"] == "building"
    assert det1["score"] == 0.84
    assert det1["mask"] is not None
    assert "polygon" in det1["mask"]
    assert "rle" in det1["mask"]
    assert "bounds" in det1["mask"]
    assert det1["mask_area_pixels"] > 0
    # Expected area for 80x80 box = 6400 pixels
    assert det1["mask_area_pixels"] == 6400

    # Verify overlay preview is generated
    assert meta["overlay_preview"] is not None
    assert meta["overlay_preview"].startswith("data:image/png;base64,")


# =====================================================================
# 4. CLASS FILTERING (SEGMENTABLE CLASSES)
# =====================================================================
def test_class_filtering():
    """
    Avoids running SAM2 blindly on abstract/unconfigured classes.
    Concrete classes like 'building' and 'water' are segmented.
    Non-segmentable classes are skipped with mask=None.
    """
    assert is_class_segmentable("building") is True
    assert is_class_segmentable("water") is True
    assert is_class_segmentable("forest") is True
    assert is_class_segmentable("vehicle") is True
    assert is_class_segmentable("unspecified_hazard") is False

    img = Image.new("RGB", (400, 400))
    detections = [
        {"id": "det-1", "label": "building", "score": 0.90, "bbox": [50, 50, 100, 100]},
        {"id": "det-2", "label": "unspecified_hazard", "score": 0.70, "bbox": [150, 150, 200, 200]},
    ]

    mock_pred = MockSAM2Predictor()
    res_dets, meta = segment_detections(
        image=img,
        detections=detections,
        enable_segmentation=True,
        predictor_override=mock_pred,
    )

    assert meta["segmented_count"] == 1  # Only building was segmented
    assert res_dets[0]["mask"] is not None
    assert res_dets[1]["mask"] is None
    assert res_dets[1]["mask_area_pixels"] == 0


# =====================================================================
# 5. MASK ENCODING: RLE AND POLYGON EXTRACTION
# =====================================================================
def test_mask_to_rle_and_polygon():
    """Verifies pure-Python and NumPy RLE encoding and polygon approximation."""
    # 4x4 boolean mask
    # 0 0 0 0
    # 0 1 1 0
    # 0 1 1 0
    # 0 0 0 0
    mask = [
        [False, False, False, False],
        [False, True, True, False],
        [False, True, True, False],
        [False, False, False, False],
    ]

    rle = mask_to_rle(mask)
    assert rle["size"] == [4, 4]
    # In flat order: 5 zeros, 2 ones, 2 zeros, 2 ones, 5 zeros
    assert sum(rle["counts"]) == 16
    assert compute_mask_area(mask) == 4

    # Polygon extraction from box fallback
    poly = mask_to_polygon(mask, box=[1.0, 1.0, 3.0, 3.0])
    assert len(poly) >= 4
    bounds = compute_mask_bounds(poly)
    assert bounds == [1.0, 1.0, 3.0, 3.0]


# =====================================================================
# 6. TRANSPARENT OVERLAY PREVIEW GENERATION
# =====================================================================
def test_generate_overlay_preview():
    """Verifies overlay generation returns base64 data URL and respects memory dimensions."""
    img = Image.new("RGB", (600, 400), color=(40, 70, 100))
    segmented_dets = [
        {
            "label": "building",
            "mask": {
                "polygon": [[50.0, 50.0], [150.0, 50.0], [150.0, 150.0], [50.0, 150.0]],
            },
        },
        {
            "label": "water",
            "mask": {
                "polygon": [[200.0, 100.0], [300.0, 100.0], [300.0, 200.0], [200.0, 200.0]],
            },
        },
    ]

    data_url = generate_overlay_preview(img, segmented_dets, max_dimension=800)
    assert data_url is not None
    assert data_url.startswith("data:image/png;base64,")

    # Oversized image downscaling check (e.g. 3000x2000 scaled to 800)
    large_img = Image.new("RGB", (3000, 2000), color=(100, 100, 100))
    large_url = generate_overlay_preview(large_img, segmented_dets, max_dimension=800)
    assert large_url is not None
    assert large_url.startswith("data:image/png;base64,")


# =====================================================================
# 7. EMPTY DETECTIONS HANDLING
# =====================================================================
def test_empty_detections_segmentation():
    """Empty detection lists return empty results with clean metadata without raising errors."""
    img = Image.new("RGB", (200, 200))
    dets, meta = segment_detections(img, [], enable_segmentation=True)
    assert dets == []
    assert meta["segmented_count"] == 0
    assert meta["total_detections"] == 0
    assert meta["overlay_preview"] is None

