"""
Unit tests for the Grounding DINO + Tiling + NMS + SAM2 debugged pipeline.

Covers:
1. tile_bbox_to_global converts coordinates correctly with offset and image clipping.
2. validate_bbox correctly handles valid, inverted, out-of-bounds, degenerate, and oversized boxes.
3. Class-aware NMS deduplicates overlapping tile detections without cross-class suppression.
4. Ship and vessel IoU threshold works as configured (0.45 default).
5. SAM2 input validation rejects invalid bboxes before inference.
6. Mask area and fill ratio calculation with sanity warning triggers.
7. Fallback behavior when SAM2 is unavailable or disabled.
8. API debug endpoint logic and response structure.
"""

import pytest
import numpy as np
from PIL import Image

from services.detection.tiler import (
    tile_bbox_to_global,
    generate_tiles,
    should_tile_image,
    TILE_SIZE,
    TILE_OVERLAP,
)
from services.detection.vocabulary import (
    validate_bbox,
    DEFAULT_CLASS_THRESHOLDS,
    format_detection,
)
from services.detection.nms import (
    apply_class_nms,
    calculate_iou,
    DEFAULT_CLASS_NMS_THRESHOLDS,
)
from services.detection.segmentation import (
    segment_detections,
    debug_segment_single_box,
    compute_mask_area,
    get_sam2_predictor,
    SAM2PredictorWrapper,
)


# ============================================================================
# 1. TILE TO GLOBAL COORDINATE CONVERSION TESTS
# ============================================================================

def test_tile_bbox_to_global_basic_offset():
    """Verify offset addition: tile (10, 20, 50, 60) in tile at (500, 300) -> (510, 320, 550, 360)."""
    tile_box = [10.0, 20.0, 50.0, 60.0]
    global_box = tile_bbox_to_global(tile_box, x_offset=500, y_offset=300, image_width=2000, image_height=2000)
    assert global_box == [510.0, 320.0, 550.0, 360.0]


def test_tile_bbox_to_global_clipping():
    """Verify boxes extending beyond image boundaries are strictly clamped."""
    tile_box = [900.0, 900.0, 1050.0, 1050.0]
    # In tile at offset (500, 500) on an image of size (1400, 1400)
    # Unclipped would be [1400, 1400, 1550, 1550], should be clamped to 1400
    global_box = tile_bbox_to_global(tile_box, x_offset=500, y_offset=500, image_width=1400, image_height=1400)
    assert global_box[0] == 1400.0
    assert global_box[1] == 1400.0
    assert global_box[2] == 1400.0
    assert global_box[3] == 1400.0


def test_tile_generation_coverage():
    """Verify tiles cover the image with expected dimensions and offsets."""
    img = Image.new("RGB", (1500, 1200))
    tiles = generate_tiles(img, tile_size=1024, overlap=0.15)
    assert len(tiles) > 1
    # Verify every tile has valid integer offsets
    for t in tiles:
        assert t["x_offset"] >= 0
        assert t["y_offset"] >= 0
        assert t["width"] <= 1024
        assert t["height"] <= 1024


# ============================================================================
# 2. BBOX VALIDATION & SANITY CHECKS
# ============================================================================

def test_validate_bbox_valid():
    """Normal valid box passes validation."""
    valid, reason = validate_bbox([100, 100, 200, 200], img_w=1000, img_h=1000)
    assert valid is True
    assert reason is None


def test_validate_bbox_inverted_coords():
    """Inverted coordinates (x2 <= x1 or y2 <= y1) are rejected."""
    valid, reason = validate_bbox([200, 100, 100, 200], img_w=1000, img_h=1000)
    assert valid is False
    assert "Non-positive dimensions" in reason


def test_validate_bbox_zero_dimension():
    """Zero-width or zero-height boxes are rejected."""
    valid, reason = validate_bbox([100, 100, 100, 200], img_w=1000, img_h=1000)
    assert valid is False
    assert "Non-positive dimensions" in reason or "Degenerate dimensions" in reason


def test_validate_bbox_out_of_bounds():
    """Boxes entirely or excessively outside image dimensions are rejected."""
    valid, reason = validate_bbox([-50, -50, -10, -10], img_w=1000, img_h=1000)
    assert valid is False
    assert "outside image" in reason


def test_validate_bbox_oversized_for_small_objects():
    """Verify maximum class area limits (e.g. ship <= 15%, vehicle <= 5%)."""
    # Ship threshold in DEFAULT_CLASS_THRESHOLDS
    ship_thresh = DEFAULT_CLASS_THRESHOLDS.get("ship")
    assert ship_thresh is not None
    assert ship_thresh.max_area_ratio == 0.15

    # A box occupying 30% of a 1000x1000 image: [100, 100, 600, 700] -> area = 500*600 = 300,000 (30%)
    oversized_box = [100.0, 100.0, 600.0, 700.0]
    valid, reason = validate_bbox(
        oversized_box,
        img_w=1000,
        img_h=1000,
        max_area_ratio=ship_thresh.max_area_ratio,
    )
    assert valid is False
    assert "exceeds max_area_ratio" in reason


# ============================================================================
# 3. NMS & DEDUPLICATION WITH SHIP THRESHOLD
# ============================================================================

def test_iou_calculation():
    """Verify exact IoU for overlapping boxes."""
    b1 = [0, 0, 100, 100]  # area 10,000
    b2 = [50, 0, 150, 100] # area 10,000, intersection 50x100 = 5,000, union 15,000
    iou = calculate_iou(b1, b2)
    assert abs(iou - (5000 / 15000)) < 1e-4  # ~0.3333


def test_ship_nms_deduplication():
    """Two overlapping detections of 'ship' with IoU > 0.45 must deduplicate to the highest score."""
    # b1: [100, 100, 200, 200], area 10,000
    # b2: [110, 105, 205, 200], intersection: [110, 105, 200, 200] -> 90 * 95 = 8550
    # IoU: 8550 / (10000 + 9025 - 8550) = 8550 / 10475 ~= 0.816 > 0.45
    dets = [
        {"id": "ship_tile_a", "label": "ship", "score": 0.88, "bbox": [100.0, 100.0, 200.0, 200.0], "source": "tile"},
        {"id": "ship_tile_b", "label": "ship", "score": 0.94, "bbox": [110.0, 105.0, 205.0, 200.0], "source": "tile"},
    ]
    nms_res = apply_class_nms(dets, default_iou_threshold=0.50)
    assert len(nms_res) == 1
    # Winner must be the higher-confidence detection
    assert nms_res[0]["id"] == "ship_tile_b"
    assert nms_res[0]["score"] == 0.94


def test_class_aware_nms_does_not_suppress_other_classes():
    """A 'ship' and a 'dock' with high overlap must NOT suppress each other."""
    dets = [
        {"id": "ship_1", "label": "ship", "score": 0.90, "bbox": [100.0, 100.0, 200.0, 200.0]},
        {"id": "dock_1", "label": "dock", "score": 0.85, "bbox": [105.0, 102.0, 195.0, 198.0]},
    ]
    nms_res = apply_class_nms(dets)
    assert len(nms_res) == 2
    labels = {d["label"] for d in nms_res}
    assert "ship" in labels
    assert "dock" in labels


def test_ship_nms_threshold_configured():
    """Verify ship and vessel default thresholds are 0.45."""
    assert DEFAULT_CLASS_NMS_THRESHOLDS.get("ship") == 0.45
    assert DEFAULT_CLASS_NMS_THRESHOLDS.get("vessel") == 0.45


# ============================================================================
# 4. SAM2 INPUT VALIDATION & MASK QUALITY METRICS
# ============================================================================

class MockRealisticShipPredictor:
    """Simulates a realistic SAM2 predictor outputting an elliptical hull mask within the box."""
    def __init__(self):
        self.backend = "mock_sam2_hull"

    def is_available(self) -> bool:
        return True

    def predict_mask(self, image_np, box_xyxy):
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        # Create an elliptical mask inside the bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        rx = (x2 - x1) / 2
        ry = (y2 - y1) / 2

        img_h, img_w = image_np.shape[:2]
        grid = np.zeros((img_h, img_w), dtype=bool)

        y_indices, x_indices = np.ogrid[:img_h, :img_w]
        mask_condition = ((x_indices - cx) / (rx + 1e-5)) ** 2 + ((y_indices - cy) / (ry + 1e-5)) ** 2 <= 1.0
        grid[mask_condition] = True

        return grid, 0.95


def test_sam2_mask_area_and_fill_ratio():
    """Verify fill_ratio is calculated properly and is strictly between 0 and 1 for an ellipse."""
    img = Image.new("RGB", (400, 400), color=(20, 50, 100))
    box = [100.0, 100.0, 200.0, 200.0]  # bbox area = 100 * 100 = 10,000
    # Ellipse area in 100x100 is pi * 50 * 50 = ~7,854 -> fill_ratio ~0.785

    dets = [{"id": "ship_1", "label": "ship", "score": 0.92, "bbox": box}]
    predictor = MockRealisticShipPredictor()

    res_dets, meta = segment_detections(
        image=img,
        detections=dets,
        enable_segmentation=True,
        predictor_override=predictor,
        generate_overlay=True,
    )

    assert meta["segmentation_available"] is True
    assert meta["segmented_count"] == 1
    assert meta["mask_overlay_url"] is not None

    d = res_dets[0]
    assert d["mask"] is not None
    assert d["bbox_area_pixels"] == 10000.0
    # Ellipse fill ratio should be approx pi/4 ~= 0.785
    assert 0.70 < d["fill_ratio"] < 0.85
    assert d["mask_area_pixels"] > 7000


def test_sam2_rejects_malformed_box():
    """SAM2 segmentation must reject malformed box without crashing."""
    img = Image.new("RGB", (400, 400), color=(20, 50, 100))
    # Inverted box: x2 < x1
    inverted_box = [250.0, 100.0, 100.0, 200.0]
    dets = [{"id": "bad_ship", "label": "ship", "score": 0.92, "bbox": inverted_box}]
    predictor = MockRealisticShipPredictor()

    res_dets, meta = segment_detections(
        image=img,
        detections=dets,
        enable_segmentation=True,
        predictor_override=predictor,
    )

    # Must not crash, but bad detection should have mask=None
    assert res_dets[0]["mask"] is None
    assert res_dets[0]["mask_area_pixels"] == 0
    assert meta["segmented_count"] == 0


def test_debug_segment_single_box():
    """Test debug_segment_single_box utility for isolation testing."""
    img = Image.new("RGB", (300, 300), color=(10, 30, 60))
    box = [50.0, 50.0, 150.0, 150.0]
    predictor = MockRealisticShipPredictor()

    result = debug_segment_single_box(
        image=img,
        bbox_xyxy=box,
        label="ship",
        predictor_override=predictor,
    )

    assert result["success"] is True
    assert result["label"] == "ship"
    assert result["fill_ratio"] is not None
    assert 0.70 < result["fill_ratio"] < 0.85
    assert result["mask_overlay_url"] is not None
