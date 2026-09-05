"""
Unit tests for Global Class-Aware NMS and Detection Deduplication.
Covers:
1. Exact duplicate removal (same class, identical coords, score prioritization)
2. Partial overlap deduplication (user example: [100,100,180,180] vs [105,102,181,179])
3. Different classes isolation ('building' and 'vehicle' never suppress each other)
4. No overlap preservation (distant objects of the same class kept)
5. Nested boxes handling (high-IoU nested duplicates vs low-IoU scale variations)
6. Empty detections handling (safe return with correct debug schema)
7. Centralized class-specific NMS thresholds (building=0.45, vehicle=0.40, water=0.55)
8. Weighted box merging mode vs standard NMS mode
9. Debug metrics verification (raw_detection_count, final_detection_count, duplicates_removed)
10. Non-tiled direct detection compatibility
"""

import pytest
from services.detection.nms import (
    calculate_iou,
    box_iou,
    apply_class_nms,
    get_class_nms_threshold,
    get_deduplication_stats,
    DEFAULT_CLASS_NMS_THRESHOLDS,
    DEFAULT_NMS_IOU_THRESHOLD,
)
from services.detection.vocabulary import filter_and_format_detections, remove_duplicate_detections


# =====================================================================
# 1. IOU CALCULATION
# =====================================================================
def test_calculate_iou_exact_and_partial():
    """Verifies IoU calculation for identical, partial, and non-overlapping boxes."""
    box_a = [100.0, 100.0, 200.0, 200.0]
    box_b = [100.0, 100.0, 200.0, 200.0]
    # Identical boxes have IoU = 1.0
    assert calculate_iou(box_a, box_b) == 1.0
    assert box_iou(box_a, box_b) == 1.0

    # User example: [100, 100, 180, 180] and [105, 102, 181, 179]
    # Box 1: 80x80 = 6400 area
    # Box 2: 76x77 = 5852 area
    # Intersection: [105, 102, 180, 179] -> 75x77 = 5775 area
    # Union: 6400 + 5852 - 5775 = 6477
    # IoU = 5775 / 6477 ~= 0.8916
    user_box1 = [100.0, 100.0, 180.0, 180.0]
    user_box2 = [105.0, 102.0, 181.0, 179.0]
    iou = calculate_iou(user_box1, user_box2)
    assert 0.88 <= iou <= 0.90

    # Non-overlapping boxes
    distant_box = [300.0, 300.0, 400.0, 400.0]
    assert calculate_iou(box_a, distant_box) == 0.0

    # Degenerate boxes
    zero_box = [100.0, 100.0, 100.0, 100.0]
    assert calculate_iou(box_a, zero_box) == 0.0
    assert calculate_iou([], box_a) == 0.0


# =====================================================================
# 2. EXACT DUPLICATES
# =====================================================================
def test_nms_exact_duplicate():
    """Verifies that duplicate detections with identical boxes are deduplicated, keeping highest score."""
    detections = [
        {"label": "building", "score": 0.65, "box": [100.0, 100.0, 200.0, 200.0]},
        {"label": "building", "score": 0.92, "box": [100.0, 100.0, 200.0, 200.0]},
    ]
    kept = apply_class_nms(detections)
    assert len(kept) == 1
    assert kept[0]["score"] == 0.92
    assert kept[0]["box"] == [100.0, 100.0, 200.0, 200.0]


# =====================================================================
# 3. PARTIAL OVERLAP (TILE SEAM DEDUPLICATION)
# =====================================================================
def test_nms_partial_overlap():
    """
    Simulates user scenario:
    Tile A: building [100, 100, 180, 180] (score 0.87)
    Tile B: same building [105, 102, 181, 179] (score 0.81)
    Expected: only one building kept, preferring highest score (0.87).
    """
    tile_a = {"label": "building", "score": 0.87, "box": [100.0, 100.0, 180.0, 180.0]}
    tile_b = {"label": "building", "score": 0.81, "box": [105.0, 102.0, 181.0, 179.0]}

    kept = apply_class_nms([tile_a, tile_b])
    assert len(kept) == 1
    assert kept[0]["score"] == 0.87
    assert kept[0]["box"] == [100.0, 100.0, 180.0, 180.0]


# =====================================================================
# 4. DIFFERENT CLASSES (NO CROSS-CLASS SUPPRESSION)
# =====================================================================
def test_nms_different_classes():
    """
    Crucial requirement: 'building' and 'vehicle' must NEVER suppress one another,
    even when bounding boxes heavily overlap (e.g. car parked right next to building).
    """
    building = {"label": "building", "score": 0.95, "box": [100.0, 100.0, 180.0, 180.0]}
    vehicle = {"label": "vehicle", "score": 0.82, "box": [105.0, 102.0, 181.0, 179.0]}

    kept = apply_class_nms([building, vehicle])
    assert len(kept) == 2
    labels = {d["label"] for d in kept}
    assert "building" in labels
    assert "vehicle" in labels


# =====================================================================
# 5. NO OVERLAP
# =====================================================================
def test_nms_no_overlap():
    """Detections of the same class at distinct locations are all preserved."""
    b1 = {"label": "building", "score": 0.85, "box": [100.0, 100.0, 150.0, 150.0]}
    b2 = {"label": "building", "score": 0.88, "box": [300.0, 300.0, 360.0, 360.0]}
    b3 = {"label": "building", "score": 0.79, "box": [600.0, 100.0, 650.0, 150.0]}

    kept = apply_class_nms([b1, b2, b3])
    assert len(kept) == 3


# =====================================================================
# 6. NESTED BOXES
# =====================================================================
def test_nms_nested_boxes():
    """
    Case A: High-overlap nested duplicate (e.g. tile crop jitter [102, 102, 198, 198] inside [100, 100, 200, 200])
            IoU > 0.45 threshold -> suppressed, highest score kept.
    Case B: Low-overlap multi-scale feature (e.g. small feature [100, 100, 120, 120] inside large zone [100, 100, 500, 500])
            IoU < threshold -> both kept under standard IoU NMS.
    """
    # High-overlap nested duplicate
    big_box = {"label": "building", "score": 0.91, "box": [100.0, 100.0, 200.0, 200.0]}
    jittered_box = {"label": "building", "score": 0.75, "box": [102.0, 102.0, 198.0, 198.0]}
    kept_a = apply_class_nms([big_box, jittered_box])
    assert len(kept_a) == 1
    assert kept_a[0]["score"] == 0.91

    # Low-overlap distinct scale
    large_structure = {"label": "building", "score": 0.88, "box": [100.0, 100.0, 500.0, 500.0]}
    small_annex = {"label": "building", "score": 0.82, "box": [100.0, 100.0, 130.0, 130.0]}
    kept_b = apply_class_nms([large_structure, small_annex])
    assert len(kept_b) == 2


# =====================================================================
# 7. EMPTY DETECTIONS
# =====================================================================
def test_nms_empty_detections():
    """Empty list inputs return empty lists and correct debug metrics without crashing."""
    assert apply_class_nms([]) == []
    empty_res, stats = apply_class_nms([], return_debug_info=True)
    assert empty_res == []
    assert stats["raw_detection_count"] == 0
    assert stats["final_detection_count"] == 0
    assert stats["duplicates_removed"] == 0


# =====================================================================
# 8. CENTRALIZED CLASS-SPECIFIC NMS THRESHOLDS
# =====================================================================
def test_centralized_class_thresholds():
    """Verifies class-specific thresholds for building (0.45), vehicle (0.40), water (0.55)."""
    assert get_class_nms_threshold("building") == 0.45
    assert get_class_nms_threshold("vehicle") == 0.40
    assert get_class_nms_threshold("water") == 0.55
    assert get_class_nms_threshold("water body") == 0.55
    assert get_class_nms_threshold("unspecified_class") == DEFAULT_NMS_IOU_THRESHOLD

    # Custom threshold dictionary override
    custom = {"building": 0.30}
    assert get_class_nms_threshold("building", custom_thresholds=custom) == 0.30

    # Explicit fallback threshold
    assert get_class_nms_threshold("random_item", fallback_threshold=0.60) == 0.60


# =====================================================================
# 9. WEIGHTED BOX MERGING VS STANDARD NMS
# =====================================================================
def test_weighted_box_merging_mode():
    """
    Tests score-weighted coordinate fusion when merge_mode='weighted'.
    Weights: w_i = score_i
    """
    det_high = {"label": "building", "score": 0.80, "box": [100.0, 100.0, 200.0, 200.0]}
    det_low = {"label": "building", "score": 0.20, "box": [110.0, 110.0, 210.0, 210.0]}

    # Standard NMS preserves exact winner coordinates
    std_kept = apply_class_nms([det_high, det_low], merge_mode="standard")
    assert len(std_kept) == 1
    assert std_kept[0]["box"] == [100.0, 100.0, 200.0, 200.0]

    # Weighted NMS calculates weighted average coordinates:
    # x1 = (0.80*100 + 0.20*110)/1.0 = 102.0
    # y1 = (0.80*100 + 0.20*110)/1.0 = 102.0
    # x2 = (0.80*200 + 0.20*210)/1.0 = 202.0
    # y2 = (0.80*200 + 0.20*210)/1.0 = 202.0
    weighted_kept = apply_class_nms([det_high, det_low], merge_mode="weighted")
    assert len(weighted_kept) == 1
    assert weighted_kept[0]["box"] == [102.0, 102.0, 202.0, 202.0]
    assert weighted_kept[0]["score"] == 0.80


# =====================================================================
# 10. DEBUG INFORMATION METRICS
# =====================================================================
def test_debug_info_metrics():
    """Verifies that debug information accurately reports raw, final, and duplicates_removed counts."""
    # 3 overlapping buildings (2 duplicates), 2 vehicles (1 duplicate, 1 separate)
    dets = [
        {"label": "building", "score": 0.90, "box": [100, 100, 200, 200]},
        {"label": "building", "score": 0.85, "box": [105, 105, 205, 205]},
        {"label": "building", "score": 0.70, "box": [102, 102, 198, 198]},
        {"label": "vehicle", "score": 0.88, "box": [50, 50, 80, 80]},
        {"label": "vehicle", "score": 0.75, "box": [52, 51, 81, 79]},
    ]
    kept, debug = apply_class_nms(dets, return_debug_info=True)

    assert len(kept) == 2  # 1 building, 1 vehicle
    assert debug["raw_detection_count"] == 5
    assert debug["final_detection_count"] == 2
    assert debug["duplicates_removed"] == 3


# =====================================================================
# 11. DIRECT NON-TILED DETECTION COMPATIBILITY
# =====================================================================
def test_direct_non_tiled_detection_integration():
    """Verifies filter_and_format_detections functions on single non-tiled image detections."""
    raw = [
        {"label": "building", "score": 0.92, "box": [50.0, 50.0, 120.0, 120.0]},
        {"label": "building", "score": 0.78, "box": [52.0, 51.0, 121.0, 119.0]},  # duplicate
        {"label": "vehicle", "score": 0.84, "box": [200.0, 200.0, 230.0, 220.0]},
    ]
    final, stats = filter_and_format_detections(
        raw_detections=raw,
        width=800,
        height=600,
        return_dedup_info=True,
    )

    assert len(final) == 2
    assert stats["raw_detection_count"] == 3
    assert stats["final_detection_count"] == 2
    assert stats["duplicates_removed"] == 1
    assert final[0]["id"] == "det-1"
    assert final[1]["id"] == "det-2"
