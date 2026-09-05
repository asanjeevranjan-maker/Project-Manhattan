"""
Grounding DINO Satellite Class Vocabulary & Preset Configuration for ai-service.
Re-exports centralized vocabulary from backend.services.detection.vocabulary
with fallback for standalone execution.
"""

import sys
from pathlib import Path

# Ensure backend is in sys.path so centralized vocabulary is imported
root_backend = Path(__file__).resolve().parent.parent / "backend"
if str(root_backend) not in sys.path:
    sys.path.insert(0, str(root_backend))

try:
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
        validate_bbox,
    )
    from services.detection.tiler import (
        TILE_SIZE,
        TILE_OVERLAP,
        ENABLE_TILING,
        MIN_IMAGE_SIZE_FOR_TILING,
        MAX_TILES,
        should_tile_image,
        calculate_tile_grid,
        iter_tiles,
        generate_tiles,
        tile_bbox_to_global,
        format_tile_metadata,
    )
    from services.detection.nms import (
        DEFAULT_NMS_IOU_THRESHOLD,
        DEFAULT_CLASS_NMS_THRESHOLDS,
        get_class_nms_threshold,
        calculate_iou,
        apply_class_nms,
        get_deduplication_stats,
    )
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
        get_sam2_predictor,
        debug_segment_single_box,
    )
    from services.detection.land_cover import (
        LandCoverAnalyzer,
        calculate_land_cover,
        DEFAULT_CATEGORY_PRIORITY,
        LABEL_TO_CATEGORY,
        LAND_COVER_COLORS,
        map_label_to_category,
    )
    from services.detection.verifier import (
        VERIFIER_AVAILABLE,
        ENABLE_VERIFICATION,
        VERIFICATION_THRESHOLD,
        DEFAULT_VERIFICATION_CLASSES,
        CLASS_NEGATIVE_ALTERNATIVES,
        crop_detection_box,
        get_alternatives_for_class,
        verify_detection,
        verify_detections,
        get_verifier,
    )
    from services.vision.context_builder import build_vision_context
except ImportError:
    # Direct relative import fallback
    from services.detection.vocabulary import *  # type: ignore
    from services.detection.tiler import *  # type: ignore
    from services.detection.nms import *  # type: ignore
    from services.detection.segmentation import *  # type: ignore
    from services.detection.land_cover import *  # type: ignore
    from services.detection.verifier import *  # type: ignore
    from services.vision.context_builder import *  # type: ignore

__all__ = [
    "SATELLITE_CLASSES",
    "ANALYSIS_PRESETS",
    "ABSTRACT_CONCEPTS",
    "DEFAULT_CLASS_THRESHOLDS",
    "ClassThreshold",
    "get_class_threshold",
    "normalize_label",
    "sanitize_prompt",
    "map_score_to_confidence_level",
    "compute_relative_location",
    "format_detection",
    "filter_and_format_detections",
    "remove_duplicate_detections",
    "box_iou",
    "validate_bbox",
    "TILE_SIZE",
    "TILE_OVERLAP",
    "ENABLE_TILING",
    "MIN_IMAGE_SIZE_FOR_TILING",
    "MAX_TILES",
    "should_tile_image",
    "calculate_tile_grid",
    "iter_tiles",
    "generate_tiles",
    "tile_bbox_to_global",
    "format_tile_metadata",
    "DEFAULT_NMS_IOU_THRESHOLD",
    "DEFAULT_CLASS_NMS_THRESHOLDS",
    "get_class_nms_threshold",
    "calculate_iou",
    "apply_class_nms",
    "get_deduplication_stats",
    "SAM2_AVAILABLE",
    "SAM2_BACKEND",
    "ENABLE_SEGMENTATION",
    "DEFAULT_SEGMENTABLE_CLASSES",
    "CLASS_OVERLAY_COLORS",
    "is_class_segmentable",
    "mask_to_rle",
    "mask_to_polygon",
    "compute_mask_bounds",
    "compute_mask_area",
    "generate_overlay_preview",
    "segment_detections",
    "get_sam2_predictor",
    "debug_segment_single_box",
    "LandCoverAnalyzer",
    "calculate_land_cover",
    "DEFAULT_CATEGORY_PRIORITY",
    "LABEL_TO_CATEGORY",
    "LAND_COVER_COLORS",
    "map_label_to_category",
    "build_vision_context",
    "VERIFIER_AVAILABLE",
    "ENABLE_VERIFICATION",
    "VERIFICATION_THRESHOLD",
    "DEFAULT_VERIFICATION_CLASSES",
    "CLASS_NEGATIVE_ALTERNATIVES",
    "crop_detection_box",
    "get_alternatives_for_class",
    "verify_detection",
    "verify_detections",
    "get_verifier",
]


