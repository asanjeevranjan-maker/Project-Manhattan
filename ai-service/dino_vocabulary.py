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
except ImportError:
    # Direct relative import fallback
    from services.detection.vocabulary import *  # type: ignore
    from services.detection.tiler import *  # type: ignore

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
]

