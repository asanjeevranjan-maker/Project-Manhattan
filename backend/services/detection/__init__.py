"""
Detection services module for SatQuery.
Provides satellite class vocabulary, presets, class-specific thresholds,
label normalization, geometry validation, and detection formatting.
"""

from .vocabulary import (
    SATELLITE_CLASSES,
    ANALYSIS_PRESETS,
    ABSTRACT_CONCEPTS,
    DEFAULT_CLASS_THRESHOLDS,
    ClassThreshold,
    normalize_label,
    sanitize_prompt,
    map_score_to_confidence_level,
    compute_relative_location,
    format_detection,
    filter_and_format_detections,
    remove_duplicate_detections,
    box_iou,
)

__all__ = [
    "SATELLITE_CLASSES",
    "ANALYSIS_PRESETS",
    "ABSTRACT_CONCEPTS",
    "DEFAULT_CLASS_THRESHOLDS",
    "ClassThreshold",
    "normalize_label",
    "sanitize_prompt",
    "map_score_to_confidence_level",
    "compute_relative_location",
    "format_detection",
    "filter_and_format_detections",
    "remove_duplicate_detections",
    "box_iou",
]
