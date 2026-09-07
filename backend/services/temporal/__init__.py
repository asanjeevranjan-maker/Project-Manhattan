"""
Temporal services package for SatQuery.
Provides bi-temporal multimodal analysis, optical + SAR fusion,
co-registration, spatial object matching, land-cover delta accounting,
standalone SAR scattering analysis, and visual overlay generation.
"""

from .fusion import (
    fuse_optical_and_sar,
    normalize_sar_backscatter,
)

from .registration import (
    register_temporal_scenes,
)

from .bitemporal_matcher import (
    match_bitemporal_detections,
    box_iou,
    box_center_distance,
    get_quadrant_location,
)

from .land_cover_change import (
    calculate_land_cover_deltas,
)

from .sar_change import (
    analyze_sar_differential_scattering,
)

from .overlay_generator import (
    generate_bitemporal_overlays,
)

from .normalization import (
    normalize_spatial_resolution,
    normalize_radiometry,
)

from .nuisance_masking import (
    compute_nuisance_masks,
)

from .change_detection import (
    detect_multisignal_changes,
    detect_structural_changes,
)

from .confidence_engine import (
    calibrate_change_confidence,
    classify_persistence,
    TEMPORARY_CLASSES,
    PERMANENT_CLASSES,
    ENVIRONMENTAL_CLASSES,
)

from .bitemporal_analyzer import (
    BiTemporalMultimodalAnalyzer,
    bitemporal_analyzer,
)

__all__ = [
    "fuse_optical_and_sar",
    "normalize_sar_backscatter",
    "register_temporal_scenes",
    "match_bitemporal_detections",
    "box_iou",
    "box_center_distance",
    "get_quadrant_location",
    "calculate_land_cover_deltas",
    "analyze_sar_differential_scattering",
    "generate_bitemporal_overlays",
    "normalize_spatial_resolution",
    "normalize_radiometry",
    "compute_nuisance_masks",
    "detect_multisignal_changes",
    "detect_structural_changes",
    "calibrate_change_confidence",
    "classify_persistence",
    "TEMPORARY_CLASSES",
    "PERMANENT_CLASSES",
    "ENVIRONMENTAL_CLASSES",
    "BiTemporalMultimodalAnalyzer",
    "bitemporal_analyzer",
]

