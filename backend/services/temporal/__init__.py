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
    "BiTemporalMultimodalAnalyzer",
    "bitemporal_analyzer",
]
