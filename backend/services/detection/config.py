"""
Centralized detection pipeline configuration for SatQuery.

Single source of truth for all pipeline thresholds and limits so they are
NOT scattered across the codebase. Every value is overridable via environment
variable for deployment-time tuning.

Values
------
- DINO_BOX_THRESHOLD      : Grounding DINO box confidence candidate cutoff.
                            Must stay BELOW every class min_score so per-class
                            filtering (not the raw cutoff) decides the intent pool.
- DINO_TEXT_THRESHOLD     : Grounding DINO text-attention threshold.
- NMS_IOU_THRESHOLD       : Default duplicate-suppression IoU.
- DINO_IOU_THRESHOLD      : Alias of NMS_IOU_THRESHOLD (backward compat).
- SIGLIP_THRESHOLD        : SigLIP verification acceptance threshold.
- MAX_VISIBLE_DETECTIONS  : Max bounding-box overlays rendered on the frontend.
                            Does NOT cap the total verified count.
- OVERALL_CONFIDENCE_WEIGHTS: transparent formula used to derive the overall
                            confidence from measured pipeline stages.
"""

import os


def _f(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, ""))
    except (TypeError, ValueError):
        return default


def _i(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, ""))
    except (TypeError, ValueError):
        return default


# --- Grounding DINO ---------------------------------------------------------
# Broad candidate collection; the per-class min_score is the real filter.
DINO_BOX_THRESHOLD: float = _f("DINO_BOX_THRESHOLD", 0.12)
DINO_TEXT_THRESHOLD: float = _f("DINO_TEXT_THRESHOLD", 0.10)

# --- NMS --------------------------------------------------------------------
NMS_IOU_THRESHOLD: float = _f("NMS_IOU_THRESHOLD", 0.45)
DINO_IOU_THRESHOLD: float = NMS_IOU_THRESHOLD  # alias

# --- SigLIP -----------------------------------------------------------------
SIGLIP_THRESHOLD: float = _f("SIGLIP_THRESHOLD", 0.35)

# --- Giant-box sanity validation (false "full-image box" rejection) ---------
# A box covering more than MAX_BOX_AREA_RATIO of the image is suspicious.
# It is rejected unless its detector confidence is at least
# LARGE_BOX_MIN_CONFIDENCE AND it does not hug the image borders
# (more than MAX_EDGE_TOUCHES_FOR_LARGE_BOX boundary touches).
MAX_BOX_AREA_RATIO: float = _f("MAX_BOX_AREA_RATIO", 0.65)
LARGE_BOX_MIN_CONFIDENCE: float = _f("LARGE_BOX_MIN_CONFIDENCE", 0.80)
MAX_EDGE_TOUCHES_FOR_LARGE_BOX: int = _i("MAX_EDGE_TOUCHES_FOR_LARGE_BOX", 2)

# --- Frontend / overlay caps ------------------------------------------------
# Only caps how many bounding-box overlays are VISUALIZED.
# The full verified detection set is always returned to the frontend.
MAX_VISIBLE_DETECTIONS: int = _i("MAX_VISIBLE_DETECTIONS", 25)

# Confidence formula weights. Only fields that actually exist are used; if a
# stage is unavailable its weight is redistributed to the remaining stages.
OVERALL_CONFIDENCE_WEIGHTS = {
    "detection": _f("CONF_DETECTION_WEIGHT", 0.60),
    "verification": _f("CONF_VERIFICATION_WEIGHT", 0.25),
    "segmentation": _f("CONF_SEGMENTATION_WEIGHT", 0.15),
}