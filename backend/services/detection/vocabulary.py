"""
Satellite Class Vocabulary, Presets, and Detection Post-Processing.
Enforces detection of physically observable features instead of abstract concepts.
Provides class-specific thresholding, label normalization, and clean output formatting.
"""

import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set

logger = logging.getLogger("satquery.detection.vocabulary")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")


# =====================================================================
# 1. REUSABLE SATELLITE CLASS VOCABULARY
# =====================================================================
# Maps canonical observable classes to their concrete synonyms and subclasses.
# Unrelated classes are strictly NOT merged:
# - river (flowing linear) is distinct from water body (standing lake/reservoir/pond)
# - road is distinct from railway
# - vehicle, ship, aircraft are distinct
# - vegetation is distinct from forest and field
# - building is distinct from tower and construction area
SATELLITE_CLASSES: Dict[str, List[str]] = {
    "building": [
        "building", "house", "structure", "roof", "rooftop",
        "warehouse", "factory", "residential building", "commercial building",
        "hangar", "shed", "facility"
    ],
    "vehicle": [
        "vehicle", "car", "truck", "bus", "van", "automobile", "lorry"
    ],
    "road": [
        "road", "highway", "street", "paved road", "roadway", "pathway", "path",
        "freeway", "avenue", "lane"
    ],
    "bridge": [
        "bridge", "overpass", "viaduct", "flyover", "trestle", "causeway"
    ],
    "river": [
        "river", "stream", "creek", "canal", "waterway", "drainage channel"
    ],
    "water body": [
        "water body", "lake", "pond", "reservoir", "water basin", "lagoon", "bay"
    ],
    "vegetation": [
        "vegetation", "greenery", "trees", "tree canopy", "shrub", "scrub",
        "vegetated area"
    ],
    "forest": [
        "forest", "woods", "woodland", "dense forest", "timberland", "rainforest"
    ],
    "field": [
        "field", "cropland", "farmland", "agricultural field", "pasture",
        "paddy field", "meadow", "cultivated plot"
    ],
    "bare soil": [
        "bare soil", "dirt", "ground", "cleared land", "excavation",
        "bare land", "barren ground", "sand"
    ],
    "construction area": [
        "construction area", "construction site", "construction",
        "earthworks", "building site", "work zone"
    ],
    "ship": [
        "ship", "vessel", "boat", "cargo ship", "tanker", "freighter",
        "ferry", "container ship", "barge", "watercraft"
    ],
    "aircraft": [
        "aircraft", "airplane", "plane", "jet", "airliner", "cargo plane",
        "helicopter"
    ],
    "railway": [
        "railway", "railroad", "train tracks", "rail line", "tracks",
        "railway track"
    ],
    "tower": [
        "tower", "transmission tower", "water tower", "antenna tower",
        "pylon", "mast", "radio tower", "cooling tower"
    ],
}

# Reverse lookup: maps every alias (lowercased) to its canonical class name
ALIAS_TO_CANONICAL: Dict[str, str] = {}
for canonical, aliases in SATELLITE_CLASSES.items():
    ALIAS_TO_CANONICAL[canonical.lower()] = canonical
    for alias in aliases:
        ALIAS_TO_CANONICAL[alias.lower()] = canonical


# =====================================================================
# 2. ANALYSIS-SPECIFIC CLASS PRESETS
# =====================================================================
ANALYSIS_PRESETS: Dict[str, List[str]] = {
    "general": [
        "building", "vehicle", "road", "river", "bridge", "vegetation"
    ],
    "urban": [
        "building", "road", "vehicle", "bridge", "construction area"
    ],
    "water": [
        "river", "water body", "bridge", "ship"
    ],
    "agriculture": [
        "field", "vegetation", "building", "water body", "road"
    ],
    "infrastructure": [
        "road", "bridge", "building", "railway", "tower", "vehicle"
    ],
    "disaster": [
        "building", "road", "bridge", "river", "water body", "bare soil"
    ],
    "maritime": [
        "ship", "water body", "bridge", "building"
    ],
}


# =====================================================================
# 3. ABSTRACT CONCEPTS PREVENTION & TRANSLATION
# =====================================================================
# Grounding DINO should NEVER be asked to detect abstract/subjective interpretations.
# These patterns detect abstract queries and translate them into physical features.
ABSTRACT_CONCEPT_MAP: Dict[str, List[str]] = {
    "flood danger area": ["river", "water body", "bridge", "road", "building"],
    "flood danger": ["river", "water body", "bridge", "road", "building"],
    "flood": ["river", "water body", "bridge", "road", "building"],
    "flooding": ["river", "water body", "bridge", "road", "building"],
    "inundation": ["river", "water body", "bridge", "road", "building"],
    "vulnerable settlement": ["building", "road", "river", "bridge"],
    "low-lying floodplain": ["river", "water body", "field", "bare soil"],
    "floodplain": ["river", "water body", "field", "bare soil"],
    "high-risk zone": ["building", "road", "bridge", "river"],
    "risk zone": ["building", "road", "bridge", "river"],
    "danger zone": ["building", "road", "bridge", "river"],
    "damage": ["building", "road", "bridge", "bare soil"],
    "destruction": ["building", "road", "bridge", "bare soil"],
    "evacuation zone": ["road", "bridge", "building", "vehicle"],
}

ABSTRACT_CONCEPTS: Set[str] = set(ABSTRACT_CONCEPT_MAP.keys())

# Stopwords to discard when validating decoded text labels from Grounding DINO
STOPWORDS: Set[str] = {
    "a", "an", "the", "all", "of", "in", "at", "to", "and", "or", "is",
    "are", "on", "by", "for", "with", "from", "as", "it", "this", "that",
    "some", "any", "no", "not", "each", "every", "both", "either",
}


# =====================================================================
# 4. CLASS-SPECIFIC CONFIGURABLE THRESHOLDS
# =====================================================================
@dataclass
class ClassThreshold:
    box_threshold: float
    text_threshold: float
    min_score: float
    max_area_ratio: float = 0.15
    min_area_pixels: int = 64  # min 8x8 px
    min_aspect: float = 0.05
    max_aspect: float = 20.0


DEFAULT_CLASS_THRESHOLDS: Dict[str, ClassThreshold] = {
    # Dense rooftop structures
    "building": ClassThreshold(
        box_threshold=0.32,
        text_threshold=0.25,
        min_score=0.35,
        max_area_ratio=0.15,
        min_aspect=0.20,
        max_aspect=5.0,
    ),
    # Compact mobile objects (small footprint)
    "vehicle": ClassThreshold(
        box_threshold=0.30,
        text_threshold=0.25,
        min_score=0.32,
        max_area_ratio=0.025,
        min_aspect=0.25,
        max_aspect=4.0,
    ),
    # Elongated transport networks
    "road": ClassThreshold(
        box_threshold=0.28,
        text_threshold=0.24,
        min_score=0.30,
        max_area_ratio=0.12,
        min_aspect=0.05,
        max_aspect=20.0,
    ),
    "railway": ClassThreshold(
        box_threshold=0.28,
        text_threshold=0.24,
        min_score=0.30,
        max_area_ratio=0.12,
        min_aspect=0.05,
        max_aspect=20.0,
    ),
    "bridge": ClassThreshold(
        box_threshold=0.32,
        text_threshold=0.25,
        min_score=0.35,
        max_area_ratio=0.12,
        min_aspect=0.10,
        max_aspect=15.0,
    ),
    # Hydrology & water
    "river": ClassThreshold(
        box_threshold=0.26,
        text_threshold=0.22,
        min_score=0.28,
        max_area_ratio=0.20,
        min_aspect=0.05,
        max_aspect=20.0,
    ),
    "water body": ClassThreshold(
        box_threshold=0.26,
        text_threshold=0.22,
        min_score=0.28,
        max_area_ratio=0.30,
        min_aspect=0.10,
        max_aspect=10.0,
    ),
    # Landcover & environmental
    "vegetation": ClassThreshold(
        box_threshold=0.30,
        text_threshold=0.25,
        min_score=0.32,
        max_area_ratio=0.25,
    ),
    "forest": ClassThreshold(
        box_threshold=0.30,
        text_threshold=0.25,
        min_score=0.32,
        max_area_ratio=0.30,
    ),
    "field": ClassThreshold(
        box_threshold=0.28,
        text_threshold=0.24,
        min_score=0.30,
        max_area_ratio=0.30,
    ),
    "bare soil": ClassThreshold(
        box_threshold=0.28,
        text_threshold=0.24,
        min_score=0.30,
        max_area_ratio=0.25,
    ),
    "construction area": ClassThreshold(
        box_threshold=0.30,
        text_threshold=0.25,
        min_score=0.32,
        max_area_ratio=0.20,
    ),
    # Maritime & aviation
    "ship": ClassThreshold(
        box_threshold=0.32,
        text_threshold=0.25,
        min_score=0.35,
        max_area_ratio=0.12,
        min_aspect=0.15,
        max_aspect=8.0,
    ),
    "aircraft": ClassThreshold(
        box_threshold=0.32,
        text_threshold=0.25,
        min_score=0.35,
        max_area_ratio=0.06,
        min_aspect=0.20,
        max_aspect=5.0,
    ),
    # Tall structures
    "tower": ClassThreshold(
        box_threshold=0.30,
        text_threshold=0.25,
        min_score=0.35,
        max_area_ratio=0.05,
    ),
    # Default fallback
    "default": ClassThreshold(
        box_threshold=0.30,
        text_threshold=0.25,
        min_score=0.32,
        max_area_ratio=0.15,
    ),
}


def get_class_threshold(canonical_label: str) -> ClassThreshold:
    """Returns the configured threshold for a canonical class or default."""
    return DEFAULT_CLASS_THRESHOLDS.get(canonical_label, DEFAULT_CLASS_THRESHOLDS["default"])


# =====================================================================
# 5. LABEL NORMALIZATION & JUNK SUPPRESSION
# =====================================================================
def normalize_label(raw_label: Any) -> Tuple[Optional[str], str]:
    """
    Normalizes Grounding DINO text label into a canonical class while preserving raw_label.
    Fixes truncated strings like 'A', 'L', 'ALL' and stopwords.

    Returns:
        (canonical_label, cleaned_raw_label)
        If the label is junk, returns (None, cleaned_raw_label).
    """
    if raw_label is None:
        return None, ""

    # Clean punctuation, whitespace, special tokens
    cleaned = str(raw_label).strip().lower()
    cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Rule 1: Reject single characters (e.g. 'a', 'l')
    if len(cleaned) <= 1:
        return None, cleaned

    # Rule 2: Reject stopwords (e.g. 'all', 'the', 'and')
    if cleaned in STOPWORDS:
        return None, cleaned

    # Rule 3: Direct exact alias match
    if cleaned in ALIAS_TO_CANONICAL:
        return ALIAS_TO_CANONICAL[cleaned], cleaned

    # Rule 4: Substring match against known aliases
    # Checks if any known alias is contained as a word boundary
    for alias, canonical in ALIAS_TO_CANONICAL.items():
        if re.search(rf"\b{re.escape(alias)}\b", cleaned):
            return canonical, cleaned

    # Rule 5: If not in satellite vocabulary, keep cleaned string if >= 3 characters and not a stopword
    if len(cleaned) >= 3 and cleaned.isalnum():
        return cleaned, cleaned

    return None, cleaned


# =====================================================================
# 6. PROMPT SANITIZER & BUILDER
# =====================================================================
def sanitize_prompt(prompt: Optional[str] = None, preset: Optional[str] = None) -> str:
    """
    Constructs a short, concrete Grounding DINO prompt separated by ' . '.
    - Prevents abstract concepts ('flood danger area', 'vulnerable settlement').
    - Translates abstract queries or presets into physically observable features.
    - Ensures each phrase is properly delimited with periods.

    Example output:
        'building . vehicle . road . river . bridge .'
    """
    # 1. Preset override
    if preset and preset.lower() in ANALYSIS_PRESETS:
        classes = ANALYSIS_PRESETS[preset.lower()]
        return " . ".join(classes) + " ."

    # 2. Check if prompt itself is a preset name
    cleaned_input = (prompt or "").strip().lower()
    if cleaned_input in ANALYSIS_PRESETS:
        classes = ANALYSIS_PRESETS[cleaned_input]
        return " . ".join(classes) + " ."

    if not cleaned_input:
        return " . ".join(ANALYSIS_PRESETS["general"]) + " ."

    # 3. Check for abstract concepts
    for abstract_term, physical_replacements in ABSTRACT_CONCEPT_MAP.items():
        if abstract_term in cleaned_input:
            logger.info(
                f"[DINO Vocabulary] Abstract query detected: '{abstract_term}'. "
                f"Translating to physical observables: {physical_replacements}"
            )
            return " . ".join(physical_replacements) + " ."

    # 4. Extract individual terms from user input
    # Split by period, comma, or 'and'
    raw_tokens = re.split(r"[,.;]+|\band\b", cleaned_input)
    selected_classes: List[str] = []
    seen_classes: Set[str] = set()

    for token in raw_tokens:
        tok = token.strip().lower()
        if not tok or tok in STOPWORDS:
            continue

        canonical, _ = normalize_label(tok)
        if canonical and canonical not in seen_classes:
            seen_classes.add(canonical)
            selected_classes.append(canonical)

    # Fallback to general preset if no observable classes were parsed
    if not selected_classes:
        logger.info(
            f"[DINO Vocabulary] Prompt '{prompt}' contained no recognizable observables. "
            f"Defaulting to general preset."
        )
        return " . ".join(ANALYSIS_PRESETS["general"]) + " ."

    return " . ".join(selected_classes) + " ."


# =====================================================================
# 7. CONFIDENCE LEVEL & LOCATION MAPPING
# =====================================================================
def map_score_to_confidence_level(score: float) -> str:
    """
    Maps continuous model confidence score to qualitative level.
    NOTE: Grounding DINO scores are logits-derived matching scores, not calibrated probabilities.
    """
    if score >= 0.65:
        return "high"
    elif score >= 0.40:
        return "medium"
    return "low"


def compute_relative_location(box: List[float], width: int, height: int) -> str:
    """
    Determines qualitative spatial quadrant of a detection based on its center point.
    Returns: 'upper-left' | 'upper-right' | 'lower-left' | 'lower-right' | 'center'
    """
    if width <= 0 or height <= 0:
        return "center"

    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0

    nx = cx / float(width)
    ny = cy / float(height)

    # Center zone: 35% to 65% in both dimensions
    if 0.35 <= nx <= 0.65 and 0.35 <= ny <= 0.65:
        return "center"

    vert = "upper" if ny < 0.5 else "lower"
    horiz = "left" if nx < 0.5 else "right"
    return f"{vert}-{horiz}"


def box_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection over Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    intersection = iw * ih

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - intersection
    if union <= 0.0:
        return 0.0

    return intersection / union


# =====================================================================
# 8. DETECTION FORMATTING & FILTERING
# =====================================================================
def format_detection(
    det_id: str,
    raw_label: Any,
    score: float,
    box: List[float],
    width: int,
    height: int,
    custom_thresholds: Optional[Dict[str, ClassThreshold]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Validates, normalizes, and formats a single Grounding DINO detection.
    Enforces class-specific score thresholds, geometry checks, and produces
    the unified output format with both modern and legacy backward-compatible fields.

    Returns None if detection violates thresholds or geometry.
    """
    # 1. Normalize label and reject truncated / stopword junk
    canonical_label, cleaned_raw = normalize_label(raw_label)
    if not canonical_label:
        return None

    # 2. Geometry bounds check
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    # 3. Class-specific threshold check
    thresholds_map = custom_thresholds or DEFAULT_CLASS_THRESHOLDS
    thresh = thresholds_map.get(canonical_label, thresholds_map.get("default", DEFAULT_CLASS_THRESHOLDS["default"]))

    if score < thresh.min_score:
        return None

    # 4. Aspect ratio and area ratio checks
    area = bw * bh
    image_area = max(1, width * height)
    area_ratio = area / float(image_area)

    if area_ratio > thresh.max_area_ratio:
        return None

    aspect = bw / float(bh) if bh > 0 else 0.0
    if aspect < thresh.min_aspect or aspect > thresh.max_aspect:
        return None

    # 5. Spatial location & center point
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    rel_loc = compute_relative_location(box, width, height)

    rounded_box = [round(float(v), 1) for v in box]

    return {
        "id": det_id,
        "label": canonical_label,
        "raw_label": cleaned_raw,
        "score": round(float(score), 3),
        "confidence": round(float(score), 3),  # Backward compatibility alias
        "confidence_level": map_score_to_confidence_level(score),
        "bbox": rounded_box,
        "box": rounded_box,  # Backward compatibility alias
        "center": [round(cx, 1), round(cy, 1)],
        "relative_location": rel_loc,
    }


def remove_duplicate_detections(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Removes duplicate overlapping bounding boxes of the same canonical class using greedy NMS.
    """
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d.get("score", d.get("confidence", 0.0)), reverse=True)
    kept: List[Dict[str, Any]] = []

    for det in sorted_dets:
        box = det.get("bbox") or det.get("box")
        label = det.get("label", "").lower()
        if not box:
            continue

        duplicate = False
        for existing in kept:
            existing_box = existing.get("bbox") or existing.get("box")
            existing_label = existing.get("label", "").lower()

            if label != existing_label:
                continue

            if existing_box and box_iou(box, existing_box) > iou_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(det)

    return kept


def filter_and_format_detections(
    raw_detections: List[Dict[str, Any]],
    width: int,
    height: int,
    iou_threshold: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Processes a list of raw detections from Grounding DINO:
    1. Normalizes and validates each detection with class-specific thresholds
    2. Drops truncated or invalid labels
    3. Performs NMS duplicate removal
    4. Re-assigns clean sequential IDs ('det-1', 'det-2', etc.)
    """
    formatted: List[Dict[str, Any]] = []

    for idx, raw in enumerate(raw_detections):
        box = raw.get("box") or raw.get("bbox")
        if not box or len(box) != 4:
            continue

        score = float(raw.get("score") or raw.get("confidence") or 0.0)
        raw_label = raw.get("label") or raw.get("raw_label") or ""

        det = format_detection(
            det_id=f"raw-{idx + 1}",
            raw_label=raw_label,
            score=score,
            box=box,
            width=width,
            height=height,
        )
        if det:
            formatted.append(det)

    # Deduplicate via NMS
    deduped = remove_duplicate_detections(formatted, iou_threshold=iou_threshold)

    # Re-index with clean IDs
    for idx, d in enumerate(deduped):
        d["id"] = f"det-{idx + 1}"

    return deduped

