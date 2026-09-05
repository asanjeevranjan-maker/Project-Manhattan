"""
Global Class-Aware NMS & Detection Deduplication Module for Grounding DINO.

Handles:
1. Reusable IoU calculation between bounding boxes [x1, y1, x2, y2]
2. Class-aware grouping (unrelated classes like 'building' and 'vehicle' never suppress each other)
3. Centralized class-specific NMS thresholds (e.g., building=0.45, vehicle=0.40, water=0.55)
4. Score-prioritized greedy selection (highest confidence detection preserved)
5. Optional weighted box merging (coordinates smoothed by confidence scores)
6. Deduplication debug metrics (raw_detection_count, final_detection_count, duplicates_removed)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict

logger = logging.getLogger("satquery.detection.nms")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")


# =====================================================================
# 1. CENTRALIZED CLASS-SPECIFIC NMS THRESHOLDS
# =====================================================================
# Default fallback IoU threshold
DEFAULT_NMS_IOU_THRESHOLD: float = 0.45

# Class-specific IoU thresholds for satellite feature characteristics
DEFAULT_CLASS_NMS_THRESHOLDS: Dict[str, float] = {
    # Dense rigid structures: 0.45 allows adjacent buildings without cross-suppression
    "building": 0.45,
    "house": 0.45,
    "structure": 0.45,
    "facility": 0.45,
    "industrial facility": 0.45,
    # Small compact mobile objects: 0.40 tightly separates packed vehicles/boats
    "vehicle": 0.40,
    "car": 0.40,
    "truck": 0.40,
    "boat": 0.40,
    "vessel": 0.40,
    "airplane": 0.40,
    "aircraft": 0.40,
    # Elongated infrastructure: 0.35 handles connected road segments
    "road": 0.35,
    "highway": 0.35,
    "bridge": 0.35,
    "runway": 0.35,
    # Large amorphous natural features: 0.55 tolerates broader tile seam overlap
    "water": 0.55,
    "water body": 0.55,
    "river": 0.55,
    "lake": 0.55,
    "ocean": 0.55,
    "reservoir": 0.55,
    "flooded area": 0.55,
    "vegetation": 0.50,
    "tree": 0.50,
    "forest": 0.50,
    "field": 0.50,
    "agricultural field": 0.50,
    "bare land": 0.50,
    # Default fallback
    "default": DEFAULT_NMS_IOU_THRESHOLD,
}


def get_class_nms_threshold(
    canonical_label: str,
    custom_thresholds: Optional[Dict[str, float]] = None,
    fallback_threshold: Optional[float] = None,
) -> float:
    """
    Retrieves the NMS IoU threshold for a given class.
    Checks custom thresholds first, then centralized defaults, and falls back to default.
    """
    label_lower = canonical_label.strip().lower()
    thresholds = custom_thresholds or DEFAULT_CLASS_NMS_THRESHOLDS

    if label_lower in thresholds:
        return thresholds[label_lower]

    # Check partial / singular match
    for k, v in thresholds.items():
        if k in label_lower or label_lower in k:
            return v

    if fallback_threshold is not None:
        return fallback_threshold

    return thresholds.get("default", DEFAULT_NMS_IOU_THRESHOLD)


# =====================================================================
# 2. REUSABLE IOU CALCULATION
# =====================================================================
def calculate_iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Boxes format: [x1, y1, x2, y2].
    Returns float in range [0.0, 1.0].
    """
    if not box_a or not box_b or len(box_a) != 4 or len(box_b) != 4:
        return 0.0

    # Intersection rectangle coordinates
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    intersection = iw * ih

    if intersection <= 0.0:
        return 0.0

    # Areas of both boxes
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))

    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0

    return float(intersection / union)


# Alias for backward compatibility
box_iou = calculate_iou


# =====================================================================
# 3. HELPER FUNCTIONS FOR EXTRACTION & MERGING
# =====================================================================
def _extract_box(det: Dict[str, Any]) -> List[float]:
    """Extracts [x1, y1, x2, y2] from detection dict supporting 'box' and 'bbox' keys."""
    b = det.get("box") or det.get("bbox")
    if b and len(b) == 4:
        return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
    return []


def _extract_score(det: Dict[str, Any]) -> float:
    """Extracts confidence score from detection dict supporting 'score' and 'confidence'."""
    return float(det.get("score") or det.get("confidence") or 0.0)


def _extract_canonical_label(det: Dict[str, Any]) -> str:
    """
    Extracts canonical label for class grouping.
    Falls back to raw_label or label if canonicalization is already done or unavailable.
    """
    lbl = det.get("label") or det.get("raw_label") or ""
    return str(lbl).strip().lower()


def _merge_cluster_boxes(cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Performs score-weighted bounding box merging for an overlapping duplicate cluster.
    Weights: w_i = max(score_i, 1e-4)
    merged_box[k] = sum(w_i * box_i[k]) / sum(w_i)
    Winner's metadata (id, label, confidence level) is preserved.
    """
    if not cluster:
        raise ValueError("Cannot merge empty cluster")

    # The first item is always the highest-scoring detection (cluster leader)
    leader = dict(cluster[0])

    if len(cluster) == 1:
        return leader

    total_weight = 0.0
    w_x1 = 0.0
    w_y1 = 0.0
    w_x2 = 0.0
    w_y2 = 0.0

    for det in cluster:
        box = _extract_box(det)
        if not box:
            continue
        score = max(_extract_score(det), 1e-4)
        w_x1 += box[0] * score
        w_y1 += box[1] * score
        w_x2 += box[2] * score
        w_y2 += box[3] * score
        total_weight += score

    if total_weight > 0:
        merged_box = [
            round(w_x1 / total_weight, 1),
            round(w_y1 / total_weight, 1),
            round(w_x2 / total_weight, 1),
            round(w_y2 / total_weight, 1),
        ]
        leader["box"] = merged_box
        leader["bbox"] = merged_box

        # Update center point if present
        if "center" in leader:
            cx = (merged_box[0] + merged_box[2]) / 2.0
            cy = (merged_box[1] + merged_box[3]) / 2.0
            leader["center"] = [round(cx, 1), round(cy, 1)]

    return leader


def get_deduplication_stats(
    raw_detection_count: int,
    final_detection_count: int,
) -> Dict[str, int]:
    """Returns standardized deduplication debug metrics."""
    return {
        "raw_detection_count": raw_detection_count,
        "final_detection_count": final_detection_count,
        "duplicates_removed": max(0, raw_detection_count - final_detection_count),
    }


# =====================================================================
# 4. GLOBAL CLASS-AWARE NMS
# =====================================================================
def apply_class_nms(
    detections: List[Dict[str, Any]],
    iou_threshold: Optional[float] = None,
    class_thresholds: Optional[Dict[str, float]] = None,
    merge_mode: str = "standard",
    return_debug_info: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, int]]]:
    """
    Applies class-aware Non-Maximum Suppression (NMS) to eliminate duplicate detections.

    Key Features:
    1. Class Isolation: Detections are grouped by canonical class first.
       Unrelated classes (e.g. 'building' and 'vehicle') never suppress each other.
    2. Score Prioritization: Highest-confidence candidate is retained first.
    3. Class-Specific Thresholds: Uses centralized thresholds (e.g. building=0.45,
       vehicle=0.40, water=0.55) unless an explicit iou_threshold is provided.
    4. Merge Modes:
       - 'standard' (default): Greedy suppression keeps the exact winner bounding box.
       - 'weighted': Smooths duplicate bounding boxes using score-weighted coordinates.
    5. Debug Information:
       Returns (kept_detections, debug_info) if return_debug_info=True.

    Parameters:
        detections: List of detection dictionaries containing bounding box, score, and class label.
        iou_threshold: Optional global IoU threshold override. If None, class-specific defaults are used.
        class_thresholds: Optional custom mapping of class -> IoU threshold.
        merge_mode: 'standard' or 'weighted'.
        return_debug_info: Whether to return (detections, debug_stats) tuple.

    Returns:
        List of deduplicated detections, or tuple of (detections, debug_stats).
    """
    if not detections:
        stats = get_deduplication_stats(0, 0)
        return ([], stats) if return_debug_info else []

    raw_count = len(detections)

    # 1. Group detections by canonical class
    class_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for det in detections:
        canonical_label = _extract_canonical_label(det)
        class_groups[canonical_label].append(det)

    final_detections: List[Dict[str, Any]] = []

    # 2. Process each class independently (no cross-class suppression)
    for class_name, group in class_groups.items():
        # Determine the effective IoU threshold for this class
        effective_thresh = get_class_nms_threshold(
            canonical_label=class_name,
            custom_thresholds=class_thresholds,
            fallback_threshold=iou_threshold,
        )

        # Sort candidate detections in descending order of score
        candidates = sorted(group, key=_extract_score, reverse=True)
        suppressed_indices = set()
        kept_in_class: List[Dict[str, Any]] = []

        for i in range(len(candidates)):
            if i in suppressed_indices:
                continue

            winner = candidates[i]
            box_winner = _extract_box(winner)
            if not box_winner:
                continue

            cluster = [winner]

            # Compare against remaining candidates in the same class
            for j in range(i + 1, len(candidates)):
                if j in suppressed_indices:
                    continue

                box_j = _extract_box(candidates[j])
                if not box_j:
                    continue

                iou = calculate_iou(box_winner, box_j)
                if iou > effective_thresh:
                    suppressed_indices.add(j)
                    cluster.append(candidates[j])

            # Apply merge mode
            if merge_mode == "weighted" and len(cluster) > 1:
                merged = _merge_cluster_boxes(cluster)
                kept_in_class.append(merged)
            else:
                # Standard NMS: keep winner untouched
                kept_in_class.append(winner)

        final_detections.extend(kept_in_class)

    # 3. Sort final detections descending by score
    final_detections.sort(key=_extract_score, reverse=True)

    # 4. Generate deduplication debug metrics
    final_count = len(final_detections)
    stats = get_deduplication_stats(raw_count, final_count)

    logger.debug(
        f"[NMS] Raw detections: {raw_count} | Final kept: {final_count} | "
        f"Duplicates removed: {stats['duplicates_removed']}"
    )

    if return_debug_info:
        return final_detections, stats

    return final_detections

