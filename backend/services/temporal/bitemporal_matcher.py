"""
Bi-Temporal Object Matcher Module
Matches Grounding DINO object detections across bi-temporal satellite scenes (T1 and T2),
classifying objects into:
- appeared: newly detected in T2, absent in T1
- disappeared: present in T1, absent in T2
- persisted: stable object confirmed across both timestamps
- possibly_changed: matched candidate with notable area expansion/contraction or spatial shift

Uses spatial matching with class compatibility, box IoU, and normalized centroid distance.
"""

import math
from typing import List, Dict, Any, Optional, Tuple

try:
    from ..detection.vocabulary import normalize_label
except ImportError:
    try:
        from services.detection.vocabulary import normalize_label  # type: ignore
    except ImportError:
        def normalize_label(label: str) -> Tuple[str, str]:
            lbl = (label or "").strip().lower()
            return lbl, lbl


def box_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection over Union between two [x1, y1, x2, y2] boxes."""
    if not box1 or not box2 or len(box1) < 4 or len(box2) < 4:
        return 0.0

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_w = max(0.0, x2 - x1)
    intersection_h = max(0.0, y2 - y1)
    intersection = intersection_w * intersection_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - intersection
    if union <= 0.0:
        return 0.0

    return float(intersection / union)


def box_center_distance(
    box1: List[float], box2: List[float], img_w: float, img_h: float
) -> float:
    """Normalized Euclidean distance between centers of two boxes [0.0 - 1.414]."""
    cx1 = (box1[0] + box1[2]) / 2.0 / (img_w or 1.0)
    cy1 = (box1[1] + box1[3]) / 2.0 / (img_h or 1.0)
    cx2 = (box2[0] + box2[2]) / 2.0 / (img_w or 1.0)
    cy2 = (box2[1] + box2[3]) / 2.0 / (img_h or 1.0)
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def get_quadrant_location(box: List[float], img_w: float, img_h: float) -> str:
    """Determines the human-readable spatial quadrant of a box."""
    if not box or len(box) < 4:
        return "center"
    cx = (box[0] + box[2]) / 2.0 / (img_w or 1.0)
    cy = (box[1] + box[3]) / 2.0 / (img_h or 1.0)

    horiz = "left" if cx < 0.40 else ("right" if cx > 0.60 else "center")
    vert = "upper" if cy < 0.40 else ("lower" if cy > 0.60 else "central")

    if horiz == "center" and vert == "central":
        return "center"
    if horiz == "center":
        return f"{vert} region"
    if vert == "central":
        return f"{horiz} side"
    return f"{vert}-{horiz} region"


def match_bitemporal_detections(
    detections_t1: List[Dict[str, Any]],
    detections_t2: List[Dict[str, Any]],
    img_width: int = 1000,
    img_height: int = 1000,
    iou_match_threshold: float = 0.20,
    distance_threshold: float = 0.10,
    date_t1: str = "T1",
    date_t2: str = "T2",
) -> Dict[str, Any]:
    """
    Spatially matches detections from Time T1 and Time T2.

    Returns:
    {
        "objects": {
            "appeared": [...],
            "disappeared": [...],
            "persisted": [...],
            "possibly_changed": [...]
        },
        "all_items": [...],
        "summary": {
            "appeared_count": int,
            "disappeared_count": int,
            "persisted_count": int,
            "possibly_changed_count": int,
            "total_t1": int,
            "total_t2": int,
        }
    }
    """
    dets_t1 = list(detections_t1 or [])
    dets_t2 = list(detections_t2 or [])

    matched_t1_indices = set()
    matched_t2_indices = set()

    # Pre-normalize labels
    norm_labels_t1 = [normalize_label(d.get("label", ""))[0] or d.get("label", "").lower() for d in dets_t1]
    norm_labels_t2 = [normalize_label(d.get("label", ""))[0] or d.get("label", "").lower() for d in dets_t2]

    candidates = []

    for i, d1 in enumerate(dets_t1):
        box1 = d1.get("box") or d1.get("bbox") or []
        if len(box1) < 4:
            continue
        l1 = norm_labels_t1[i]

        for j, d2 in enumerate(dets_t2):
            box2 = d2.get("box") or d2.get("bbox") or []
            if len(box2) < 4:
                continue
            l2 = norm_labels_t2[j]

            # Label compatibility check
            compatible = (l1 == l2) or (l1 in l2) or (l2 in l1)
            if not compatible:
                continue

            iou = box_iou(box1, box2)
            dist = box_center_distance(box1, box2, img_width, img_height)

            if iou >= iou_match_threshold or dist <= distance_threshold:
                # Combined similarity score
                sim = iou * 0.70 + max(0.0, 1.0 - (dist / distance_threshold)) * 0.30
                candidates.append((sim, i, j, iou, dist))

    # Greedy bipartite assignment by similarity descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    persisted: List[Dict[str, Any]] = []
    possibly_changed: List[Dict[str, Any]] = []

    for sim, i, j, iou, dist in candidates:
        if i in matched_t1_indices or j in matched_t2_indices:
            continue

        matched_t1_indices.add(i)
        matched_t2_indices.add(j)

        d1 = dets_t1[i]
        d2 = dets_t2[j]
        box1 = d1.get("box") or d1.get("bbox")
        box2 = d2.get("box") or d2.get("bbox")

        w1 = max(1.0, box1[2] - box1[0])
        h1 = max(1.0, box1[3] - box1[1])
        area1 = w1 * h1

        w2 = max(1.0, box2[2] - box2[0])
        h2 = max(1.0, box2[3] - box2[1])
        area2 = w2 * h2

        area_ratio = max(area1, area2) / min(area1, area2)
        quad = get_quadrant_location(box2, img_width, img_height)
        label = d2.get("label", d1.get("label", "object"))
        mean_conf = round(((d1.get("confidence") or d1.get("score") or 0.8) + (d2.get("confidence") or d2.get("score") or 0.8)) / 2.0, 3)

        # Classification criterion:
        # High overlap & stable area -> persisted
        # Lower overlap or significant area ratio (> 1.35x) -> possibly_changed
        if iou >= 0.40 and area_ratio <= 1.35:
            item = {
                "id": f"persisted-{len(persisted)+1}",
                "type": "persisted",
                "label": label,
                "confidence": mean_conf,
                "box_t1": box1,
                "box_t2": box2,
                "current_box": box2,
                "iou": round(iou, 3),
                "area_ratio": round(area_ratio, 2),
                "location": quad,
                "details": f"Stable {label} confirmed across both timestamps (IoU: {iou:.2f}) in {quad}.",
            }
            persisted.append(item)
        else:
            diff_desc = "expansion" if area2 > area1 else "shrinkage"
            item = {
                "id": f"changed-{len(possibly_changed)+1}",
                "type": "possibly_changed",
                "label": label,
                "confidence": mean_conf,
                "box_t1": box1,
                "box_t2": box2,
                "current_box": box2,
                "iou": round(iou, 3),
                "area_ratio": round(area_ratio, 2),
                "location": quad,
                "details": f"Altered {label} observed in {quad} ({diff_desc} {area_ratio:.2f}x, IoU: {iou:.2f}).",
            }
            possibly_changed.append(item)

    # Detections in T2 without match in T1 -> APPEARED
    appeared: List[Dict[str, Any]] = []
    for j, d2 in enumerate(dets_t2):
        if j not in matched_t2_indices:
            box2 = d2.get("box") or d2.get("bbox")
            if not box2:
                continue
            label = d2.get("label", "object")
            conf = round(float(d2.get("confidence") or d2.get("score") or 0.8), 3)
            quad = get_quadrant_location(box2, img_width, img_height)

            item = {
                "id": f"appeared-{len(appeared)+1}",
                "type": "appeared",
                "label": label,
                "confidence": conf,
                "box_t1": None,
                "box_t2": box2,
                "current_box": box2,
                "iou": 0.0,
                "area_ratio": None,
                "location": quad,
                "details": f"Newly appeared {label} detected in {quad} at {date_t2}.",
            }
            appeared.append(item)

    # Detections in T1 without match in T2 -> DISAPPEARED
    disappeared: List[Dict[str, Any]] = []
    for i, d1 in enumerate(dets_t1):
        if i not in matched_t1_indices:
            box1 = d1.get("box") or d1.get("bbox")
            if not box1:
                continue
            label = d1.get("label", "object")
            conf = round(float(d1.get("confidence") or d1.get("score") or 0.8), 3)
            quad = get_quadrant_location(box1, img_width, img_height)

            item = {
                "id": f"disappeared-{len(disappeared)+1}",
                "type": "disappeared",
                "label": label,
                "confidence": conf,
                "box_t1": box1,
                "box_t2": None,
                "current_box": box1,
                "iou": 0.0,
                "area_ratio": None,
                "location": quad,
                "details": f"Historical {label} from {date_t1} is no longer present in {quad} at {date_t2}.",
            }
            disappeared.append(item)

    summary = {
        "appeared_count": len(appeared),
        "disappeared_count": len(disappeared),
        "persisted_count": len(persisted),
        "possibly_changed_count": len(possibly_changed),
        "total_t1": len(dets_t1),
        "total_t2": len(dets_t2),
    }

    all_items = appeared + disappeared + possibly_changed + persisted

    return {
        "objects": {
            "appeared": appeared,
            "disappeared": disappeared,
            "persisted": persisted,
            "possibly_changed": possibly_changed,
        },
        "all_items": all_items,
        "summary": summary,
    }

