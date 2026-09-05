"""
Temporal Object Matcher Module
Matches Grounding DINO object detections across bi-temporal satellite scenes,
classifies objects into NEW, REMOVED, UNCHANGED, and POSSIBLY_MODIFIED,
and maps bounding-box centers to geographical coordinates (WGS84 lat/lon).
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from satellite.provider_base import AOIBoundingBox


def box_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection over Union between two [x1, y1, x2, y2] boxes."""
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

    return intersection / union


def box_center_distance(box1: List[float], box2: List[float], img_w: float, img_h: float) -> float:
    """Normalized Euclidean distance between centers of two boxes."""
    cx1 = (box1[0] + box1[2]) / 2.0 / (img_w or 1.0)
    cy1 = (box1[1] + box1[3]) / 2.0 / (img_h or 1.0)
    cx2 = (box2[0] + box2[2]) / 2.0 / (img_w or 1.0)
    cy2 = (box2[1] + box2[3]) / 2.0 / (img_h or 1.0)
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def pixel_to_geo_coords(
    box: List[float],
    img_w: int,
    img_h: int,
    aoi: Optional[AOIBoundingBox] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Maps bounding box center (px) to approximate latitude and longitude (WGS84).
    """
    if not aoi or img_w <= 0 or img_h <= 0:
        return None, None

    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0

    norm_x = min(max(cx / img_w, 0.0), 1.0)
    norm_y = min(max(cy / img_h, 0.0), 1.0)

    # Latitude decreases going down (north -> south)
    lat = aoi.north - norm_y * (aoi.north - aoi.south)
    # Longitude increases going right (west -> east)
    lon = aoi.west + norm_x * (aoi.east - aoi.west)

    return round(lat, 6), round(lon, 6)


def match_temporal_detections(
    detections_t1: List[Dict[str, Any]],
    detections_t2: List[Dict[str, Any]],
    img_width: int,
    img_height: int,
    aoi: Optional[AOIBoundingBox] = None,
    historical_date: str = "T1",
    latest_date: str = "T2",
    iou_match_threshold: float = 0.25,
    distance_threshold: float = 0.08,
) -> Dict[str, Any]:
    """
    Matches detections from T1 (Historical) and T2 (Latest).
    Classifies objects into:
      - 'new': detected in T2, not present in T1
      - 'removed': present in T1, absent in T2
      - 'modified': overlapping/near but significant area/shape/confidence change
      - 'unchanged': stable object across both timestamps
    """
    matched_t1_indices = set()
    matched_t2_indices = set()
    changes: List[Dict[str, Any]] = []

    # Build cost/similarity matrix between each T1 and T2 pair
    pair_candidates = []
    for i, d1 in enumerate(detections_t1):
        for j, d2 in enumerate(detections_t2):
            # Same or compatible class label
            l1 = d1["label"].lower().strip()
            l2 = d2["label"].lower().strip()
            label_compatible = (l1 == l2) or (l1 in l2) or (l2 in l1)
            if not label_compatible:
                continue

            iou = box_iou(d1["box"], d2["box"])
            dist = box_center_distance(d1["box"], d2["box"], img_width, img_height)

            if iou >= iou_match_threshold or dist <= distance_threshold:
                score = iou * 0.7 + (1.0 - min(dist / distance_threshold, 1.0)) * 0.3
                pair_candidates.append((score, i, j, iou, dist))

    # Sort greedy matching by best match score descending
    pair_candidates.sort(key=lambda x: x[0], reverse=True)

    for score, i, j, iou, dist in pair_candidates:
        if i in matched_t1_indices or j in matched_t2_indices:
            continue

        matched_t1_indices.add(i)
        matched_t2_indices.add(j)

        d1 = detections_t1[i]
        d2 = detections_t2[j]

        # Area comparison
        w1 = max(1.0, d1["box"][2] - d1["box"][0])
        h1 = max(1.0, d1["box"][3] - d1["box"][1])
        area1 = w1 * h1

        w2 = max(1.0, d2["box"][2] - d2["box"][0])
        h2 = max(1.0, d2["box"][3] - d2["box"][1])
        area2 = w2 * h2

        area_ratio = max(area1, area2) / min(area1, area2)
        conf_diff = abs(d1["confidence"] - d2["confidence"])

        lat, lon = pixel_to_geo_coords(d2["box"], img_width, img_height, aoi)

        # Classification
        if iou >= 0.40 and area_ratio <= 1.40:
            change_type = "unchanged"
            details = f"Stable structure confirmed across both timestamps (IoU: {iou:.2f})"
        else:
            change_type = "modified"
            details = (
                f"Structure expansion/modification observed (Area ratio: {area_ratio:.2f}x, "
                f"center shift: {dist*100:.1f}% of image)"
            )

        changes.append({
            "id": f"chg-m-{len(changes) + 1}",
            "type": change_type,
            "label": d2["label"],
            "confidence": round((d1["confidence"] + d2["confidence"]) / 2.0, 3),
            "boxT1": d1["box"],
            "boxT2": d2["box"],
            "currentBox": d2["box"],
            "latitude": lat,
            "longitude": lon,
            "details": details,
            "historicalDate": historical_date,
            "latestDate": latest_date,
            "metrics": {
                "iou": round(iou, 3),
                "areaRatio": round(area_ratio, 2),
                "confidenceT1": round(d1["confidence"], 3),
                "confidenceT2": round(d2["confidence"], 3),
            }
        })

    # Unmatched T2 detections are NEW objects
    for j, d2 in enumerate(detections_t2):
        if j not in matched_t2_indices:
            lat, lon = pixel_to_geo_coords(d2["box"], img_width, img_height, aoi)
            changes.append({
                "id": f"chg-new-{len(changes) + 1}",
                "type": "new",
                "label": d2["label"],
                "confidence": round(d2["confidence"], 3),
                "boxT1": None,
                "boxT2": d2["box"],
                "currentBox": d2["box"],
                "latitude": lat,
                "longitude": lon,
                "details": f"New construction / appearance detected in latest imagery ({latest_date})",
                "historicalDate": historical_date,
                "latestDate": latest_date,
                "metrics": {
                    "confidenceT2": round(d2["confidence"], 3),
                }
            })

    # Unmatched T1 detections are REMOVED objects
    for i, d1 in enumerate(detections_t1):
        if i not in matched_t1_indices:
            lat, lon = pixel_to_geo_coords(d1["box"], img_width, img_height, aoi)
            changes.append({
                "id": f"chg-rem-{len(changes) + 1}",
                "type": "removed",
                "label": d1["label"],
                "confidence": round(d1["confidence"], 3),
                "boxT1": d1["box"],
                "boxT2": None,
                "currentBox": d1["box"],
                "latitude": lat,
                "longitude": lon,
                "details": f"Object observed on {historical_date} is no longer present in latest imagery ({latest_date})",
                "historicalDate": historical_date,
                "latestDate": latest_date,
                "metrics": {
                    "confidenceT1": round(d1["confidence"], 3),
                }
            })

    new_count = sum(1 for c in changes if c["type"] == "new")
    removed_count = sum(1 for c in changes if c["type"] == "removed")
    modified_count = sum(1 for c in changes if c["type"] == "modified")
    unchanged_count = sum(1 for c in changes if c["type"] == "unchanged")

    # Sort changes: new first, then removed, modified, unchanged
    type_priority = {"new": 0, "removed": 1, "modified": 2, "unchanged": 3}
    changes.sort(key=lambda c: (type_priority.get(c["type"], 4), -c["confidence"]))

    summary = {
        "totalBefore": len(detections_t1),
        "totalLatest": len(detections_t2),
        "newCount": new_count,
        "removedCount": removed_count,
        "modifiedCount": modified_count,
        "unchangedCount": unchanged_count,
        "totalChanges": new_count + removed_count + modified_count,
    }

    return {
        "summary": summary,
        "changes": changes,
        "rawT1Count": len(detections_t1),
        "rawT2Count": len(detections_t2),
    }

