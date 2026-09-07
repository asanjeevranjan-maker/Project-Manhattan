"""
Temporal Object Matcher Module (Enhanced).
Matches Grounding DINO object detections across bi-temporal satellite scenes,
classifies objects into NEW, REMOVED, UNCHANGED, and POSSIBLY_MODIFIED,
and maps bounding-box centers to geographical coordinates (WGS84 lat/lon).

Integrated with Calibrated Confidence Engine, Crop Visual Verification,
and Strict Gating to prevent false-positive claims like "NEW Building (88%)".
"""

import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None

from satellite.provider_base import AOIBoundingBox

logger = logging.getLogger("satquery.temporal_matcher")

# Nuisance & temporary classes
TEMPORARY_CLASSES = {
    "ship", "boat", "vessel", "cargo ship", "tanker", "container ship",
    "vehicle", "car", "truck", "bus", "aircraft", "airplane", "container",
    "movable", "crane", "machinery"
}


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
    """Maps bounding box center (px) to approximate latitude and longitude (WGS84)."""
    if not aoi or img_w <= 0 or img_h <= 0:
        return None, None

    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0

    norm_x = min(max(cx / img_w, 0.0), 1.0)
    norm_y = min(max(cy / img_h, 0.0), 1.0)

    lat = aoi.north - norm_y * (aoi.north - aoi.south)
    lon = aoi.west + norm_x * (aoi.east - aoi.west)

    return round(lat, 6), round(lon, 6)


def _crop_similarity(image_t1: Image.Image, image_t2: Image.Image, box: List[float]) -> float:
    """Computes structural similarity on the cropped patch."""
    if not (CV2_AVAILABLE and NUMPY_AVAILABLE):
        return 0.50
    try:
        w, h = image_t1.size
        x1 = max(0, min(int(box[0]), w - 2))
        y1 = max(0, min(int(box[1]), h - 2))
        x2 = max(x1 + 2, min(int(box[2]), w))
        y2 = max(y1 + 2, min(int(box[3]), h))

        t1_arr = np.array(image_t1.convert("RGB"))[y1:y2, x1:x2]
        t2_arr = np.array(image_t2.convert("RGB"))[y1:y2, x1:x2]

        if t1_arr.size == 0 or t2_arr.size == 0:
            return 0.50

        g1 = cv2.cvtColor(t1_arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(t2_arr, cv2.COLOR_RGB2GRAY).astype(np.float32)

        s1, s2 = np.std(g1), np.std(g2)
        if s1 > 1.0 and s2 > 1.0:
            corr = float(np.mean((g1 - np.mean(g1)) * (g2 - np.mean(g2))) / (s1 * s2))
            return max(0.0, min(1.0, (corr + 1.0) / 2.0))
        mean_diff = float(np.mean(np.abs(g1 - g2)))
        return max(0.0, 1.0 - (mean_diff / 50.0))
    except Exception:
        return 0.50


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
    image_t1: Optional[Image.Image] = None,
    image_t2: Optional[Image.Image] = None,
    registration_quality: float = 0.85,
    pixel_change_mask: Optional["np.ndarray"] = None,
) -> Dict[str, Any]:
    """
    Matches detections from T1 (Historical) and T2 (Latest).
    Classifies objects into:
      - 'new': verified new appearance in T2
      - 'removed': confirmed absent in T2
      - 'modified': overlapping/near but altered
      - 'unchanged': stable object across both timestamps
    """
    matched_t1_indices = set()
    matched_t2_indices = set()
    changes: List[Dict[str, Any]] = []

    pair_candidates = []
    for i, d1 in enumerate(detections_t1):
        for j, d2 in enumerate(detections_t2):
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

    pair_candidates.sort(key=lambda x: x[0], reverse=True)

    for score, i, j, iou, dist in pair_candidates:
        if i in matched_t1_indices or j in matched_t2_indices:
            continue

        matched_t1_indices.add(i)
        matched_t2_indices.add(j)

        d1 = detections_t1[i]
        d2 = detections_t2[j]

        w1 = max(1.0, d1["box"][2] - d1["box"][0])
        h1 = max(1.0, d1["box"][3] - d1["box"][1])
        area1 = w1 * h1

        w2 = max(1.0, d2["box"][2] - d2["box"][0])
        h2 = max(1.0, d2["box"][3] - d2["box"][1])
        area2 = w2 * h2

        area_ratio = max(area1, area2) / min(area1, area2)
        lat, lon = pixel_to_geo_coords(d2["box"], img_width, img_height, aoi)
        lbl = d2["label"]

        if iou >= 0.40 and area_ratio <= 1.40:
            change_type = "unchanged"
            details = f"Stable structure confirmed across both timestamps (IoU: {iou:.2f})"
            conf = round((d1["confidence"] + d2["confidence"]) / 2.0, 3)
            v_status = "CONFIRMED_UNCHANGED"
        else:
            change_type = "modified"
            details = f"Structure alteration observed (Area ratio: {area_ratio:.2f}x, IoU: {iou:.2f})"
            conf = round(min(0.85, (d1["confidence"] + d2["confidence"]) / 2.0), 3)
            v_status = "LIKELY_CHANGE"

        changes.append({
            "id": f"chg-m-{len(changes) + 1}",
            "type": change_type,
            "label": lbl,
            "confidence": conf,
            "verificationStatus": v_status,
            "persistence": "TEMPORARY" if any(t in lbl.lower() for t in TEMPORARY_CLASSES) else "PERMANENT",
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

    # Unmatched T2 detections: verify whether they are genuinely NEW
    for j, d2 in enumerate(detections_t2):
        if j in matched_t2_indices:
            continue

        lat, lon = pixel_to_geo_coords(d2["box"], img_width, img_height, aoi)
        lbl = d2["label"].lower().strip()
        d2_raw_conf = float(d2["confidence"])
        is_temp = any(t in lbl for t in TEMPORARY_CLASSES)

        # Check crop similarity if images provided
        patch_sim = 0.50
        if image_t1 is not None and image_t2 is not None:
            patch_sim = _crop_similarity(image_t1, image_t2, d2["box"])

        # Check change mask overlap if provided
        change_overlap = 0.50
        if pixel_change_mask is not None:
            b = d2["box"]
            x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
            x2, y2 = min(img_width, int(b[2])), min(img_height, int(b[3]))
            sub = pixel_change_mask[y1:y2, x1:x2]
            if sub.size > 0:
                change_overlap = float(np.sum(sub > 0)) / sub.size

        # DROPOUT RECOVERY:
        # If the T1 image crop already has high similarity (> 0.80) and low change overlap,
        # DINO missed the object in T1. DO NOT declare false "NEW Building"!
        if patch_sim > 0.80 and change_overlap < 0.20:
            changes.append({
                "id": f"chg-u-{len(changes) + 1}",
                "type": "unchanged",
                "label": d2["label"],
                "confidence": round(d2_raw_conf * 0.90, 3),
                "verificationStatus": "CONFIRMED_UNCHANGED",
                "persistence": "TEMPORARY" if is_temp else "PERMANENT",
                "boxT1": d2["box"],
                "boxT2": d2["box"],
                "currentBox": d2["box"],
                "latitude": lat,
                "longitude": lon,
                "details": f"Pre-existing {d2['label']} confirmed in T1 (detection dropout compensated by visual verification).",
                "historicalDate": historical_date,
                "latestDate": latest_date,
                "metrics": {
                    "patchSimilarity": round(patch_sim, 3),
                    "dropoutCorrected": True,
                }
            })
            continue

        # TEMPORARY OBJECT HANDLING:
        if is_temp:
            changes.append({
                "id": f"chg-tmp-{len(changes) + 1}",
                "type": "new",
                "label": d2["label"],
                "confidence": round(min(0.85, d2_raw_conf), 3),
                "verificationStatus": "CONFIRMED_CHANGE",
                "persistence": "TEMPORARY",
                "boxT1": None,
                "boxT2": d2["box"],
                "currentBox": d2["box"],
                "latitude": lat,
                "longitude": lon,
                "details": f"Mobile asset ({d2['label']}) observed at berth/location; not permanent construction.",
                "historicalDate": historical_date,
                "latestDate": latest_date,
                "metrics": {
                    "confidenceT2": round(d2_raw_conf, 3),
                    "persistence": "TEMPORARY",
                }
            })
            continue

        # STRICT CONFIDENCE GATING FOR PERMANENT CHANGES:
        # Cap confidence at 0.78 unless multi-signal evidence is proven
        calib_conf = min(d2_raw_conf, 0.78)
        if registration_quality < 0.70:
            calib_conf = min(calib_conf, 0.55)

        changes.append({
            "id": f"chg-new-{len(changes) + 1}",
            "type": "new",
            "label": d2["label"],
            "confidence": round(calib_conf, 3),
            "verificationStatus": "LIKELY_CHANGE" if calib_conf >= 0.70 else "POSSIBLE_CHANGE",
            "persistence": "PERMANENT",
            "boxT1": None,
            "boxT2": d2["box"],
            "currentBox": d2["box"],
            "latitude": lat,
            "longitude": lon,
            "details": f"Potential new structure ({d2['label']}) observed in latest observation ({latest_date}).",
            "historicalDate": historical_date,
            "latestDate": latest_date,
            "metrics": {
                "confidenceT2": round(d2_raw_conf, 3),
                "calibratedConfidence": round(calib_conf, 3),
                "patchSimilarityT1": round(patch_sim, 3),
            }
        })

    # Unmatched T1 detections: verify removal
    for i, d1 in enumerate(detections_t1):
        if i in matched_t1_indices:
            continue

        lat, lon = pixel_to_geo_coords(d1["box"], img_width, img_height, aoi)
        lbl = d1["label"].lower().strip()
        d1_raw_conf = float(d1["confidence"])
        is_temp = any(t in lbl for t in TEMPORARY_CLASSES)

        patch_sim = 0.50
        if image_t1 is not None and image_t2 is not None:
            patch_sim = _crop_similarity(image_t1, image_t2, d1["box"])

        if patch_sim > 0.80:
            changes.append({
                "id": f"chg-u-{len(changes) + 1}",
                "type": "unchanged",
                "label": d1["label"],
                "confidence": round(d1_raw_conf * 0.90, 3),
                "verificationStatus": "CONFIRMED_UNCHANGED",
                "persistence": "TEMPORARY" if is_temp else "PERMANENT",
                "boxT1": d1["box"],
                "boxT2": d1["box"],
                "currentBox": d1["box"],
                "latitude": lat,
                "longitude": lon,
                "details": f"Persistent {d1['label']} confirmed in both scenes (DINO missed in T2; patch confirmed).",
                "historicalDate": historical_date,
                "latestDate": latest_date,
                "metrics": {
                    "patchSimilarity": round(patch_sim, 3),
                    "dropoutCorrected": True,
                }
            })
            continue

        changes.append({
            "id": f"chg-rem-{len(changes) + 1}",
            "type": "removed",
            "label": d1["label"],
            "confidence": round(min(0.85, d1_raw_conf), 3),
            "verificationStatus": "LIKELY_CHANGE",
            "persistence": "TEMPORARY" if is_temp else "PERMANENT",
            "boxT1": d1["box"],
            "boxT2": None,
            "currentBox": d1["box"],
            "latitude": lat,
            "longitude": lon,
            "details": f"Object ({d1['label']}) observed on {historical_date} is no longer present in latest imagery.",
            "historicalDate": historical_date,
            "latestDate": latest_date,
            "metrics": {
                "confidenceT1": round(d1_raw_conf, 3),
            }
        })

    new_count = sum(1 for c in changes if c["type"] == "new")
    removed_count = sum(1 for c in changes if c["type"] == "removed")
    modified_count = sum(1 for c in changes if c["type"] == "modified")
    unchanged_count = sum(1 for c in changes if c["type"] == "unchanged")

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
