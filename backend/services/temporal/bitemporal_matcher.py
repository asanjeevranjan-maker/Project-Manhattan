"""
Enhanced Bi-Temporal Object Matcher Module (Stages 9, 10, 12, 16, 17, 18, 19, 21, 22).
Matches Grounding DINO object detections across bi-temporal satellite scenes (T1 and T2).

Key Enhancements:
1. Crop-level visual verification: prevents DINO detection dropouts in T1 from being falsely reported as "NEW".
2. Multi-signal change map overlap verification: requires verified structural change inside the candidate box.
3. Nuisance discounting: filters water/tide, shadow, and cloud edge artifacts.
4. Persistence categorization: separates TEMPORARY (ships, vehicles) from PERMANENT (buildings, roads).
5. Evidence-based confidence calibration with strict gating (>= 90% requires multi-signal proof).
6. Localized bounding boxes and geographic coordinate mapping.
"""

import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

logger = logging.getLogger("satquery.temporal.matcher")

CV2_AVAILABLE = False
NUMPY_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore

from .confidence_engine import calibrate_change_confidence, classify_persistence

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
    """Determines human-readable spatial quadrant of a box."""
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


def pixel_to_geo_coords(
    box: List[float],
    img_w: int,
    img_h: int,
    aoi: Optional[Any] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Maps bounding box center (px) to approximate latitude and longitude (WGS84)."""
    if not aoi or img_w <= 0 or img_h <= 0:
        return None, None
    try:
        north = getattr(aoi, "north", None) or aoi.get("north")
        south = getattr(aoi, "south", None) or aoi.get("south")
        east = getattr(aoi, "east", None) or aoi.get("east")
        west = getattr(aoi, "west", None) or aoi.get("west")

        if north is None or south is None or east is None or west is None:
            return None, None

        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0

        norm_x = min(max(cx / img_w, 0.0), 1.0)
        norm_y = min(max(cy / img_h, 0.0), 1.0)

        lat = float(north) - norm_y * (float(north) - float(south))
        lon = float(west) + norm_x * (float(east) - float(west))

        return round(lat, 6), round(lon, 6)
    except Exception:
        return None, None


def _crop_and_compare_patches(
    image_t1: Image.Image,
    image_t2: Image.Image,
    box: List[float],
) -> Tuple[float, float, bool, bool]:
    """
    Extracts crops from T1 and T2 at the specified box coordinates and computes:
    - patch_ssim: structural similarity (1.0 = identical patch)
    - mean_diff: mean absolute pixel difference
    - has_structure_t1: True if T1 crop contains genuine structural contrast/texture
    - has_structure_t2: True if T2 crop contains genuine structural contrast/texture
    """
    if not (CV2_AVAILABLE and NUMPY_AVAILABLE):
        return 0.50, 20.0, False, False

    try:
        w, h = image_t1.size
        x1 = max(0, min(int(box[0]), w - 2))
        y1 = max(0, min(int(box[1]), h - 2))
        x2 = max(x1 + 2, min(int(box[2]), w))
        y2 = max(y1 + 2, min(int(box[3]), h))

        t1_np = np.array(image_t1.convert("RGB"))
        t2_np = np.array(image_t2.convert("RGB"))

        crop1 = t1_np[y1:y2, x1:x2]
        crop2 = t2_np[y1:y2, x1:x2]

        if crop1.size == 0 or crop2.size == 0:
            return 0.50, 20.0, False, False

        g1 = cv2.cvtColor(crop1, cv2.COLOR_RGB2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(crop2, cv2.COLOR_RGB2GRAY).astype(np.float32)

        mean_diff = float(np.mean(np.abs(g1 - g2)))
        s1, s2 = float(np.std(g1)), float(np.std(g2))

        has_structure_t1 = bool(s1 >= 5.0)
        has_structure_t2 = bool(s2 >= 5.0)

        # Local patch correlation as SSIM approximation
        if s1 > 2.0 and s2 > 2.0:
            corr = float(np.mean((g1 - np.mean(g1)) * (g2 - np.mean(g2))) / (s1 * s2))
            patch_ssim = max(0.0, min(1.0, (corr + 1.0) / 2.0))
        elif not has_structure_t1 and not has_structure_t2:
            # Low contrast / flat area in both scenes (untextured canvas)
            patch_ssim = 0.50
        else:
            # One scene has structure while the other is flat
            patch_ssim = max(0.0, 1.0 - (mean_diff / 50.0))

        return round(patch_ssim, 3), round(mean_diff, 1), has_structure_t1, has_structure_t2
    except Exception as e:
        logger.debug(f"[Patch Comparison Exception]: {e}")
        return 0.50, 20.0, False, False


def match_bitemporal_detections(
    detections_t1: List[Dict[str, Any]],
    detections_t2: List[Dict[str, Any]],
    img_width: int = 1000,
    img_height: int = 1000,
    iou_match_threshold: float = 0.20,
    distance_threshold: float = 0.10,
    date_t1: str = "T1",
    date_t2: str = "T2",
    image_t1: Optional[Image.Image] = None,
    image_t2: Optional[Image.Image] = None,
    structural_change_map: Optional["np.ndarray"] = None,
    nuisance_masks: Optional[Dict[str, Any]] = None,
    registration_quality: float = 0.85,
    sar_score: float = 0.0,
    has_sar: bool = False,
    aoi: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Spatially matches detections from Time T1 and Time T2 with crop-level verification,
    nuisance discounting, and calibrated confidence gating.
    """
    dets_t1 = list(detections_t1 or [])
    dets_t2 = list(detections_t2 or [])

    matched_t1_indices = set()
    matched_t2_indices = set()

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

            compatible = (l1 == l2) or (l1 in l2) or (l2 in l1)
            if not compatible:
                continue

            iou = box_iou(box1, box2)
            dist = box_center_distance(box1, box2, img_width, img_height)

            if iou >= iou_match_threshold or dist <= distance_threshold:
                sim = iou * 0.70 + max(0.0, 1.0 - (dist / distance_threshold)) * 0.30
                candidates.append((sim, i, j, iou, dist))

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
        persistence = classify_persistence(label)
        lat, lon = pixel_to_geo_coords(box2, img_width, img_height, aoi)

        mean_conf = round(
            ((d1.get("confidence") or d1.get("score") or 0.8) + (d2.get("confidence") or d2.get("score") or 0.8)) / 2.0,
            3,
        )

        if iou >= 0.40 and area_ratio <= 1.35:
            item = {
                "id": f"persisted-{len(persisted)+1}",
                "type": "persisted",
                "label": label,
                "confidence": mean_conf,
                "verificationStatus": "CONFIRMED_UNCHANGED",
                "persistence": persistence,
                "box_t1": box1,
                "box_t2": box2,
                "boxT1": box1,
                "boxT2": box2,
                "current_box": box2,
                "currentBox": box2,
                "latitude": lat,
                "longitude": lon,
                "iou": round(iou, 3),
                "area_ratio": round(area_ratio, 2),
                "location": quad,
                "details": f"Stable {label} confirmed across both timestamps (IoU: {iou:.2f}) in {quad}.",
                "description": f"Stable {label} confirmed across both timestamps in {quad}.",
            }
            persisted.append(item)
        else:
            diff_desc = "expansion" if area2 > area1 else "shrinkage"
            calib = calibrate_change_confidence(
                change_type="modified",
                label=label,
                registration_quality=registration_quality,
                structural_change_score=min(1.0, (area_ratio - 1.0) / 2.0),
                object_detection_confidence=mean_conf,
                temporal_object_evidence=min(1.0, 1.0 - iou),
                patch_ssim_t1_t2=0.60,
                has_sar=has_sar,
                sar_score=sar_score,
            )
            item = {
                "id": f"changed-{len(possibly_changed)+1}",
                "type": "possibly_changed",
                "label": label,
                "confidence": calib["confidence"],
                "verificationStatus": calib["verification_status"],
                "persistence": calib["persistence"],
                "box_t1": box1,
                "box_t2": box2,
                "boxT1": box1,
                "boxT2": box2,
                "current_box": box2,
                "currentBox": box2,
                "latitude": lat,
                "longitude": lon,
                "iou": round(iou, 3),
                "area_ratio": round(area_ratio, 2),
                "location": quad,
                "details": f"Altered {label} observed in {quad} ({diff_desc} {area_ratio:.2f}x, IoU: {iou:.2f}).",
                "description": calib["description"],
                "evidence": calib["evidence"],
                "penalties": calib["penalties"],
            }
            possibly_changed.append(item)

    # -----------------------------------------------------------------
    # UNMATCHED T2 DETECTIONS -> CROP-LEVEL VERIFICATION & GATING
    # -----------------------------------------------------------------
    appeared: List[Dict[str, Any]] = []

    for j, d2 in enumerate(dets_t2):
        if j in matched_t2_indices:
            continue

        box2 = d2.get("box") or d2.get("bbox")
        if not box2 or len(box2) < 4:
            continue

        label = d2.get("label", "object")
        d2_conf = float(d2.get("confidence") or d2.get("score") or 0.8)
        quad = get_quadrant_location(box2, img_width, img_height)
        lat, lon = pixel_to_geo_coords(box2, img_width, img_height, aoi)
        persistence = classify_persistence(label)

        # 1. Crop-level visual verification (Is the object physically in T1?)
        patch_ssim = 0.50
        mean_diff = 25.0
        has_structure_t1 = False
        if image_t1 is not None and image_t2 is not None:
            patch_ssim, mean_diff, has_structure_t1, _ = _crop_and_compare_patches(image_t1, image_t2, box2)

        # 2. Structural change map overlap
        change_overlap = 0.0
        nuisance_overlap = 0.0
        is_water = False

        if NUMPY_AVAILABLE and np is not None:
            w, h = img_width, img_height
            x1 = max(0, min(int(box2[0]), w - 1))
            y1 = max(0, min(int(box2[1]), h - 1))
            x2 = max(x1 + 1, min(int(box2[2]), w))
            y2 = max(y1 + 1, min(int(box2[3]), h))
            box_area = float(max(1, (x2 - x1) * (y2 - y1)))

            if structural_change_map is not None:
                sub_chg = structural_change_map[y1:y2, x1:x2]
                change_overlap = float(np.sum(sub_chg > 0)) / box_area

            if nuisance_masks is not None:
                comb_nuis = nuisance_masks.get("combined_nuisance_mask")
                if comb_nuis is not None:
                    sub_nuis = comb_nuis[y1:y2, x1:x2]
                    nuisance_overlap = float(np.sum(sub_nuis > 0)) / box_area
                w_mask = nuisance_masks.get("water_mask")
                if w_mask is not None:
                    sub_water = w_mask[y1:y2, x1:x2]
                    is_water = bool((float(np.sum(sub_water > 0)) / box_area) > 0.40)

        # -------------------------------------------------------------
        # DROPOUT RECOVERY:
        # If T1 already physically contains the object (patch SSIM > 0.80, negligible change, and T1 has structure),
        # Grounding DINO simply dropped the detection in T1.
        # DO NOT report a false "NEW BUILDING"! Reclassify as PERSISTED.
        # -------------------------------------------------------------
        if patch_ssim > 0.80 and change_overlap < 0.15 and has_structure_t1:
            logger.info(
                f"[Matcher] Detection dropout recovered: '{label}' in T2 matches T1 crop (SSIM: {patch_ssim:.2f}). "
                "Classifying as PERSISTED rather than false NEW."
            )
            item = {
                "id": f"persisted-rec-{len(persisted)+1}",
                "type": "persisted",
                "label": label,
                "confidence": round(d2_conf * 0.90, 2),
                "verificationStatus": "CONFIRMED_UNCHANGED",
                "persistence": persistence,
                "box_t1": box2,
                "box_t2": box2,
                "boxT1": box2,
                "boxT2": box2,
                "current_box": box2,
                "currentBox": box2,
                "latitude": lat,
                "longitude": lon,
                "iou": 1.0,
                "area_ratio": 1.0,
                "location": quad,
                "details": f"Existing {label} confirmed in both images (pre-existing in T1; detection dropout resolved).",
                "description": f"Existing {label} confirmed present across both timestamps in {quad}.",
                "evidence": {
                    "patch_ssim_t1_t2": patch_ssim,
                    "change_overlap": round(change_overlap, 3),
                    "dropout_corrected": True,
                },
            }
            persisted.append(item)
            continue

        # Run Calibrated Confidence Engine
        temporal_evidence = max(0.0, min(1.0, (1.0 - patch_ssim) * 0.60 + change_overlap * 0.40))
        calib = calibrate_change_confidence(
            change_type="new",
            label=label,
            registration_quality=registration_quality,
            structural_change_score=change_overlap,
            object_detection_confidence=d2_conf,
            temporal_object_evidence=temporal_evidence,
            patch_ssim_t1_t2=patch_ssim,
            segmentation_quality=0.85,
            optical_score=0.85,
            sar_score=sar_score,
            has_sar=has_sar,
            nuisance_overlap_ratio=nuisance_overlap,
            is_water_dominated=is_water,
        )

        item = {
            "id": f"appeared-{len(appeared)+1}",
            "type": "appeared",
            "label": label,
            "confidence": calib["confidence"],
            "verificationStatus": calib["verification_status"],
            "persistence": calib["persistence"],
            "isGated": calib["is_gated"],
            "gateReason": calib["gate_reason"],
            "box_t1": None,
            "box_t2": box2,
            "boxT1": None,
            "boxT2": box2,
            "current_box": box2,
            "currentBox": box2,
            "latitude": lat,
            "longitude": lon,
            "iou": 0.0,
            "area_ratio": None,
            "location": quad,
            "details": f"{calib['verification_status'].replace('_', ' ').title()} - {calib['description']}",
            "description": calib["description"],
            "evidence": calib["evidence"],
            "penalties": calib["penalties"],
        }
        appeared.append(item)

    # -----------------------------------------------------------------
    # UNMATCHED T1 DETECTIONS -> VERIFY REMOVAL
    # -----------------------------------------------------------------
    disappeared: List[Dict[str, Any]] = []

    for i, d1 in enumerate(dets_t1):
        if i in matched_t1_indices:
            continue

        box1 = d1.get("box") or d1.get("bbox")
        if not box1 or len(box1) < 4:
            continue

        label = d1.get("label", "object")
        d1_conf = float(d1.get("confidence") or d1.get("score") or 0.8)
        quad = get_quadrant_location(box1, img_width, img_height)
        lat, lon = pixel_to_geo_coords(box1, img_width, img_height, aoi)
        persistence = classify_persistence(label)

        patch_ssim = 0.50
        has_structure_t2 = False
        if image_t1 is not None and image_t2 is not None:
            patch_ssim, _, _, has_structure_t2 = _crop_and_compare_patches(image_t1, image_t2, box1)

        # If T2 patch is identical to T1 patch and has structure, it was not removed!
        if patch_ssim > 0.80 and has_structure_t2:
            item = {
                "id": f"persisted-rec-{len(persisted)+1}",
                "type": "persisted",
                "label": label,
                "confidence": round(d1_conf * 0.90, 2),
                "verificationStatus": "CONFIRMED_UNCHANGED",
                "persistence": persistence,
                "box_t1": box1,
                "box_t2": box1,
                "boxT1": box1,
                "boxT2": box1,
                "current_box": box1,
                "currentBox": box1,
                "latitude": lat,
                "longitude": lon,
                "iou": 1.0,
                "area_ratio": 1.0,
                "location": quad,
                "details": f"Persistent {label} confirmed in both scenes (DINO missed in T2; patch confirmed).",
                "description": f"Persistent {label} verified across both dates in {quad}.",
            }
            persisted.append(item)
            continue

        calib = calibrate_change_confidence(
            change_type="removed",
            label=label,
            registration_quality=registration_quality,
            structural_change_score=0.70,
            object_detection_confidence=d1_conf,
            temporal_object_evidence=max(0.0, min(1.0, 1.0 - patch_ssim)),
            patch_ssim_t1_t2=patch_ssim,
            has_sar=has_sar,
            sar_score=sar_score,
        )

        item = {
            "id": f"disappeared-{len(disappeared)+1}",
            "type": "disappeared",
            "label": label,
            "confidence": calib["confidence"],
            "verificationStatus": calib["verification_status"],
            "persistence": calib["persistence"],
            "box_t1": box1,
            "box_t2": None,
            "boxT1": box1,
            "boxT2": None,
            "current_box": box1,
            "currentBox": box1,
            "latitude": lat,
            "longitude": lon,
            "iou": 0.0,
            "area_ratio": None,
            "location": quad,
            "details": f"Demolition / disappearance: {label} present at {date_t1} is no longer observed at {date_t2}.",
            "description": f"Removal or demolition of {label} observed between {date_t1} and {date_t2}.",
            "evidence": calib["evidence"],
            "penalties": calib["penalties"],
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
        "appeared": appeared,
        "disappeared": disappeared,
        "persisted": persisted,
        "possibly_changed": possibly_changed,
        "all_items": all_items,
        "summary": summary,
    }
