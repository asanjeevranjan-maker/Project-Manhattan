"""
Compact Vision Context Builder for Satellite Imagery
Synthesizes machine-generated Grounding DINO detections, SAM2 segmentation,
and mask-derived land-cover statistics into token-efficient LLM context.
"""

from typing import Dict, Any, Optional, List, Tuple, Union


def _pluralize(label: str, count: int) -> str:
    """Provides natural pluralization for common satellite objects."""
    if count == 1:
        if label.lower() == "water" or label.lower() == "water body":
            return "1 water region"
        return f"1 {label}"
    
    lbl = label.lower().strip()
    if lbl in ("water", "water body"):
        return f"{count} water regions"
    if lbl in ("vegetation", "forest"):
        return f"{count} {lbl} zones"
    if lbl.endswith("s") or lbl.endswith("sh") or lbl.endswith("ch"):
        return f"{count} {lbl}es"
    if lbl.endswith("y") and not lbl.endswith("ay") and not lbl.endswith("ey"):
        return f"{count} {lbl[:-1]}ies"
    return f"{count} {lbl}s"


def _calculate_spatial_distribution(
    boxes: List[List[float]],
    img_width: float,
    img_height: float,
) -> str:
    """
    Computes a concise, natural spatial description for a cluster of bounding boxes.
    Outputs expressions like 'mostly left side', 'near center-left', 'crossing center-right'.
    """
    if not boxes or img_width <= 0 or img_height <= 0:
        return "distributed across the scene"

    # Single large spanning feature (e.g. river, wide water body, linear transport corridor)
    if len(boxes) == 1:
        b = boxes[0]
        bw = abs(b[2] - b[0])
        bh = abs(b[3] - b[1])
        cx = (b[0] + b[2]) / 2.0 / img_width
        cy = (b[1] + b[3]) / 2.0 / img_height

        if bw / img_width > 0.45 or bh / img_height > 0.45:
            if cx >= 0.45:
                return "crossing center-right"
            if cx <= 0.55:
                return "crossing center-left"
            return "spanning across the scene"

        # Smaller single feature
        if cx < 0.35:
            loc_x = "left side"
        elif cx > 0.65:
            loc_x = "right side"
        elif cx < 0.5:
            loc_x = "center-left"
        elif cx > 0.5:
            loc_x = "center-right"
        else:
            loc_x = "center"

        if cy < 0.35:
            loc_y = "upper"
        elif cy > 0.65:
            loc_y = "lower"
        else:
            loc_y = ""

        if loc_y and loc_x != "center":
            return f"{loc_y} {loc_x}"
        return f"near {loc_x}" if not loc_y else f"{loc_y} {loc_x}"

    # Multiple objects: analyze centroid distribution
    cx_rel: List[float] = []
    cy_rel: List[float] = []
    for b in boxes:
        cx_rel.append(((b[0] + b[2]) / 2.0) / img_width)
        cy_rel.append(((b[1] + b[3]) / 2.0) / img_height)

    total = len(boxes)
    left_count = sum(1 for x in cx_rel if x < 0.30)
    right_count = sum(1 for x in cx_rel if x > 0.70)
    center_left_count = sum(1 for x in cx_rel if 0.25 <= x < 0.50)
    center_right_count = sum(1 for x in cx_rel if 0.50 <= x <= 0.75)
    center_count = sum(1 for x in cx_rel if 0.35 <= x <= 0.65)

    upper_count = sum(1 for y in cy_rel if y < 0.35)
    lower_count = sum(1 for y in cy_rel if y > 0.65)

    # Predominant horizontal placement
    if left_count / total >= 0.60:
        return "mostly left side"
    if right_count / total >= 0.60:
        return "mostly right side"
    if center_left_count / total >= 0.60:
        return "near center-left"
    if center_right_count / total >= 0.60:
        return "near center-right"
    if center_count / total >= 0.60:
        return "concentrated in center"

    # Predominant vertical placement
    if upper_count / total >= 0.65:
        return "concentrated in upper section"
    if lower_count / total >= 0.65:
        return "concentrated in lower section"

    return "distributed across the scene"


def build_vision_context(
    detections: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
    segmentation_summary: Optional[Dict[str, Any]] = None,
    land_cover: Optional[Dict[str, Any]] = None,
    change_detection: Optional[Dict[str, Any]] = None,
    image_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """
    Builds a compact, token-efficient machine evidence summary for LLMs.
    Ensures models treat detector outputs as ground-truth evidence without hallucinating.

    Returns:
    {
      "summary_text": str (formatted prompt-ready text),
      "detected_objects_summary": List[str],
      "land_cover_summary": List[str],
      "segmentation_summary": Optional[str],
      "change_summary": Optional[str],
      "object_counts": Dict[str, int],
      "bbox_distribution": Dict[str, str],
      "evidence_flags": {
          "detections_used": bool,
          "segmentation_used": bool,
          "land_cover_used": bool,
          "change_detection_used": bool,
      }
    }
    """
    # 1. Image dimensions
    img_w, img_h = 1000.0, 1000.0
    if image_size and len(image_size) == 2:
        img_w, img_h = float(image_size[0]), float(image_size[1])
    elif isinstance(detections, dict):
        img_w = float(detections.get("width", 1000) or 1000)
        img_h = float(detections.get("height", 1000) or 1000)

    # 2. Extract detection items
    det_list: List[Dict[str, Any]] = []
    if isinstance(detections, list):
        det_list = detections
    elif isinstance(detections, dict):
        det_list = detections.get("detections", [])

    # Group detections by normalized label
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for d in det_list:
        lbl = str(d.get("label") or "object").strip().lower()
        by_label.setdefault(lbl, []).append(d)

    object_counts: Dict[str, int] = {}
    bbox_distribution: Dict[str, str] = {}
    detected_lines: List[str] = []

    for lbl, items in sorted(by_label.items(), key=lambda x: len(x[1]), reverse=True):
        cnt = len(items)
        object_counts[lbl] = cnt

        # Extract boxes
        boxes: List[List[float]] = []
        for it in items:
            b = it.get("box") or it.get("bbox")
            if b and len(b) >= 4:
                boxes.append([float(v) for v in b[:4]])

        dist_str = _calculate_spatial_distribution(boxes, img_w, img_h)
        bbox_distribution[lbl] = dist_str

        plural_phrase = _pluralize(lbl, cnt)
        detected_lines.append(f"- {plural_phrase}, {dist_str}")

    detections_used = len(det_list) > 0

    # 3. Extract Land Cover
    land_cover_lines: List[str] = []
    land_cover_used = False

    if isinstance(land_cover, dict):
        lc_meta = land_cover.get("land_cover")
        if isinstance(lc_meta, dict):
            # Dict form: {"built_up": {"percentage": 31.2}, "vegetation": ...}
            items_to_sort: List[Tuple[str, float]] = []
            for k, val in lc_meta.items():
                if isinstance(val, dict) and "percentage" in val:
                    pct = float(val["percentage"])
                    items_to_sort.append((k.replace("_", "-"), pct))

            items_to_sort.sort(key=lambda x: x[1], reverse=True)
            for cat_name, pct in items_to_sort:
                if pct > 0.05:  # Only report meaningful percentages
                    land_cover_lines.append(f"- {cat_name}: {pct:.1f}%")

            if land_cover_lines:
                land_cover_used = bool(land_cover.get("measured_from_masks", True))

        elif "coverage" in land_cover and isinstance(land_cover["coverage"], list):
            # Coverage array form: [{"class": "vegetation", "coverage": 0.448}, ...]
            cov_items = sorted(
                land_cover["coverage"],
                key=lambda x: x.get("coverage", 0.0),
                reverse=True,
            )
            for item in cov_items:
                c_name = item.get("class", "other")
                c_val = float(item.get("coverage", 0.0))
                pct = c_val * 100.0 if c_val <= 1.0 else c_val
                if pct > 0.05:
                    land_cover_lines.append(f"- {c_name}: {pct:.1f}%")

            if land_cover_lines:
                land_cover_used = bool(land_cover.get("measured_from_masks", True))

    # 4. Extract Segmentation Summary
    seg_summary_text: Optional[str] = None
    segmentation_used = False
    if isinstance(segmentation_summary, dict) and segmentation_summary.get("segmentation_available"):
        seg_cnt = segmentation_summary.get("segmented_count", 0)
        classes = segmentation_summary.get("classes_segmented", [])
        if seg_cnt > 0:
            cls_str = ", ".join(classes) if classes else "target objects"
            seg_summary_text = f"SAM2 instance segmentation active: {seg_cnt} masks generated ({cls_str})."
            segmentation_used = True
        else:
            seg_summary_text = "SAM2 instance segmentation active (0 high-confidence masks generated)."
    elif any("mask" in d for d in det_list):
        mask_cnt = sum(1 for d in det_list if d.get("mask"))
        if mask_cnt > 0:
            seg_summary_text = f"Instance segmentation active: {mask_cnt} masks generated."
            segmentation_used = True

    # 5. Extract Change Detection Summary
    change_summary_lines: List[str] = []
    change_detection_used = False

    if isinstance(change_detection, dict):
        change_detection_used = True

        # Check for multi-modal bi-temporal result
        lc_deltas = change_detection.get("land_cover_change")
        if isinstance(lc_deltas, dict):
            change_summary_lines.append("Land-cover changes (T1 to T2):")
            for cat, delta in lc_deltas.items():
                if delta != 0.0:
                    sign = f"+{delta}" if delta > 0 else f"{delta}"
                    change_summary_lines.append(f"- {cat}: {sign} percentage points")

        objs_summary = change_detection.get("objects_summary") or change_detection.get("summary")
        if isinstance(objs_summary, dict):
            app_cnt = objs_summary.get("appeared_count", objs_summary.get("newCount", 0))
            dis_cnt = objs_summary.get("disappeared_count", objs_summary.get("removedCount", 0))
            per_cnt = objs_summary.get("persisted_count", objs_summary.get("unchangedCount", 0))
            chg_cnt = objs_summary.get("possibly_changed_count", objs_summary.get("modifiedCount", 0))
            change_summary_lines.append(
                f"Object dynamics: {app_cnt} appeared, {dis_cnt} disappeared, {chg_cnt} modified, {per_cnt} persisted."
            )

        # Detailed object changes by location
        objs_dict = change_detection.get("objects")
        if isinstance(objs_dict, dict):
            app_list = objs_dict.get("appeared", [])
            if app_list:
                loc_summary = ", ".join(f"{item.get('label')} in {item.get('location', 'scene')}" for item in app_list[:4])
                change_summary_lines.append(f"- Newly appeared objects: {loc_summary}")
            dis_list = objs_dict.get("disappeared", [])
            if dis_list:
                loc_summary = ", ".join(f"{item.get('label')} in {item.get('location', 'scene')}" for item in dis_list[:4])
                change_summary_lines.append(f"- Disappeared objects: {loc_summary}")

        # SAR radar scattering analysis
        sar_analysis = change_detection.get("sar_analysis")
        if isinstance(sar_analysis, dict) and sar_analysis.get("available"):
            radar_sum = sar_analysis.get("radar_summary")
            if radar_sum:
                change_summary_lines.append(f"SAR radar backscatter evidence: {radar_sum}")

        # Primary shift
        primary_shift = change_detection.get("primary_shift")
        if primary_shift:
            change_summary_lines.append(f"Primary shift: {primary_shift}")

        # Fallback to simple pixel change percentage if present
        change_pct = change_detection.get("changePercentage") or change_detection.get("change_percentage")
        if change_pct is not None and not lc_deltas:
            change_summary_lines.append(f"Radiometric change: {change_pct}% of surface area altered.")

    change_summary_text = "\n".join(change_summary_lines) if change_summary_lines else None

    # 6. Compose Structured Prompt Text
    prompt_lines: List[str] = []

    if detected_lines or land_cover_lines or seg_summary_text or change_summary_text:
        prompt_lines.append("MACHINE-GENERATED SATELLITE EVIDENCE:")
        prompt_lines.append("Treat detection and segmentation results as machine-generated evidence.")
        prompt_lines.append("Use them when answering the user's question.")
        prompt_lines.append("Do not invent exact object counts unless they come from supplied detection results.")
        prompt_lines.append("If visual interpretation disagrees with supplied detections, state uncertainty.")
        prompt_lines.append("")

        if detected_lines:
            prompt_lines.append("Detected objects:")
            prompt_lines.extend(detected_lines)
            prompt_lines.append("")

        if land_cover_lines:
            prompt_lines.append("Land cover:")
            prompt_lines.extend(land_cover_lines)
            prompt_lines.append("")

        if seg_summary_text:
            prompt_lines.append(seg_summary_text)
            prompt_lines.append("")

        if change_summary_text:
            prompt_lines.append(change_summary_text)
            prompt_lines.append("")

    summary_text = "\n".join(prompt_lines).strip()

    return {
        "summary_text": summary_text,
        "detected_objects_summary": detected_lines,
        "land_cover_summary": land_cover_lines,
        "segmentation_summary": seg_summary_text,
        "change_summary": change_summary_text,
        "object_counts": object_counts,
        "bbox_distribution": bbox_distribution,
        "evidence_flags": {
            "detections_used": detections_used,
            "segmentation_used": segmentation_used,
            "land_cover_used": land_cover_used,
            "change_detection_used": change_detection_used,
        },
    }
