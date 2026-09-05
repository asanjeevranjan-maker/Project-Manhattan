"""
Land-Cover Change & Mask Delta Analysis Module
Computes truthful, mask-measured percentage deltas between Time T1 and Time T2
for standard land-cover classes:
- water (expansion / reduction)
- built_up (urban growth / demolition)
- vegetation (greening / deforestation / loss)
- bare_soil (soil exposure / development)
- other

Identifies contiguous spatial change regions with quadrant-level localization.
"""

import math
from typing import Dict, Any, Optional, List, Tuple


def calculate_land_cover_deltas(
    land_cover_t1: Optional[Dict[str, Any]],
    land_cover_t2: Optional[Dict[str, Any]],
    detections_t1: Optional[List[Dict[str, Any]]] = None,
    detections_t2: Optional[List[Dict[str, Any]]] = None,
    image_size: Tuple[int, int] = (1000, 1000),
) -> Dict[str, Any]:
    """
    Computes objective percentage point deltas between T1 and T2 land-cover distributions.
    Extracts spatial change regions with qualitative localization (e.g. lower-right region).

    Returns:
    {
        "land_cover_change": {
            "water": float (e.g. +5.2),
            "built_up": float (e.g. +2.1),
            "vegetation": float (e.g. -4.6),
            "bare_soil": float (e.g. +0.8),
            "other": float (e.g. -3.5)
        },
        "coverage_t1": Dict[str, float],
        "coverage_t2": Dict[str, float],
        "change_regions": List[Dict[str, Any]],
        "significant_changes": List[str],
        "primary_shift": Optional[str],
    }
    """
    cats = ["water", "built_up", "vegetation", "bare_soil", "other"]

    # Extract percentage distributions
    cov1: Dict[str, float] = {}
    cov2: Dict[str, float] = {}

    if land_cover_t1 and "coverage" in land_cover_t1:
        c_dict = land_cover_t1["coverage"]
        for cat in cats:
            val = c_dict.get(cat, {})
            cov1[cat] = float(val.get("percentage", 0.0)) if isinstance(val, dict) else float(val or 0.0)
    else:
        # Default baseline if no masks
        cov1 = {"water": 0.0, "built_up": 0.0, "vegetation": 0.0, "bare_soil": 0.0, "other": 100.0}

    if land_cover_t2 and "coverage" in land_cover_t2:
        c_dict = land_cover_t2["coverage"]
        for cat in cats:
            val = c_dict.get(cat, {})
            cov2[cat] = float(val.get("percentage", 0.0)) if isinstance(val, dict) else float(val or 0.0)
    else:
        cov2 = dict(cov1)

    deltas: Dict[str, float] = {}
    for cat in cats:
        d = round(cov2.get(cat, 0.0) - cov1.get(cat, 0.0), 1)
        deltas[cat] = d

    # Analyze spatial location of changes using detection distribution
    img_w, img_h = float(image_size[0]), float(image_size[1])

    def _get_category_location(dets: List[Dict[str, Any]], category_label: str) -> str:
        matching = [
            d for d in dets
            if category_label in str(d.get("label", "")).lower() or
            (category_label == "built_up" and any(k in str(d.get("label", "")).lower() for k in ("building", "house", "structure", "road"))) or
            (category_label == "water" and any(k in str(d.get("label", "")).lower() for k in ("water", "river", "lake", "flood"))) or
            (category_label == "vegetation" and any(k in str(d.get("label", "")).lower() for k in ("vegetation", "tree", "forest", "field")))
        ]
        if not matching:
            return "distributed across scene"

        centers_x = []
        centers_y = []
        for d in matching:
            box = d.get("box") or d.get("bbox") or []
            if len(box) >= 4:
                centers_x.append((box[0] + box[2]) / 2.0 / img_w)
                centers_y.append((box[1] + box[3]) / 2.0 / img_h)

        if not centers_x:
            return "distributed across scene"

        avg_x = sum(centers_x) / len(centers_x)
        avg_y = sum(centers_y) / len(centers_y)

        horiz = "left" if avg_x < 0.40 else ("right" if avg_x > 0.60 else "central")
        vert = "upper" if avg_y < 0.40 else ("lower" if avg_y > 0.60 else "middle")

        if horiz == "central" and vert == "middle":
            return "center region"
        if horiz == "central":
            return f"{vert} region"
        if vert == "middle":
            return f"{horiz} side"
        return f"{vert}-{horiz} region"

    change_regions: List[Dict[str, Any]] = []
    significant_statements: List[str] = []

    # Sort categories by absolute magnitude of delta
    sorted_cats = sorted(cats, key=lambda c: abs(deltas[c]), reverse=True)

    for cat in sorted_cats:
        delta = deltas[cat]
        if abs(delta) < 0.5:
            continue

        action = "increased" if delta > 0 else "decreased"
        sign_str = f"+{delta}" if delta > 0 else f"{delta}"

        # Determine qualitative location from T2 detections if expanding, or T1 if receding
        relevant_dets = (detections_t2 or []) if delta > 0 else (detections_t1 or [])
        loc_str = _get_category_location(relevant_dets, cat)

        region_entry = {
            "category": cat,
            "delta": delta,
            "delta_display": f"{sign_str}%",
            "action": action,
            "location": loc_str,
            "description": f"{cat.replace('_', ' ').capitalize()} coverage {action} by {abs(delta):.1f} percentage points, concentrated primarily in the {loc_str}.",
        }
        change_regions.append(region_entry)
        significant_statements.append(
            f"{cat.replace('_', ' ').capitalize()} {action} by {abs(delta):.1f}% ({sign_str} pts in {loc_str})"
        )

    primary_shift = change_regions[0]["description"] if change_regions else "No major land-cover shift observed."

    return {
        "land_cover_change": deltas,
        "coverage_t1": cov1,
        "coverage_t2": cov2,
        "change_regions": change_regions,
        "significant_changes": significant_statements,
        "primary_shift": primary_shift,
    }
