"""
Land-Cover Coverage Analyzer for SatQuery.

Computes objective, pixel-accurate land-cover coverage from segmentation masks.
Eliminates hallucinated/guessed model estimates by calculating exact pixel percentages.

Key Architecture:
1. Priority-based multi-class rasterization: avoids double-counting overlapping masks.
2. Core land-cover categories:
   - water
   - built_up
   - vegetation
   - bare_soil (if present)
   - other (unassigned background)
3. Sum constraint: guarantees percentages total 100.00% within strict tolerance.
4. Truthful availability: returns {"available": False, "reason": "Segmentation unavailable"}
   if segmentation is missing or disabled, never fabricating estimates.
5. Overlay visualization: produces a color-coded transparent composite preview.
"""

import io
import math
import base64
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from PIL import Image, ImageDraw

logger = logging.getLogger("satquery.detection.land_cover")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")


# =====================================================================
# 1. CATEGORY MAPPINGS & PRIORITY HIERARCHY
# =====================================================================
# Priority hierarchy for resolving overlapping masks without double-counting.
# Water has highest priority (flood/water delineation is decisive),
# followed by built structures, vegetation, bare soil, and unassigned "other".
DEFAULT_CATEGORY_PRIORITY: List[str] = [
    "water",
    "built_up",
    "vegetation",
    "bare_soil",
    "other",
]

# Mapping from canonical observable detection labels to high-level land cover classes
LABEL_TO_CATEGORY: Dict[str, str] = {
    # Water categories
    "water": "water",
    "water body": "water",
    "river": "water",
    "lake": "water",
    "ocean": "water",
    "reservoir": "water",
    "flooded area": "water",
    "canal": "water",
    "stream": "water",
    "pond": "water",

    # Built-up / Impervious structures
    "building": "built_up",
    "house": "built_up",
    "structure": "built_up",
    "facility": "built_up",
    "industrial facility": "built_up",
    "road": "built_up",
    "highway": "built_up",
    "bridge": "built_up",
    "runway": "built_up",
    "vehicle": "built_up",
    "car": "built_up",
    "truck": "built_up",
    "boat": "built_up",
    "vessel": "built_up",
    "airplane": "built_up",
    "aircraft": "built_up",
    "construction": "built_up",
    "construction site": "built_up",

    # Vegetation / Green cover
    "vegetation": "vegetation",
    "tree": "vegetation",
    "forest": "vegetation",
    "field": "vegetation",
    "agricultural field": "vegetation",
    "grass": "vegetation",
    "park": "vegetation",
    "mangrove": "vegetation",
    "crop": "vegetation",

    # Bare soil / Earthworks
    "bare soil": "bare_soil",
    "bare land": "bare_soil",
    "sand": "bare_soil",
    "dirt": "bare_soil",
    "soil": "bare_soil",
}

# Hex and RGBA colors for land cover visualization
LAND_COVER_COLORS: Dict[str, str] = {
    "water": "#06b6d4",        # Cyan / Blue
    "built_up": "#f97316",     # Orange
    "vegetation": "#10b981",   # Emerald green
    "bare_soil": "#eab308",    # Amber / Gold
    "other": "#6b7280",        # Neutral gray
}

LAND_COVER_RGBA_COLORS: Dict[str, Tuple[int, int, int, int]] = {
    "water": (6, 182, 212, 140),
    "built_up": (249, 115, 22, 130),
    "vegetation": (16, 185, 129, 130),
    "bare_soil": (234, 179, 8, 120),
    "other": (107, 114, 128, 60),
}


def map_label_to_category(label: str) -> str:
    """Resolves any canonical or raw detection label to its core land-cover category."""
    if not label:
        return "other"
    lbl = label.strip().lower()
    if lbl in LABEL_TO_CATEGORY:
        return LABEL_TO_CATEGORY[lbl]

    # Substring / partial matching
    if any(k in lbl for k in ("water", "river", "lake", "flood", "ocean", "pond")):
        return "water"
    if any(k in lbl for k in ("build", "house", "road", "bridge", "vehic", "car", "truck", "struct", "construct")):
        return "built_up"
    if any(k in lbl for k in ("vegetat", "tree", "forest", "field", "crop", "grass")):
        return "vegetation"
    if any(k in lbl for k in ("soil", "sand", "dirt", "bare")):
        return "bare_soil"

    return "other"


# =====================================================================
# 2. LAND-COVER ANALYZER ENGINE
# =====================================================================
class LandCoverAnalyzer:
    """
    Analyzes instance segmentation masks to generate objective,
    pixel-counted land-cover statistics with zero double-counting.
    """
    def __init__(
        self,
        category_priority: Optional[List[str]] = None,
        max_overlay_dimension: int = 1280,
    ):
        self.priority = category_priority or DEFAULT_CATEGORY_PRIORITY
        self.max_overlay_dimension = max_overlay_dimension

        # Category ID mapping for rasterization (1..4, 0 is unassigned)
        self.category_ids = {
            "water": 1,
            "built_up": 2,
            "vegetation": 3,
            "bare_soil": 4,
        }
        self.id_to_category = {v: k for k, v in self.category_ids.items()}

    def _rasterize_detection_mask(
        self,
        mask_data: Any,
        bbox: Optional[List[float]],
        width: int,
        height: int,
    ) -> Optional[Image.Image]:
        """
        Rasterizes a detection mask (polygon, RLE, or bounding box fallback)
        into a binary PIL Image of mode '1' of dimension (width, height).
        """
        if width <= 0 or height <= 0:
            return None

        # 1. RLE format (exact pixel-level representation from SAM2)
        if isinstance(mask_data, dict) and "rle" in mask_data:
            rle = mask_data["rle"]
            counts = rle.get("counts", [])
            size = rle.get("size", [height, width])
            rh, rw = size[0], size[1]
            if counts and rh > 0 and rw > 0:
                flat = []
                val = False
                for c in counts:
                    flat.extend([val] * int(c))
                    val = not val
                # Ensure correct total size
                target_len = rh * rw
                if len(flat) < target_len:
                    flat.extend([False] * (target_len - len(flat)))
                elif len(flat) > target_len:
                    flat = flat[:target_len]

                # Convert flat list to byte buffer
                b_img = Image.new("1", (rw, rh), 0)
                b_img.putdata([1 if v else 0 for v in flat])
                if (rw, rh) != (width, height):
                    b_img = b_img.resize((width, height), Image.Resampling.NEAREST)
                return b_img

        # 2. Direct 2D array / list of lists (exact binary mask)
        if isinstance(mask_data, list) and mask_data and isinstance(mask_data[0], list):
            mh = len(mask_data)
            mw = len(mask_data[0]) if mh > 0 else 0
            if mh > 0 and mw > 0:
                flat = [1 if v else 0 for row in mask_data for v in row]
                b_img = Image.new("1", (mw, mh), 0)
                b_img.putdata(flat)
                if (mw, mh) != (width, height):
                    b_img = b_img.resize((width, height), Image.Resampling.NEAREST)
                return b_img

        # 3. Polygon format (contour approximation or vector coordinates)
        if isinstance(mask_data, dict) and "polygon" in mask_data:
            poly = mask_data["polygon"]
            if poly and len(poly) >= 3:
                img = Image.new("1", (width, height), 0)
                draw = ImageDraw.Draw(img)
                flat_poly = [(float(pt[0]), float(pt[1])) for pt in poly]
                draw.polygon(flat_poly, fill=1)
                return img

        # 4. Fallback to bounding box rectangle if available
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            left, right = max(0, min(x1, x2)), min(width, max(x1, x2))
            top, bottom = max(0, min(y1, y2)), min(height, max(y1, y2))
            if right > left and bottom > top:
                img = Image.new("1", (width, height), 0)
                draw = ImageDraw.Draw(img)
                # PIL rectangle coordinates are inclusive of both corners
                draw.rectangle([left, top, right - 1, bottom - 1], fill=1)
                return img

        return None

    def analyze(
        self,
        image_size: Tuple[int, int],
        detections: List[Dict[str, Any]],
        segmentation_available: bool = True,
        generate_visualization: bool = True,
        base_image: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """
        Calculates exact, non-overlapping land-cover coverage metrics.

        Parameters:
            image_size: (width, height) of the satellite image.
            detections: List of detection dictionaries (with optional 'mask' properties).
            segmentation_available: Whether SAM2 segmentation succeeded.
            generate_visualization: Whether to render an overlay visualization preview.
            base_image: Optional PIL Image to overlay colors on top of.

        Returns:
            Dict conforming to the required schema.
        """
        # 1. Truthful handling if segmentation is unavailable: NEVER fabricate estimates
        if not segmentation_available:
            return {
                "available": False,
                "reason": "Segmentation unavailable",
                "measured_from_masks": False,
                "estimated": False,
            }

        width, height = image_size
        total_pixels = width * height
        if total_pixels <= 0:
            return {
                "available": False,
                "reason": "Invalid image dimensions",
                "measured_from_masks": False,
                "estimated": False,
            }

        # Filter detections that contain valid mask data
        masked_detections = [
            d for d in (detections or [])
            if d.get("mask") is not None
        ]

        # 2. Master pixel canvas initialized to 0 ('other' / unassigned)
        # We use PIL mode 'L' (8-bit grayscale) where 0=other, 1=water, 2=built_up, 3=vegetation, 4=bare_soil
        canvas = Image.new("L", (width, height), 0)
        draw_canvas = ImageDraw.Draw(canvas)

        # 3. Process each category in strict priority order (prevents double-counting)
        category_pixel_counts: Dict[str, int] = {
            "water": 0,
            "built_up": 0,
            "vegetation": 0,
            "bare_soil": 0,
            "other": 0,
        }

        # Priority order: water -> built_up -> vegetation -> bare_soil
        for category in self.priority:
            if category == "other":
                continue

            cat_id = self.category_ids.get(category)
            if not cat_id:
                continue

            # Gather all detections mapped to this category
            cat_dets = [
                d for d in masked_detections
                if map_label_to_category(d.get("label") or d.get("raw_label") or "") == category
            ]
            if not cat_dets:
                continue

            # Combine all masks for this category into a single union mask
            cat_union = Image.new("1", (width, height), 0)
            for det in cat_dets:
                mask_data = det.get("mask")
                bbox = det.get("bbox") or det.get("box")
                r_mask = self._rasterize_detection_mask(mask_data, bbox, width, height)
                if r_mask:
                    # Bitwise OR with union
                    cat_union = Image.frombytes(
                        "1",
                        (width, height),
                        bytes(a | b for a, b in zip(cat_union.tobytes(), r_mask.tobytes()))
                    )

            # Apply priority assignment: only assign pixels where canvas is currently 0
            canvas_bytes = bytearray(canvas.tobytes())
            union_bytes = cat_union.tobytes()

            # Pack/unpack bits from cat_union
            # Since PIL '1' mode packs 8 pixels per byte, convert union to 'L' mode for fast byte indexing
            union_L = cat_union.convert("L").tobytes()
            assigned_count = 0

            for i in range(len(canvas_bytes)):
                # If union mask is positive (255 in 'L') and canvas is unassigned (0)
                if union_L[i] > 0 and canvas_bytes[i] == 0:
                    canvas_bytes[i] = cat_id
                    assigned_count += 1

            canvas = Image.frombytes("L", (width, height), bytes(canvas_bytes))
            category_pixel_counts[category] = assigned_count

        # 4. Count remaining unassigned pixels as 'other'
        assigned_total = sum(
            category_pixel_counts[c] for c in ("water", "built_up", "vegetation", "bare_soil")
        )
        other_pixels = max(0, total_pixels - assigned_total)
        category_pixel_counts["other"] = other_pixels

        # 5. Calculate percentages and enforce sum to exactly 100.00%
        percentages: Dict[str, float] = {}
        for c in ("built_up", "vegetation", "water", "bare_soil"):
            pct = round((category_pixel_counts[c] / float(total_pixels)) * 100.0, 2)
            percentages[c] = pct

        # 'other' percentage is the remainder to guarantee total = 100.00%
        other_pct = round(100.0 - sum(percentages.values()), 2)
        other_pct = max(0.0, other_pct)
        percentages["other"] = other_pct

        # 6. Generate overlay visualization
        overlay_visualization = None
        if generate_visualization:
            overlay_visualization = self._generate_visualization_preview(
                canvas=canvas,
                base_image=base_image,
                width=width,
                height=height,
            )

        # 7. Construct standard output structure
        land_cover_output = {
            "built_up": {
                "pixels": category_pixel_counts["built_up"],
                "percentage": percentages["built_up"],
            },
            "vegetation": {
                "pixels": category_pixel_counts["vegetation"],
                "percentage": percentages["vegetation"],
            },
            "water": {
                "pixels": category_pixel_counts["water"],
                "percentage": percentages["water"],
            },
            "other": {
                "pixels": category_pixel_counts["other"],
                "percentage": percentages["other"],
            },
        }

        # Include bare_soil if detected or configured
        if category_pixel_counts["bare_soil"] > 0:
            land_cover_output["bare_soil"] = {
                "pixels": category_pixel_counts["bare_soil"],
                "percentage": percentages["bare_soil"],
            }

        # Backward compatible coverage list for frontend
        coverage_list = [
            {"class": "built-up", "coverage": round(percentages["built_up"] / 100.0, 4), "color": LAND_COVER_COLORS["built_up"]},
            {"class": "vegetation", "coverage": round(percentages["vegetation"] / 100.0, 4), "color": LAND_COVER_COLORS["vegetation"]},
            {"class": "water", "coverage": round(percentages["water"] / 100.0, 4), "color": LAND_COVER_COLORS["water"]},
            {"class": "bare soil", "coverage": round(percentages["bare_soil"] / 100.0, 4), "color": LAND_COVER_COLORS["bare_soil"]},
            {"class": "other", "coverage": round(percentages["other"] / 100.0, 4), "color": LAND_COVER_COLORS["other"]},
        ]

        logger.info(
            f"[LandCover] Analysis complete. Built-up: {percentages['built_up']}%, "
            f"Vegetation: {percentages['vegetation']}%, Water: {percentages['water']}%, "
            f"Other: {percentages['other']}% (Total Pixels: {total_pixels})"
        )

        return {
            "available": True,
            "measured_from_masks": True,
            "estimated": False,
            "total_pixels": total_pixels,
            "land_cover": land_cover_output,
            "coverage": coverage_list,
            "overlay_visualization": overlay_visualization,
        }

    def _generate_visualization_preview(
        self,
        canvas: Image.Image,
        base_image: Optional[Image.Image],
        width: int,
        height: int,
    ) -> Optional[str]:
        """
        Creates a color-coded transparent preview of the land-cover classification.
        Constrains output to max_overlay_dimension for memory safety.
        """
        try:
            # Determine scaling factor
            scale = 1.0
            longest = max(width, height)
            if longest > self.max_overlay_dimension:
                scale = float(self.max_overlay_dimension) / float(longest)
                target_w = max(1, int(width * scale))
                target_h = max(1, int(height * scale))
                render_canvas = canvas.resize((target_w, target_h), Image.Resampling.NEAREST)
                if base_image:
                    bg = base_image.resize((target_w, target_h), Image.Resampling.BILINEAR).convert("RGBA")
                else:
                    bg = Image.new("RGBA", (target_w, target_h), (25, 30, 36, 255))
            else:
                render_canvas = canvas
                if base_image:
                    bg = base_image.copy().convert("RGBA")
                else:
                    bg = Image.new("RGBA", (width, height), (25, 30, 36, 255))

            # Build color-coded RGBA layer
            rw, rh = render_canvas.size
            canvas_bytes = render_canvas.tobytes()
            rgba_bytes = bytearray(rw * rh * 4)

            for i in range(rw * rh):
                val = canvas_bytes[i]
                cat = self.id_to_category.get(val, "other")
                color = LAND_COVER_RGBA_COLORS[cat]
                idx = i * 4
                rgba_bytes[idx] = color[0]
                rgba_bytes[idx + 1] = color[1]
                rgba_bytes[idx + 2] = color[2]
                rgba_bytes[idx + 3] = color[3]

            mask_layer = Image.frombytes("RGBA", (rw, rh), bytes(rgba_bytes))
            composite = Image.alpha_composite(bg, mask_layer)

            # Encode to PNG base64 data URL
            buf = io.BytesIO()
            composite.save(buf, format="PNG", optimize=True)
            encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"

        except Exception as e:
            logger.warning(f"[LandCover] Visualization preview error: {e}")
            return None


# =====================================================================
# 3. CONVENIENCE FUNCTION
# =====================================================================
def calculate_land_cover(
    image_size: Tuple[int, int],
    detections: List[Dict[str, Any]],
    segmentation_available: bool = True,
    generate_visualization: bool = True,
    base_image: Optional[Image.Image] = None,
    category_priority: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper to execute LandCoverAnalyzer and obtain
    objective pixel-counted land cover estimates.
    """
    analyzer = LandCoverAnalyzer(category_priority=category_priority)
    return analyzer.analyze(
        image_size=image_size,
        detections=detections,
        segmentation_available=segmentation_available,
        generate_visualization=generate_visualization,
        base_image=base_image,
    )
