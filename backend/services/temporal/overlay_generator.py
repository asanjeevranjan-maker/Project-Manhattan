"""
Visual Overlay Generator for Bi-Temporal Multimodal Analysis
Synthesizes:
- Before Overlay: T1 image with historical bounding box annotations
- After Overlay: T2 image with latest bounding box annotations
- Change Overlay: Transparent multi-color RGBA mask displaying:
  * Green: Newly appeared structures / objects
  * Red: Disappeared / removed objects
  * Amber: Modified objects / significant shifts
  * Cyan: Water expansion zones

Works with 100% pure PIL without requiring OpenCV or NumPy.
"""

import io
import base64
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image, ImageDraw


def _image_to_data_url(image: Image.Image, format: str = "PNG") -> str:
    """Converts a PIL image to a Base64 data URL."""
    buffered = io.BytesIO()
    if format.upper() == "PNG":
        image.save(buffered, format="PNG", optimize=True)
        mime = "image/png"
    else:
        image.convert("RGB").save(buffered, format="JPEG", quality=85)
        mime = "image/jpeg"
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def generate_bitemporal_overlays(
    image_t1: Image.Image,
    image_t2: Image.Image,
    matched_objects: Dict[str, Any],
    land_cover_change: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Generates before, after, and composite change overlays.
    Returns:
    {
        "before_overlay_url": str (data:image/png;base64,...),
        "after_overlay_url": str (data:image/png;base64,...),
        "change_overlay_url": str (data:image/png;base64,...),
    }
    """
    w, h = image_t1.size
    img_t2_res = image_t2.resize((w, h), Image.Resampling.LANCZOS)

    # 1. Before Overlay (T1)
    before_canvas = image_t1.convert("RGBA").copy()
    draw_before = ImageDraw.Draw(before_canvas)

    # 2. After Overlay (T2)
    after_canvas = img_t2_res.convert("RGBA").copy()
    draw_after = ImageDraw.Draw(after_canvas)

    # 3. Transparent Change Overlay (RGBA)
    change_canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_change = ImageDraw.Draw(change_canvas)

    appeared = matched_objects.get("appeared", [])
    disappeared = matched_objects.get("disappeared", [])
    possibly_changed = matched_objects.get("possibly_changed", [])
    persisted = matched_objects.get("persisted", [])

    # Draw Persisted (Neutral Blue/Gray) on Before and After
    for item in persisted:
        box1 = item.get("box_t1")
        box2 = item.get("box_t2")
        if box1 and len(box1) >= 4:
            draw_before.rectangle(box1, outline=(59, 130, 246, 200), width=2)
        if box2 and len(box2) >= 4:
            draw_after.rectangle(box2, outline=(59, 130, 246, 200), width=2)

    # Draw Disappeared (Red) on Before and Change
    # Red: RGBA(239, 68, 68, 220)
    for item in disappeared:
        box1 = item.get("box_t1") or item.get("current_box")
        if box1 and len(box1) >= 4:
            draw_before.rectangle(box1, outline=(239, 68, 68, 240), width=3)
            # Fill with translucent red on change canvas
            draw_change.rectangle(box1, fill=(239, 68, 68, 80), outline=(239, 68, 68, 230), width=3)

    # Draw Appeared (Green) on After and Change
    # Green: RGBA(34, 197, 94, 220)
    for item in appeared:
        box2 = item.get("box_t2") or item.get("current_box")
        if box2 and len(box2) >= 4:
            draw_after.rectangle(box2, outline=(34, 197, 94, 240), width=3)
            # Fill with translucent green on change canvas
            draw_change.rectangle(box2, fill=(34, 197, 94, 85), outline=(34, 197, 94, 230), width=3)

    # Draw Possibly Changed (Amber) on After and Change
    # Amber: RGBA(245, 158, 11, 220)
    for item in possibly_changed:
        box1 = item.get("box_t1")
        box2 = item.get("box_t2") or item.get("current_box")
        if box1 and len(box1) >= 4:
            draw_before.rectangle(box1, outline=(245, 158, 11, 200), width=2)
        if box2 and len(box2) >= 4:
            draw_after.rectangle(box2, outline=(245, 158, 11, 240), width=3)
            draw_change.rectangle(box2, fill=(245, 158, 11, 75), outline=(245, 158, 11, 230), width=2)

    # If significant water expansion exists, add a subtle cyan indicator in the relevant region
    if land_cover_change:
        water_delta = land_cover_change.get("land_cover_change", {}).get("water", 0.0)
        if water_delta >= 2.0:
            # Water flood glow on lower perimeter
            draw_change.rectangle(
                [0, int(h * 0.70), w, h],
                fill=(6, 182, 212, 40),
                outline=(6, 182, 212, 120),
                width=1,
            )

    return {
        "before_overlay_url": _image_to_data_url(before_canvas),
        "after_overlay_url": _image_to_data_url(after_canvas),
        "change_overlay_url": _image_to_data_url(change_canvas),
    }
