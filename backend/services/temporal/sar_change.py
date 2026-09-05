"""
Standalone SAR Differential Scattering Analysis Module
Analyzes changes in Synthetic Aperture Radar (SAR) backscatter between Time T1 and Time T2.

Cloud-Independent Intelligence:
- Bright corner scatterers added (+delta): Reveals new metallic / vertical structural additions,
  industrial installations, or vessels even under complete optical cloud cover.
- Dark specular scatterers added (-delta): Reveals new standing water / flood inundation
  where smooth water surfaces reflect microwave pulses specularly away from the sensor.
"""

import math
from typing import Optional, Dict, Any, Tuple
from PIL import Image, ImageChops

NUMPY_AVAILABLE = False
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore


def analyze_sar_differential_scattering(
    sar_t1: Optional[Image.Image],
    sar_t2: Optional[Image.Image],
    target_size: Optional[Tuple[int, int]] = None,
    threshold: int = 40,
) -> Dict[str, Any]:
    """
    Computes differential radar backscatter between aligned SAR images.

    Returns:
    {
        "available": bool,
        "total_pixels": int,
        "bright_scatterers_added_pixels": int,
        "bright_scatterers_percentage": float,
        "dark_scatterers_added_pixels": int,
        "dark_scatterers_percentage": float,
        "mean_backscatter_t1": float,
        "mean_backscatter_t2": float,
        "radar_detected_flood_inundation": bool,
        "radar_detected_new_structures": bool,
        "radar_summary": str,
    }
    """
    if sar_t1 is None or sar_t2 is None:
        return {
            "available": False,
            "reason": "SAR imagery not provided for both timestamps.",
            "radar_summary": "No dual-temporal SAR imagery available.",
            "bright_scatterers_percentage": 0.0,
            "dark_scatterers_percentage": 0.0,
        }

    # Normalize to matching dimensions and grayscale
    w, h = target_size if target_size else sar_t1.size
    g1 = sar_t1.convert("L").resize((w, h), Image.Resampling.LANCZOS)
    g2 = sar_t2.convert("L").resize((w, h), Image.Resampling.LANCZOS)

    total_pixels = w * h

    if NUMPY_AVAILABLE and np is not None:
        arr1 = np.array(g1, dtype=np.float32)
        arr2 = np.array(g2, dtype=np.float32)

        diff = arr2 - arr1
        mean1 = round(float(np.mean(arr1)), 1)
        mean2 = round(float(np.mean(arr2)), 1)

        bright_mask = diff >= threshold
        dark_mask = diff <= -threshold

        bright_pixels = int(np.sum(bright_mask))
        dark_pixels = int(np.sum(dark_mask))
    else:
        # Pure PIL fallback without deprecation warning
        if hasattr(g1, "get_flattened_data"):
            p1 = list(g1.get_flattened_data())
            p2 = list(g2.get_flattened_data())
        else:
            p1 = list(g1.getdata())
            p2 = list(g2.getdata())
        mean1 = round(sum(p1) / len(p1), 1)
        mean2 = round(sum(p2) / len(p2), 1)

        bright_pixels = 0
        dark_pixels = 0
        for v1, v2 in zip(p1, p2):
            d = v2 - v1
            if d >= threshold:
                bright_pixels += 1
            elif d <= -threshold:
                dark_pixels += 1

    bright_pct = round((bright_pixels / total_pixels) * 100.0, 2)
    dark_pct = round((dark_pixels / total_pixels) * 100.0, 2)

    has_flooding = bool(dark_pct >= 1.5)
    has_structures = bool(bright_pct >= 1.0)

    summary_items = []
    if has_structures:
        summary_items.append(
            f"Radar backscatter surge (+{bright_pct}%): High corner reflections indicate new metallic or concrete structural development."
        )
    if has_flooding:
        summary_items.append(
            f"Radar backscatter drop (-{dark_pct}%): Specular microwave absorption confirms standing water or flood inundation."
        )

    if not summary_items:
        summary_items.append("Radar scattering signature remained largely stable across both dates.")

    radar_summary = " ".join(summary_items)

    return {
        "available": True,
        "total_pixels": total_pixels,
        "bright_scatterers_added_pixels": bright_pixels,
        "bright_scatterers_percentage": bright_pct,
        "dark_scatterers_added_pixels": dark_pixels,
        "dark_scatterers_percentage": dark_pct,
        "mean_backscatter_t1": mean1,
        "mean_backscatter_t2": mean2,
        "radar_detected_flood_inundation": has_flooding,
        "radar_detected_new_structures": has_structures,
        "radar_summary": radar_summary,
    }
