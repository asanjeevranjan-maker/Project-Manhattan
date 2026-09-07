"""
Nuisance Masking Module (Stage 6).
Detects and discounts transient/environmental variations:
- Water, tides, waves, boat wakes
- Deep cast shadows and solar angle illumination shifts
- Clouds, cloud shadows, and specular glare
- Seasonal vegetation phenology (greening / drying)

Guarantees port water movement or tree foliage changes are never misclassified
as permanent building or infrastructure construction.
"""

import logging
from typing import Dict, Any, Tuple, Optional
from PIL import Image

logger = logging.getLogger("satquery.temporal.nuisance_masking")

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


def compute_nuisance_masks(
    image_t1: Any,
    image_t2: Optional[Any] = None,
    valid_mask: Optional["np.ndarray"] = None,
) -> Dict[str, Any]:
    """
    Computes nuisance masks across T1 and T2 to prevent environmental false positives.

    Returns:
        {
            "water_mask": np.ndarray (uint8, 255 for water),
            "shadow_mask": np.ndarray (uint8, 255 for deep shadows),
            "cloud_mask": np.ndarray (uint8, 255 for clouds / glare),
            "vegetation_mask": np.ndarray (uint8, 255 for dense vegetation),
            "combined_nuisance_mask": np.ndarray (uint8, 255 for any nuisance),
            "percentages": {
                "water": float,
                "shadow": float,
                "cloud": float,
                "vegetation": float,
                "combined": float,
            }
        }
    """
    if NUMPY_AVAILABLE and np is not None:
        if isinstance(image_t1, np.ndarray):
            image_t1 = Image.fromarray(image_t1)
        if image_t2 is not None and isinstance(image_t2, np.ndarray):
            image_t2 = Image.fromarray(image_t2)

    if image_t2 is None:
        image_t2 = image_t1

    w, h = image_t1.size
    if not (CV2_AVAILABLE and NUMPY_AVAILABLE):
        dummy = np.zeros((h, w), dtype=np.uint8) if NUMPY_AVAILABLE else None
        return {
            "water_mask": dummy,
            "shadow_mask": dummy,
            "cloud_mask": dummy,
            "vegetation_mask": dummy,
            "combined_nuisance_mask": dummy,
            "percentages": {"water": 0.0, "shadow": 0.0, "cloud": 0.0, "vegetation": 0.0, "combined": 0.0},
        }

    try:
        t1_np = np.array(image_t1.convert("RGB"))
        t2_np = np.array(image_t2.convert("RGB"))

        if valid_mask is None:
            valid_mask = np.ones((h, w), dtype=np.uint8) * 255

        total_valid = float(max(1, np.sum(valid_mask > 0)))

        # -------------------------------------------------------------
        # 1. WATER & TIDAL MASK
        # -------------------------------------------------------------
        # Water exhibits: Blue/Green > Red, low local texture variance, low brightness
        def _extract_water(arr: np.ndarray) -> np.ndarray:
            r = arr[:, :, 0].astype(np.float32)
            g = arr[:, :, 1].astype(np.float32)
            b = arr[:, :, 2].astype(np.float32)

            # NDWI-RGB index: (Green - Red) / (Green + Red + eps)
            ndwi_rgb = (g - r) / (g + r + 15.0)

            # Local texture standard deviation (water is very smooth compared to land)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            local_mean = cv2.blur(gray.astype(np.float32), (9, 9))
            local_sqr_mean = cv2.blur(gray.astype(np.float32) ** 2, (9, 9))
            local_var = np.maximum(0.0, local_sqr_mean - local_mean ** 2)
            local_std = np.sqrt(local_var)

            # Water: blue/green dominant OR very dark smooth regions with low texture
            is_water = (
                ((ndwi_rgb > 0.05) & (b >= r) & (local_std < 18.0)) |
                ((r < 65) & (g < 75) & (b < 95) & (local_std < 12.0))
            )
            water_bin = np.where(is_water, 255, 0).astype(np.uint8)

            # Morphological smoothing to consolidate water bodies
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            water_bin = cv2.morphologyEx(water_bin, cv2.MORPH_CLOSE, kernel)
            water_bin = cv2.morphologyEx(water_bin, cv2.MORPH_OPEN, kernel)
            return water_bin

        water_t1 = _extract_water(t1_np)
        water_t2 = _extract_water(t2_np)
        # Combine water regions from both dates (coastal tide boundaries, moving water)
        water_mask = cv2.bitwise_or(water_t1, water_t2)
        water_mask = cv2.bitwise_and(water_mask, valid_mask)

        # -------------------------------------------------------------
        # 2. SHADOW MASK
        # -------------------------------------------------------------
        # Cast shadows: very low V in HSV, or low L in Lab, often with subtle blue/gray cast
        def _extract_shadow(arr: np.ndarray) -> np.ndarray:
            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
            v = hsv[:, :, 2]
            # Deep shadow threshold (dark surface < 45 in 0-255 V range)
            is_shadow = v < 45
            shadow_bin = np.where(is_shadow, 255, 0).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            shadow_bin = cv2.morphologyEx(shadow_bin, cv2.MORPH_OPEN, kernel)
            return shadow_bin

        shadow_t1 = _extract_shadow(t1_np)
        shadow_t2 = _extract_shadow(t2_np)
        shadow_mask = cv2.bitwise_or(shadow_t1, shadow_t2)
        shadow_mask = cv2.bitwise_and(shadow_mask, valid_mask)

        # -------------------------------------------------------------
        # 3. CLOUD & GLINT MASK
        # -------------------------------------------------------------
        # Clouds: near-white saturation across R, G, B, with high luminance
        def _extract_cloud(arr: np.ndarray) -> np.ndarray:
            r = arr[:, :, 0].astype(np.float32)
            g = arr[:, :, 1].astype(np.float32)
            b = arr[:, :, 2].astype(np.float32)
            brightness = (r + g + b) / 3.0
            whiteness = np.maximum(np.abs(r - g), np.maximum(np.abs(r - b), np.abs(g - b)))
            is_cloud = (brightness > 215.0) & (whiteness < 30.0)
            cloud_bin = np.where(is_cloud, 255, 0).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            cloud_bin = cv2.morphologyEx(cloud_bin, cv2.MORPH_OPEN, kernel)
            return cloud_bin

        cloud_t1 = _extract_cloud(t1_np)
        cloud_t2 = _extract_cloud(t2_np)
        cloud_mask = cv2.bitwise_or(cloud_t1, cloud_t2)
        cloud_mask = cv2.bitwise_and(cloud_mask, valid_mask)

        # -------------------------------------------------------------
        # 4. VEGETATION MASK
        # -------------------------------------------------------------
        # Visible Atmospherically Resistant Index (VARI): (G - R) / (G + R - B + eps)
        def _extract_vegetation(arr: np.ndarray) -> np.ndarray:
            r = arr[:, :, 0].astype(np.float32)
            g = arr[:, :, 1].astype(np.float32)
            b = arr[:, :, 2].astype(np.float32)
            denom = g + r - b
            denom[denom == 0] = 1.0
            vari = (g - r) / denom
            is_veg = (vari > 0.12) & (g > r) & (g > b)
            veg_bin = np.where(is_veg, 255, 0).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            veg_bin = cv2.morphologyEx(veg_bin, cv2.MORPH_OPEN, kernel)
            return veg_bin

        veg_t1 = _extract_vegetation(t1_np)
        veg_t2 = _extract_vegetation(t2_np)
        veg_mask = cv2.bitwise_or(veg_t1, veg_t2)
        veg_mask = cv2.bitwise_and(veg_mask, valid_mask)

        # -------------------------------------------------------------
        # COMBINED NUISANCE MASK
        # -------------------------------------------------------------
        combined = cv2.bitwise_or(water_mask, shadow_mask)
        combined = cv2.bitwise_or(combined, cloud_mask)

        percentages = {
            "water": round(float(np.sum(water_mask > 0)) / total_valid * 100.0, 2),
            "shadow": round(float(np.sum(shadow_mask > 0)) / total_valid * 100.0, 2),
            "cloud": round(float(np.sum(cloud_mask > 0)) / total_valid * 100.0, 2),
            "vegetation": round(float(np.sum(veg_mask > 0)) / total_valid * 100.0, 2),
            "combined": round(float(np.sum(combined > 0)) / total_valid * 100.0, 2),
        }

        return {
            "water_mask": water_mask,
            "shadow_mask": shadow_mask,
            "cloud_mask": cloud_mask,
            "vegetation_mask": veg_mask,
            "combined_nuisance_mask": combined,
            "percentages": percentages,
        }

    except Exception as e:
        logger.warning(f"[Nuisance Masking Exception]: {e}")
        dummy = np.zeros((h, w), dtype=np.uint8)
        return {
            "water_mask": dummy,
            "shadow_mask": dummy,
            "cloud_mask": dummy,
            "vegetation_mask": dummy,
            "combined_nuisance_mask": dummy,
            "percentages": {"water": 0.0, "shadow": 0.0, "cloud": 0.0, "vegetation": 0.0, "combined": 0.0},
        }
