"""
Optical + SAR Imagery Fusion Module
Combines optical multispectral RGB imagery with Synthetic Aperture Radar (SAR) backscatter.
Supports cloud-penetrating structural enhancement via Intensity-Hue-Saturation (IHS) fusion
and high-frequency backscatter edge injection.

Works seamlessly in both full scientific (OpenCV/NumPy) and lightweight serverless (pure PIL) environments.
"""

import math
import logging
from typing import Optional, Dict, Any, Tuple
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

logger = logging.getLogger("satquery.temporal.fusion")

# Optional NumPy / OpenCV acceleration
NUMPY_AVAILABLE = False
CV2_AVAILABLE = False
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


def normalize_sar_backscatter(sar_image: Image.Image) -> Image.Image:
    """
    Normalizes radar backscatter intensity into standard 8-bit dynamic range [0, 255].
    Applies percentile contrast stretching (2% - 98%) to eliminate speckle outliers.
    """
    gray_sar = sar_image.convert("L")

    if NUMPY_AVAILABLE and np is not None:
        arr = np.array(gray_sar, dtype=np.float32)
        p2 = float(np.percentile(arr, 2))
        p98 = float(np.percentile(arr, 98))
        if p98 > p2:
            stretched = np.clip((arr - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(stretched, mode="L")

    # Pure PIL fallback: Autocontrast with 2% cutoff
    try:
        return ImageOps.autocontrast(gray_sar, cutoff=2)
    except Exception:
        return gray_sar


def fuse_optical_and_sar(
    optical_image: Optional[Image.Image],
    sar_image: Optional[Image.Image],
    sar_weight: float = 0.30,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Fuses Optical and SAR imagery into a single comprehensive representation.

    Modes:
    1. Both Optical and SAR:
       - Resizes SAR to Optical dimensions.
       - Normalizes SAR backscatter.
       - Merges via Intensity-Hue-Saturation (IHS/HSV) fusion:
         H & S are preserved from Optical (preserving natural spectral colors),
         while V (Value) is modulated with SAR backscatter:
         V_fused = (1 - sar_weight) * V_opt + sar_weight * SAR_norm
       - Injects high-frequency structural edges from SAR to highlight metal,
         buildings, bridges, and vessels even under hazy or cloudy conditions.
    2. Optical only:
       - Returns Optical RGB image directly.
    3. SAR only:
       - Normalizes SAR and converts to 3-channel RGB representation.
    """
    if optical_image is None and sar_image is None:
        raise ValueError("At least one of optical_image or sar_image must be provided.")

    # Mode 2: Optical Only
    if sar_image is None:
        opt_rgb = optical_image.convert("RGB")
        if target_size:
            opt_rgb = opt_rgb.resize(target_size, Image.Resampling.LANCZOS)
        return opt_rgb, {
            "modality": "optical_only",
            "has_optical": True,
            "has_sar": False,
            "fusion_method": "none",
            "width": opt_rgb.width,
            "height": opt_rgb.height,
        }

    # Mode 3: SAR Only
    if optical_image is None:
        norm_sar = normalize_sar_backscatter(sar_image)
        if target_size:
            norm_sar = norm_sar.resize(target_size, Image.Resampling.LANCZOS)
        sar_rgb = norm_sar.convert("RGB")
        return sar_rgb, {
            "modality": "sar_only",
            "has_optical": False,
            "has_sar": True,
            "fusion_method": "grayscale_expansion",
            "width": sar_rgb.width,
            "height": sar_rgb.height,
        }

    # Mode 1: Optical + SAR Multi-modal Fusion
    opt_rgb = optical_image.convert("RGB")
    w, h = target_size if target_size else opt_rgb.size
    opt_resized = opt_rgb.resize((w, h), Image.Resampling.LANCZOS)

    norm_sar = normalize_sar_backscatter(sar_image).resize((w, h), Image.Resampling.LANCZOS)

    # Perform HSV-based fusion
    # Convert optical to HSV
    opt_hsv = opt_resized.convert("HSV")
    h_chan, s_chan, v_chan = opt_hsv.split()

    weight = max(0.05, min(0.60, float(sar_weight)))

    # Blend Value channel with SAR backscatter
    # V_fused = (1 - weight) * V_optical + weight * SAR_norm
    blended_v = Image.blend(v_chan, norm_sar, alpha=weight)

    # High-frequency structural detail injection:
    # High-pass filter on SAR to capture corner reflections (bridges, roofs, ships)
    sar_detail = norm_sar.filter(ImageFilter.FIND_EDGES)
    sar_detail_dim = ImageEnhance.Brightness(sar_detail).enhance(0.40)
    enhanced_v = Image.blend(blended_v, sar_detail_dim, alpha=0.15)

    fused_hsv = Image.merge("HSV", (h_chan, s_chan, enhanced_v))
    fused_rgb = fused_hsv.convert("RGB")

    metadata = {
        "modality": "optical+sar",
        "has_optical": True,
        "has_sar": True,
        "fusion_method": "ihs_backscatter_injection",
        "sar_weight": weight,
        "width": w,
        "height": h,
    }

    logger.info(
        f"[Multimodal Fusion] Successfully fused Optical and SAR ({w}x{h}, weight={weight:.2f})."
    )

    return fused_rgb, metadata

